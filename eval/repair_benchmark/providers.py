"""Episode transports. The official DS route and numerical workers have separate access to credentials."""

import time
from contextlib import contextmanager

from tsmcp.cleaning.common import locked, read, write


class EpisodeLimit(Exception):
    pass


class DeepSeek:
    mode = "live"
    endpoint = "https://api.deepseek.com/anthropic"

    def __init__(self, budget_root, episode_cap, *, experiment_budget=None):
        self.budget_root = budget_root
        self.episode_cap = episode_cap
        self.experiment_budget = experiment_budget
        self.calls = []
        self.models = []

    @contextmanager
    def session(self, folder, deadline, max_requests):
        from tsmcp.cleaning.agent import BudgetClient

        # Use the same lock, shared cap and uncertain-charge reserve as the production Reviewer.
        if not (self.budget_root / "api_budget.json").is_file():
            raise ValueError("Expected the existing shared DeepSeek budget ledger")
        with locked(self.budget_root):
            ledger = read(self.budget_root / "api_budget.json")
            if not 0 < self.episode_cap <= ledger["cap_usd"] <= 5:
                raise ValueError("Episode and shared budget caps must be within the run authorization")
            self.client = BudgetClient(folder, cap=ledger["cap_usd"], budget_root=self.budget_root, compact_tools=True)
            start = len(self.client.ledger["calls"])
            first = start
            if self.experiment_budget is not None:
                binding = dict(
                    shared_ledger=str(self.client.path.resolve()), cap_usd=self.episode_cap, model=self.client.model
                )
                if not self.experiment_budget.exists():
                    write(self.experiment_budget, dict(**binding, first_call=start))
                limit = read(self.experiment_budget)
                if any(limit.get(k) != v for k, v in binding.items()) or not 0 <= limit["first_call"] <= start:
                    self.client.close()
                    raise ValueError("Experiment budget or shared ledger changed")
                first = limit["first_call"]

            def debit():
                calls = self.client.ledger["calls"][start:]
                reserved = sum(
                    c.get("estimated_peak_usd", c["reserve_usd"]) for c in self.client.ledger["calls"][first:]
                )
                if len(calls) >= max_requests or reserved + self.client.RESERVE > self.episode_cap:
                    raise EpisodeLimit("provider_request_or_spend_limit")
                if time.monotonic() >= deadline:
                    raise TimeoutError("episode deadline")

            self.client.before_attempt = debit
            self.client.output_limit = 4096
            self.client.deadline = deadline
            try:
                yield self
            finally:
                self.calls = self.client.ledger["calls"][start:]
                self.client.close()

    def response(self, system, messages, tools, request_hash):
        try:
            r = self.client.request(messages, tools=tools, system=system)
        except RuntimeError as exc:
            if str(exc) == "api_budget_exhausted":
                raise EpisodeLimit(str(exc)) from exc
            raise
        self.last_id = r.id
        self.models.append(r.model)
        if (
            not r.model
            or not r.model.lower().startswith("deepseek-")
            or "flash" not in r.model.lower()
            or r.model != self.models[0]
        ):
            raise ValueError("Endpoint did not retain the requested DeepSeek Flash family and returned model")
        return [b.model_dump(exclude_none=True) for b in r.content if b.type in ("text", "tool_use")]

    def accounting(self):
        returned = [c for c in self.calls if c["status"] == "returned"]
        complete = len(returned) == len(self.calls)

        def tokens(field):
            return (
                sum(c["usage"][field] for c in returned)
                if complete and all(isinstance(c["usage"].get(field), int) for c in returned)
                else None
            )

        accounting = dict(
            model_requests=len(self.calls),
            input_tokens=tokens("input_tokens"),
            output_tokens=tokens("output_tokens"),
            reasoning_tokens=None,
            cached_input_tokens=tokens("cache_read_input_tokens"),
            usd_cost=sum(c["estimated_peak_usd"] for c in returned) if complete else None,
            unknown_fields=["provider_confirmed_charge", "reasoning_tokens"]
            + ([] if complete else ["failed_request_usage"]),
        )
        receipt = dict(
            provider="deepseek",
            base_url=self.endpoint,
            requested_model_id="deepseek-flash",
            returned_model_id=self.models[-1] if self.models else None,
            request_ids=[c["request_id"] for c in returned],
        )
        return (
            accounting,
            receipt,
            dict(
                endpoint_family="anthropic_messages",
                requested_model="deepseek-flash",
                estimated_price_usd_per_million=dict(input=0.30, cache_read=0.006, output=1.20),
                reported_charge_usd=None,
                transport_attempts=self.calls,
                returned_models=self.models,
                unknown_charge_reserve_usd=sum(c["reserve_usd"] for c in self.calls if c["status"] != "returned"),
                requested_controls=dict(max_tokens=4096, thinking="disabled"),
                effective_controls="not_reported",
            ),
        )


class Replay:
    mode = "replay"

    def __init__(self, path):
        self.frames = iter(read(path)["frames"])

    @contextmanager
    def session(self, folder, deadline, max_requests):
        yield self
        if next(self.frames, None) is not None:
            raise ValueError("Episode ended before consuming all recorded responses")

    def response(self, system, messages, tools, request_hash):
        frame = next(self.frames, None)
        if frame is None or frame["request_hash"] != request_hash:
            raise ValueError("Recorded request does not match the current prompt, tools or evidence")
        self.last_id = frame.get("request_id")
        return frame["content"]

    def accounting(self):
        return (
            dict(
                model_requests=0,
                input_tokens=0,
                output_tokens=0,
                reasoning_tokens=0,
                cached_input_tokens=0,
                usd_cost=0,
                unknown_fields=[],
            ),
            None,
            dict(transport="local_replay"),
        )
