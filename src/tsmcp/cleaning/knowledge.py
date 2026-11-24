"""Role-scoped experience and versioned, validated skill suggestions. No registry entry executes model code."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from urllib import parse, request

import numpy as np
from pydantic import Field

from .common import digest, implementation_hash, locked, read, sha, write
from .contracts import Contract
from .quality_gate import OPERATIONS


class Skill(Contract):
    skill_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    role: Literal["repair", "review"]
    problem_type: str
    modalities: list[str] = Field(min_length=1)
    applicability: dict
    prerequisites: list[str]
    tools: dict[str, str] = Field(min_length=1)
    parameters: dict
    read_scope: str
    write_scope: str
    expected_effects: list[str]
    stop_conditions: list[str] = Field(min_length=1)
    acceptance_criteria: dict
    regression_suite: str
    known_failure_modes: list[str]
    sources: list[str] = Field(min_length=1)


class AlgorithmCard(Contract):
    title: str
    doi: str | None
    version: str
    source_refs: list[str] = Field(min_length=1)
    formula_or_steps: list[str] = Field(min_length=1)
    windows: str
    parameters: dict
    boundary_handling: str
    applicability: str
    implementation_differences: str
    validation_data: str
    limitations: list[str]


class Knowledge:
    def __init__(self, path):
        self.path = Path(path)
        self.registry = self.path / "skills.json"

    def record(self, role, episode):
        if role not in ("repair", "review", "shared"):
            raise ValueError("Unknown role namespace")
        key = digest(episode)
        p = self.path / role / (key + ".json")
        with locked(self.path):
            if not p.exists():
                write(p, dict(id=key, role=role, created=datetime.now(UTC).isoformat(), **episode))
        return key

    def freeze(self, root, offline=None):
        path = root / "knowledge-snapshot.json"
        if path.exists():
            snap = read(path)
            if digest(snap["content"]) != snap["id"]:
                raise ValueError("Changed knowledge snapshot")
            return snap
        if offline:
            frozen = read(Path(offline) / path.name)
            if digest(frozen["content"]) != frozen["id"]:
                raise ValueError("Changed replay knowledge")
        else:
            with locked(self.path):
                content = {}
                implementation = implementation_hash()
                for role in ("repair", "review"):
                    memory = [read(f) for f in (self.path / role).glob("*.json")]
                    skills = read(self.registry).values() if self.registry.exists() else []
                    content[role] = dict(
                        memory=sorted(memory, key=lambda e: (e["created"], e["id"]), reverse=True),
                        skills=[
                            s["card"]
                            for s in sorted(skills, key=lambda s: s["card"]["skill_id"])
                            if s["state"] == "active"
                            and s["card"]["role"] == role
                            and s["trial"]["implementation_hash"] == implementation
                        ],
                    )
            frozen = dict(id=digest(content), content=content)
        write(path, frozen)
        return frozen

    @staticmethod
    def context(frozen, role, task, trigger):
        branch = frozen["content"][role]
        return dict(
            snapshot=frozen["id"],
            memory=[r for r in branch["memory"] if r["modality"] == task and r["trigger"] == trigger][:2],
            skills=[s for s in branch["skills"] if task in s["modalities"] and s["problem_type"] == trigger][:2],
        )

    def propose(self, card):
        card = Skill.model_validate(card)
        if set(card.tools) - OPERATIONS:
            raise ValueError("Skill tool is not registered")
        key = f"{card.skill_id}@{card.version}"
        with locked(self.path):
            catalog = read(self.registry) if self.registry.exists() else {}
            if key in catalog:
                raise ValueError("Use a new skill version")
            catalog[key] = dict(state="proposed", card=card.model_dump(), history=["proposed"])
            write(self.registry, catalog)
        return key

    def transition(self, skill_id, version, state, *, workspace=None, version_id=None, benchmark=None):
        if not skill_id.replace("-", "").replace("_", "").isalnum() or not all(x.isdigit() for x in version.split(".")):
            raise ValueError("Invalid skill/version")
        key = f"{skill_id}@{version}"
        with locked(self.path):
            catalog = read(self.registry)
            item = catalog[key]
            old = item["state"]
            next_states = {
                "proposed": {"sandboxed"},
                "sandboxed": {"evaluated"},
                "evaluated": {"approved"},
                "approved": {"active"},
                "active": {"deprecated"},
                "deprecated": {"active"},
            }
            if state not in next_states[old]:
                raise ValueError("Invalid skill transition")
            if state == "sandboxed":
                if not workspace or not version_id or Path(version_id).name != version_id:
                    raise ValueError("Provide the isolated recipe trial workspace and version")
                trial = read(Path(workspace) / "versions" / version_id / "version.json")
                results = trial["results"]
                if (
                    not results
                    or trial["recipe"]["kind"] != "python"
                    or not trial["report"]["program_checks_passed"]
                    or not all(
                        r.get("operator_trace") and r.get("program_checks", {}).get("passed") for r in results.values()
                    )
                ):
                    raise ValueError("An executed and verified isolated recipe trial is required")
                actual = {o["op"] for r in results.values() for o in r["operations"]}
                actual_versions = {(o["op"], o["version"]) for r in results.values() for o in r["operations"]}
                if not actual or not actual_versions <= set(item["card"]["tools"].items()):
                    raise ValueError("Trial tools differ from skill")
                item["trial"] = dict(
                    recipe_hash=trial["recipe"]["hash"],
                    implementation_hash=trial["implementation_hash"],
                    version=version_id,
                )
            if state == "evaluated":
                item["validation"] = self.validate_skill(item, benchmark)
            if state == "active":
                for k, other in catalog.items():
                    if k != key and other["card"]["skill_id"] == skill_id and other["state"] == "active":
                        other["state"] = "deprecated"
                        other["history"].append("deprecated")
            item["state"] = state
            item["history"].append(state)
            write(self.registry, catalog)
        return item

    @staticmethod
    def validate_skill(item, benchmark):
        """Read reference arrays in the evaluator only, post hoc against verified sandbox artifacts."""
        from .engine import load_arrays

        if not benchmark:
            raise ValueError("Validation requires separate regression and holdout references")
        manifest = read(benchmark)
        base = Path(benchmark).resolve().parent
        groups = {"regression": set(), "holdout": set()}
        metrics = []
        if manifest["suite"] != item["card"]["regression_suite"]:
            raise ValueError("Unexpected regression suite")
        for entry in manifest["cases"]:
            root = (base / entry["workspace"]).resolve()
            ver = read(root / "versions" / entry["version_id"] / "version.json")
            if (
                ver["recipe"]["hash"] != item["trial"]["recipe_hash"]
                or ver["implementation_hash"] != item["trial"]["implementation_hash"]
                or not ver["report"]["program_checks_passed"]
            ):
                raise ValueError("Evaluation must use the same verified tool implementation")
            audited = read(root / "dataset_manifest.json")
            c = next(c for c in audited["cases"] if c["id"] == entry["case"])
            source_record = next((r for r in audited.get("records", []) if r.get("record_id") == c["record_id"]), {})
            subject = source_record.get("patient_key") or c.get("patient_key") or c["record_id"]
            if entry["subject"] != subject:
                raise ValueError("Subject differs from audited record")
            groups[entry["split"]].add(subject)
            artifact = ver["results"][c["id"]]
            source, t, y, valid, estimated = load_arrays(root, c, artifact)
            from .common import array_hash

            if array_hash(y, t, valid, estimated) != artifact["content_hash"]:
                raise ValueError("Changed evaluation artifact")
            path = (base / entry["reference"]).resolve()
            if (root / "public").resolve() in path.parents:
                raise ValueError("Hidden references must be outside worker inputs")
            if sha(path) != entry["reference_sha256"]:
                raise ValueError("Changed reference")
            with np.load(path, allow_pickle=False) as z:
                truth = z["clean"]
                bad = z["affected"].astype(bool)
            if truth.shape != y.shape or bad.shape != y.shape or not bad.any() or bad.all():
                raise ValueError("Incomplete independent reference")
            from .common import changed

            if not valid[bad].all():
                raise ValueError("Masked targets cannot count as recovered")
            gain = float(np.mean(abs(source[bad] - truth[bad])) - np.mean(abs(y[bad] - truth[bad])))
            _, _, _, original_valid, original_estimated = load_arrays(root, c)
            goodloss = float(
                (changed(source, y) | (valid != original_valid) | (estimated != original_estimated))[~bad].mean()
            )
            criteria = item["card"]["acceptance_criteria"]
            if not (
                gain > 0 and gain >= criteria["min_mae_gain"] and goodloss <= criteria["max_clean_change_fraction"]
            ):
                raise ValueError("Reference acceptance criteria failed")
            metrics.append(
                dict(
                    split=entry["split"],
                    reference_sha256=entry["reference_sha256"],
                    gain=gain,
                    goodloss=goodloss,
                    input=c["input_sha256"],
                    subject=entry["subject"],
                )
            )
        if (
            not all(groups.values())
            or groups["regression"] & groups["holdout"]
            or len({m["input"] for m in metrics}) != len(metrics)
        ):
            raise ValueError("Regression and holdout subjects and inputs must be independent")
        return dict(manifest=sha(benchmark), metrics=metrics)

    def paper(self, card):
        value = AlgorithmCard.model_validate(card).model_dump()
        key = digest(value)
        write(self.path / "research" / (key + ".json"), value)
        return key


def paper_search(query):
    url = "https://api.crossref.org/works?" + parse.urlencode(
        {"query.bibliographic": query, "rows": 5, "select": "DOI,title,URL,published,link"}
    )
    with request.urlopen(url, timeout=15) as response:
        return json.load(response)["message"]["items"]


def cli(args):
    store = Knowledge(args.library)
    if args.action == "search":
        return paper_search(args.query)
    if args.action == "card":
        return store.paper(read(args.input))
    if args.action == "propose":
        return str(store.propose(read(args.input)))
    if args.action == "trial":
        from .engine import execute_version, register_recipe

        workspace = args.workspace.resolve()
        recipe = register_recipe(workspace, source=args.input.read_text())
        return execute_version(
            workspace, recipe, read(workspace / "dataset_manifest.json")["cases"], stage="skill_trial"
        )
    return store.transition(
        args.skill,
        args.version,
        args.action,
        workspace=args.workspace,
        version_id=args.trial_version,
        benchmark=args.input,
    )
