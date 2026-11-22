"""Receive Responses SSE while passing only the complete native response to the existing driver.

Transport-only opt-in. Keep completed native items (including opaque encrypted
reasoning) in memory; never print or export stream deltas or reasoning contents.
"""

import json
import time

import httpx2


def stream_final_post(driver, body):
    completed_items = {}
    with driver._http.stream(
        "POST", driver.url, json={**body, "stream": True}, headers={"Authorization": f"Bearer {driver.key}"}
    ) as response:
        if not response.is_success:
            detail = response.read().decode(errors="replace").replace(driver.key, "[redacted]")[:300]
            raise RuntimeError(f"HTTP {response.status_code} from gateway: {detail}")
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            event = json.loads(data)
            kind = event.get("type")
            if kind == "response.output_item.done":
                completed_items[event["output_index"]] = event["item"]
            elif kind in ("response.completed", "response.incomplete", "response.failed"):
                final = event["response"]
                items = dict(enumerate(final.get("output") or []))
                items.update(completed_items)
                final["output"] = [items[i] for i in sorted(items)]
                return final
            elif kind == "error":
                message = str(event.get("message", "stream error")).replace(driver.key, "[redacted]")[:300]
                raise RuntimeError(message)
    raise RuntimeError("Responses stream ended without a terminal response")


def stream_final_post_with_reconnect(driver, body):
    """At most two transport-only retries; no tool action is executed before a terminal response."""
    for attempt in range(3):
        try:
            return stream_final_post(driver, body)
        except httpx2.RemoteProtocolError:
            driver._network_disconnects = getattr(driver, "_network_disconnects", 0) + 1
            if attempt == 2:
                raise
            driver._http.close()
            driver._http = httpx2.Client(
                timeout=300.0,
                transport=httpx2.HTTPTransport(retries=2, limits=httpx2.Limits(max_keepalive_connections=0)),
            )
            time.sleep(0.5 * (attempt + 1))
