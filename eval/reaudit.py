"""Re-apply the CURRENT contamination audit to finished runs, so every case is judged by the same rules.

  uv run --group eval python eval/reaudit.py [--sandboxed] eval/runs/<run> [...]
Updates status 'ok' <-> 'contaminated' and contamination_flags in results.jsonl; prints every change.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.run_claude_code import NETWORK, SUSPICIOUS  # noqa: E402


def flags_for(transcript: Path) -> list[str]:
    inputs = []
    for line in transcript.read_text().splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("type") == "assistant":
            inputs += [b.get("input") for b in e["message"].get("content", []) if b.get("type") == "tool_use"]
    blob = re.sub(r"/tmp/tsmcp-[a-z]-[\w]+(/home)?", "<work>", json.dumps(inputs).lower())
    return [pat for pat in SUSPICIOUS if re.search(pat, blob)]


def main() -> None:
    sandboxed = "--sandboxed" in sys.argv  # runs made with the Bash network sandbox: network attempts are not leaks
    for run in map(Path, [a for a in sys.argv[1:] if not a.startswith("--")]):
        p = run / "results.jsonl"
        rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        for r in rows:
            if r.get("arm") not in ("D", "E") or r.get("status") not in ("ok", "contaminated"):
                continue
            if r.get("audit_override"):  # a human looked at the transcript; keep that verdict
                continue
            t = run / "transcripts" / f"{r['case_id']}.jsonl"
            if not t.exists():
                continue
            new = flags_for(t)
            attempted = [f for f in new if f in NETWORK] if sandboxed else []
            new = [f for f in new if f not in attempted]
            if attempted:
                r["attempted_network"] = attempted
            status = "contaminated" if new else "ok"
            if status != r["status"] or new != r.get("contamination_flags"):
                print(f"{run.name} {r['case_id']}: {r['status']} {r.get('contamination_flags')} -> {status} {new}")
                r["status"], r["contamination_flags"] = status, new
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


if __name__ == "__main__":
    main()
