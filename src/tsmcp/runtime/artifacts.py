"""Local artifact storage for beat tables and other arrays. The model gets summaries and stable refs, not arrays."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path


class ArtifactStore:
    def __init__(self, root: Path | None = None):
        self._root = Path(root) if root else None
        if self._root:
            self._root.mkdir(parents=True, exist_ok=True)
        self._mem: dict[str, dict] = {}
        self._lock = threading.Lock()

    def put(self, kind: str, columns: list[str], rows: list[dict], meta: dict | None = None) -> str:
        body = {"kind": kind, "columns": list(columns), "rows": rows, "meta": meta or {}}
        raw = json.dumps(body, sort_keys=True, default=str).encode()
        ref = "art-" + hashlib.sha256(raw).hexdigest()[:12]
        with self._lock:
            if ref not in self._mem:
                self._mem[ref] = body
                if self._root:
                    (self._root / f"{ref}.json").write_bytes(raw)
        return ref

    def get(self, ref: str) -> dict | None:
        with self._lock:
            if ref in self._mem:
                return self._mem[ref]
        if self._root and ref.startswith("art-") and ref[4:].isalnum():
            p = self._root / f"{ref}.json"
            if p.exists():
                body = json.loads(p.read_text())
                with self._lock:
                    self._mem[ref] = body
                return body
        return None

    def summary(self, ref: str) -> dict | None:
        body = self.get(ref)
        if body is None:
            return None
        rows = body["rows"]
        acc = [r for r in rows if r.get("accepted") is True]
        reasons: dict[str, int] = {}
        for r in rows:
            if r.get("accepted") is False:
                reasons[r.get("reject_reason", "unspecified")] = (
                    reasons.get(r.get("reject_reason", "unspecified"), 0) + 1
                )
        return {
            "artifact_ref": ref,
            "kind": body["kind"],
            "columns": body["columns"],
            "n_rows": len(rows),
            "n_accepted": len(acc),
            "n_rejected": len(rows) - len(acc),
            "rejection_reasons": reasons,
            "meta": body["meta"],
        }

    def rows(self, ref: str, start: int = 0, limit: int = 50, rejected_only: bool = False) -> dict | None:
        """Authorized retrieval: exclusions are always visible (rejected rows are returned with their reason)."""
        body = self.get(ref)
        if body is None:
            return None
        rows = body["rows"]
        if rejected_only:
            rows = [r for r in rows if r.get("accepted") is False]
        start = max(0, int(start))
        limit = max(1, min(int(limit), 500))
        page = rows[start : start + limit]
        return {
            "artifact_ref": ref,
            "start": start,
            "returned": len(page),
            "total": len(rows),
            "rows": page,
            "truncated": start + len(page) < len(rows),
        }
