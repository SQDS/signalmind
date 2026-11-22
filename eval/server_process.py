"""Run a selected checkout's console entrypoint and report its import provenance.

Invoked with that checkout's venv Python. The imported package stays in sys.modules
when the console entrypoint runs, so the report describes the actual server process.
"""

from __future__ import annotations

import argparse
import importlib
import json
import runpy
import subprocess
import sys
from pathlib import Path


def server_info(root: Path) -> dict:
    root = root.expanduser().resolve()
    package = importlib.import_module("tsmcp")
    package_path = Path(package.__file__).resolve()
    if not package_path.is_relative_to(root):
        raise RuntimeError(f"tsmcp imported from {package_path}, outside server root {root}")
    sha = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    return {"server_root": str(root), "server_git_sha": sha, "server_package_path": str(package_path)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-root", type=Path, required=True)
    ap.add_argument("--provenance", type=Path)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("server_args", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    root = args.server_root.expanduser().resolve()
    info = server_info(root)
    if args.check_only:
        print(json.dumps(info))
        return
    if args.provenance is None:
        ap.error("--provenance is required when launching the server")
    args.provenance.write_text(json.dumps(info))
    entrypoint = root / ".venv" / "bin" / "tsmcp"
    server_args = args.server_args[1:] if args.server_args[:1] == ["--"] else args.server_args
    sys.argv = [str(entrypoint), *server_args]
    runpy.run_path(str(entrypoint), run_name="__main__")


if __name__ == "__main__":
    main()
