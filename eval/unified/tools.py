"""Tool sets for the unified loop. Both expose `schemas` (Anthropic tool definitions) and async `call(name, args)`
returning (content_blocks, is_error).

PythonTools (Base arms): `python` runs code in a fresh sandboxed process (bubblewrap: read-only filesystem except the
case work directory and a private home, network unshared) with numpy/scipy/neurokit2/wfdb (the D environment);
`view_image` shows a PNG/JPEG from the work directory (vision models only).
SignalMindTools (Agent arms): the TS-MCP server in-process (the same tools, plugins, TaskSpec, ledger, verifier and
ts_submit contract as the MCP deployment); tool definitions come from the server's own tool list.
"""

from __future__ import annotations

import base64
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VENV_D = ROOT / "eval" / ".venv-d"
HOME = Path.home()
PY_TIMEOUT_S = 120
MAX_OUTPUT = 20000


def _clip(text: str) -> str:
    return text if len(text) <= MAX_OUTPUT else text[:MAX_OUTPUT] + f"\n[... {len(text) - MAX_OUTPUT} more characters]"


def sandbox_prefix(work: Path, fake_home: Path) -> list[str]:
    args = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", str(HOME), "--tmpfs", "/tmp",
            "--unshare-net", "--die-with-parent"]  # fmt: skip
    for p in (VENV_D, HOME / ".local" / "share" / "uv" / "python"):
        if p.exists():
            args += ["--ro-bind", str(p), str(p)]
    args += ["--bind", str(work), str(work), "--bind", str(fake_home), str(fake_home), "--setenv", "HOME", str(fake_home),
             "--setenv", "MPLBACKEND", "Agg", "--chdir", str(work)]  # fmt: skip
    return args + ["--"]


class PythonTools:
    def __init__(self, work: Path, vision: bool, isolate: bool = True):
        self.work, self.vision, self.isolate = Path(work), vision, isolate
        self.fake_home = self.work.parent / "home"
        self.fake_home.mkdir(exist_ok=True)
        self.schemas = [{
            "name": "python",
            "description": "Run Python 3 code in a sandbox (no network). numpy, scipy, matplotlib, neurokit2 and wfdb are "
            f"installed. Each call is a fresh process; files written to {self.work} persist between calls. Print what "
            f"you need; output is truncated at {MAX_OUTPUT} characters; time limit {PY_TIMEOUT_S} s per call.",
            "input_schema": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
        }]  # fmt: skip
        if vision:
            self.schemas.append({
                "name": "view_image",
                "description": f"Show an image file (PNG or JPEG) saved under {self.work}, e.g. a matplotlib figure.",
                "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
            })  # fmt: skip

    async def call(self, name: str, args: dict) -> tuple[list[dict], bool]:
        if name == "python":
            code = args.get("code")
            if not isinstance(code, str):
                return [{"type": "text", "text": "python needs a 'code' string"}], True
            cmd = [str(VENV_D / "bin" / "python"), "-I", "-"]
            if self.isolate:
                cmd = sandbox_prefix(self.work, self.fake_home) + cmd
            try:
                env = {"PATH": os.defpath, "HOME": str(self.fake_home), "MPLBACKEND": "Agg", "LANG": "C.UTF-8"}
                p = subprocess.run(
                    cmd, input=code, capture_output=True, text=True, timeout=PY_TIMEOUT_S, cwd=self.work, env=env
                )
            except subprocess.TimeoutExpired:
                return [{"type": "text", "text": f"timed out after {PY_TIMEOUT_S} s"}], True
            out = (p.stdout or "") + (("\n[stderr]\n" + p.stderr) if p.stderr else "")
            return [{"type": "text", "text": _clip(out) or "(no output)"}], p.returncode != 0
        if name == "view_image" and self.vision:
            path = (
                (self.work / args.get("path", "")).resolve()
                if not str(args.get("path", "")).startswith("/")
                else Path(args["path"]).resolve()
            )
            if not path.is_relative_to(self.work.resolve()) or not path.is_file():
                return [{"type": "text", "text": "image must be an existing file inside the work directory"}], True
            media = (
                "image/png"
                if path.suffix.lower() == ".png"
                else "image/jpeg"
                if path.suffix.lower() in (".jpg", ".jpeg")
                else None
            )
            if media is None or path.stat().st_size > 5_000_000:
                return [{"type": "text", "text": "only PNG/JPEG files up to 5 MB"}], True
            data = base64.b64encode(path.read_bytes()).decode()
            return [{"type": "image", "source": {"type": "base64", "media_type": media, "data": data}}], False
        return [{"type": "text", "text": f"unknown tool {name!r}"}], True

    async def close(self):
        return None


class SignalMindTools:
    """The TS-MCP server in-process through the MCP client (exactly what an MCP client would see)."""

    def __init__(self, work: Path, runtime_dir: Path, plugins: list[str], mode: str, task_spec: Path | None,
                 images: bool, run_id: str):  # fmt: skip
        self.args = dict(work=Path(work), runtime_dir=Path(runtime_dir), plugins=plugins, mode=mode, task_spec=task_spec,
                         images=images, run_id=run_id)  # fmt: skip
        self.schemas: list[dict] = []
        self.instructions = ""
        self._client = None
        self._cm = None

    async def open(self):
        from mcp import Client

        from tsmcp.core.store import SignalStore
        from tsmcp.server.app import build_server

        a = self.args
        server = build_server(SignalStore(allow_roots=[a["work"]]), images=a["images"], plugins=a["plugins"],
                              runtime=a["mode"], runtime_dir=str(a["runtime_dir"]),
                              task_spec=str(a["task_spec"]) if a["task_spec"] else None, run_id=a["run_id"])  # fmt: skip
        from tsmcp.server.app import INSTRUCTIONS

        rt = getattr(server, "runtime", None)
        self.instructions = INSTRUCTIONS + (rt.instructions() if rt is not None else "")  # what the server announces
        self._cm = Client(server)
        self._client = await self._cm.__aenter__()
        tools = (await self._client.list_tools()).tools
        self.schemas = [
            {"name": t.name, "description": t.description or "", "input_schema": t.input_schema} for t in tools
        ]
        return self

    async def call(self, name: str, args: dict) -> tuple[list[dict], bool]:
        try:
            r = await self._client.call_tool(name, args)
        except Exception as e:  # noqa: BLE001 - surfaced to the model as a tool error
            return [{"type": "text", "text": f"{type(e).__name__}: {e}"}], True
        blocks = []
        for c in r.content:
            if c.type == "text":
                blocks.append({"type": "text", "text": c.text})
            elif c.type == "image":
                blocks.append(
                    {"type": "image", "source": {"type": "base64", "media_type": c.mime_type, "data": c.data}}
                )
        return blocks or [{"type": "text", "text": "(empty)"}], bool(r.is_error)

    async def close(self):
        if self._cm is not None:
            await self._cm.__aexit__(None, None, None)


def copy_case(case_dir: Path, base: Path) -> Path:
    work = base / "work"
    work.mkdir(parents=True, exist_ok=True)
    shutil.copy(case_dir / "signal.npy", work / "signal.npy")
    return work
