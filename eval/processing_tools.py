"""Strict single-case tool environments for the processing benchmark.

Only system runtimes, the pinned numerical environment, one read-only public
case and one writable output directory are mounted. SignalMind's service also
gets its implementation source, never the benchmark's private/ or baseline code.
"""

from __future__ import annotations

import base64
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV = ROOT / "eval/.venv-processing"
PYTHON = ENV / "bin/python"
MAX_OUTPUT = 20000


def sandbox_command(case: Path, output: Path, *, agent: bool = False) -> list[str]:
    prefix = subprocess.check_output([str(PYTHON), "-c", "import sys;print(sys.base_prefix)"], text=True).strip()
    cmd = ["bwrap", "--unshare-net", "--unshare-pid", "--new-session", "--die-with-parent", "--clearenv"]
    for folder in ("/usr", "/lib", "/lib64", "/bin", "/sbin"):
        if Path(folder).exists():
            cmd += ["--ro-bind", folder, folder]
    cmd += ["--dir", "/etc"]
    for file in ("/etc/ld.so.cache", "/etc/fonts", "/etc/passwd", "/etc/group", "/etc/localtime"):
        if Path(file).exists():
            cmd += ["--ro-bind", file, file]
    cmd += [
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        "--tmpfs",
        "/tmp",
        "--ro-bind",
        str(ENV),
        str(ENV),
        "--ro-bind",
        prefix,
        prefix,
        "--ro-bind",
        str(case.resolve()),
        "/case",
        "--bind",
        str(output.resolve()),
        "/output",
        "--setenv",
        "PATH",
        "/usr/bin:/bin",
        "--setenv",
        "HOME",
        "/tmp",
        "--setenv",
        "MPLBACKEND",
        "Agg",
        "--setenv",
        "MPLCONFIGDIR",
        "/tmp/mpl",
        "--setenv",
        "LANG",
        "C.UTF-8",
        "--setenv",
        "OPENBLAS_NUM_THREADS",
        "1",
        "--setenv",
        "OMP_NUM_THREADS",
        "1",
        "--setenv",
        "MKL_NUM_THREADS",
        "1",
        "--chdir",
        "/case",
    ]
    interpreter_alias = PYTHON.readlink().parent.parent
    if str(interpreter_alias) != prefix:
        cmd += ["--ro-bind", str(interpreter_alias), str(interpreter_alias)]
    if agent:
        cmd += [
            "--ro-bind",
            str(ROOT / "src"),
            "/opt/signalmind/src",
            "--ro-bind",
            str(ROOT / "eval/processing_worker.py"),
            "/opt/signalmind/worker.py",
        ]
    return cmd + ["--"]


def clipped(text: str) -> str:
    return text if len(text) <= MAX_OUTPUT else text[:MAX_OUTPUT] + f"\n[truncated {len(text) - MAX_OUTPUT} characters]"


class ProcessingPythonTools:
    def __init__(self, case: Path, output: Path):
        self.case, self.output = case, output
        self.command = sandbox_command(case, output) + [str(PYTHON), "-I", "-"]
        self.schemas = [
            {
                "name": "python",
                "description": "Run Python in a fresh network-isolated process. Pinned numpy,scipy,matplotlib "
                "are available. Read current case at /case; save all work/output under /output. Input is read-only. "
                "Files under /output persist across calls. To import the public helper, insert /case into sys.path. "
                "Print useful results. Each call has120s wall limit and20000 output characters.",
                "input_schema": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
            },
            {
                "name": "view_image",
                "description": "View a PNG/JPEG under /case or /output.",
                "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
            },
        ]

    async def call(self, name, args):
        if name == "python":
            try:
                r = subprocess.run(
                    self.command,
                    input=args["code"],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
                )
            except subprocess.TimeoutExpired:
                return [{"type": "text", "text": "Python execution exceeded120s"}], True
            message = r.stdout + ("\n[stderr]\n" + r.stderr if r.stderr else "")
            return [{"type": "text", "text": clipped(message) or "(no output)"}], r.returncode != 0
        if name == "view_image":
            path = Path(args["path"])
            roots = {"case": self.case, "output": self.output}
            if not path.is_absolute() or len(path.parts) < 3 or path.parts[1] not in roots:
                return [{"type": "text", "text": "image must be under /case or /output"}], True
            root = roots[path.parts[1]].resolve()
            resolved = (root / Path(*path.parts[2:])).resolve()
            if not resolved.is_relative_to(root) or not resolved.is_file():
                return [{"type": "text", "text": "image outside assigned directories or not found"}], True
            if resolved.suffix.lower() not in (".png", ".jpg", ".jpeg") or resolved.stat().st_size > 5_000_000:
                return [{"type": "text", "text": "use a PNG/JPEG no larger than5MB"}], True
            media = "image/png" if resolved.suffix.lower() == ".png" else "image/jpeg"
            return [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media,
                        "data": base64.b64encode(resolved.read_bytes()).decode(),
                    },
                }
            ], False
        return [{"type": "text", "text": f"unknown tool {name}"}], True

    async def close(self):
        pass


class ProcessingSignalMindTools:
    def __init__(self, case: Path, output: Path):
        self.case, self.output = case, output
        self.client = None
        self.schemas = []

    async def open(self):
        from mcp import Client, StdioServerParameters

        bootstrap = "import sys,runpy;sys.path.insert(0,'/opt/signalmind/src');runpy.run_path('/opt/signalmind/worker.py',run_name='__main__')"
        command = sandbox_command(self.case, self.output, agent=True) + [
            str(PYTHON),
            "-I",
            "-c",
            bootstrap,
            "--case",
            "/case",
            "--output",
            "/output",
        ]
        self.cm = Client(
            StdioServerParameters(
                command=command[0], args=command[1:], env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}
            ),
            read_timeout_seconds=120,
        )
        self.client = await self.cm.__aenter__()
        self.schemas = [
            {"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
            for t in (await self.client.list_tools()).tools
        ]
        return self

    async def call(self, name, args):
        try:
            reply = await self.client.call_tool(name, args)
        except Exception as exc:
            return [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}], True
        blocks = []
        for part in reply.content:
            if part.type == "text":
                blocks.append({"type": "text", "text": part.text})
            elif part.type == "image":
                blocks.append(
                    {"type": "image", "source": {"type": "base64", "media_type": part.mime_type, "data": part.data}}
                )
        return blocks, bool(reply.is_error)

    async def close(self):
        if self.client is not None:
            await self.cm.__aexit__(None, None, None)
