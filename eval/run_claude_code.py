"""Headline arms through the real client users run: Claude Code headless (`claude -p`).

  B: fixed frozen-R evidence bundle, one response with no tools
  C: TS-MCP tools only (no built-in tools), cold start (no workflow text injected)
  D: Bash + Read with a Python env (numpy/scipy/matplotlib/wfdb/neurokit2), can view its own plots
  E: D + `import tsmcp` available

  uv run --group eval python eval/run_claude_code.py --suite p1 --arm C [--limit 5] [--effort medium]

Isolation: each case runs in a fresh dir holding only signal.npy; no user settings, no other MCP servers.
D/E run inside bwrap with the home directory hidden (annotations/answers unreachable); tool calls are also
audited and suspicious cases are flagged `contaminated` and excluded from scoring.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.common import (  # noqa: E402
    ROOT,
    RUNS,
    append_jsonl,
    balance,
    case_prompt,
    cost_usd,
    load_provider,
    load_suite,
    parse_answer,
    read_jsonl,
)

REPO = ROOT.parent
HOME = Path.home()
VENV_D = ROOT / ".venv-d"  # created by `eval/setup_venv_d.sh`
# socat (needed by Claude Code's Bash sandbox on Linux), installed user-level: conda create -p <this> -c conda-forge socat
SANDBOX_TOOLS = HOME / ".local" / "share" / "tsmcp-sandbox-tools"
# Pre-approve every tool of the tsmcp server (not a hand-kept list: a new tool once silently stayed blocked)
ALLOW_C = "mcp__tsmcp"
CASE_TIMEOUT_S = 900
SERVER_LAUNCHER = ROOT / "server_process.py"
# Regexes over the (lower-cased) tool inputs. Matching = the model tried to reach data outside its case directory.
# NETWORK patterns are blocked by the Bash sandbox since 078218b: an attempt there is logged, not a leak.
SUSPICIOUS = (
    r"/home\b",
    r"\bfind /(?!tmp\b)",  # find over the filesystem; /tmp (its own scratch) is fine
    r"\.atr\b",
    r"\.hea\b",
    r"physionet",
    r"\.answers\.jsonl",
    r"\bmitdb\b",
    r"\bafdb\b",
    r"\bltafdb\b",
    r"\bucr\b",
    r"\brdann\b",
    r"\bcurl ",
    r"\bwget ",
    r"\bcases/",
)
NETWORK = (
    r"physionet",
    r"\bcurl ",
    r"\bwget ",
)  # answer-access patterns only; installing packages (pip install) is not a leak
E_HINT = (
    "\nThe `tsmcp` Python package is available: `from tsmcp.core import SignalStore` "
    "(`SignalStore(allow_roots=[dir]).load(source=path, fs=fs)`), `tsmcp.core.views.describe.describe(sig)`, "
    "`tsmcp.core.measure.MEASURES` (period, dominant_freq, amplitude_pp, rms, mean: f(x, fs, unit) -> dict), "
    "`tsmcp.core.views.render.render(channels, t0, t1)` -> PNG bytes."
)


def e_provenance(e_root: Path) -> dict:
    """Which tsmcp checkout arm E imports (via PYTHONPATH=<e_root>/src) and the ECG plugin text it would document."""
    src = e_root.expanduser().resolve() / "src"
    probe = (
        "import json, tsmcp\n"
        "from tsmcp.plugins.ecg import PLUGIN\n"
        "print(json.dumps({'package_path': tsmcp.__file__, 'summary': PLUGIN.summary,"
        " 'detectors_summary': PLUGIN.detectors_summary}))"
    )
    env = {**os.environ, "PYTHONPATH": str(src)}
    out = subprocess.run(
        [str(VENV_D / "bin" / "python"), "-c", probe], env=env, capture_output=True, text=True, check=True
    )
    info = json.loads(out.stdout)
    if not Path(info["package_path"]).resolve().is_relative_to(src):
        raise SystemExit(f"arm E would import tsmcp from {info['package_path']}, not {src}")
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=e_root, capture_output=True, text=True, check=True)
    return {"e_root": str(e_root.expanduser().resolve()), "e_git_sha": sha.stdout.strip(), **info}


def e_plugin_hint(info: dict) -> str:
    """The same ECG summaries arm C sees in its ts_measure / ts_find_events descriptions, as Python calls."""
    return (
        "\nECG functions (the same algorithms as the TS-MCP tools): `from tsmcp.plugins.ecg import PLUGIN`; "
        "`PLUGIN.measures[name](x, fs, unit)` -> dict, `PLUGIN.detectors[name](x, fs, unit)` -> list of events, "
        "`PLUGIN.series[name](x, fs, unit)` -> dict (x: 1-D numpy array, unit: 'mV'; each call recomputes). "
        f"Measures — {info['summary']}. Detectors — {info['detectors_summary']}"
    )


# Variables that attach a child `claude` to the *calling* Claude Code session (its session id, pid, messaging socket,
# IDE bridge). Inherited, the evaluated client registered as a child session of the operator's session and could be
# torn down from there. Behaviour-relevant settings (entrypoint, effort, provider URL/token) are left untouched so arms
# stay comparable with earlier runs.
HOST_SESSION_VARS = (
    "CLAUDE_CODE_CHILD_SESSION",
    "CLAUDE_CODE_SESSION_ID",
    "CLAUDE_PID",
    "CLAUDE_CODE_MESSAGING_SOCKET",
    "CLAUDE_CODE_MESSAGING_TOKEN",
    "CLAUDE_CODE_SESSION_ATTENDED",
    "VSCODE_IPC_HOOK_CLI",
    "VSCODE_CLI_REQUIRE_TOKEN",
)


def claude_bin() -> str:
    b = shutil.which("claude")
    if not b:
        sys.exit("claude CLI not found on PATH")
    return str(Path(b).resolve())


def bwrap_prefix(work: Path, fake_home: Path, extra_ro: list[Path]) -> list[str]:
    node_root = Path(claude_bin()).parents[2]  # .../node/vXX (bin/, lib/)
    args = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", str(HOME), "--tmpfs", "/tmp"]
    for p in [node_root, HOME / ".local" / "share" / "uv" / "python", *extra_ro]:
        if p.exists():
            args += ["--ro-bind", str(p), str(p)]
    args += [
        "--bind",
        str(work),
        str(work),
        "--bind",
        str(fake_home),
        str(fake_home),
        "--setenv",
        "HOME",
        str(fake_home),
    ]
    return args + ["--"]


# Claude Code's own Bash sandbox: no network for the model's commands (Claude Code itself still reaches its API).
# Needed because a model once downloaded the dataset's expert annotations from PhysioNet (outer bwrap hides files,
# not the network).
BASH_SANDBOX = {
    "sandbox": {
        "enabled": True,
        "autoAllowBashIfSandboxed": True,
        "allowUnsandboxedCommands": False,
        "failIfUnavailable": True,
        "network": {"strictAllowlist": True, "allowedDomains": []},
    }
}
NO_VISION_NOTE = "\nNote: you cannot view images in this session; analyze numerically."


def provider_env(env: dict, prov: dict) -> None:
    """Point Claude Code at another Anthropic-compatible backend (as cc-switch does) for the child process only."""
    if not prov["base_url"]:
        return
    env["ANTHROPIC_BASE_URL"] = prov["base_url"]
    env["ANTHROPIC_AUTH_TOKEN"] = prov["api_key"]
    env.pop("ANTHROPIC_API_KEY", None)
    for k in ("ANTHROPIC_MODEL", "ANTHROPIC_SMALL_FAST_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL",
              "ANTHROPIC_DEFAULT_SONNET_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL"):  # fmt: skip
        env[k] = prov["model"]


def server_command(server_root: Path) -> list[str]:
    """Use the selected venv, without inheriting a different checkout through PYTHONPATH."""
    return [str(server_root / ".venv" / "bin" / "python"), "-I", str(SERVER_LAUNCHER),
            "--server-root", str(server_root)]  # fmt: skip


def check_server(server_root: Path) -> dict:
    """Check imports before spending on a model; the real server independently reports them again."""
    for name in ("python", "tsmcp"):
        if not (server_root / ".venv" / "bin" / name).is_file():
            raise SystemExit(f"missing {server_root / '.venv' / 'bin' / name}")
    p = subprocess.run(server_command(server_root) + ["--check-only"], capture_output=True, text=True, check=False)
    if p.returncode:
        raise SystemExit(f"server provenance check failed: {p.stderr.strip()}")
    return json.loads(p.stdout)


def prepare_b_case(case: dict, prov: dict, effort: str | None, server_root: Path,
                   interface: str | None, b_workflow: str = "v0") -> dict:  # fmt: skip
    from eval import b_revisions as br
    from eval import b_workflow as bw

    effective = br.effective_workflow(b_workflow, case["qtype"])
    if interface not in (None, "v1"):
        raise ValueError("arm B is fixed to interface v1")
    if not prov["vision"]:
        raise ValueError("arm B requires a provider accepting the specified image evidence")
    if server_root == REPO:
        server_root = bw.SERVER_ROOT
    if b_workflow == "v0":
        bundle, meta = bw.build_in_worker(case, server_root)
    else:
        bundle, meta = br.build_in_worker(case, server_root=server_root, workflow=b_workflow)
    meta = {**meta, "b_workflow": b_workflow, "b_workflow_effective": effective}
    fs = case["visible"]["channels"][0]["fs"]
    prompt = (
        case_prompt(case["visible"], case["question"], case["qtype"])
        + f"\nThe signal file is {bw.signal_path(case)} (sampling rate {fs:g} Hz)."
    )
    stdin = bw.user_message(prompt, bundle)  # fail closed before preparing any model process
    base = Path(tempfile.mkdtemp(prefix="tsmcp-B-"))
    work = base / "work"
    work.mkdir()
    env = {k: v for k, v in os.environ.items() if k not in HOST_SESSION_VARS}
    provider_env(env, prov)
    command = [
        claude_bin(),
        "-p",
        "--output-format",
        "stream-json",
        "--verbose",
        "--model",
        prov["model"],
        *(["--effort", effort] if effort else []),
        "--no-session-persistence",
        "--setting-sources",
        "project",
        "--strict-mcp-config",
        "--tools",
        "",
        "--input-format",
        "stream-json",
        "--max-turns",
        "1",
    ]
    provenance = {
        **meta["identity"]["provenance"],
        "bundle_sha256": meta["bundle_sha256"],
        "bundle_request_sha256": meta["request_sha256"],
        "workflow_sha256": meta["identity"]["workflow_sha256"],
        "b_workflow": b_workflow,
        "b_workflow_effective": effective,
        "bundle_sizes": meta["sizes"],
        "bundle_dropped": meta["dropped"],
    }
    return {
        "base": base,
        "cwd": work,
        "command": command,
        "env": env,
        "stdin": stdin,
        "mcp_config": None,
        "provenance": provenance,
        "bundle_manifest": meta,
    }


def prepare_case(case: dict, arm: str, prov: dict, effort: str | None, isolate: bool,
                 server_root: Path = REPO, e_info: dict | None = None,
                 interface: str | None = None, b_workflow: str = "v0") -> dict:  # fmt: skip
    """Materialize exactly the command, configs and stdin used for a case, without invoking Claude."""
    server_root = server_root.expanduser().resolve()
    if arm == "B":
        return prepare_b_case(case, prov, effort, server_root, interface, b_workflow)
    model = prov["model"]
    v = case["visible"]
    fs = v["channels"][0]["fs"]
    base = Path(tempfile.mkdtemp(prefix=f"tsmcp-{arm}-"))
    work, fake_home = base / "work", base / "home"
    work.mkdir()
    fake_home.mkdir()
    shutil.copy(Path(case["dir"]) / v["file"], work / "signal.npy")
    prompt = (
        case_prompt(v, case["question"], case["qtype"])
        + f"\nThe signal file is {work / 'signal.npy'} (sampling rate {fs:g} Hz)."
    )
    cmd = [
        claude_bin(),
        "-p",
        "--output-format",
        "stream-json",
        "--verbose",
        "--model",
        model,
        *(["--effort", effort] if effort else []),
        "--no-session-persistence",
        "--setting-sources",
        "project",
        "--strict-mcp-config",
    ]
    env = {k: v for k, v in os.environ.items() if k not in HOST_SESSION_VARS}
    env.setdefault("MCP_TOOL_TIMEOUT", "600000")  # first beat detection on a 24 h record takes ~15 s; never cut it off
    provider_env(env, prov)
    prefix: list[str] = []
    if not prov["vision"]:
        prompt += NO_VISION_NOTE
    mcp_config = None
    provenance = base / "server-provenance.json"
    if arm == "C":
        cfg = base / "mcp.json"
        launch = server_command(server_root)
        mcp_config = {"mcpServers": {"tsmcp": {
            "command": launch[0],
            "args": launch[1:] + ["--provenance", str(provenance), "--", "--allow-root", str(work)]
            + ([] if prov["vision"] else ["--no-images"])
            + (["--interface", interface] if interface else []),
        }}}  # fmt: skip
        cfg.write_text(json.dumps(mcp_config))
        cmd += [
            "--mcp-config",
            str(cfg),
            "--tools",
            "",
            "--allowedTools",
            ALLOW_C,
        ]
    else:
        if not (VENV_D / "bin" / "python").exists():
            sys.exit(f"missing {VENV_D}; run eval/setup_venv_d.sh")
        tools = "Bash,Read" if prov["vision"] else "Bash"  # Read of a PNG would send an image the model cannot take
        cmd += ["--tools", tools, "--allowedTools", tools, "--settings", json.dumps(BASH_SANDBOX)]
        env["PATH"] = f"{VENV_D / 'bin'}:{SANDBOX_TOOLS / 'bin'}:{env['PATH']}"
        env["MPLBACKEND"] = "Agg"
        extra = [VENV_D, SANDBOX_TOOLS]
        if arm == "E":
            e_src = Path(e_info["e_root"]) / "src" if e_info else REPO / "src"
            env["PYTHONPATH"] = str(e_src)
            extra.append(e_src)
            prompt += E_HINT
            if e_info and e_info.get("plugin_hint"):
                prompt += e_plugin_hint(e_info)
        else:
            env.pop("PYTHONPATH", None)
        if isolate:
            prefix = bwrap_prefix(work, fake_home, extra)
    # prompt goes via stdin: --allowedTools/--tools are variadic and would swallow a positional prompt
    return {"base": base, "cwd": work, "command": prefix + cmd, "env": env, "stdin": prompt,
            "mcp_config": mcp_config, "provenance": provenance}  # fmt: skip


def run_isolated_session(command: list[str], cwd: Path, env: dict, stdin: str, timeout_s: int,
                         stream_to: Path | None = None) -> tuple[str, str, int]:  # fmt: skip
    """Run the client in its own session/process group. A process-group kill inside the client (e.g. of a timed-out
    Bash command) then cannot reach this harness, and on our timeout the whole group is killed, not just the client."""
    sink = open(stream_to, "w") if stream_to else None  # noqa: SIM115 - closed below; survives a harness kill
    p = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.PIPE, stdout=sink or subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, start_new_session=True)  # fmt: skip
    try:
        out, err = p.communicate(stdin, timeout=timeout_s)
        if sink:
            sink.close()
            out = Path(stream_to).read_text()
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        p.communicate()
        if sink:
            sink.close()
        raise
    return out, err, p.returncode


def run_case(case: dict, arm: str, prov: dict, effort: str | None, run_dir: Path, isolate: bool,
             timeout_s: int = CASE_TIMEOUT_S, server_root: Path = REPO,
             expected_server: dict | None = None, e_info: dict | None = None,
             interface: str | None = None, b_workflow: str = "v0") -> dict:  # fmt: skip
    plan = prepare_case(case, arm, prov, effort, isolate, server_root, e_info, interface, b_workflow)
    provenance = plan["provenance"] if arm == "B" else {}
    if arm == "B" and timeout_s != CASE_TIMEOUT_S:
        shutil.rmtree(plan["base"], ignore_errors=True)
        raise ValueError("arm B timeout is fixed at 900 seconds")
    if arm == "E" and e_info:
        provenance = {k: e_info[k] for k in ("e_root", "e_git_sha", "package_path", "plugin_hint")}

    t = time.monotonic()
    try:
        (run_dir / "transcripts").mkdir(parents=True, exist_ok=True)
        partial = run_dir / "transcripts" / f"{case['case_id']}.partial.jsonl"
        out, err, rc = run_isolated_session(
            plan["command"], plan["cwd"], plan["env"], plan["stdin"], timeout_s, stream_to=partial
        )
        partial.unlink(missing_ok=True)
        if arm == "C":
            try:
                provenance = json.loads(plan["provenance"].read_text())
                if Path(provenance["server_root"]) != server_root.expanduser().resolve():
                    raise ValueError("server root does not match requested checkout")
                if (
                    not Path(provenance["server_package_path"])
                    .resolve()
                    .is_relative_to(server_root.expanduser().resolve())
                ):
                    raise ValueError("server package is outside requested checkout")
                if not re.fullmatch(r"[0-9a-f]{40,64}", provenance["server_git_sha"]):
                    raise ValueError("server reported an invalid git SHA")
                if expected_server is not None and provenance != expected_server:
                    raise ValueError("server provenance changed since the preflight check")
            except (OSError, ValueError, KeyError, TypeError) as e:
                return {"error": "server_provenance", "stderr": f"{e}; {err[-500:]}"}
    except subprocess.TimeoutExpired as e:
        return {**(provenance if arm == "B" else {}), "error": "timeout", "stderr": str(e)[:300]}
    finally:
        wall = round(time.monotonic() - t, 1)
        shutil.rmtree(plan["base"], ignore_errors=True)
    (run_dir / "transcripts").mkdir(parents=True, exist_ok=True)
    (run_dir / "transcripts" / f"{case['case_id']}.jsonl").write_text(out)
    events = []
    for line in out.splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    result = next((e for e in reversed(events) if e.get("type") == "result"), None)
    if result is None:
        return {**(provenance if arm == "B" else {}), "error": f"no result event (rc={rc})", "stderr": err[-800:]}
    served = {e["message"].get("model") for e in events if e.get("type") == "assistant" and "message" in e}
    tool_calls = [
        {"name": b.get("name"), "input": b.get("input")}
        for e in events
        if e.get("type") == "assistant"
        for b in e["message"].get("content", [])
        if b.get("type") == "tool_use"
    ]
    if arm == "B" and (tool_calls or result.get("num_turns", 0) > 1):
        return {**provenance, "error": "B_single_turn_contract", "stderr": "B response used tools or multiple turns"}
    # the case's own sandbox dir (/tmp/tsmcp-...) is fine to explore; mask it before matching suspicious patterns
    blob = re.sub(r"/tmp/tsmcp-[a-z]-[\w]+(/home)?", "<work>", json.dumps([c["input"] for c in tool_calls]).lower())
    flags = [pat for pat in SUSPICIOUS if arm != "C" and re.search(pat, blob)]
    usage = result.get("usage") or {}
    return {
        **provenance,
        "served_models": sorted(m for m in served if m),
        "text": result.get("result") or "",
        "is_error": result.get("is_error"),
        "subtype": result.get("subtype"),
        "turns": result.get("num_turns"),
        "usage": usage,
        "cc_cost_usd": result.get("total_cost_usd"),
        "model_usage": result.get("modelUsage"),
        "tool_calls": [c["name"] for c in tool_calls],
        "contamination_flags": flags,
        "wall_s": wall,
    }


def run_directory(name: str, label: str | None) -> Path:
    if label is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", label):
        raise SystemExit("--label must use letters, digits, underscores, dots or hyphens, starting with a letter/digit")
    return RUNS / (f"{name}-{label}" if label else name)


def print_dry_run(plan: dict, case_id: str, run_dir: Path) -> None:
    """Leave local artifacts in place, so the printed commands/configs remain inspectable."""
    prompt_path = plan["base"] / "prompt.txt"
    prompt_path.write_text(plan["stdin"])
    changed_env = {k: v for k, v in plan["env"].items() if os.environ.get(k) != v}
    # Provider tokens are required only for real execution; never print credentials.
    changed_env = {k: ("<redacted>" if "TOKEN" in k or "KEY" in k else v) for k, v in changed_env.items()}
    display_stdin = plan["stdin"]
    if "bundle_manifest" in plan:
        from eval.b_workflow import summarize_images

        display_stdin = summarize_images(display_stdin)
    print(json.dumps({
        "dry_run": True, "case_id": case_id, "run_directory": str(run_dir), "cwd": str(plan["cwd"]),
        "command": plan["command"], "shell_command": shlex.join(plan["command"]) + f" < {shlex.quote(str(prompt_path))}",
        "environment_overrides": changed_env,
        "environment_removed": sorted(set(os.environ) - set(plan["env"])),
        "mcp_config": plan["mcp_config"], "stdin_path": str(prompt_path), "stdin": display_stdin,
        **({"bundle_manifest": plan["bundle_manifest"]} if "bundle_manifest" in plan else {}),
    }, indent=2))  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--arm", choices=["B", "C", "D", "E"], required=True)
    ap.add_argument(
        "--provider", default="anthropic", help="anthropic (default endpoint) or a key of eval/providers.local.json"
    )
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", nargs="*", default=None, help="case ids")
    ap.add_argument("--no-isolate", action="store_true", help="skip bwrap for D/E (audit still runs)")
    ap.add_argument("--case-timeout", type=int, default=CASE_TIMEOUT_S, help="seconds per case (wall clock)")
    ap.add_argument("--run", default=None)
    ap.add_argument("--server-root", type=Path, default=REPO, help="checkout whose .venv/bin/tsmcp serves arm C")
    ap.add_argument("--label", help="suffix appended to the run directory (for example M or R)")
    ap.add_argument("--e-root", type=Path, default=REPO, help="checkout whose src/ arm E imports as tsmcp")
    ap.add_argument("--interface", choices=("v1", "v2"), help="arm C: pass --interface to the server (unset = default)")
    ap.add_argument(
        "--e-plugin-hint", action="store_true", help="arm E: also document the ECG plugin (same text arm C sees)"
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="print commands/configs for one case; no model or network calls"
    )
    ap.add_argument("--cases-root", type=Path, help="arm B: public development cases root")
    ap.add_argument("--b-workflow", choices=("v0", "r1", "r2", "final"), default="v0",
                    help="arm B: deterministic evidence version (default: frozen v0)")  # fmt: skip
    args = ap.parse_args()
    if args.arm == "B" and args.case_timeout != CASE_TIMEOUT_S:
        ap.error("arm B timeout is fixed at 900 seconds")

    prov = load_provider(args.provider)
    model = prov["model"]
    effort = prov["effort"] if isinstance(prov["effort"], str) else (args.effort if prov["effort"] else None)
    server_root = args.server_root.expanduser().resolve()
    e_info = None
    if args.arm == "E":
        e_info = {**e_provenance(args.e_root), "plugin_hint": args.e_plugin_hint}
    tag = model if args.provider == "anthropic" else f"{args.provider}_{model}"
    run = run_directory(args.run or f"{args.suite}-{args.arm}-{tag}-{effort or 'default'}", args.label)
    if args.arm == "B":
        from eval.b_workflow import public_cases

        suite_cases = public_cases(args.cases_root or ROOT / "cases", args.suite)
    else:
        suite_cases = load_suite(args.suite)
    cases = [c for c in suite_cases if not args.only or c["case_id"] in args.only][: args.limit]
    if args.dry_run:
        if not cases:
            raise SystemExit("no matching cases")
        plan = prepare_case(cases[0], args.arm, prov, effort, not args.no_isolate, server_root,
                            e_info, args.interface, args.b_workflow)  # fmt: skip
        print_dry_run(plan, cases[0]["case_id"], run)
        return
    expected_server = check_server(server_root) if args.arm == "C" else None
    if args.arm == "B":
        from eval.b_revisions import effective_workflow, workflow_hash
        from eval.b_workflow import FROZEN_R, SERVER_ROOT

        if server_root == REPO:
            server_root = SERVER_ROOT
        expected_server = check_server(server_root)
        if expected_server["server_git_sha"] != FROZEN_R:
            raise SystemExit("arm B requires the frozen R tree")
        expected_server.update(interface="v1", workflow_sha256=workflow_hash(args.b_workflow),
                               b_workflow=args.b_workflow)  # fmt: skip
    previous = read_jsonl(run / "results.jsonl")
    if expected_server and any(any(r.get(k) != v for k, v in expected_server.items()) for r in previous):
        raise SystemExit("existing run has different or missing server provenance; choose a new --label")
    if args.arm == "B" and any(
        r.get("b_workflow_effective") != effective_workflow(args.b_workflow, r["qtype"]) for r in previous
    ):
        raise SystemExit("existing run has different or missing B workflow provenance; choose a new --label")
    done = {r["case_id"] for r in previous}
    bal0 = balance(prov)
    if bal0:
        print("balance before:", bal0, flush=True)
    spent = 0.0
    for c in cases:
        if c["case_id"] in done:
            continue
        r = run_case(c, args.arm, prov, effort, run, isolate=not args.no_isolate, timeout_s=args.case_timeout,
                     server_root=server_root, expected_server=expected_server, e_info=e_info,
                     interface=args.interface, b_workflow=args.b_workflow)  # fmt: skip
        if "error" in r:
            append_jsonl(
                run / "errors.jsonl",
                {
                    "case_id": c["case_id"],
                    "class": r["error"],
                    "msg": r.get("stderr", ""),
                    **({k: v for k, v in r.items() if k not in ("error", "stderr")} if args.arm == "B" else {}),
                },
            )
            print(c["case_id"], "ERROR", r["error"], r.get("stderr", "")[-300:], flush=True)
            continue
        mismatch = [m for m in r["served_models"] if m != model and not m.startswith("<")]
        if mismatch and args.provider != "anthropic":
            print(c["case_id"], "note: provider reports model", mismatch, flush=True)
        elif mismatch:
            append_jsonl(
                run / "errors.jsonl",
                {
                    "case_id": c["case_id"],
                    "class": "served_model_mismatch",
                    "msg": r["served_models"],
                    **({k: r[k] for k in ("b_workflow", "b_workflow_effective")} if args.arm == "B" else {}),
                },
            )
            print(c["case_id"], "MODEL MISMATCH", r["served_models"], flush=True)
            continue
        if r["is_error"] and "API Error" in (r["text"] or "")[:300]:
            # the request never reached a model (gateway down, quota, ...): infrastructure, not a model answer
            append_jsonl(
                run / "errors.jsonl",
                {
                    "case_id": c["case_id"],
                    "class": "api_error",
                    "msg": r["text"][:300],
                    **({k: r[k] for k in ("b_workflow", "b_workflow_effective")} if args.arm == "B" else {}),
                },
            )
            print(c["case_id"], "ERROR api", r["text"][:120], flush=True)
            continue
        cost = cost_usd(model, r["usage"], prov.get("prices"))
        spent += cost
        status = "contaminated" if r["contamination_flags"] else ("error" if r["is_error"] else "ok")
        row = {
            "case_id": c["case_id"],
            "arm": args.arm,
            "qtype": c["qtype"],
            "model": model,
            "provider": args.provider,
            "server_root": r.get("server_root"),
            "server_git_sha": r.get("server_git_sha"),
            "server_package_path": r.get("server_package_path"),
            "e_root": r.get("e_root"),
            "e_git_sha": r.get("e_git_sha"),
            "e_package_path": r.get("package_path"),
            "e_plugin_hint": r.get("plugin_hint"),
            "host_session_env_stripped": True,
            "interface": args.interface or "default",
            "served_models": r["served_models"],
            "effort": effort,
            "pred": parse_answer(r["text"]),
            "status": status,
            "usage": r["usage"],
            "cost_usd": round(cost, 5),
            "cc_cost_usd": r["cc_cost_usd"],
            "latency_s": r["wall_s"],
            "turns": r["turns"],
            "tool_calls": r["tool_calls"],
            "contamination_flags": r["contamination_flags"],
            "text": r["text"][-3000:],
        }
        if args.arm == "B":
            row.update(
                {
                    key: r[key]
                    for key in (
                        "bundle_sha256",
                        "bundle_request_sha256",
                        "workflow_sha256",
                        "b_workflow",
                        "b_workflow_effective",
                        "bundle_sizes",
                        "bundle_dropped",
                        "interface",
                    )
                }
            )
        append_jsonl(run / "results.jsonl", row)
        print(c["case_id"], c["qtype"], row["pred"], status, f"turns={r['turns']}", f"${cost:.3f}", flush=True)
    print(f"spend this invocation: ${spent:.3f} -> {run}")
    if bal0:
        print("balance after:", balance(prov), "(before:", bal0 + ")", flush=True)


if __name__ == "__main__":
    main()
