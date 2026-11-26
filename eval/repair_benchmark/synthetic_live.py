"""One bounded official DeepSeek development comparison. All signal samples are generated here from formulas."""

import argparse
import base64
import json
import secrets
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from tsmcp.cleaning.common import digest, locked, read, sha, write

from .episode import ARMS, run_episode
from .providers import DeepSeek
from .public import SIMULATOR, prepare
from .scoring import report


def formula():
    i = np.arange(3000)
    phase = (i % 100) - 18
    x = (
        np.exp(-((phase / 2.2) ** 2))
        - 0.16 * np.exp(-(((phase + 3) / 1.2) ** 2))
        - 0.22 * np.exp(-(((phase - 4) / 1.6) ** 2))
        + 0.14 * np.exp(-(((phase - 28) / 7) ** 2))
        + 0.055 * np.exp(-(((phase + 14) / 3.2) ** 2))
        + 0.015 * np.sin(i / 135)
    )
    return i / 125, x


def build(root, cap):
    t, ref = formula()
    # Predeclared unlabelled case IDs, one replicate per arm. All variants use the same virtual record.
    design = [
        ("C031", "between_beats_biphasic", 1448, [1.0, -1.0]),
        ("C084", "unchanged", None, []),
        ("C052", "near_qrs_biphasic", 1519, [1.5, -1.5]),
        ("C017", "between_beats_single_sign", 1648, [0.65, 0.65]),
    ]
    cases = []
    private = root / "organizer"
    private.mkdir(parents=True)
    for cid, stratum, site, noise in design:
        x = ref.copy()
        mask = np.zeros(len(t), bool)
        if site is not None:
            x[site : site + len(noise)] += noise
            mask[site : site + len(noise)] = True
        pub = prepare(
            root / cid / "public",
            cid,
            "ECG",
            "synthetic units",
            125,
            t,
            x,
            np.ones(len(t), bool),
            np.zeros(len(t), bool),
            acquisition=SIMULATOR,
        )
        truth = private / (cid + ".npz")
        np.savez(
            truth,
            reference=ref,
            damaged=mask,
            reference_support=np.ones(len(t), bool),
            event_samples=np.arange(18, len(t), 100),
        )
        cases.append(
            dict(
                case_id=cid,
                subject="virtual-01",
                record="formula-01",
                stratum=stratum,
                noise_source=None if site is None else "formula-spike",
                split="development",
                input_hash=pub["input_hash"],
                public_dir=f"../{cid}/public",
                private_npz=truth.name,
                private_sha256=sha(truth),
                outputs={arm: f"../{cid}/{arm}" for arm in ARMS},
            )
        )
    manifest = dict(
        arms=list(ARMS),
        arm_modes=dict.fromkeys(ARMS, "live"),
        repetitions=1,
        cases=cases,
        cap_usd=cap,
        scope="analytic ECG only; one virtual record; synthetic development profile",
        generator_sha256=sha(__file__),
        reference_scale_floors={},
        budget=dict(max_turns=8, max_tools=16, max_candidates=2, seconds=180),
        price_basis="shared ledger peak USD per million input/cache read/output: 0.30/0.006/1.20, verified 2026-11-25",
        price_source="https://api-docs.deepseek.com/quick_start/pricing/",
    )
    write(private / "manifest.json", manifest)
    return manifest


def capability(folder, provider):
    folder.mkdir()
    code = secrets.token_hex(3).upper()
    tool_nonce = secrets.token_hex(4).upper()
    t, x = formula()
    fig = Figure(figsize=(5, 2.6), layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    ax.plot(t[:240], x[:240])
    ax.set(title="Synthetic ECG   " + code, xlabel="Time (s)")
    png = folder / "probe.png"
    fig.savefig(png)
    messages = [
        dict(
            role="user",
            content=[
                dict(type="text", text="Read the six-character title code and call read_code."),
                dict(
                    type="image",
                    source=dict(
                        type="base64", media_type="image/png", data=base64.b64encode(png.read_bytes()).decode()
                    ),
                ),
            ],
        )
    ]
    schema = dict(
        type="object", properties=dict(code={"type": "string"}), required=["code"], additionalProperties=False
    )
    ok_image = ok_roundtrip = False
    error = None
    frames = []
    try:
        with provider.session(folder, time.monotonic() + 120, 2):
            for name, expected in (("read_code", code), ("confirm_nonce", tool_nonce)):
                tools = [
                    dict(
                        name=name,
                        description="Submit the requested code. Calling this tool is required to complete this step.",
                        input_schema=schema,
                    )
                ]
                blocks = provider.response(
                    "Complete each step by calling the one provided tool exactly once, without a text-only answer.",
                    messages,
                    tools,
                    digest(messages),
                )
                frames.append(dict(request_hash=digest(messages), content=blocks, target=name))
                calls = [b for b in blocks if b["type"] == "tool_use"]
                if len(calls) != 1 or calls[0]["name"] != name or calls[0]["input"] != {"code": expected}:
                    raise ValueError(name + "_mismatch")
                if name == "read_code":
                    ok_image = True
                    messages.extend(
                        [
                            dict(role="assistant", content=blocks),
                            dict(
                                role="user",
                                content=[
                                    dict(
                                        type="tool_result",
                                        tool_use_id=calls[0]["id"],
                                        content=json.dumps({"nonce": tool_nonce}),
                                    ),
                                    dict(
                                        type="text",
                                        text="Call confirm_nonce now. Its code argument must be the exact nonce from the preceding tool result. The title code was only for the first step.",
                                    ),
                                ],
                            ),
                        ]
                    )
                else:
                    ok_roundtrip = True
    except Exception as exc:
        error = str(exc) if isinstance(exc, ValueError) and str(exc).endswith("_mismatch") else type(exc).__name__
    usage, provider_receipt, account = provider.accounting()
    record = dict(
        passed=ok_image and ok_roundtrip,
        image_verified=ok_image,
        tool_roundtrip_verified=ok_roundtrip,
        image_sha256=sha(png),
        error=error,
        usage=usage,
        provider_receipt=provider_receipt,
        accounting=account,
    )
    write(folder / "probe.json", record)
    write(folder / "transport.json", dict(frames=frames))
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--budget-root", type=Path, required=True)
    p.add_argument("--cap-usd", type=float, required=True)
    p.add_argument("--live", action="store_true", required=True)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if not 0 < args.cap_usd <= 0.50:
        p.error("This synthetic development run has a maximum incremental cap of USD 0.50")
    root = args.out
    root.mkdir(parents=True, exist_ok=args.resume)
    with locked(root / "run-lock"):
        run_frozen(root, args)


def run_frozen(root, args):
    from tsmcp.cleaning.common import implementation_hash
    from tsmcp.cleaning.provenance import freeze_implementation

    code = Path(__file__).parent
    frozen = dict(
        production=implementation_hash(),
        method_files={
            str(p.relative_to(code)): sha(p)
            for p in sorted(code.rglob("*"))
            if p.is_file() and p.suffix in (".py", ".json", ".txt") and p.name != "synthetic_live.py"
        },
    )
    if args.resume:
        with tempfile.TemporaryDirectory() as temp:
            expected = build(Path(temp), args.cap_usd)
        spec = read(root / "organizer" / "manifest.json")
        # Probe wording can change between failed attempts. Inputs, arms, budgets and scoring assignments are fixed.
        if {k: v for k, v in spec.items() if k != "generator_sha256"} != {
            k: v for k, v in expected.items() if k != "generator_sha256"
        }:
            raise ValueError("Regenerated synthetic assignment differs from frozen run")
        for c in spec["cases"]:
            from tsmcp.cleaning.common import array_hash

            public = root / c["case_id"] / "public"
            case = read(public / "case.json")
            with np.load(public / "observed.npz", allow_pickle=False) as z:
                actual = array_hash(*(z[k] for k in ("t", "x", "valid", "estimated")))
            if (
                case["input_hash"] != c["input_hash"]
                or digest({k: v for k, v in case.items() if k != "input_hash"}) != c["input_hash"]
                or actual != case["observed_hash"]
            ):
                raise ValueError("Public input no longer matches the formula-generated case")
        if (root / "frozen-methods.json").exists() and read(root / "frozen-methods.json") != frozen:
            raise ValueError("Method implementation changed; resume with the archived source")
        if any(root.glob("capability*")) and not (root / "experiment-budget.json").exists():
            raise ValueError("Cannot resume paid calls without the original experiment ledger window")
    else:
        spec = build(root, args.cap_usd)
        shutil.copytree(code, root / "method-source", ignore=shutil.ignore_patterns("__pycache__"))
        freeze_implementation(root)
        write(root / "frozen-methods.json", frozen)

    def provider():
        return DeepSeek(args.budget_root, args.cap_usd, experiment_budget=root / "experiment-budget.json")

    probes = sorted(f for f in root.glob("capability*") if f.is_dir())
    checked = next(
        (read(f / "probe.json") for f in probes if (f / "probe.json").exists() and read(f / "probe.json")["passed"]),
        None,
    )
    if checked is None:
        folder = root / ("capability" if not probes else f"capability-{len(probes) + 1}")
        checked = capability(folder, provider())
    if not checked["passed"]:
        raise RuntimeError("Synthetic capability check did not pass; see capability/probe.json")
    # Rotate arm order across cases. One shared ledger window includes preflight, retries and failed transports.
    for i, c in enumerate(spec["cases"]):
        for arm in ARMS[i % len(ARMS) :] + ARMS[: i % len(ARMS)]:
            out = root / c["case_id"] / arm
            if (out / "result.json").is_file():
                existing = read(out / "result.json")
                if existing["input_hash"] != c["input_hash"] or existing["arm_id"] != arm:
                    raise ValueError("Unrelated result in frozen assignment")
                continue
            r = run_episode(
                root / c["case_id"] / "public",
                out,
                arm,
                provider(),
                episode_id=digest([c["case_id"], 1, "synthetic-live-v1"])[:24],
                **spec["budget"],
            )
            print(c["case_id"], arm, r["status"], r["usage"]["model_requests"], r["usage"]["usd_cost"], flush=True)
    path = root / "organizer" / "manifest.json"
    if read(path) != spec:
        raise ValueError("Frozen assignment manifest changed")
    write(root / "organizer" / "report.json", report(path))


if __name__ == "__main__":
    main()
