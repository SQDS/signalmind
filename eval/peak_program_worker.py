"""Run model-authored marker code inside the existing isolated processing environment."""

import importlib.util
import json
import random
import resource
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, "/opt/signalmind/src")
from tsmcp.peak_policy import outside_rr  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def load_case(uid):
    with np.load(Path("/case") / uid / "input.npz", allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files}


def normalize(result, previous, old, n, fs):
    points = np.asarray(result["new_peak_samples"])
    if points.shape != old.shape or not np.isfinite(points).all() or not np.equal(points, np.round(points)).all():
        raise ValueError("Return one integer sample coordinate per original marker, or -1 for deletion")
    points = points.astype(np.int64)
    if np.any((points < -1) | (points >= n)) or len(np.unique(points[points >= 0])) != int((points >= 0).sum()):
        raise ValueError("Output contains out-of-range or duplicate retained coordinates")
    if not isinstance(result["exclude"], (bool, np.bool_)):
        raise ValueError("exclude must be a boolean")
    queue = result.get("review_queue", [])
    if not isinstance(queue, list):
        raise ValueError("review_queue must be a list")
    return {
        **previous,
        "new_peak_samples": points.tolist(),
        "exclude": bool(result["exclude"]),
        "exclude_reasons": result.get("exclude_reasons", []),
        "review_queue": queue,
        "review_required": bool(result.get("review_required", False) or queue),
        "deleted_count": int((points < 0).sum()),
        "moved_count": int(((points >= 0) & (points != old)).sum()),
        "features": {
            **previous["features"],
            "removed_fraction": float(np.mean(points < 0)) if len(points) else 0,
            "kept_peak_count": int((points >= 0).sum()),
            "kept_rr_outside_fraction": outside_rr(points[points >= 0], fs),
        },
        "waveform_values_modified": False,
    }


def main():
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    np.random.seed(0)
    random.seed(0)
    job = read("/output/job.json")
    labels = read("/case/development_labels.json")
    spec = importlib.util.spec_from_file_location("agent_program", "/output/program.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    development = []
    for uid, reference in labels.items():
        development.append({**load_case(uid), "reference": reference, "record_id": reference["subject_key"]})
    state = module.fit(development, job["fs_hz"]) if hasattr(module, "fit") else None
    predictions, errors = {}, []
    for uid in job["case_ids"]:
        try:
            data = load_case(uid)
            previous = job["previous"][uid]
            result = module.process(data["ecg"], data["abp"], data["old_peaks"], job["fs_hz"], previous, state)
            predictions[uid] = normalize(result, previous, data["old_peaks"], len(data["ecg"]), job["fs_hz"])
        except Exception as exc:
            errors.append(
                {"id": uid, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-2000:]}
            )
    Path("/output/result.json").write_text(json.dumps({"predictions": predictions, "errors": errors}, allow_nan=False))
    print(json.dumps({"completed": len(predictions), "errors": len(errors)}))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
