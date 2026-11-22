"""Check accepted ECG trial against preceding observed peaks; the comparison has no human reference."""

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from scipy.signal import find_peaks

from tsmcp.cleaning.common import read, write

folder = Path(__file__).resolve().parent
root = folder.parents[1] / "runs" / folder.name / "zoom"
trials = [z for f in root.glob("records/*/spike-trials.json") for z in read(f) if z["status"] == "accepted"]
rows = []
for trial in trials:
    case, a, b = trial["case"], trial["a"], trial["b"]
    fs = next(c["fs"] for c in read(root / "dataset_manifest.json")["cases"] if c["id"] == case)
    with np.load(root / "public" / case / "input.npz") as z:
        x, t = z["values"], z["time"]
    art = read(root / "artifacts" / trial["candidate"] / "artifact.json")
    with np.load(root / art["arrays"]) as z:
        y = z["values"]
    peak = a + int(x[a:b].argmax())
    # Independent local-maxima check: keep preceding peaks of same polarity and comparable amplitude.
    peers, _ = find_peaks(x[:peak], distance=round(0.45 * fs), height=0.55 * x[peak])
    peers = peers[peers >= 5][-6:]
    offsets = np.arange(-5, 6)
    corr = [float(np.corrcoef(x[q + offsets], x[peak + offsets])[0, 1]) for q in peers]
    periods = (np.diff(np.r_[peers, peak]) / fs * 1000).tolist()
    result = dict(
        case=case,
        a=a,
        b=b,
        source_start=float(t[a]),
        source_stop=float(t[b - 1] + 1 / fs),
        peak=float(x[peak]),
        released_core=y[a:b].tolist(),
        preceding_peak_samples=peers.tolist(),
        preceding_peak_amplitudes=x[peers].tolist(),
        preceding_intervals_ms=periods,
        correlation_on_11_samples=corr,
        assessment="suspected removal of a genuine beat; retain original for analysis pending independent review",
    )
    rows.append(result)
    fig = Figure(figsize=(11, 4.1), layout="constrained")
    left, right = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.65, 1]})
    i = max(0, peers[0] - int(0.25 * fs))
    j = min(len(x), b + int(0.2 * fs))
    left.plot(t[i:j] - t[peak], x[i:j], lw=1.1, color="#365a78")
    left.scatter(t[peers] - t[peak], x[peers], s=15, color="#728b80", label="Previous six peaks")
    left.axvspan(t[a] - t[peak], t[b] - t[peak], color="#e4ad5c", alpha=0.4)
    left.scatter(0, x[peak], s=30, color="#b65226", label="Agent-selected core")
    left.set(
        xlabel="Seconds relative to target peak", ylabel="ECG (source units)", title="Timing in the observed signal"
    )
    left.legend(fontsize=8)
    z = np.arange(-8, min(9, len(x) - peak))
    for n, q in enumerate(peers):
        right.plot(
            1000 * z / fs, x[q + z], lw=1, color="#a3acb6", alpha=0.7, label="Preceding beats" if n == 0 else None
        )
    right.plot(1000 * z / fs, x[peak + z], ".-", lw=1.5, color="#285b79", label="Observed target")
    right.plot(1000 * z / fs, y[peak + z], "--", lw=1.9, color="#c6542b", label="Agent release")
    right.axvspan((a - peak - 0.5) * 1000 / fs, (b - peak - 0.5) * 1000 / fs, color="#edc575", alpha=0.23)
    right.set(xlabel="Milliseconds from aligned peaks", title="Observed shapes and actual numerical change")
    right.legend(fontsize=8)
    for ax in (left, right):
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Accepted ECG trial   |   Source time {t[a]:.3f}–{t[b]:.3f} s")
    fig.savefig(folder / f"accepted-{case}-{a}.png", dpi=145)
write(folder / "accepted-peak-audit.json", rows)
