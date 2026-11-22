# Experimental signal-processing API

This opt-in processing layer creates new float64 signal versions and leaves the original and previous versions
unchanged. It is separate from the default TS-MCP analysis server. The initial version handles one channel with a
fixed sampling rate and keeps its sample count, time axis and unit. SciPy is provided by the optional `processing`
dependency group; the processing-pack experiment uses its own pinned numerical environment.

```python
import numpy as np
from tsmcp.processing import ProcessingSession

raw = np.load("signal.npy", allow_pickle=False)  # one-dimensional float64 waveform
session = ProcessingSession(
    raw, fs=250, channel="abp", unit="mmHg",
    source_id="my-record", max_gap_samples=5, max_transient_samples=20,
)
print(session.summary())
new = session.process(session.raw.id, "interpolate", [[100, 103]])
np.savez("processed.npz", values=new.values, valid_mask=new.valid,
         artifact_mask=new.artifact, repaired_mask=new.repaired)
```

The caller must determine appropriate operations and ranges. `interpolate` uses finite valid anchors and respects
the declared repair-length limit, protected intervals and confirmed saturation. `max_gap_samples` limits original
missing runs; the optional `max_transient_samples` separately limits the full finite-artifact repair interval,
including any short gap and contaminated anchors within it. Without that option, the original common limit applies.
`mark_invalid` preserves observed values. `notch` uses a forward/backward IIR notch with explicit frequency and Q;
`lowpass` uses a forward/backward Butterworth filter with cutoff `frequency_hz` and `filter_order` (default4).
`remove_tone` fits a sinusoid plus intercept and subtracts only the sinusoidal component. Optional `window_s`
uses overlapping local fits with normalized Hann blending; omitting it retains one fit over each selected valid run.
Filtering operates separately on contiguous valid runs and leaves samples outside the selected ranges unchanged.
These operations do not automatically classify the signal or guarantee that processing improves it.

`tsmcp.server.processing.build_processing_server` exposes the same implementation over MCP:

| Tool | Purpose |
|---|---|
| `ts_describe` | Original/version handle, sample metadata, masks, quantiles and largest adjacent steps |
| `ts_samples` | Exact sample values and masks in a requested interval |
| `ts_spectrum` | Welch spectrum peaks and 50/60 Hz neighborhood measurements |
| `ts_zoom` | Waveform view, optionally with original and processed panels |
| `ts_process` | Apply an explicitly selected operation and return a new handle |
| `ts_measure` | Call the configured public measurement function |
| `ts_export` | Call the configured output adapter with values, masks and actual operations |

All processing intervals use zero-based half-open sample indices. Handles live for the server session; exported
arrays and provenance are the persistent artifacts. The two callbacks configure measurements and file format,
not processing choices. For the supplied test pack, `eval/processing_worker.py` uses only the current public
`task.json`, `input.npz` and `submission_helper.py` to implement those callbacks. It never imports private references,
the generator, fixed baseline, or grader. `eval/processing_tools.py` runs that real MCP service in the isolated
single-case environment, and runs the Python comparison arm in the same pinned numerical environment.

The test pack's `modified_mask` is computed by its shared helper from actual input/output differences. Its before/
after metrics may use different valid supports; the separate grader checks fixed reference domains. Generic
processing logs are not a replacement for independent quality scoring. This first version is an engineering
prototype, with no claim of clinical calibration or support for multichannel resampling/real-time filtering.
