# TS-MCP public conventions (v0, provisional)

These rules are binding for every tool. The plugin protocol and the `tsmcp.core`
Python API are **provisional** during 0.x; tool names/params may evolve (log in CHANGELOG).

## Tools (names use underscores, never dots)
`ts_load`, `ts_describe`, `ts_zoom`, `ts_measure`, `ts_forget` (P1), `ts_find_events` (P2).
The tool list is static for the process lifetime. Plugin detectors/measures appear as enum values.

## Handles
- `id = sha1(abs_path, size, mtime_ns, load_params)[:8]` for files; `sha1(array bytes, fs)[:8]` for inline arrays.
- On load, a JSON recipe `{source, format, fs, channels, params}` is written to
  `$TSMCP_CACHE_DIR` (default `~/.cache/tsmcp/handles/<id>.json`); inline arrays are saved next to it as `.npy`.
- An unknown id is lazily reloaded from its recipe. Handles never depend on process-lifetime state.
- The in-memory store is an LRU with a byte cap; eviction is safe because of lazy reload.

## Refs (evidence)
Self-describing, stable across processes: `<id>:<kind>@<where>`
- measurement: `a3f9c2d1:m.period@300-330` (name after `m.`, window in seconds)
- event: `a3f9c2d1:ev.spike@21603.4`
- view: `a3f9c2d1:zoom@300-330`
No process-wide counters.

## Time
- Relative seconds from record start. Windows are half-open `[t0, t1)`.
- Out-of-range windows are clipped and a `warn` string is added; fully-outside windows are an error.
- Absolute start time (if the file has one) is reported once by `ts_load`/`ts_describe` as `start`, verbatim, no timezone.

## Precision (type-aware, never a global rounding)
- times: decimals = clamp(ceil(log10(fs)), 1, 4) (e.g. fs=250 → 3 decimals, fs=4 → 1)
- measurements & stats: 3 significant figures
- encoding parameters (e.g. numeric-mode `mid`/`half`): full-precision **strings** (repr of float64)

## Channels
- Each channel has its own `fs` and `unit` (unit may be `"?"`).
- Default describe policy: ≤ 8 channels → all; more → the 8 with highest robust std, plus `omitted: N`.
- NaN: allowed; reported as `nan_frac`; measures ignore NaN; a window with >50% NaN is an error for measures.
- Irregular sampling (timestamps given with non-constant spacing, >1% jitter) → error asking the user to resample.

## Quality (only defined quantities, no pseudo-scores)
`nan_frac`, `flat_frac` (fraction of samples in runs ≥ 0.1 s with zero first difference),
`clip_frac` (fraction of samples within 0.1% of min or max of the channel when that value occurs in runs ≥ 3 samples).
No unitless `sqi` / `stationarity` / `regimes` in v0.

## Output format
- Every tool returns pre-serialized compact JSON in one `TextContent` (separators `,` `:` , no spaces),
  plus `ImageContent` (PNG) for image views. No MCP structured output in v0.
- Arrays longer than `top_k` → `"truncated":true,"total":N`.
- Tool annotations: `readOnlyHint=True`, `openWorldHint=False`.

## Errors
Raise `mcp.server.mcpserver.exceptions.ToolError` with an actionable message written for the model
(e.g. `window [700, 800) is outside the signal (0–600 s)`). Other exceptions are masked by the SDK.
Never echo raw file bytes or header free-text in errors.

## Security
- File reads only under `--allow-root` directories (default: current working directory).
- `np.load(..., allow_pickle=False)`; URL sources are rejected in v0.
- Max file size (default 2 GB) and max inline samples (default 100k).
- Header free text (patient fields, comments) is not returned; channel names are sanitized
  (printable ASCII/Unicode letters, digits, `-_ .()/`, max 32 chars).

## Images
- Rendering uses `matplotlib.figure.Figure` + `FigureCanvasAgg` (never `pyplot`), guarded by a lock.
- Default width 1200 px (`detail="normal"`), 2400 px (`detail="high"`); long edge ≤ 2576 px.
- Per-pixel min/max envelope for long windows (reshape, no precomputed pyramid).
- `--no-images` server flag turns image views into a compact text summary.

## Candidate inspection
- `ts_find_events(..., snippets=True)` adds one PNG with up to four returned candidates, ranked by score,
  with each event shaded and two event-window lengths of context on either side. The text response is identical
  with and without snippets. No image is emitted when images are disabled or there are no candidates.
  Point events use their reported resolution (or one second) as a centred display window. Maximum: 1200 × 680 px.
- `ts_zoom(..., view="verify", window=[t0, t1])` shows a focused waveform above frequency and RMS (standard
  deviation) over ten window lengths of context on each side. The focus is shaded in all three panels.
  Feature windows match the focus duration, with a minimum of eight samples. Maximum: 1200 × 860 px, including
  `detail="high"`. This registered view is image-only; `--no-images` returns an actionable error.
- Detectors may return a short `caveat`, passed through unchanged in each event. Candidate scores are relative
  detector statistics, not probabilities or proof of an anomaly. Inspect candidates before making that conclusion.
- Generic `discord` compares pooled, z-normalised windows of one dominant period. Without a period, it uses
  n/200 samples clipped to [16, 2000]; the final window is clipped to [8, n/8]. Scores are nearest-neighbour
  distances divided by their median. Selection suppresses two window lengths on either side on the sampled grid.
  `detail` retains the original window description. Multi-scale and fusion experiments did not improve dev top-3
  recall, so the original ranking and scores are restored; see `docs/tasks/generic-anomaly-report.md`.

## AF burden denominators

`ecg_af_burden` estimates AF-like RR support on the requested, clipped window.
`pct_of_window = 100 * af_s / window_s` answers whole-record burden questions
when the window is the whole record. `pct_of_readable = 100 * af_s / readable_s`
uses only readable time as denominator. Legacy `value` remains the readable-time
percentage, `unit` remains `%`, and `irregular_s` aliases `af_s`. `readable_s`,
`window_s` and `unreadable_s = window_s - readable_s` are seconds (time precision for `fs`);
percentages are reported to 3 significant figures. With no readable time, the library returns `None` for the readable
percentage (omitted in compact tool JSON); whole-window burden remains zero.
Unknown time is not evidence of absence of AF.

`ecg_af_episodes` lists episodes by decreasing merged span, with supported
`af_s` excluding gaps. The default grouping merges gaps strictly below 30 s.
`ecg_irregular` uses the same episodes. Summing per-episode
`af_s` gives the burden numerator; summing episode spans does not. The AF support
thresholds remain those of the frozen round2 detector; this is an RR-based
estimate and can miss regular flutter. No diagnostic probability guarantee is
implied by the inherited development calibration.

## Opt-in compact interface v2

`tsmcp --interface v2` (or `build_server(store, interface="v2")`) selects the versioned presentation.
The default is `v1`: its tool schemas, descriptions and successful response text retain the frozen R
contract, including the absence of a version field. Every successful v2 tool JSON response includes
`"interface_version":"v2"`. Errors remain MCP tool errors. The detector and measurement functions are
identical in both interfaces; only the server presentation changes.

In v2, `ts_measure(..., what=["ecg_af_episodes"])` returns a complete index for each selected channel:
`name`, `channel`, `count`, `units:"s"`, `window`, `ranking`, `ref`, and `episodes`. Each episode has exactly
`id`, `start`, `end`, `span_s`, `af_s`, `unknown_s`, and `merged_from`. There is no `top_k` truncation:
every candidate is retained. Order is decreasing **unrounded** span, with earliest unrounded start breaking
ties; do not re-rank using the displayed rounded values. `merged_from` counts constituent AF support
intervals. `unknown_s` is the intersection of the episode span with the detector's unknown-quality intervals;
readable non-AF gaps are not unknown time. The index omits support arrays and duplicate start/span aliases.

Times and durations in seconds, including `start`, `end`, `span_s`, `af_s`, and `unknown_s`, use the existing
sampling-rate precision rule; non-time measurements use three significant figures. Round only after
ranking and interval integration. The per-channel `window` is the actual sample-aligned analysed window
(floor the start sample, ceil the end sample, clip to the channel). The outer response `window` remains the
requested window clipped to signal duration, as in v1. These can differ for sub-sample requests or channels
of different lengths. Burden fields and their numerical precision remain unchanged from v1.

Retrieve full evidence explicitly with
`ts_measure(id="<signal>", what=["ecg_af_episode_detail"], episode="<episode id>")`.
The ID binds the signal, channel, analysis sample bounds, and original episode content. It resolves across
server restarts through the signal's existing load recipe; repeat neither `window` nor `channels` in a
detail request. A malformed ID, changed episode content, or conflicting arguments is an error.
The detail measurement includes `episode`, `channel`, `window`, `units`, `ref`, `legacy_episode`,
`readable_intervals_s`, and `unknown_intervals_s`. `legacy_episode` reproduces every original v1 episode
field, including `intervals_s`, aliases, scores, and lead evidence. The quality arrays are clipped to the
episode span. **Archival detail is the precision exception:** original floats are retained exactly, so the
full v1 evidence can be recovered; the compact index uses the precision rules above.
The detail `ref` equals its opaque episode ID, a v2 exception to the general `id:kind@where` reference
format. Pass that ID back unchanged; clients need not parse its internal encoding.

Every v2 AF measurement/event response states `af_conventions` once, shared by all its AF results:
episodes merge AF support separated by gaps **strictly less than 30 seconds**; `af_s` excludes gaps while
`span_s` includes them; `pct_of_window = 100*af_s/window_s` uses the entire analysed window and is the
whole-record burden when that window is the full record; `pct_of_readable = value = 100*af_s/readable_s`;
unknown time is not evidence of absence of AF. The tool descriptions state these conventions too.
The burden response retains both denominators and all legacy numeric fields, including `irregular_s`;
only its repeated formula-bearing `method` text is shortened to `"AF support"` in v2.
