import json

import numpy as np
import pytest

from tsmcp.core import SignalStore
from tsmcp.core.io.edf import load_edf
from tsmcp.core.io.wfdb import load_wfdb
from tsmcp.core.views.describe import describe


def write_edf(path, signals, fs_list, units, labels, rec_dur=1.0, patient="Jane DOE 1970 secret", phys=(-500.0, 500.0)):
    """Independent EDF writer straight from the spec (the oracle for load_edf)."""
    ns = len(signals)
    spr = [int(fs * rec_dur) for fs in fs_list]
    n_rec = min(len(s) // k for s, k in zip(signals, spr, strict=True))
    f = lambda s, w: s.ljust(w)[:w].encode("ascii")  # noqa: E731
    head = f("0", 8) + f(patient, 80) + f("Startdate 01-JAN-2020 X X X", 80) + f("01.01.20", 8) + f("22.30.00", 8)
    head += f(str(256 * (ns + 1)), 8) + f("", 44) + f(str(n_rec), 8) + f(f"{rec_dur:g}", 8) + f(str(ns), 4)
    sh = b"".join(f(v, 16) for v in labels) + b"".join(f("AgAgCl", 80) for _ in range(ns))
    sh += b"".join(f(u, 8) for u in units) + b"".join(f(f"{phys[0]:g}", 8) for _ in range(ns))
    sh += b"".join(f(f"{phys[1]:g}", 8) for _ in range(ns)) + b"".join(f("-32768", 8) for _ in range(ns))
    sh += b"".join(f("32767", 8) for _ in range(ns)) + b"".join(f("HP:0.1Hz", 80) for _ in range(ns))
    sh += b"".join(f(str(k), 8) for k in spr) + b"".join(f("", 32) for _ in range(ns))
    gain = (phys[1] - phys[0]) / 65535
    body = []
    for r in range(n_rec):
        for s, k in zip(signals, spr, strict=True):
            d = np.round((s[r * k : (r + 1) * k] - phys[0]) / gain - 32768).astype("<i2")
            body.append(d.tobytes())
    path.write_bytes(head + sh + b"".join(body))
    return gain


def test_edf_roundtrip_multirate_and_no_patient_leak(tmp_path):
    rng = np.random.default_rng(0)
    a, b = 100 * rng.standard_normal(256 * 30), 50 * np.sin(np.arange(4 * 30))
    p = tmp_path / "x.edf"
    gain = write_edf(p, [a, b], [256, 4], ["uV", "mV"], ["EEG C3", "Resp"])
    chans, start = load_edf(p, None, None, None)
    assert [(c.name, c.fs, c.unit) for c in chans] == [("EEG C3", 256, "uV"), ("Resp", 4, "mV")]
    assert np.max(np.abs(chans[0].data - a)) <= gain
    assert start == "01.01.20 22.30.00"
    s = SignalStore(allow_roots=[tmp_path]).load(source=str(p))
    out = json.dumps(describe(s))
    assert "Jane" not in out and "secret" not in out and "22.30.00" in out


def test_edf_channel_selection(tmp_path):
    p = tmp_path / "x.edf"
    write_edf(p, [np.zeros(512), np.ones(512)], [256, 256], ["uV", "uV"], ["A", "B"])
    chans, _ = load_edf(p, None, ["B"], None)
    assert [c.name for c in chans] == ["B"]


@pytest.mark.parametrize("fmt", ["16", "212"])
def test_wfdb_matches_wfdb_package(tmp_path, fmt):
    wfdb = pytest.importorskip("wfdb")
    rng = np.random.default_rng(1)
    sig = np.column_stack([np.sin(np.arange(3000) / 20), 0.5 * rng.standard_normal(3000)])
    wfdb.wrsamp("rec", fs=360, units=["mV", "mV"], sig_name=["MLII", "V5"], p_signal=sig, fmt=[fmt, fmt],
                adc_gain=[200, 200], baseline=[1024, 1024] if fmt == "212" else [0, 0], write_dir=str(tmp_path))  # fmt: skip
    ref = wfdb.rdrecord(str(tmp_path / "rec")).p_signal
    chans, _ = load_wfdb(tmp_path / "rec.hea", None, None, None)
    assert [c.name for c in chans] == ["MLII", "V5"] and chans[0].fs == 360
    for i, c in enumerate(chans):
        assert np.max(np.abs(c.data - ref[:, i])) < 1e-5
