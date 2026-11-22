import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("TSMCP_CACHE_DIR", str(tmp_path / "cache"))


@pytest.fixture
def sine_npy(tmp_path):
    """10 min, 250 Hz: 1.2 Hz sine, frequency drops to 0.8 Hz at 300 s, plus mild noise."""
    fs = 250.0
    t = np.arange(int(600 * fs)) / fs
    f = np.where(t < 300, 1.2, 0.8)
    phase = 2 * np.pi * np.cumsum(f) / fs
    x = np.sin(phase) + 0.05 * np.random.default_rng(0).standard_normal(t.size)
    p = tmp_path / "sig.npy"
    np.save(p, x.astype(np.float32))
    return p, fs
