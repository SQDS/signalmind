"""Boundaries: the runtime never sees evaluation data, never imports mcp, and the sealed files are untouched."""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = sorted((ROOT / "src/tsmcp/runtime").glob("*.py")) + sorted((ROOT / "src/tsmcp/plugins/pressure").glob("*.py"))


def test_runtime_imports_neither_eval_nor_mcp():
    for f in RUNTIME:
        src = f.read_text()
        assert not re.search(r"^\s*(from|import)\s+eval\b", src, re.M), f
        assert not re.search(r"^\s*(from|import)\s+mcp\b", src, re.M), f
        assert not re.search(r"\beval\(|\bexec\(", src), f
        for word in ("answers.jsonl", "p9level2", "manifest.json", "expected_answer", "grounding_rubric"):
            assert word not in src, (f, word)


def test_runtime_has_no_new_runtime_dependencies():
    allowed = {"numpy", "tsmcp"}
    for f in RUNTIME + [ROOT / "src/tsmcp/plugins/optional.py"]:
        for m in re.finditer(r"^\s*(?:from|import)\s+([\w.]+)", f.read_text(), re.M):
            top = m.group(1).split(".")[0]
            assert top in allowed or top in {
                "__future__", "collections", "dataclasses", "decimal", "enum", "hashlib", "json", "math", "pathlib",
                "threading", "time", "typing", "weakref", "secrets",
            }, (f, top)  # fmt: skip


def test_sealed_files_are_byte_identical_to_baseline_tag():
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", "baseline-R-2026-10-12", "src/tsmcp/core", "src/tsmcp/plugins"],
        cwd=ROOT, text=True,
    ).split()  # fmt: skip
    for path in paths:
        baseline = subprocess.check_output(["git", "show", f"baseline-R-2026-10-12:{path}"], cwd=ROOT)
        assert (ROOT / path).read_bytes() == baseline, path


def test_frozen_harness_unchanged():
    import hashlib

    assert hashlib.sha256((ROOT / "eval/run_claude_code.py").read_bytes()).hexdigest().startswith("7e4bfeea06d2742a")


def test_default_pyproject_dependencies_unchanged():
    text = (ROOT / "pyproject.toml").read_text()
    block = text[text.index("dependencies = [") : text.index("]", text.index("dependencies = ["))]
    assert sorted(re.findall(r'"([a-z]+)', block)) == ["matplotlib", "mcp", "numpy"]
