#!/usr/bin/env bash
# Python env for arms D/E: what a coding agent would typically have. tsmcp is NOT installed here (E adds it via PYTHONPATH).
set -euo pipefail
cd "$(dirname "$0")"
uv venv --python 3.12 .venv-d
uv pip install --python .venv-d/bin/python numpy scipy matplotlib wfdb neurokit2 pandas
