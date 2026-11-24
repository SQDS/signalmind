"""Engineering inspection profiles. Lengths and hops are in seconds; thresholds are not clinical cutoffs."""

from functools import cache
from pathlib import Path

import yaml


@cache
def profiles():
    return yaml.safe_load(Path(__file__).with_name("inspection_profiles.yaml").read_text())["profiles"]


def profile(name):
    return profiles()["deep" if name == "thorough" else name]
