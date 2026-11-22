"""Local Draft 2020-12 registry, distributed from SignalMind Development Kit v1.0."""

import json
import math
from functools import cache
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource


def strict_loads(raw):
    def finite(token):
        value = float(token)
        if not math.isfinite(value):
            raise ValueError(f"Nonfinite number: {token}")
        return value

    def unique(pairs):
        obj = dict(pairs)
        if len(obj) != len(pairs):
            raise ValueError("Duplicate JSON key")
        return obj

    return json.loads(raw, parse_float=finite, parse_constant=finite, object_pairs_hook=unique)


@cache
def validators():
    schemas = [strict_loads(p.read_text()) for p in Path(__file__).with_name("schemas").glob("*.schema.json")]
    registry = Registry().with_resources((s["$id"], Resource.from_contents(s)) for s in schemas)
    return {s["title"]: Draft202012Validator(s, registry=registry, format_checker=FormatChecker()) for s in schemas}


def validate(name, document):
    json.dumps(document, allow_nan=False)
    validators()[name].validate(document)
