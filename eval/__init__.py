"""Evaluation extensions; preserve the historical common.py prompt source bytes."""

from .audit_formats import AUDIT_ANSWER_FORMATS
from .common import ANSWER_FORMATS

# Only new question types are registered. Existing prompts and saved provenance
# remain byte-identical; registration requires no changes to the C harness.
if ANSWER_FORMATS.keys() & AUDIT_ANSWER_FORMATS.keys():
    raise RuntimeError("Audit answer formats must not replace existing qtypes")
ANSWER_FORMATS.update(AUDIT_ANSWER_FORMATS)
