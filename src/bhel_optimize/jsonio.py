"""JSON transport helpers.

Python's ``json`` module happily emits ``Infinity``/``NaN`` — which is **not
valid JSON** and makes browsers fail with "unexpected character" while
``json.loads`` (and Flask's test client) accept it silently. That combination
once shipped a broken ``/api/live`` payload: a fault held "until stopped" has
``duration_s = inf``.

Rule: nothing non-finite ever reaches the wire. ``json_safe`` maps non-finite
floats to ``None``, and the app factories set ``allow_nan = False`` so any
future leak fails loudly in tests instead of quietly in a browser.
"""

from __future__ import annotations

import math
from typing import Any


def json_safe(value: Any) -> Any:
    """Recursively replace non-finite floats with ``None`` for JSON output."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value
