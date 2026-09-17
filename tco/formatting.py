"""Display formatting helpers. Calculations keep full precision; only these round."""

from __future__ import annotations

import math


def _is_missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def currency(value: object, decimals: int = 0) -> str:
    """Format dollars, for example ``$41,238``."""
    if _is_missing(value):
        return "—"
    return f"${float(value):,.{decimals}f}"


def per_mile(value: object) -> str:
    """Format a cost per mile in cents, for example ``$0.612/mi``."""
    if _is_missing(value):
        return "—"
    return f"${float(value):,.3f}/mi"


def percent(value: object, decimals: int = 2) -> str:
    """Format a 0-1 fraction as a percentage."""
    if _is_missing(value):
        return "—"
    return f"{float(value) * 100:.{decimals}f}%"
