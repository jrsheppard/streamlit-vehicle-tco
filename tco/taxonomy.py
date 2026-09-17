"""Editable taxonomy: body segments and brand groups.

Both are data files rather than conditionals in code so that membership stays
visible and editable in the app. A make with no mapping stays in
``Other / unclassified`` instead of being forced into a group.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tco.paths import BODY_SEGMENTS_PATH, BRAND_GROUPS_PATH

UNCLASSIFIED = "Other / unclassified"

DEFAULT_BODY_SEGMENTS: tuple[str, ...] = (
    "Small sedan",
    "Midsize sedan",
    "Large sedan",
    "Hatchback",
    "Small SUV / crossover",
    "Midsize SUV",
    "Large SUV",
    "Pickup",
    "Minivan",
    "Sports / performance",
    UNCLASSIFIED,
)

DEFAULT_BRAND_GROUPS: tuple[str, ...] = (
    "Mainstream Japanese",
    "Mainstream American",
    "Mainstream Korean",
    "Mainstream European",
    "Luxury German",
    "Luxury Japanese",
    "Other Luxury",
    "EV specialist",
    UNCLASSIFIED,
)


def read_body_segments(path: Path | str = BODY_SEGMENTS_PATH) -> pd.DataFrame:
    """Read the editable body-segment taxonomy."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(
            {
                "body_segment": list(DEFAULT_BODY_SEGMENTS),
                "description": [""] * len(DEFAULT_BODY_SEGMENTS),
            }
        )
    frame = pd.read_csv(path, dtype="string")
    if "body_segment" not in frame.columns:
        raise ValueError("Body-segment taxonomy needs a 'body_segment' column.")
    if "description" not in frame.columns:
        frame["description"] = ""
    return frame


def read_brand_groups(path: Path | str = BRAND_GROUPS_PATH) -> pd.DataFrame:
    """Read the editable make-to-brand-group mapping."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame({"make": pd.Series(dtype="string"), "brand_group": pd.Series(dtype="string")})
    frame = pd.read_csv(path, dtype="string")
    missing = sorted({"make", "brand_group"} - set(frame.columns))
    if missing:
        raise ValueError(f"Brand-group taxonomy is missing columns: {missing}")
    frame["make"] = frame["make"].str.strip()
    frame["brand_group"] = frame["brand_group"].str.strip()
    return frame


def brand_group_map(frame: pd.DataFrame) -> dict[str, str]:
    """Case-insensitive make -> brand group lookup."""
    return {
        str(make).strip().casefold(): str(group).strip()
        for make, group in zip(frame["make"], frame["brand_group"])
        if str(make).strip() and str(group).strip()
    }


def assign_brand_group(make: object, mapping: dict[str, str]) -> str:
    """Brand group for a make, or ``Other / unclassified`` when unmapped."""
    key = "" if make is None else str(make).strip().casefold()
    return mapping.get(key, UNCLASSIFIED)
