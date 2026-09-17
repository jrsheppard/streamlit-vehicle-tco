"""Optional Washington State model context.

``static/wa_registered_models.csv`` lists the unique battery electric and
plug-in hybrid model configurations registered in Washington State, derived from
the Department of Licensing Electric Vehicle Population Data by
``scripts/reduce_registry.py``.

It answers "which models are registered in Washington", never "how many". Every
per-vehicle identifier, registration count, and location field is dropped during
the reduction, so this file cannot be read as sales volume or market share. It
carries no price, efficiency, or cost data and never overrides EPA
specifications.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from tco.paths import WA_MODELS_PATH

REQUIRED_COLUMNS: tuple[str, ...] = (
    "model_year",
    "make",
    "model",
    "ev_type",
    "cafv_eligibility",
    "electric_range_mi",
)


@dataclass(frozen=True)
class ModelContext:
    """The unique registered models plus the few aggregates the app shows."""

    models: pd.DataFrame
    total_models: int
    models_by_make: pd.DataFrame
    models_by_ev_type: pd.DataFrame
    unknown_range_models: int
    model_year_range: tuple[int, int] | None


def read_registered_models(path: Path | str = WA_MODELS_PATH) -> pd.DataFrame:
    """Read the reduced unique-model list."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Washington model list not found at {path}. Run "
            "`venv/bin/python -m scripts.reduce_registry` to derive it from the "
            "raw snapshot."
        )
    frame = pd.read_csv(
        path,
        dtype={
            "model_year": "Int64",
            "make": "string",
            "model": "string",
            "ev_type": "string",
            "cafv_eligibility": "string",
            "electric_range_mi": "Int64",
        },
    )
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"The Washington model list is missing columns: {missing}")
    return frame


def summarize_models(frame: pd.DataFrame) -> ModelContext:
    """Count distinct model configurations, never registrations."""
    by_make = (
        frame.groupby("make", dropna=True)
        .size()
        .reset_index(name="model_configurations")
        .sort_values("model_configurations", ascending=False)
        .reset_index(drop=True)
    )
    by_type = (
        frame.groupby("ev_type", dropna=True)
        .size()
        .reset_index(name="model_configurations")
        .sort_values("model_configurations", ascending=False)
        .reset_index(drop=True)
    )
    years = frame["model_year"].dropna()
    year_range = (int(years.min()), int(years.max())) if not years.empty else None

    return ModelContext(
        models=frame,
        total_models=int(len(frame)),
        models_by_make=by_make,
        models_by_ev_type=by_type,
        unknown_range_models=int(frame["electric_range_mi"].isna().sum()),
        model_year_range=year_range,
    )
