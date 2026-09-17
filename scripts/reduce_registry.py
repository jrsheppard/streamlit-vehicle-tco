"""Reduce the Washington registry snapshot to its unique model configurations.

The supplied ``static/extract.csv`` has one row per registered vehicle (294,193
rows, 80 MB) and repeats the same model thousands of times. The app only needs
the distinct models, so this script collapses it to one row per unique model
configuration and drops every per-vehicle identifier and location field.

    venv/bin/python -m scripts.reduce_registry

The reduced file is what the app reads and what the repository commits. The raw
snapshot stays local and is not required at runtime.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.refresh_data import record_provenance, write_csv_atomically  # noqa: E402
from tco.paths import STATIC_DIR, WA_MODELS_PATH  # noqa: E402

#: Columns that describe the model rather than an individual registration.
MODEL_COLUMNS: tuple[str, ...] = (
    "Model Year",
    "Make",
    "Model",
    "Electric Vehicle Type",
    "Clean Alternative Fuel Vehicle (CAFV) Eligibility",
    "Electric Range",
)

#: Per-vehicle identifiers and location fields that are deliberately dropped.
DROPPED_COLUMNS: tuple[str, ...] = (
    "VIN (1-10)",
    "DOL Vehicle ID",
    "County",
    "City",
    "State",
    "Postal Code",
    "Legislative District",
    "Vehicle Location",
    "Electric Utility",
    "2020 GEOID",
)

OUTPUT_COLUMNS: tuple[str, ...] = (
    "model_year",
    "make",
    "model",
    "ev_type",
    "cafv_eligibility",
    "electric_range_mi",
)


def find_source() -> Path:
    """Locate the raw snapshot, accepting either the plain or gzipped copy."""
    for name in ("extract.csv", "extract.csv.gz"):
        candidate = STATIC_DIR / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No registry snapshot found in {STATIC_DIR}. Expected extract.csv or "
        "extract.csv.gz."
    )


def reduce_to_models(raw: pd.DataFrame) -> pd.DataFrame:
    """Collapse registration rows to one row per unique model configuration."""
    missing = [column for column in MODEL_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(
            f"The snapshot is missing expected columns: {missing}. "
            f"Found: {list(raw.columns)}"
        )

    unique = raw.loc[:, list(MODEL_COLUMNS)].drop_duplicates()
    unique.columns = list(OUTPUT_COLUMNS)

    unique["model_year"] = pd.to_numeric(unique["model_year"], errors="coerce").astype(
        "Int64"
    )
    electric_range = pd.to_numeric(unique["electric_range_mi"], errors="coerce")
    # A zero means "not researched" in this source, so it stays unknown.
    unique["electric_range_mi"] = electric_range.where(electric_range > 0).astype(
        "Int64"
    )

    return (
        unique.dropna(subset=["make", "model"])
        .sort_values(["make", "model", "model_year"])
        .reset_index(drop=True)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=None, help="Raw snapshot to reduce."
    )
    args = parser.parse_args(argv)

    source = args.source or find_source()
    raw = pd.read_csv(source, dtype="string")
    models = reduce_to_models(raw)
    if models.empty:
        raise SystemExit("Reduction produced zero rows; the existing file was kept.")

    write_csv_atomically(models, WA_MODELS_PATH)
    record_provenance(
        "wa_registered_models",
        {
            "source_name": (
                "Washington State Department of Licensing, Electric Vehicle "
                "Population Data"
            ),
            "source_file": source.name,
            "derived_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source_rows": int(len(raw)),
            "unique_models": int(len(models)),
            "normalized_file": str(WA_MODELS_PATH.relative_to(STATIC_DIR.parent)),
            "dropped_columns": list(DROPPED_COLUMNS),
            "limitation_note": (
                "One row per unique registered model configuration. Per-vehicle "
                "identifiers, counts, and locations are removed, so this file "
                "shows which models are registered in Washington and never how "
                "many. An electric range of 0 in the source means 'not "
                "researched' and is stored as unknown."
            ),
        },
    )
    print(
        f"Reduced {len(raw):,} registration rows to {len(models):,} unique models "
        f"at {WA_MODELS_PATH.relative_to(STATIC_DIR.parent)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
