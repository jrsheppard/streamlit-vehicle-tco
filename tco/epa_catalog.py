"""Normalized FuelEconomy.gov vehicle-specification catalog.

The raw download is produced by ``scripts/refresh_data.py``. This module owns the
source-field mapping, the deterministic powertrain classifier, and the reader for
the normalized local file. Nothing here touches the network or Streamlit.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from tco.paths import EPA_CATALOG_PATH

BEV = "BEV"
PHEV = "PHEV"
HEV = "HEV"
GASOLINE = "Gasoline"
OTHER = "Other"

#: Powertrains the calculation engine can cost. ``Other`` is kept for auditing.
SUPPORTED_POWERTRAINS: tuple[str, ...] = (BEV, PHEV, HEV, GASOLINE)

GASOLINE_FUELS = frozenset(
    {"Regular Gasoline", "Midgrade Gasoline", "Premium Gasoline"}
)

#: App concept -> FuelEconomy.gov source field.
SOURCE_FIELD_MAP: dict[str, str] = {
    "source_vehicle_id": "id",
    "year": "year",
    "make": "make",
    "model": "model",
    "base_model": "baseModel",
    "fuel_type": "fuelType",
    "fuel_type1": "fuelType1",
    "fuel_type2": "fuelType2",
    "atv_type": "atvType",
    "epa_vclass": "VClass",
    "drive": "drive",
    "transmission": "trany",
    "comb_mpg_rounded": "comb08",
    "comb_mpg_unrounded": "comb08U",
    "comb_kwh_per_100mi": "combE",
    "combined_uf": "combinedUF",
    "range_fuel1": "range",
    "range_fuel2": "rangeA",
    "charge_120v_hours": "charge120",
    "charge_240v_hours": "charge240",
    "ev_motor": "evMotor",
    "source_created_on": "createdOn",
    "source_modified_on": "modifiedOn",
}

#: Columns the normalized catalog file must contain.
CATALOG_COLUMNS: tuple[str, ...] = (
    "source_vehicle_id",
    "year",
    "make",
    "model",
    "base_model",
    "powertrain",
    "fuel_type",
    "fuel_type1",
    "fuel_type2",
    "atv_type",
    "epa_vclass",
    "drive",
    "transmission",
    "epa_comb_mpg",
    "epa_kwh_per_100mi",
    "epa_combined_uf",
    "epa_electric_range_mi",
    "epa_gasoline_range_mi",
    "charge_120v_hours",
    "charge_240v_hours",
    "ev_motor",
    "source_created_on",
    "source_modified_on",
)


def _clean(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def classify_powertrain(
    atv_type: object,
    fuel_type1: object,
    fuel_type2: object = None,
    fuel_type: object = None,
) -> str:
    """Classify one EPA record into exactly one powertrain.

    Precedence is fixed so that overlapping source fields cannot produce two
    answers: BEV, then PHEV, then HEV, then gasoline, then ``Other``.

    ``HEV`` and ``Gasoline`` additionally require a gasoline primary fuel because
    the calculation engine only prices gasoline and electricity. Diesel,
    flex-fuel, CNG, LPG, hydrogen, and ambiguous records fall through to
    ``Other``.
    """
    atv = _clean(atv_type)
    primary = _clean(fuel_type1)
    secondary = _clean(fuel_type2)
    combined = _clean(fuel_type)

    if atv.upper() == "EV" or primary == "Electricity":
        return BEV
    if atv == "Plug-in Hybrid":
        return PHEV
    if atv == "Hybrid":
        return HEV if primary in GASOLINE_FUELS else OTHER
    if atv:
        # FFV, Diesel, CNG, Bifuel, FCV, eFCV and any future value.
        return OTHER
    if secondary or "/" in combined:
        # Dual-fuel record without an atvType label.
        return OTHER
    if primary in GASOLINE_FUELS:
        return GASOLINE
    return OTHER


def _positive_or_nan(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.where(numeric > 0)


def normalize_epa_vehicles(raw: pd.DataFrame) -> pd.DataFrame:
    """Map raw FuelEconomy.gov rows onto the app's normalized schema.

    Zero and blank source values become missing rather than a real zero, because
    the source uses ``0`` for "not applicable" and "not researched".
    """
    missing = sorted(
        source for source in SOURCE_FIELD_MAP.values() if source not in raw.columns
    )
    if missing:
        raise ValueError(f"Source data is missing required columns: {missing}")

    renamed = raw.rename(
        columns={source: app for app, source in SOURCE_FIELD_MAP.items()}
    )
    out = pd.DataFrame(index=renamed.index)

    out["source_vehicle_id"] = pd.to_numeric(
        renamed["source_vehicle_id"], errors="coerce"
    ).astype("Int64")
    out["year"] = pd.to_numeric(renamed["year"], errors="coerce").astype("Int64")

    for column in (
        "make",
        "model",
        "base_model",
        "fuel_type",
        "fuel_type1",
        "fuel_type2",
        "atv_type",
        "epa_vclass",
        "drive",
        "transmission",
        "ev_motor",
        "source_created_on",
        "source_modified_on",
    ):
        out[column] = renamed[column].astype("string").str.strip()

    out["powertrain"] = [
        classify_powertrain(atv, fuel1, fuel2, fuel)
        for atv, fuel1, fuel2, fuel in zip(
            renamed["atv_type"],
            renamed["fuel_type1"],
            renamed["fuel_type2"],
            renamed["fuel_type"],
        )
    ]

    unrounded = _positive_or_nan(renamed["comb_mpg_unrounded"])
    rounded = _positive_or_nan(renamed["comb_mpg_rounded"])
    # comb08 is MPGe for battery electric records, so it is not a gasoline MPG.
    mpg = unrounded.fillna(rounded)
    out["epa_comb_mpg"] = mpg.where(out["powertrain"] != BEV)

    out["epa_kwh_per_100mi"] = _positive_or_nan(renamed["comb_kwh_per_100mi"])
    out["epa_combined_uf"] = _positive_or_nan(renamed["combined_uf"]).where(
        out["powertrain"] == PHEV
    )

    range_fuel1 = _positive_or_nan(renamed["range_fuel1"])
    range_fuel2 = _positive_or_nan(renamed["range_fuel2"])
    # For a BEV, fuel type 1 is electricity. For a PHEV, fuel type 2 is.
    out["epa_electric_range_mi"] = range_fuel1.where(
        out["powertrain"] == BEV, range_fuel2.where(out["powertrain"] == PHEV)
    )
    out["epa_gasoline_range_mi"] = range_fuel1.where(out["powertrain"] == PHEV)

    out["charge_120v_hours"] = _positive_or_nan(renamed["charge_120v_hours"])
    out["charge_240v_hours"] = _positive_or_nan(renamed["charge_240v_hours"])

    out = out.dropna(subset=["source_vehicle_id"])
    out = out.drop_duplicates(subset=["source_vehicle_id"], keep="last")
    return out.loc[:, list(CATALOG_COLUMNS)].sort_values(
        ["year", "make", "model"], ascending=[False, True, True]
    ).reset_index(drop=True)


def vehicle_label(row: pd.Series) -> str:
    """Human-readable label for one catalog row."""
    year = row.get("year")
    year_text = "" if pd.isna(year) else f"{int(year)} "
    return f"{year_text}{row.get('make', '')} {row.get('model', '')}".strip()


def read_vehicle_catalog(path: Path | str = EPA_CATALOG_PATH) -> pd.DataFrame:
    """Read the normalized catalog written by the refresh script."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Normalized vehicle catalog not found at {path}. "
            "Run `venv/bin/python -m scripts.refresh_data` to create it."
        )
    frame = pd.read_csv(
        path,
        dtype={
            "source_vehicle_id": "Int64",
            "year": "Int64",
            "make": "string",
            "model": "string",
            "base_model": "string",
            "powertrain": "string",
            "fuel_type": "string",
            "fuel_type1": "string",
            "fuel_type2": "string",
            "atv_type": "string",
            "epa_vclass": "string",
            "drive": "string",
            "transmission": "string",
            "ev_motor": "string",
            "source_created_on": "string",
            "source_modified_on": "string",
        },
    )
    missing = [column for column in CATALOG_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Vehicle catalog at {path} is missing columns: {missing}")
    frame["vehicle_label"] = frame.apply(vehicle_label, axis=1)
    return frame
