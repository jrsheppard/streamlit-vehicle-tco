"""The editable cost-assumption layer and its validation rules.

Sourced vehicle specifications live in :mod:`tco.epa_catalog`. This module owns
the separate, user-editable cost layer keyed by a stable ``vehicle_id`` and
linked to EPA records through ``source_vehicle_id``.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from tco.engine import VehicleAssumptions
from tco.epa_catalog import BEV, GASOLINE, HEV, PHEV, SUPPORTED_POWERTRAINS
from tco.paths import COST_ASSUMPTIONS_PATH, FIELD_PROVENANCE_PATH

ERROR = "error"
INCOMPLETE = "incomplete"

#: Fields a user may override per vehicle without editing the catalog.
OVERRIDE_FIELDS: tuple[str, ...] = (
    "purchase_price_usd",
    "mpg",
    "kwh_per_100mi",
    "phev_electric_share",
    "annual_depreciation_rate",
    "annual_maintenance_usd",
    "annual_insurance_usd",
    "annual_registration_fees_usd",
    "resale_value_usd",
)

VALUE_STATUSES: tuple[str, ...] = (
    "Sourced",
    "Derived",
    "Illustrative default",
    "User override",
)

TEXT_COLUMNS: tuple[str, ...] = (
    "vehicle_id",
    "make",
    "model",
    "trim",
    "powertrain",
    "body_segment",
    "brand_group",
    "data_source",
    "value_status",
    "notes",
)

NUMERIC_COLUMNS: tuple[str, ...] = (
    "model_year",
    "source_vehicle_id",
    "purchase_price_usd",
    "kwh_per_100mi",
    "mpg",
    "phev_electric_share",
    "electric_range_mi",
    "annual_depreciation_rate",
    "annual_maintenance_usd",
    "annual_insurance_usd",
    "annual_registration_fees_usd",
    "purchase_fees_usd",
)

REQUIRED_COLUMNS: tuple[str, ...] = (
    "vehicle_id",
    "model_year",
    "make",
    "model",
    "powertrain",
    "body_segment",
    "brand_group",
    "purchase_price_usd",
    "annual_depreciation_rate",
    "annual_maintenance_usd",
    "annual_insurance_usd",
    "annual_registration_fees_usd",
    "data_source",
    "value_status",
)

CATALOG_COLUMNS: tuple[str, ...] = (
    "vehicle_id",
    "source_vehicle_id",
    "model_year",
    "make",
    "model",
    "trim",
    "powertrain",
    "body_segment",
    "brand_group",
    "purchase_price_usd",
    "purchase_fees_usd",
    "kwh_per_100mi",
    "mpg",
    "phev_electric_share",
    "electric_range_mi",
    "annual_depreciation_rate",
    "annual_maintenance_usd",
    "annual_insurance_usd",
    "annual_registration_fees_usd",
    "data_source",
    "value_status",
    "notes",
)

NONNEGATIVE_COLUMNS: tuple[str, ...] = (
    "purchase_price_usd",
    "purchase_fees_usd",
    "annual_maintenance_usd",
    "annual_insurance_usd",
    "annual_registration_fees_usd",
    "kwh_per_100mi",
    "mpg",
    "electric_range_mi",
)

BOUNDED_COLUMNS: tuple[str, ...] = (
    "annual_depreciation_rate",
    "phev_electric_share",
)


@dataclass
class ValidationReport:
    """Outcome of validating a cost-assumption catalog."""

    frame: pd.DataFrame
    issues: pd.DataFrame
    missing_columns: tuple[str, ...] = ()

    @property
    def errors(self) -> pd.DataFrame:
        if self.issues.empty:
            return self.issues
        return self.issues.loc[self.issues["severity"] == ERROR]

    @property
    def incomplete(self) -> pd.DataFrame:
        if self.issues.empty:
            return self.issues
        return self.issues.loc[self.issues["severity"] == INCOMPLETE]

    @property
    def is_valid(self) -> bool:
        return not self.missing_columns and self.errors.empty


def _issue(row_number: object, vehicle_id: object, column: str, severity: str, message: str) -> dict[str, object]:
    return {
        "row": row_number,
        "vehicle_id": "" if vehicle_id is None else str(vehicle_id),
        "field": column,
        "severity": severity,
        "message": message,
    }


def _empty_issues() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row": pd.Series(dtype="Int64"),
            "vehicle_id": pd.Series(dtype="string"),
            "field": pd.Series(dtype="string"),
            "severity": pd.Series(dtype="string"),
            "message": pd.Series(dtype="string"),
        }
    )


def coerce_catalog_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """Add missing optional columns and apply the catalog's dtypes."""
    out = frame.copy()
    for column in CATALOG_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    for column in TEXT_COLUMNS:
        out[column] = out[column].astype("string").str.strip()
    for column in NUMERIC_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["model_year"] = out["model_year"].astype("Int64")
    out["source_vehicle_id"] = out["source_vehicle_id"].astype("Int64")
    return out.loc[:, list(CATALOG_COLUMNS)]


def validate_catalog(frame: pd.DataFrame) -> ValidationReport:
    """Validate a cost-assumption catalog and classify every problem found.

    ``error`` issues make the catalog unusable and must block a replacement.
    ``incomplete`` issues only mean a vehicle is not comparison-ready yet;
    unknown values stay unknown and are never converted to zero.
    """
    missing = tuple(
        column for column in REQUIRED_COLUMNS if column not in frame.columns
    )
    if missing:
        issues = pd.DataFrame(
            [
                _issue(
                    pd.NA,
                    "",
                    column,
                    ERROR,
                    f"Required column '{column}' is missing.",
                )
                for column in missing
            ]
        )
        return ValidationReport(
            frame=pd.DataFrame(columns=list(CATALOG_COLUMNS)),
            issues=issues,
            missing_columns=missing,
        )

    raw = frame.copy()
    records: list[dict[str, object]] = []

    # Numeric conversion failures must be reported before coercion hides them.
    for column in NUMERIC_COLUMNS:
        if column not in raw.columns:
            continue
        original = raw[column]
        converted = pd.to_numeric(original, errors="coerce")
        broken = converted.isna() & original.notna() & (
            original.astype("string").str.strip() != ""
        )
        for position in raw.index[broken]:
            records.append(
                _issue(
                    position + 2,
                    raw.at[position, "vehicle_id"],
                    column,
                    ERROR,
                    f"'{original[position]}' is not a number.",
                )
            )

    clean = coerce_catalog_dtypes(raw)

    blank_ids = clean["vehicle_id"].isna() | (clean["vehicle_id"] == "")
    for position in clean.index[blank_ids]:
        records.append(
            _issue(position + 2, "", "vehicle_id", ERROR, "Vehicle ID is blank.")
        )

    duplicates = clean.loc[~blank_ids, "vehicle_id"].duplicated(keep=False)
    for position in duplicates.index[duplicates]:
        records.append(
            _issue(
                position + 2,
                clean.at[position, "vehicle_id"],
                "vehicle_id",
                ERROR,
                f"Vehicle ID '{clean.at[position, 'vehicle_id']}' is used more than once.",
            )
        )

    unsupported = ~clean["powertrain"].isin(SUPPORTED_POWERTRAINS)
    for position in clean.index[unsupported]:
        records.append(
            _issue(
                position + 2,
                clean.at[position, "vehicle_id"],
                "powertrain",
                ERROR,
                "Powertrain must be one of "
                f"{', '.join(SUPPORTED_POWERTRAINS)}; got "
                f"'{clean.at[position, 'powertrain']}'.",
            )
        )

    for column in NONNEGATIVE_COLUMNS:
        negative = clean[column].notna() & (clean[column] < 0)
        for position in clean.index[negative]:
            records.append(
                _issue(
                    position + 2,
                    clean.at[position, "vehicle_id"],
                    column,
                    ERROR,
                    f"Value {clean.at[position, column]} must be zero or greater.",
                )
            )

    for column in BOUNDED_COLUMNS:
        out_of_range = clean[column].notna() & (
            (clean[column] < 0) | (clean[column] > 1)
        )
        for position in clean.index[out_of_range]:
            records.append(
                _issue(
                    position + 2,
                    clean.at[position, "vehicle_id"],
                    column,
                    ERROR,
                    f"Value {clean.at[position, column]} must be between 0 and 1.",
                )
            )

    zero_efficiency = clean["mpg"].notna() & (clean["mpg"] == 0)
    for position in clean.index[zero_efficiency]:
        records.append(
            _issue(
                position + 2,
                clean.at[position, "vehicle_id"],
                "mpg",
                ERROR,
                "MPG must be greater than zero. Leave it blank when unknown.",
            )
        )
    zero_consumption = clean["kwh_per_100mi"].notna() & (clean["kwh_per_100mi"] == 0)
    for position in clean.index[zero_consumption]:
        records.append(
            _issue(
                position + 2,
                clean.at[position, "vehicle_id"],
                "kwh_per_100mi",
                ERROR,
                "kWh/100 mi must be greater than zero. Leave it blank when unknown.",
            )
        )

    bev = clean["powertrain"] == BEV
    combustion_only = clean["powertrain"].isin([HEV, GASOLINE])
    for position in clean.index[bev & clean["mpg"].notna()]:
        records.append(
            _issue(
                position + 2,
                clean.at[position, "vehicle_id"],
                "mpg",
                ERROR,
                "Battery electric vehicles must not carry a gasoline MPG value.",
            )
        )
    for position in clean.index[bev & clean["phev_electric_share"].notna()]:
        records.append(
            _issue(
                position + 2,
                clean.at[position, "vehicle_id"],
                "phev_electric_share",
                ERROR,
                "Electric-driving share applies to plug-in hybrids only.",
            )
        )
    for position in clean.index[combustion_only & clean["kwh_per_100mi"].notna()]:
        records.append(
            _issue(
                position + 2,
                clean.at[position, "vehicle_id"],
                "kwh_per_100mi",
                ERROR,
                f"{clean.at[position, 'powertrain']} vehicles must not carry a "
                "kWh/100 mi value.",
            )
        )
    for position in clean.index[combustion_only & clean["phev_electric_share"].notna()]:
        records.append(
            _issue(
                position + 2,
                clean.at[position, "vehicle_id"],
                "phev_electric_share",
                ERROR,
                "Electric-driving share applies to plug-in hybrids only.",
            )
        )
    for position in clean.index[combustion_only & clean["electric_range_mi"].notna()]:
        records.append(
            _issue(
                position + 2,
                clean.at[position, "vehicle_id"],
                "electric_range_mi",
                ERROR,
                f"{clean.at[position, 'powertrain']} vehicles have no electric range.",
            )
        )

    required_values = (
        "model_year",
        "make",
        "model",
        "body_segment",
        "brand_group",
        "purchase_price_usd",
        "annual_depreciation_rate",
        "annual_maintenance_usd",
        "annual_insurance_usd",
        "annual_registration_fees_usd",
        "data_source",
        "value_status",
    )
    for column in required_values:
        blank = clean[column].isna() | (
            clean[column].astype("string").str.strip() == ""
        )
        for position in clean.index[blank]:
            records.append(
                _issue(
                    position + 2,
                    clean.at[position, "vehicle_id"],
                    column,
                    INCOMPLETE,
                    f"'{column}' is required before this vehicle can be compared.",
                )
            )

    unknown_status = clean["value_status"].notna() & ~clean["value_status"].isin(
        VALUE_STATUSES
    )
    for position in clean.index[unknown_status]:
        records.append(
            _issue(
                position + 2,
                clean.at[position, "vehicle_id"],
                "value_status",
                INCOMPLETE,
                "Value status should be one of "
                f"{', '.join(VALUE_STATUSES)}.",
            )
        )

    issues = pd.DataFrame(records) if records else _empty_issues()
    return ValidationReport(frame=clean, issues=issues, missing_columns=())


def read_cost_assumptions(path: Path | str = COST_ASSUMPTIONS_PATH) -> pd.DataFrame:
    """Read the seed cost-assumption catalog from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Cost-assumption catalog not found at {path}. "
            "Run `venv/bin/python -m scripts.refresh_data` to create it."
        )
    return coerce_catalog_dtypes(pd.read_csv(path))


def parse_uploaded_catalog(data: bytes | str) -> ValidationReport:
    """Parse and validate an uploaded catalog CSV without applying it."""
    if isinstance(data, bytes):
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise ValueError(f"The file is not UTF-8 text: {error}") from error
    else:
        text = data
    try:
        frame = pd.read_csv(io.StringIO(text))
    except Exception as error:  # pandas raises several parser error types
        raise ValueError(f"The file could not be read as CSV: {error}") from error
    if frame.empty:
        raise ValueError("The uploaded catalog has no rows.")
    return validate_catalog(frame)


def read_field_provenance(path: Path | str = FIELD_PROVENANCE_PATH) -> pd.DataFrame:
    """Read the long-form per-field provenance table."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "vehicle_id",
                "field",
                "value_status",
                "source_name",
                "source_url",
                "effective_date",
                "units",
                "notes",
            ]
        )
    return pd.read_csv(path, dtype="string").fillna("")


@dataclass(frozen=True)
class EffectiveSpec:
    """An efficiency value together with where it came from."""

    value: float | None
    origin: str


def _pick(catalog_value: object, epa_value: object) -> EffectiveSpec:
    if catalog_value is not None and catalog_value == catalog_value and catalog_value != "":
        return EffectiveSpec(float(catalog_value), "Catalog")
    if epa_value is not None and epa_value == epa_value and epa_value != "":
        return EffectiveSpec(float(epa_value), "EPA")
    return EffectiveSpec(None, "Unknown")


def merge_with_epa(
    catalog: pd.DataFrame, epa: pd.DataFrame
) -> pd.DataFrame:
    """Join cost assumptions to EPA specifications and resolve efficiencies.

    Catalog values win when present; otherwise the EPA value is used. The origin
    of each resolved value is recorded so the app can label it.
    """
    epa_columns = [
        "source_vehicle_id",
        "epa_comb_mpg",
        "epa_kwh_per_100mi",
        "epa_combined_uf",
        "epa_electric_range_mi",
        "epa_vclass",
        "drive",
        "transmission",
        "powertrain",
        "year",
        "make",
        "model",
    ]
    available = [column for column in epa_columns if column in epa.columns]
    merged = catalog.merge(
        epa.loc[:, available].rename(
            columns={
                "powertrain": "epa_powertrain",
                "year": "epa_year",
                "make": "epa_make",
                "model": "epa_model",
            }
        ),
        on="source_vehicle_id",
        how="left",
    )

    mpg = [_pick(row.mpg, getattr(row, "epa_comb_mpg", None)) for row in merged.itertuples()]
    kwh = [
        _pick(row.kwh_per_100mi, getattr(row, "epa_kwh_per_100mi", None))
        for row in merged.itertuples()
    ]
    share = [
        _pick(row.phev_electric_share, getattr(row, "epa_combined_uf", None))
        for row in merged.itertuples()
    ]
    electric_range = [
        _pick(row.electric_range_mi, getattr(row, "epa_electric_range_mi", None))
        for row in merged.itertuples()
    ]

    merged["effective_mpg"] = [item.value for item in mpg]
    merged["mpg_origin"] = [item.origin for item in mpg]
    merged["effective_kwh_per_100mi"] = [item.value for item in kwh]
    merged["kwh_origin"] = [item.origin for item in kwh]
    merged["effective_electric_share"] = [item.value for item in share]
    merged["electric_share_origin"] = [item.origin for item in share]
    merged["effective_electric_range_mi"] = [item.value for item in electric_range]
    merged["electric_range_origin"] = [item.origin for item in electric_range]

    merged["epa_linked"] = merged["source_vehicle_id"].notna() & merged.get(
        "epa_powertrain", pd.Series(pd.NA, index=merged.index)
    ).notna()
    merged["vehicle_label"] = [
        " ".join(
            part
            for part in (
                "" if pd.isna(row.model_year) else str(int(row.model_year)),
                str(row.make or ""),
                str(row.model or ""),
                "" if pd.isna(row.trim) else str(row.trim),
            )
            if part
        ).strip()
        for row in merged.itertuples()
    ]
    return merged


SPEC_FIELDS: dict[str, tuple[str, str, str]] = {    "effective_mpg": ("mpg_origin", "MPG", "Combined gasoline efficiency"),
    "effective_kwh_per_100mi": (
        "kwh_origin",
        "kWh/100 mi",
        "Combined electric consumption",
    ),
    "effective_electric_share": (
        "electric_share_origin",
        "fraction of miles",
        "Electric-driving share",
    ),
    "effective_electric_range_mi": (
        "electric_range_origin",
        "miles",
        "Electric or charge-depleting range",
    ),
}

COST_FIELDS: dict[str, str] = {
    "purchase_price_usd": "USD",
    "purchase_fees_usd": "USD",
    "annual_depreciation_rate": "fraction per year",
    "annual_maintenance_usd": "USD per year",
    "annual_insurance_usd": "USD per year",
    "annual_registration_fees_usd": "USD per year",
}


def build_field_provenance(
    merged: pd.DataFrame,
    epa_meta: dict[str, object] | None = None,
    overrides: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Long-form provenance, one row per vehicle and field.

    Values resolved from the EPA catalog carry EPA provenance; cost assumptions
    carry the catalog's own source and value status. Rows in ``overrides`` (from
    ``data/field_provenance.csv``) replace the derived row for the same vehicle
    and field, so verified sources can be recorded without code changes.
    """
    epa_meta = epa_meta or {}
    epa_source = str(epa_meta.get("source_name", "FuelEconomy.gov"))
    epa_url = str(epa_meta.get("source_url", ""))
    epa_date = str(
        epa_meta.get("http_last_modified") or epa_meta.get("retrieved_at") or ""
    )

    rows: list[dict[str, str]] = []
    for row in merged.itertuples():
        vehicle_id = str(row.vehicle_id)
        catalog_source = str(getattr(row, "data_source", "") or "")
        catalog_status = str(getattr(row, "value_status", "") or "")
        for field_name, units in COST_FIELDS.items():
            value = getattr(row, field_name, None)
            rows.append(
                {
                    "vehicle_id": vehicle_id,
                    "field": field_name,
                    "value": "" if pd.isna(value) else str(value),
                    "value_status": "Unknown" if pd.isna(value) else catalog_status,
                    "source_name": catalog_source,
                    "source_url": "",
                    "effective_date": "",
                    "units": units,
                    "notes": str(getattr(row, "notes", "") or ""),
                }
            )
        for field_name, (origin_field, units, label) in SPEC_FIELDS.items():
            value = getattr(row, field_name, None)
            origin = str(getattr(row, origin_field, "Unknown"))
            if origin == "EPA":
                status, source, url, date = "Sourced", epa_source, epa_url, epa_date
            elif origin == "Catalog":
                status, source, url, date = catalog_status, catalog_source, "", ""
            elif origin == "User override":
                status, source, url, date = "User override", "User input", "", ""
            else:
                status, source, url, date = "Unknown", "", "", ""
            rows.append(
                {
                    "vehicle_id": vehicle_id,
                    "field": field_name,
                    "value": "" if value is None or pd.isna(value) else str(value),
                    "value_status": status,
                    "source_name": source,
                    "source_url": url,
                    "effective_date": date,
                    "units": units,
                    "notes": label,
                }
            )

    derived = pd.DataFrame(rows)
    if overrides is None or overrides.empty:
        return derived
    keys = ["vehicle_id", "field"]
    trimmed = derived.merge(
        overrides.loc[:, keys].drop_duplicates(),
        on=keys,
        how="left",
        indicator=True,
    )
    derived = derived.loc[(trimmed["_merge"] == "left_only").to_numpy()]
    return pd.concat([derived, overrides], ignore_index=True).sort_values(
        keys
    ).reset_index(drop=True)


def comparison_readiness(merged: pd.DataFrame) -> pd.DataFrame:
    """Flag which merged rows have everything the engine needs."""
    out = merged.copy()
    reasons: list[str] = []
    ready: list[bool] = []
    for row in out.itertuples():
        missing: list[str] = []
        if pd.isna(row.purchase_price_usd):
            missing.append("purchase price")
        if pd.isna(row.annual_depreciation_rate):
            missing.append("depreciation rate")
        if pd.isna(row.annual_maintenance_usd):
            missing.append("maintenance cost")
        if pd.isna(row.annual_insurance_usd):
            missing.append("insurance cost")
        if pd.isna(row.annual_registration_fees_usd):
            missing.append("registration fees")
        powertrain = row.powertrain
        if powertrain == BEV:
            if not row.effective_kwh_per_100mi:
                missing.append("kWh/100 mi")
        elif powertrain in (HEV, GASOLINE):
            if not row.effective_mpg:
                missing.append("MPG")
        elif powertrain == PHEV:
            if not row.effective_kwh_per_100mi:
                missing.append("kWh/100 mi")
            if not row.effective_mpg:
                missing.append("MPG")
            if row.effective_electric_share is None or pd.isna(
                row.effective_electric_share
            ):
                missing.append("electric-driving share")
        else:
            missing.append("supported powertrain")
        ready.append(not missing)
        reasons.append("Missing: " + ", ".join(missing) if missing else "")
    out["comparison_ready"] = ready
    out["readiness_note"] = reasons
    return out


def _number(value: object) -> float | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def build_vehicle_assumptions(
    row: pd.Series, overrides: dict[str, object] | None = None
) -> VehicleAssumptions:
    """Turn one merged catalog row plus user overrides into an engine input."""
    overrides = {
        key: value
        for key, value in (overrides or {}).items()
        if _number(value) is not None
    }

    def pick(field_name: str, fallback: object) -> float | None:
        if field_name in overrides:
            return _number(overrides[field_name])
        return _number(fallback)

    price = pick("purchase_price_usd", row.get("purchase_price_usd"))
    if price is None:
        raise ValueError(
            f"{row.get('vehicle_label', row.get('vehicle_id'))} has no purchase price."
        )

    return VehicleAssumptions(
        vehicle_id=str(row["vehicle_id"]),
        label=str(row.get("vehicle_label", row["vehicle_id"])),
        powertrain=str(row["powertrain"]),
        purchase_price=price,
        annual_depreciation_rate=pick(
            "annual_depreciation_rate", row.get("annual_depreciation_rate")
        )
        or 0.0,
        annual_maintenance=pick(
            "annual_maintenance_usd", row.get("annual_maintenance_usd")
        )
        or 0.0,
        annual_insurance=pick("annual_insurance_usd", row.get("annual_insurance_usd"))
        or 0.0,
        annual_registration_fees=pick(
            "annual_registration_fees_usd", row.get("annual_registration_fees_usd")
        )
        or 0.0,
        mpg=pick("mpg", row.get("effective_mpg")),
        kwh_per_100mi=pick("kwh_per_100mi", row.get("effective_kwh_per_100mi")),
        electric_share=pick(
            "phev_electric_share", row.get("effective_electric_share")
        ),
        purchase_fees=pick("purchase_fees_usd", row.get("purchase_fees_usd")),
        resale_override=_number(overrides.get("resale_value_usd")),
    )
