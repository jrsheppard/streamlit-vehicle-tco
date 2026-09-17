"""Derive the vehicle cost-assumption catalog for recent model years.

The EPA catalog lists specifications but no money. This script fills the cost
layer for every 2025-2027 configuration the engine can price, using two models
from :mod:`tco.cost_model`: a hedonic regression on curated MSRP anchors for
purchase price, and AAA's published category tables for depreciation,
maintenance, insurance and registration.

It reads only committed files, writes atomically, and records its inputs and
out-of-sample error in ``data/provenance.json``. Re-running it with unchanged
inputs reproduces the same output byte for byte.

    venv/bin/python -m scripts.build_cost_assumptions --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.refresh_data import record_provenance, write_csv_atomically  # noqa: E402
from tco import epa_catalog, taxonomy  # noqa: E402
from tco.assumptions import CATALOG_COLUMNS, coerce_catalog_dtypes, validate_catalog  # noqa: E402
from tco.cost_model import (  # noqa: E402
    OPERATING_TARGETS,
    OperatingCostModel,
    PriceModel,
)
from tco.epa_catalog import SUPPORTED_POWERTRAINS  # noqa: E402
from tco.paths import (  # noqa: E402
    BRAND_GROUPS_PATH,
    COST_ASSUMPTIONS_PATH,
    DATA_DIR,
    EPA_CATALOG_PATH,
    FIELD_PROVENANCE_PATH,
    SALES_TAX_PATH,
)

MODEL_YEARS: tuple[int, ...] = (2025, 2026, 2027)
COLLAPSE_KEYS: tuple[str, ...] = (
    "model_year",
    "make",
    "base_model",
    "powertrain",
    "drive",
)

AAA_COSTS_PATH = DATA_DIR / "sources" / "aaa_your_driving_costs_2025.csv"
AAA_STUDY_PATH = DATA_DIR / "sources" / "aaa_your_driving_costs_2025_study.csv"
VCLASS_SEGMENT_PATH = DATA_DIR / "taxonomy_vclass_to_segment.csv"
PRICE_ANCHORS_PATH = DATA_DIR / "price_anchors.csv"

AAA_SOURCE_NAME = "AAA Your Driving Costs 2025"
AAA_SOURCE_URL = (
    "https://newsroom.aaa.com/wp-content/uploads/2025/09/"
    "AAA-Brochure-Your-Driving-Cost-9.2025.pdf"
)
AAA_EFFECTIVE_DATE = "2025-09-16"

FIELD_UNITS: dict[str, str] = {
    "purchase_price_usd": "USD",
    "annual_depreciation_rate": "fraction per year",
    "annual_maintenance_usd": "USD per year at 15,000 mi/yr",
    "annual_insurance_usd": "USD per year",
    "annual_registration_fees_usd": "USD per year",
}


class BuildError(RuntimeError):
    """Raised when an input is missing or the derived catalog fails validation."""


def _checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(path: Path) -> Path:
    if not path.exists():
        raise BuildError(f"Required input is missing: {path}")
    return path


def load_candidates() -> pd.DataFrame:
    """One row per priceable 2025-2027 configuration, with taxonomy attached."""
    epa = epa_catalog.read_vehicle_catalog(_require(EPA_CATALOG_PATH))
    frame = epa.loc[
        epa["year"].isin(MODEL_YEARS) & epa["powertrain"].isin(SUPPORTED_POWERTRAINS)
    ].rename(columns={"year": "model_year"})

    # The representative record is the lowest EPA id in the group, which keeps the
    # choice deterministic and keeps every published spec traceable to one record
    # rather than to an average across trims.
    frame = frame.sort_values("source_vehicle_id").drop_duplicates(
        list(COLLAPSE_KEYS), keep="first"
    )

    segments = pd.read_csv(_require(VCLASS_SEGMENT_PATH))
    frame = frame.merge(
        segments.loc[:, ["epa_vclass", "body_segment", "aaa_category"]],
        on="epa_vclass",
        how="left",
    )
    unmapped = sorted(frame.loc[frame["body_segment"].isna(), "epa_vclass"].unique())
    if unmapped:
        raise BuildError(
            f"{VCLASS_SEGMENT_PATH.name} has no mapping for EPA size classes: "
            f"{unmapped}"
        )

    brands = taxonomy.read_brand_groups(_require(BRAND_GROUPS_PATH))
    if "brand_tier" not in brands.columns:
        raise BuildError(f"{BRAND_GROUPS_PATH.name} needs a 'brand_tier' column.")
    lookup = taxonomy.brand_group_map(brands)
    tiers = {
        str(m).strip().casefold(): str(t).strip()
        for m, t in zip(brands["make"], brands["brand_tier"])
    }
    frame["brand_group"] = [
        taxonomy.assign_brand_group(make, lookup) for make in frame["make"]
    ]
    frame["brand_tier"] = [
        tiers.get(str(make).strip().casefold(), taxonomy.UNCLASSIFIED)
        for make in frame["make"]
    ]
    return frame.sort_values(list(COLLAPSE_KEYS)).reset_index(drop=True)


def load_anchors(candidates: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Attach model features to each curated MSRP anchor."""
    anchors = pd.read_csv(_require(PRICE_ANCHORS_PATH))
    merged = anchors.merge(candidates, on=list(COLLAPSE_KEYS), how="left")
    missing = merged["vehicle_label"].isna()
    unmatched = [
        f"{r.model_year} {r.make} {r.base_model} ({r.powertrain}, {r.drive})"
        for r in merged.loc[missing].itertuples()
    ]
    return merged.loc[~missing].reset_index(drop=True), unmatched


def load_operating_inputs() -> tuple[pd.DataFrame, dict[str, float], float]:
    """AAA's fit rows, its study-level constants, and the national tax rate."""
    costs = pd.read_csv(_require(AAA_COSTS_PATH))
    study_rows = pd.read_csv(_require(AAA_STUDY_PATH))
    study = {
        str(r.key): float(r.value)
        for r in study_rows.itertuples()
        if str(r.value).replace(".", "", 1).isdigit()
    }

    weighted = costs.loc[
        (costs["aaa_category"] == "All categories") & (costs["powertrain"] == "All")
    ]
    if weighted.empty:
        raise BuildError(
            f"{AAA_COSTS_PATH.name} needs the sales-weighted average row to "
            "calibrate the study's loan rate."
        )
    study["weighted_average_finance_usd_per_year"] = float(
        weighted["finance_usd_per_year"].iloc[0]
    )

    observations = costs.loc[costs["use_for_fit"].astype(bool)].reset_index(drop=True)
    if observations.empty:
        raise BuildError(f"{AAA_COSTS_PATH.name} has no rows flagged use_for_fit.")

    rates = pd.read_csv(_require(SALES_TAX_PATH))
    return observations, study, float(rates["combined_rate"].mean())


def build_catalog(
    candidates: pd.DataFrame, prices: pd.DataFrame, operating: pd.DataFrame
) -> pd.DataFrame:
    """Assemble rows in the catalog's own schema."""
    frame = pd.DataFrame(index=candidates.index)
    frame["vehicle_id"] = [
        f"epa-{int(v)}" for v in candidates["source_vehicle_id"]
    ]
    frame["source_vehicle_id"] = candidates["source_vehicle_id"].astype("Int64")
    frame["model_year"] = candidates["model_year"].astype("Int64")
    frame["make"] = candidates["make"]
    frame["model"] = candidates["base_model"]
    frame["trim"] = candidates["drive"]
    frame["powertrain"] = candidates["powertrain"]
    frame["body_segment"] = candidates["body_segment"]
    frame["brand_group"] = candidates["brand_group"]
    frame["purchase_price_usd"] = prices["purchase_price_usd"].round(0)

    # Efficiency stays blank so the EPA join remains the single source for specs.
    for column in ("purchase_fees_usd", "kwh_per_100mi", "mpg", "phev_electric_share",
                   "electric_range_mi"):
        frame[column] = pd.NA

    priced = prices["purchase_price_usd"].notna()
    for target in OPERATING_TARGETS:
        values = operating[target].where(priced)
        frame[target] = values.round(4 if target.endswith("rate") else 0)

    frame["data_source"] = (
        f"Price: hedonic model on curated MSRP anchors. Operating costs: "
        f"{AAA_SOURCE_NAME} category tables."
    )
    frame["value_status"] = "Derived"
    frame["notes"] = [
        note
        if note
        else (
            f"Price estimate ${low:,.0f}-${high:,.0f} (80% interval). Operating "
            f"costs are the {segment} / {powertrain} category average, not a "
            "per-model figure."
        )
        for note, low, high, segment, powertrain in zip(
            prices["price_note"],
            prices["price_low_usd"].fillna(0),
            prices["price_high_usd"].fillna(0),
            candidates["aaa_category"],
            candidates["powertrain"],
        )
    ]
    return coerce_catalog_dtypes(frame).loc[:, list(CATALOG_COLUMNS)]


def build_field_provenance(
    catalog: pd.DataFrame, prices: pd.DataFrame, price_effective_date: str
) -> pd.DataFrame:
    """One citation row per vehicle and derived field."""
    rows: list[dict[str, str]] = []
    for position, row in enumerate(catalog.itertuples()):
        interval = (
            f"80% interval ${prices['price_low_usd'].iloc[position]:,.0f}-"
            f"${prices['price_high_usd'].iloc[position]:,.0f}"
            if pd.notna(prices["price_low_usd"].iloc[position])
            else str(prices["price_note"].iloc[position])
        )
        rows.append(
            {
                "vehicle_id": row.vehicle_id,
                "field": "purchase_price_usd",
                "value_status": "Derived",
                "source_name": "Hedonic price model on curated manufacturer MSRPs",
                "source_url": "",
                "effective_date": price_effective_date,
                "units": FIELD_UNITS["purchase_price_usd"],
                "notes": interval,
            }
        )
        for field_name in OPERATING_TARGETS:
            rows.append(
                {
                    "vehicle_id": row.vehicle_id,
                    "field": field_name,
                    "value_status": "Derived",
                    "source_name": AAA_SOURCE_NAME,
                    "source_url": AAA_SOURCE_URL,
                    "effective_date": AAA_EFFECTIVE_DATE,
                    "units": FIELD_UNITS[field_name],
                    "notes": (
                        "Stated at 15,000 mi/yr; half is rescaled to the annual "
                        "mileage in use. Category average, not specific to this "
                        "model."
                        if field_name == "annual_maintenance_usd"
                        else (
                            "Category average scaled by vehicle price; not "
                            "specific to this model."
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report diagnostics without writing any file.",
    )
    args = parser.parse_args(argv)

    try:
        candidates = load_candidates()
        anchors, unmatched = load_anchors(candidates)
        observations, study, tax_rate = load_operating_inputs()
    except (BuildError, FileNotFoundError, ValueError) as error:
        print(f"build_cost_assumptions: {error}", file=sys.stderr)
        return 1

    if unmatched:
        print(f"warning: {len(unmatched)} price anchors matched no EPA record:")
        for label in unmatched[:10]:
            print(f"  {label}")

    price_model = PriceModel().fit(anchors)
    prices = price_model.predict(candidates)

    operating_model = OperatingCostModel().fit(observations, study, tax_rate)
    operating = operating_model.predict(
        candidates.assign(purchase_price_usd=prices["purchase_price_usd"])
    )

    catalog = build_catalog(candidates, prices, operating)
    report = validate_catalog(catalog)
    if not report.is_valid:
        print("build_cost_assumptions: the derived catalog failed validation:", file=sys.stderr)
        print(report.errors.to_string(index=False), file=sys.stderr)
        return 1

    priced = int(catalog["purchase_price_usd"].notna().sum())
    diagnostics = price_model.diagnostics_
    print(f"candidates            {len(catalog):,}")
    print(f"price anchors         {diagnostics.n_anchors:,} ({len(unmatched)} unmatched)")
    print(f"priced rows           {priced:,} ({priced / len(catalog):.1%})")
    print(f"cv MAPE               {diagnostics.cv_mape:.1%}")
    print(f"cv median APE         {diagnostics.cv_median_ape:.1%}")
    print(f"cv interval coverage  {diagnostics.interval_coverage:.1%}")
    print(f"ridge alpha           {diagnostics.ridge_alpha:g}")
    print(f"implied study APR     {operating_model.implied_apr_:.3%}")
    for target in OPERATING_TARGETS:
        print(
            f"  {target:<32} elasticity {operating_model.elasticity_[target]:+.3f}"
            f"   R2 {operating_model.r_squared_[target]:.3f}"
        )
    if report.incomplete.empty:
        print("incomplete rows       0")
    else:
        print(f"incomplete rows       {report.incomplete['vehicle_id'].nunique():,}")

    if args.dry_run:
        print("\ndry run: nothing written")
        return 0

    provenance = build_field_provenance(catalog, prices, AAA_EFFECTIVE_DATE)
    write_csv_atomically(catalog, COST_ASSUMPTIONS_PATH)
    write_csv_atomically(provenance, FIELD_PROVENANCE_PATH)
    record_provenance(
        "vehicle_cost_assumptions",
        {
            "source_name": "Derived cost layer for 2025-2027 EPA configurations",
            "derived_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model_years": list(MODEL_YEARS),
            "collapse_keys": list(COLLAPSE_KEYS),
            "rows": len(catalog),
            "rows_with_price": priced,
            "normalized_file": str(COST_ASSUMPTIONS_PATH.relative_to(DATA_DIR.parent)),
            "field_provenance_file": str(
                FIELD_PROVENANCE_PATH.relative_to(DATA_DIR.parent)
            ),
            "price_model": diagnostics.as_dict(),
            "operating_cost_model": operating_model.diagnostics_.as_dict(),
            "inputs": {
                "epa_vehicles_sha256": _checksum(EPA_CATALOG_PATH),
                "price_anchors_sha256": _checksum(PRICE_ANCHORS_PATH),
                "aaa_costs_sha256": _checksum(AAA_COSTS_PATH),
                "sales_tax_rates_sha256": _checksum(SALES_TAX_PATH),
            },
            "operating_cost_source": {
                "source_name": AAA_SOURCE_NAME,
                "source_url": AAA_SOURCE_URL,
                "effective_date": AAA_EFFECTIVE_DATE,
            },
            "limitation_note": (
                "Purchase prices are modelled estimates, not quotes. Operating "
                "costs are published category averages scaled by price and are "
                "identical for every model in a segment and powertrain cell. "
                "Maintenance is stated at the study's 15,000 mi/yr basis; the "
                "engine rescales half of it to the annual mileage in use."
            ),
        },
    )
    print(f"\nwrote {COST_ASSUMPTIONS_PATH} ({len(catalog):,} rows)")
    print(f"wrote {FIELD_PROVENANCE_PATH} ({len(provenance):,} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
