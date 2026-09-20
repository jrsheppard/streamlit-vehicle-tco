import pandas as pd
import pytest

from tco.assumptions import (
    ERROR,
    INCOMPLETE,
    read_cost_assumptions,
    read_field_provenance,
    build_vehicle_assumptions,
    comparison_readiness,
    latest_model_year_rows,
    merge_with_epa,
    parse_uploaded_catalog,
    validate_catalog,
)
from tco.epa_catalog import (
    BEV,
    GASOLINE,
    HEV,
    OTHER,
    PHEV,
    classify_powertrain,
    normalize_epa_vehicles,
)
from tco.paths import COST_ASSUMPTIONS_PATH, FIELD_PROVENANCE_PATH

BASE_ROW = {
    "vehicle_id": "v1",
    "source_vehicle_id": 1,
    "model_year": 2025,
    "make": "Toyota",
    "model": "Corolla",
    "trim": "",
    "powertrain": GASOLINE,
    "body_segment": "Small sedan",
    "brand_group": "Mainstream Japanese",
    "purchase_price_usd": 24_000,
    "purchase_fees_usd": "",
    "kwh_per_100mi": "",
    "mpg": 35,
    "phev_electric_share": "",
    "electric_range_mi": "",
    "annual_depreciation_rate": 0.15,
    "annual_maintenance_usd": 800,
    "annual_insurance_usd": 1_800,
    "annual_registration_fees_usd": 160,
    "data_source": "Illustrative seed assumption",
    "value_status": "Illustrative default",
    "notes": "",
}


def catalog(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows) or [BASE_ROW])


# --------------------------------------------------------------------------- #
# Powertrain classifier
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("atv", "fuel1", "fuel2", "fuel", "expected"),
    [
        ("EV", "Electricity", None, "Electricity", BEV),
        (None, "Electricity", None, "Electricity", BEV),
        ("Plug-in Hybrid", "Regular Gasoline", "Electricity", "Gas/Electricity", PHEV),
        ("Hybrid", "Regular Gasoline", None, "Regular", HEV),
        (None, "Regular Gasoline", None, "Regular", GASOLINE),
        (None, "Premium Gasoline", None, "Premium", GASOLINE),
        (None, "Midgrade Gasoline", None, "Midgrade", GASOLINE),
        ("Diesel", "Diesel", None, "Diesel", OTHER),
        ("FFV", "Regular Gasoline", "E85", "Gasoline or E85", OTHER),
        ("CNG", "Natural Gas", None, "Natural Gas", OTHER),
        ("FCV", "Hydrogen", None, "Hydrogen", OTHER),
        ("Hybrid", "Diesel", None, "Diesel", OTHER),
        (None, None, None, None, OTHER),
    ],
)
def test_classifier_precedence(atv, fuel1, fuel2, fuel, expected):
    assert classify_powertrain(atv, fuel1, fuel2, fuel) == expected


def test_normalization_treats_source_zeros_as_unknown():
    raw = pd.DataFrame(
        [
            {
                "id": 1,
                "year": 2025,
                "make": "Tesla",
                "model": "Model 3",
                "baseModel": "Model 3",
                "fuelType": "Electricity",
                "fuelType1": "Electricity",
                "fuelType2": None,
                "atvType": "EV",
                "VClass": "Midsize Cars",
                "drive": "Rear-Wheel Drive",
                "trany": "Automatic (A1)",
                "comb08": 137,
                "comb08U": 136.79,
                "combE": 24.64,
                "combinedUF": 0,
                "range": 363,
                "rangeA": None,
                "charge120": 0,
                "charge240": 10,
                "evMotor": "150 kW",
                "createdOn": "x",
                "modifiedOn": "y",
            }
        ]
    )
    normalized = normalize_epa_vehicles(raw)
    row = normalized.iloc[0]
    assert row["powertrain"] == BEV
    # comb08 is MPGe for a BEV, so it must not become a gasoline MPG.
    assert pd.isna(row["epa_comb_mpg"])
    assert row["epa_kwh_per_100mi"] == pytest.approx(24.64)
    assert row["epa_electric_range_mi"] == 363
    assert pd.isna(row["charge_120v_hours"])


# --------------------------------------------------------------------------- #
# Catalog validation
# --------------------------------------------------------------------------- #
def test_a_clean_catalog_validates():
    report = validate_catalog(catalog())
    assert report.is_valid
    assert report.errors.empty


def test_missing_required_columns_block_the_catalog():
    frame = catalog().drop(columns=["purchase_price_usd"])
    report = validate_catalog(frame)
    assert not report.is_valid
    assert "purchase_price_usd" in report.missing_columns


def test_duplicate_vehicle_ids_are_errors():
    report = validate_catalog(catalog(BASE_ROW, dict(BASE_ROW)))
    assert not report.is_valid
    assert (report.errors["field"] == "vehicle_id").any()


def test_blank_vehicle_id_is_an_error():
    report = validate_catalog(catalog({**BASE_ROW, "vehicle_id": " "}))
    assert not report.is_valid


def test_numeric_conversion_failures_are_errors():
    report = validate_catalog(catalog({**BASE_ROW, "purchase_price_usd": "cheap"}))
    assert not report.is_valid
    assert "is not a number" in report.errors.iloc[0]["message"]


def test_negative_costs_are_errors():
    report = validate_catalog(catalog({**BASE_ROW, "annual_maintenance_usd": -50}))
    assert not report.is_valid


@pytest.mark.parametrize(
    "field", ["annual_depreciation_rate", "phev_electric_share"]
)
def test_rates_outside_zero_to_one_are_errors(field):
    row = {**BASE_ROW, "powertrain": PHEV, "kwh_per_100mi": 30, field: 1.5}
    report = validate_catalog(catalog(row))
    assert not report.is_valid


def test_zero_efficiency_is_an_error_not_an_unknown():
    report = validate_catalog(catalog({**BASE_ROW, "mpg": 0}))
    assert not report.is_valid


def test_powertrain_incompatible_fields_are_errors():
    bev = {**BASE_ROW, "powertrain": BEV, "mpg": 35, "kwh_per_100mi": 28}
    assert not validate_catalog(catalog(bev)).is_valid

    hev = {**BASE_ROW, "powertrain": HEV, "kwh_per_100mi": 30}
    assert not validate_catalog(catalog(hev)).is_valid

    gas_with_range = {**BASE_ROW, "electric_range_mi": 40}
    assert not validate_catalog(catalog(gas_with_range)).is_valid


def test_unsupported_powertrain_is_an_error():
    report = validate_catalog(catalog({**BASE_ROW, "powertrain": "Diesel"}))
    assert not report.is_valid


def test_missing_cost_values_are_incomplete_not_errors():
    report = validate_catalog(catalog({**BASE_ROW, "annual_insurance_usd": None}))
    assert report.is_valid
    assert (report.incomplete["field"] == "annual_insurance_usd").any()
    assert pd.isna(report.frame.iloc[0]["annual_insurance_usd"])


def test_issue_severities_are_only_error_or_incomplete():
    report = validate_catalog(catalog({**BASE_ROW, "value_status": "Guess"}))
    assert set(report.issues["severity"]) <= {ERROR, INCOMPLETE}


# --------------------------------------------------------------------------- #
# Uploads
# --------------------------------------------------------------------------- #
def test_a_valid_upload_parses():
    report = parse_uploaded_catalog(catalog().to_csv(index=False).encode())
    assert report.is_valid


def test_malformed_upload_raises_rather_than_replacing_data():
    with pytest.raises(ValueError):
        parse_uploaded_catalog(b"")
    with pytest.raises(ValueError):
        parse_uploaded_catalog(b"not,a,catalog\n")


def test_upload_with_duplicate_ids_is_reported_as_invalid():
    frame = catalog(BASE_ROW, dict(BASE_ROW))
    report = parse_uploaded_catalog(frame.to_csv(index=False).encode())
    assert not report.is_valid


# --------------------------------------------------------------------------- #
# Merging and readiness
# --------------------------------------------------------------------------- #
EPA = pd.DataFrame(
    [
        {
            "source_vehicle_id": 1,
            "epa_comb_mpg": 35.0,
            "epa_kwh_per_100mi": None,
            "epa_combined_uf": None,
            "epa_electric_range_mi": None,
            "epa_vclass": "Compact Cars",
            "drive": "Front-Wheel Drive",
            "transmission": "Automatic",
            "powertrain": GASOLINE,
            "year": 2025,
            "make": "Toyota",
            "model": "Corolla",
        }
    ]
)


def test_epa_values_fill_blank_catalog_efficiencies_and_are_labeled():
    frame = validate_catalog(catalog({**BASE_ROW, "mpg": ""})).frame
    merged = merge_with_epa(frame, EPA)
    assert merged.iloc[0]["effective_mpg"] == pytest.approx(35.0)
    assert merged.iloc[0]["mpg_origin"] == "EPA"


def test_catalog_values_win_over_epa_values():
    merged = merge_with_epa(validate_catalog(catalog()).frame, EPA)
    assert merged.iloc[0]["effective_mpg"] == pytest.approx(35.0)
    assert merged.iloc[0]["mpg_origin"] == "Catalog"


def test_latest_model_year_rows_keeps_newest_year_per_powertrain():
    frame = catalog(
        BASE_ROW,
        {**BASE_ROW, "vehicle_id": "v2", "model_year": 2026},
        {
            **BASE_ROW,
            "vehicle_id": "v3",
            "model_year": 2025,
            "powertrain": HEV,
        },
    )

    latest = latest_model_year_rows(frame)

    assert latest["vehicle_id"].tolist() == ["v2", "v3"]


def test_unknown_specifications_leave_a_vehicle_incomplete():
    frame = validate_catalog(catalog({**BASE_ROW, "mpg": ""})).frame
    merged = comparison_readiness(merge_with_epa(frame, EPA.iloc[0:0]))
    row = merged.iloc[0]
    assert not row["comparison_ready"]
    assert "MPG" in row["readiness_note"]
    assert pd.isna(row["effective_mpg"])


def test_missing_purchase_price_blocks_engine_input():
    frame = validate_catalog(catalog({**BASE_ROW, "purchase_price_usd": None})).frame
    merged = merge_with_epa(frame, EPA)
    with pytest.raises(ValueError):
        build_vehicle_assumptions(merged.iloc[0])


def test_overrides_replace_catalog_values():
    merged = merge_with_epa(validate_catalog(catalog()).frame, EPA)
    vehicle = build_vehicle_assumptions(
        merged.iloc[0], {"purchase_price_usd": 21_000, "mpg": 44.0}
    )
    assert vehicle.purchase_price == 21_000
    assert vehicle.mpg == 44.0


# --------------------------------------------------------------------------- #
# The generated catalog that ships with the app
# --------------------------------------------------------------------------- #
def test_the_generated_catalog_is_valid():
    frame = read_cost_assumptions(COST_ASSUMPTIONS_PATH)
    report = validate_catalog(frame)
    assert report.is_valid, report.errors.to_string(index=False)
    assert frame["vehicle_id"].is_unique
    assert set(frame["model_year"].dropna().unique()) <= {2025, 2026, 2027}
    assert set(frame["powertrain"].unique()) <= {BEV, PHEV, HEV, GASOLINE}


def test_every_generated_row_is_either_ready_or_explains_itself():
    frame = read_cost_assumptions(COST_ASSUMPTIONS_PATH)
    priced = frame["purchase_price_usd"].notna()
    # A row without a price must say why, and must carry no operating costs
    # either, so nothing downstream mistakes a partial row for a complete one.
    assert frame.loc[~priced, "notes"].str.contains("No price published").all()
    assert frame.loc[~priced, "annual_depreciation_rate"].isna().all()
    assert frame.loc[priced, "annual_insurance_usd"].notna().all()
    assert priced.mean() > 0.80


def test_the_generated_catalog_carries_field_level_provenance():
    frame = read_cost_assumptions(COST_ASSUMPTIONS_PATH)
    provenance = read_field_provenance(FIELD_PROVENANCE_PATH)
    assert set(provenance["vehicle_id"]) == set(frame["vehicle_id"])
    assert provenance["source_name"].notna().all()
    aaa = provenance[provenance["field"] == "annual_insurance_usd"]
    assert aaa["source_url"].str.startswith("https://").all()
