import pandas as pd
import pytest

from scripts.reduce_registry import MODEL_COLUMNS, reduce_to_models
from tco.registry import read_registered_models, summarize_models

RAW = pd.DataFrame(
    [
        # The same model repeated, as it appears once per registered vehicle.
        ["5YJ3E1EA5L", "King", "2020", "TESLA", "MODEL 3", "Battery Electric Vehicle (BEV)", "Clean Alternative Fuel Vehicle Eligible", "266", "102765120"],
        ["5YJ3E1EA9K", "Yakima", "2020", "TESLA", "MODEL 3", "Battery Electric Vehicle (BEV)", "Clean Alternative Fuel Vehicle Eligible", "266", "326796641"],
        ["5YJ3E1EB1K", "Kitsap", "2019", "TESLA", "MODEL 3", "Battery Electric Vehicle (BEV)", "Clean Alternative Fuel Vehicle Eligible", "220", "272512824"],
        # A zero electric range means "not researched" in this source.
        ["KM8K33AG7L", "King", "2024", "ACURA", "ZDX", "Battery Electric Vehicle (BEV)", "Eligibility unknown as battery range has not been researched", "0", "127090015"],
        ["3C3CFFGE9H", "Thurston", "2025", "TOYOTA", "RAV4 PRIME", "Plug-in Hybrid Electric Vehicle (PHEV)", "Clean Alternative Fuel Vehicle Eligible", "42", "294761365"],
    ],
    columns=[
        "VIN (1-10)",
        "County",
        "Model Year",
        "Make",
        "Model",
        "Electric Vehicle Type",
        "Clean Alternative Fuel Vehicle (CAFV) Eligibility",
        "Electric Range",
        "DOL Vehicle ID",
    ],
).astype("string")


def test_reduction_keeps_one_row_per_unique_model():
    models = reduce_to_models(RAW)
    assert len(models) == 4
    tesla = models[(models["make"] == "TESLA") & (models["model"] == "MODEL 3")]
    assert sorted(tesla["model_year"].tolist()) == [2019, 2020]


def test_reduction_drops_per_vehicle_identifiers_and_locations():
    models = reduce_to_models(RAW)
    assert set(models.columns) == {
        "model_year",
        "make",
        "model",
        "ev_type",
        "cafv_eligibility",
        "electric_range_mi",
    }
    for dropped in ("VIN (1-10)", "DOL Vehicle ID", "County"):
        assert dropped not in models.columns


def test_reduction_carries_no_registration_counts():
    models = reduce_to_models(RAW)
    assert not any("count" in column or "registration" in column for column in models.columns)


def test_a_zero_electric_range_becomes_unknown_not_zero():
    models = reduce_to_models(RAW)
    acura = models[models["make"] == "ACURA"].iloc[0]
    assert pd.isna(acura["electric_range_mi"])
    assert (models["electric_range_mi"].dropna() > 0).all()


def test_a_snapshot_missing_expected_columns_is_rejected():
    with pytest.raises(ValueError):
        reduce_to_models(RAW.drop(columns=["Electric Range"]))


def test_model_columns_exclude_identifiers():
    assert "VIN (1-10)" not in MODEL_COLUMNS
    assert "DOL Vehicle ID" not in MODEL_COLUMNS


def test_summary_counts_model_configurations_not_vehicles():
    context = summarize_models(reduce_to_models(RAW))
    assert context.total_models == 4
    tesla = context.models_by_make.set_index("make").at["TESLA", "model_configurations"]
    assert tesla == 2  # two model years, not three registrations
    assert context.unknown_range_models == 1
    assert context.model_year_range == (2019, 2025)


def test_the_shipped_model_list_loads_and_summarizes():
    frame = read_registered_models()
    assert not frame.empty
    assert frame["make"].notna().all()
    context = summarize_models(frame)
    assert context.total_models == len(frame)
    assert context.model_year_range is not None


def test_a_missing_model_list_raises_an_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="reduce_registry"):
        read_registered_models(tmp_path / "absent.csv")
