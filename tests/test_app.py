"""App-level behavior tests using Streamlit's headless AppTest.

AppTest cannot simulate dataframe or chart selections, file uploads, or browser
layout, so those are covered by unit tests and the browser smoke test instead.
"""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from tco.assumptions import (
    CATALOG_COLUMNS,
    coerce_catalog_dtypes,
    latest_model_year_rows,
)
from tco.paths import COST_ASSUMPTIONS_PATH

APP = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")

CATALOG = pd.read_csv(COST_ASSUMPTIONS_PATH)


def vehicle_id(make: str, model: str, powertrain: str, drive: str) -> str:
    """Resolve an id from the generated catalog so tests survive a rebuild."""
    rows = CATALOG[
        (CATALOG["make"] == make)
        & (CATALOG["model"] == model)
        & (CATALOG["powertrain"] == powertrain)
        & (CATALOG["trim"] == drive)
        & CATALOG["purchase_price_usd"].notna()
    ]
    assert not rows.empty, f"no priced catalog row for {make} {model} {powertrain}"
    return str(rows.sort_values("model_year", ascending=False).iloc[0]["vehicle_id"])


GASOLINE_CAR = vehicle_id("Toyota", "Corolla", "Gasoline", "Front-Wheel Drive")
ELECTRIC_CAR = vehicle_id("Tesla", "Model 3", "BEV", "Rear-Wheel Drive")


def run_app(**session_state) -> AppTest:
    app = AppTest.from_file(APP, default_timeout=60)
    for key, value in session_state.items():
        app.session_state[key] = value
    return app.run()


def metric_labels(app: AppTest) -> list[str]:
    return [item.label for item in app.metric]


def ranking_candidate_count(app: AppTest) -> int:
    caption = next(
        item.value
        for item in app.caption
        if "comparison-ready vehicle(s) matching" in item.value
    )
    match = re.search(r"Ranked across ([\d,]+)", caption)
    assert match
    return int(match.group(1).replace(",", ""))


def test_default_render_succeeds():
    app = run_app()
    assert not app.exception
    assert app.title[0].value == "Vehicle total cost of ownership"
    assert "Lowest total cost" in metric_labels(app)
    assert len(app.session_state["shortlist"]) == 4


def test_cost_rankings_render_for_filtered_catalog():
    app = run_app()
    markdown = [item.value for item in app.markdown]

    assert not app.exception
    assert "**Top 10 most expensive cars to drive**" in markdown
    assert "**Top 10 least expensive cars to drive**" in markdown
    assert len(app.get("vega_lite_chart")) == 4
    assert app.toggle(key="include_capital_costs").value is True
    assert ranking_candidate_count(app) > 10


def test_ranking_toggle_does_not_change_shortlist_tco():
    app = run_app()
    comparison_before = app.dataframe[0].value.copy()

    app = app.toggle(key="include_capital_costs").set_value(False).run()
    comparison_after = app.dataframe[0].value

    assert not app.exception
    pd.testing.assert_frame_equal(comparison_after, comparison_before)
    assert any(
        "and are excluded from these rankings" in item.value
        for item in app.caption
    )


def test_cost_rankings_follow_sidebar_filters():
    unfiltered = run_app()
    filtered = run_app(filter_powertrain=["BEV"])

    assert not filtered.exception
    assert 0 < ranking_candidate_count(filtered) < ranking_candidate_count(unfiltered)


def test_default_shortlist_spans_multiple_powertrains():
    app = run_app()
    table = app.dataframe[0].value
    assert table["powertrain"].nunique() >= 3
    assert table["total_cost"].is_monotonic_increasing


def test_filtering_by_powertrain_limits_the_options():
    unfiltered = run_app(shortlist_initialized=True, shortlist=[])
    app = run_app(filter_powertrain=["BEV"], shortlist_initialized=True, shortlist=[])
    assert not app.exception
    options = app.multiselect(key="shortlist").options
    assert 0 < len(options) < len(unfiltered.multiselect(key="shortlist").options)
    assert not any("Corolla" in label for label in options)


def test_shortlist_offers_only_the_latest_year_with_unique_labels():
    app = run_app(shortlist_initialized=True, shortlist=[])
    options = app.multiselect(key="shortlist").options
    latest = latest_model_year_rows(CATALOG)
    expected = {
        " ".join(
            [
                str(int(row.model_year)),
                str(row.make),
                str(row.model),
                "" if pd.isna(row.trim) else str(row.trim),
            ]
        ).strip()
        + f" ({row.powertrain})"
        for row in latest.itertuples()
        if pd.notna(row.purchase_price_usd)
    }

    assert not app.exception
    assert len(options) == len(set(options))
    assert set(options) <= expected


def test_filtering_by_segment_narrows_the_options():
    app = run_app(
        filter_segment=["Pickup"],
        shortlist_initialized=True,
        shortlist=[],
    )
    assert not app.exception
    options = app.multiselect(key="shortlist").options
    assert options
    pickups = set(CATALOG.loc[CATALOG["body_segment"] == "Pickup", "model"])
    assert all(any(model in label for model in pickups) for label in options)


def test_filtering_by_trim_narrows_the_options():
    app = run_app(
        filter_trim=["Rear-Wheel Drive"],
        shortlist_initialized=True,
        shortlist=[],
    )

    assert not app.exception
    options = app.multiselect(key="shortlist").options
    assert options
    assert all("Rear-Wheel Drive" in label for label in options)


def test_filtering_by_purchase_price_limits_the_options():
    latest = latest_model_year_rows(CATALOG)
    minimum = int(latest["purchase_price_usd"].min() // 1_000 * 1_000)
    maximum = 50_000
    price_by_label = {
        " ".join(
            [
                str(int(row.model_year)),
                str(row.make),
                str(row.model),
                str(row.trim),
            ]
        ).strip()
        + f" ({row.powertrain})": row.purchase_price_usd
        for row in latest.itertuples()
    }
    app = run_app(
        filter_price_range=(minimum, maximum),
        shortlist_initialized=True,
        shortlist=[],
    )

    assert not app.exception
    options = app.multiselect(key="shortlist").options
    assert options
    assert all(minimum <= price_by_label[label] <= maximum for label in options)


def test_filters_narrow_one_another_and_drop_impossible_combinations():
    app = run_app(filter_segment=["Pickup"], filter_make=["Porsche"])
    assert not app.exception
    # Porsche sells no pickup, so the brand filter is dropped rather than
    # silently producing an empty comparison.
    assert app.session_state["filter_make"] == []


def test_an_empty_catalog_shows_the_empty_state():
    empty = pd.DataFrame(columns=list(CATALOG_COLUMNS))
    app = run_app(active_catalog=coerce_catalog_dtypes(empty))
    assert not app.exception
    assert app.warning
    assert "No vehicles match" in app.warning[0].value


def test_selecting_a_shortlist_updates_the_comparison():
    app = run_app(shortlist_initialized=True, shortlist=[ELECTRIC_CAR])
    assert not app.exception
    assert len(app.dataframe[0].value) == 1
    assert app.dataframe[0].value.iloc[0]["label"].startswith(
        f"{CATALOG.loc[CATALOG['vehicle_id'] == ELECTRIC_CAR, 'model_year'].iloc[0]} "
        "Tesla Model 3"
    )


def test_no_selection_shows_guidance_instead_of_results():
    app = run_app(shortlist_initialized=True, shortlist=[])
    assert not app.exception
    assert any("Select at least one vehicle" in item.value for item in app.info)


def test_changing_the_gasoline_price_changes_gasoline_costs_only():
    baseline = run_app(shortlist_initialized=True, shortlist=[GASOLINE_CAR])
    before = baseline.dataframe[0].value.iloc[0]

    app = baseline.number_input(key="gasoline_price").set_value(6.0).run()
    after = app.dataframe[0].value.iloc[0]

    assert after["gasoline_cost"] > before["gasoline_cost"]
    assert after["total_cost"] > before["total_cost"]
    assert after["maintenance_cost"] == before["maintenance_cost"]


def test_changing_the_electricity_price_changes_electric_costs():
    baseline = run_app(shortlist_initialized=True, shortlist=[ELECTRIC_CAR])
    before = baseline.dataframe[0].value.iloc[0]

    app = baseline.number_input(key="electricity_price").set_value(0.40).run()
    after = app.dataframe[0].value.iloc[0]

    assert after["electricity_cost"] > before["electricity_cost"]
    assert after["gasoline_cost"] == 0


def test_changing_annual_mileage_changes_energy_and_cost_per_mile():
    baseline = run_app(shortlist_initialized=True, shortlist=[GASOLINE_CAR])
    before = baseline.dataframe[0].value.iloc[0]

    app = baseline.number_input(key="annual_miles").set_value(24_000).run()
    after = app.dataframe[0].value.iloc[0]

    assert after["energy_cost"] == pytest.approx(before["energy_cost"] * 2, rel=1e-6)
    assert after["cost_per_mile"] < before["cost_per_mile"]


def test_changing_the_apr_changes_financing_interest():
    baseline = run_app(shortlist_initialized=True, shortlist=[GASOLINE_CAR])
    before = baseline.dataframe[0].value.iloc[0]

    app = baseline.number_input(key="apr_percent").set_value(0.0).run()
    after = app.dataframe[0].value.iloc[0]

    assert before["financing_interest"] > 0
    assert after["financing_interest"] == pytest.approx(0.0, abs=0.01)


def test_changing_the_ownership_period_changes_annualized_cost():
    baseline = run_app(shortlist_initialized=True, shortlist=[GASOLINE_CAR])
    before = baseline.dataframe[0].value.iloc[0]

    app = baseline.slider(key="ownership_years").set_value(10).run()
    after = app.dataframe[0].value.iloc[0]

    assert after["total_cost"] > before["total_cost"]
    assert after["annual_cost"] < before["annual_cost"]


def test_a_down_payment_above_the_purchase_price_reports_an_error():
    app = run_app(shortlist_initialized=True, shortlist=[GASOLINE_CAR])
    app = app.number_input(key="down_payment").set_value(120_000.0).run()
    assert not app.exception
    assert app.error
    assert "Down payment cannot exceed" in app.error[0].value


def test_the_sales_tax_override_changes_the_tax_charged():
    baseline = run_app(shortlist_initialized=True, shortlist=[GASOLINE_CAR])
    before = baseline.dataframe[0].value.iloc[0]

    app = baseline.toggle(key="use_tax_override").set_value(True).run()
    app = app.number_input(key="tax_override_percent").set_value(0.0).run()
    after = app.dataframe[0].value.iloc[0]

    assert before["sales_tax"] > 0
    assert after["sales_tax"] == pytest.approx(0.0)


def test_an_unknown_zip_code_warns_without_crashing():
    app = run_app(shortlist_initialized=True, shortlist=[GASOLINE_CAR])
    app = app.text_input(key="zip_code").set_value("00000").run()
    assert not app.exception
    assert any("ZIP" in item.value for item in [*app.caption, *app.warning])


def test_a_filter_change_drops_an_unavailable_selection_and_says_so():
    app = run_app(shortlist_initialized=True, shortlist=[ELECTRIC_CAR])
    app = app.multiselect(key="filter_segment").select("Pickup").run()
    assert not app.exception
    assert any("Removed from the comparison" in item.value for item in app.info)
    assert ELECTRIC_CAR not in app.session_state["shortlist"]
