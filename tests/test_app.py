"""App-level behavior tests using Streamlit's headless AppTest.

AppTest cannot simulate dataframe or chart selections, file uploads, or browser
layout, so those are covered by unit tests and the browser smoke test instead.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from tco.assumptions import CATALOG_COLUMNS, coerce_catalog_dtypes

APP = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")


def run_app(**session_state) -> AppTest:
    app = AppTest.from_file(APP, default_timeout=60)
    for key, value in session_state.items():
        app.session_state[key] = value
    return app.run()


def metric_labels(app: AppTest) -> list[str]:
    return [item.label for item in app.metric]


def test_default_render_succeeds():
    app = run_app()
    assert not app.exception
    assert app.title[0].value == "Vehicle total cost of ownership"
    assert "Lowest total cost" in metric_labels(app)
    assert len(app.session_state["shortlist"]) == 4


def test_default_shortlist_spans_multiple_powertrains():
    app = run_app()
    table = app.dataframe[0].value
    assert table["powertrain"].nunique() >= 3
    assert table["total_cost"].is_monotonic_increasing


def test_filtering_by_powertrain_limits_the_options():
    app = run_app(filter_powertrain=["BEV"], shortlist_initialized=True, shortlist=[])
    assert not app.exception
    options = app.multiselect(key="shortlist").options
    assert len(options) == 6
    assert not any("Corolla" in label for label in options)


def test_filtering_by_segment_narrows_the_options():
    app = run_app(
        filter_segment=["Pickup"],
        shortlist_initialized=True,
        shortlist=[],
    )
    assert not app.exception
    options = app.multiselect(key="shortlist").options
    assert options
    assert all("F-150" in label or "Maverick" in label for label in options)


def test_filters_narrow_one_another_and_drop_impossible_combinations():
    app = run_app(filter_segment=["Pickup"], filter_make=["Toyota"])
    assert not app.exception
    # Toyota sells no pickup in the seed catalog, so the brand filter is dropped
    # rather than silently producing an empty comparison.
    assert app.session_state["filter_make"] == []


def test_an_empty_catalog_shows_the_empty_state():
    empty = pd.DataFrame(columns=list(CATALOG_COLUMNS))
    app = run_app(active_catalog=coerce_catalog_dtypes(empty))
    assert not app.exception
    assert app.warning
    assert "No vehicles match" in app.warning[0].value


def test_selecting_a_shortlist_updates_the_comparison():
    app = run_app(shortlist_initialized=True, shortlist=["epa-48765"])
    assert not app.exception
    assert len(app.dataframe[0].value) == 1
    assert app.dataframe[0].value.iloc[0]["label"].startswith("2025 Tesla Model 3")


def test_no_selection_shows_guidance_instead_of_results():
    app = run_app(shortlist_initialized=True, shortlist=[])
    assert not app.exception
    assert any("Select at least one vehicle" in item.value for item in app.info)


def test_changing_the_gasoline_price_changes_gasoline_costs_only():
    baseline = run_app(shortlist_initialized=True, shortlist=["epa-48493"])
    before = baseline.dataframe[0].value.iloc[0]

    app = baseline.number_input(key="gasoline_price").set_value(6.0).run()
    after = app.dataframe[0].value.iloc[0]

    assert after["gasoline_cost"] > before["gasoline_cost"]
    assert after["total_cost"] > before["total_cost"]
    assert after["maintenance_cost"] == before["maintenance_cost"]


def test_changing_the_electricity_price_changes_electric_costs():
    baseline = run_app(shortlist_initialized=True, shortlist=["epa-48765"])
    before = baseline.dataframe[0].value.iloc[0]

    app = baseline.number_input(key="electricity_price").set_value(0.40).run()
    after = app.dataframe[0].value.iloc[0]

    assert after["electricity_cost"] > before["electricity_cost"]
    assert after["gasoline_cost"] == 0


def test_changing_annual_mileage_changes_energy_and_cost_per_mile():
    baseline = run_app(shortlist_initialized=True, shortlist=["epa-48493"])
    before = baseline.dataframe[0].value.iloc[0]

    app = baseline.number_input(key="annual_miles").set_value(24_000).run()
    after = app.dataframe[0].value.iloc[0]

    assert after["energy_cost"] == pytest.approx(before["energy_cost"] * 2, rel=1e-6)
    assert after["cost_per_mile"] < before["cost_per_mile"]


def test_changing_the_apr_changes_financing_interest():
    baseline = run_app(shortlist_initialized=True, shortlist=["epa-48493"])
    before = baseline.dataframe[0].value.iloc[0]

    app = baseline.number_input(key="apr_percent").set_value(0.0).run()
    after = app.dataframe[0].value.iloc[0]

    assert before["financing_interest"] > 0
    assert after["financing_interest"] == pytest.approx(0.0, abs=0.01)


def test_changing_the_ownership_period_changes_annualized_cost():
    baseline = run_app(shortlist_initialized=True, shortlist=["epa-48493"])
    before = baseline.dataframe[0].value.iloc[0]

    app = baseline.slider(key="ownership_years").set_value(10).run()
    after = app.dataframe[0].value.iloc[0]

    assert after["total_cost"] > before["total_cost"]
    assert after["annual_cost"] < before["annual_cost"]


def test_a_down_payment_above_the_purchase_price_reports_an_error():
    app = run_app(shortlist_initialized=True, shortlist=["epa-48493"])
    app = app.number_input(key="down_payment").set_value(120_000.0).run()
    assert not app.exception
    assert app.error
    assert "Down payment cannot exceed" in app.error[0].value


def test_the_sales_tax_override_changes_the_tax_charged():
    baseline = run_app(shortlist_initialized=True, shortlist=["epa-48493"])
    before = baseline.dataframe[0].value.iloc[0]

    app = baseline.toggle(key="use_tax_override").set_value(True).run()
    app = app.number_input(key="tax_override_percent").set_value(0.0).run()
    after = app.dataframe[0].value.iloc[0]

    assert before["sales_tax"] > 0
    assert after["sales_tax"] == pytest.approx(0.0)


def test_an_unknown_zip_code_warns_without_crashing():
    app = run_app(shortlist_initialized=True, shortlist=["epa-48493"])
    app = app.text_input(key="zip_code").set_value("00000").run()
    assert not app.exception
    assert any("ZIP" in item.value for item in [*app.caption, *app.warning])


def test_a_filter_change_drops_an_unavailable_selection_and_says_so():
    app = run_app(shortlist_initialized=True, shortlist=["epa-48765"])
    app = app.multiselect(key="filter_segment").select("Pickup").run()
    assert not app.exception
    assert any("Removed from the comparison" in item.value for item in app.info)
    assert "epa-48765" not in app.session_state["shortlist"]
