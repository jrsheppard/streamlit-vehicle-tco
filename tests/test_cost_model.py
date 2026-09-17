"""Tests for the statistical cost layer.

The most important guarantee here is negative: operating costs must not vary
with the individual model. They are published category averages, so two
different vehicles in the same segment, powertrain and price must receive
identical depreciation, maintenance, insurance and registration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.build_cost_assumptions import (
    load_anchors,
    load_candidates,
    load_operating_inputs,
)
from tco.cost_model import (
    DEPRECIATION_BOUNDS,
    MAX_RELATIVE_HALF_WIDTH,
    OPERATING_TARGETS,
    OperatingCostModel,
    PriceModel,
    loan_total_interest,
    solve_apr,
    solve_price,
)


@pytest.fixture(scope="module")
def candidates() -> pd.DataFrame:
    return load_candidates()


@pytest.fixture(scope="module")
def price_model(candidates: pd.DataFrame) -> PriceModel:
    anchors, _ = load_anchors(candidates)
    return PriceModel().fit(anchors)


@pytest.fixture(scope="module")
def operating_model() -> OperatingCostModel:
    observations, study, tax_rate = load_operating_inputs()
    return OperatingCostModel().fit(observations, study, tax_rate)


# --------------------------------------------------------------------------- #
# Loan arithmetic
# --------------------------------------------------------------------------- #
def test_solving_for_apr_and_price_are_inverses():
    interest = loan_total_interest(40_000, 0.065, 0.15, 60)
    assert solve_apr(40_000, interest, 0.15, 60) == pytest.approx(0.065, abs=1e-4)
    assert solve_price(interest, 0.065, 0.15, 60) == pytest.approx(40_000, rel=1e-4)


def test_a_zero_rate_loan_costs_no_interest():
    assert loan_total_interest(40_000, 0.0, 0.15, 60) == 0.0


def test_the_implied_study_rate_is_a_plausible_car_loan(
    operating_model: OperatingCostModel,
):
    assert 0.03 < operating_model.implied_apr_ < 0.12


# --------------------------------------------------------------------------- #
# Price model
# --------------------------------------------------------------------------- #
def test_price_model_is_accurate_out_of_sample(price_model: PriceModel):
    diagnostics = price_model.diagnostics_
    assert diagnostics.n_anchors >= 300
    assert diagnostics.cv_mape < 0.20
    assert diagnostics.cv_median_ape < 0.15


def test_the_published_interval_is_calibrated(price_model: PriceModel):
    # An 80% interval should contain close to 80% of held-out anchors.
    assert 0.74 <= price_model.diagnostics_.interval_coverage <= 0.86


def test_anchored_vehicles_are_priced_near_their_anchor(
    price_model: PriceModel, candidates: pd.DataFrame
):
    anchors, _ = load_anchors(candidates)
    predicted = price_model.predict(anchors)["purchase_price_usd"]
    error = (predicted - anchors["msrp_usd"]).abs() / anchors["msrp_usd"]
    assert error.median() < 0.12


def test_predictions_are_deterministic(candidates: pd.DataFrame):
    anchors, _ = load_anchors(candidates)
    first = PriceModel().fit(anchors).predict(candidates)["purchase_price_usd"]
    second = PriceModel().fit(anchors).predict(candidates)["purchase_price_usd"]
    pd.testing.assert_series_equal(first, second)


def test_no_price_is_published_for_an_unanchored_cell(
    price_model: PriceModel, candidates: pd.DataFrame
):
    predicted = price_model.predict(candidates)
    unsupported = predicted["price_anchor_support"] == 0
    assert unsupported.any()
    assert predicted.loc[unsupported, "purchase_price_usd"].isna().all()
    assert (predicted.loc[unsupported, "price_note"] != "").all()


def test_published_prices_stay_within_the_interval_limit(
    price_model: PriceModel, candidates: pd.DataFrame
):
    predicted = price_model.predict(candidates)
    published = predicted["purchase_price_usd"].notna()
    assert (
        predicted.loc[published, "price_relative_half_width"]
        <= MAX_RELATIVE_HALF_WIDTH
    ).all()
    assert (
        predicted.loc[published, "price_low_usd"]
        < predicted.loc[published, "purchase_price_usd"]
    ).all()
    assert (
        predicted.loc[published, "price_high_usd"]
        > predicted.loc[published, "purchase_price_usd"]
    ).all()


def test_fitting_needs_enough_anchors(candidates: pd.DataFrame):
    anchors, _ = load_anchors(candidates)
    with pytest.raises(ValueError, match="at least"):
        PriceModel().fit(anchors.head(5))


# --------------------------------------------------------------------------- #
# Operating costs: the no-per-model-overfit guarantee
# --------------------------------------------------------------------------- #
def _cell(price: float, category: str = "Compact SUV", powertrain: str = "Gasoline"):
    return pd.DataFrame(
        {
            "aaa_category": [category],
            "powertrain": [powertrain],
            "purchase_price_usd": [price],
        }
    )


def test_two_models_in_one_cell_get_identical_operating_costs(
    operating_model: OperatingCostModel,
):
    shared = operating_model.predict(
        pd.concat([_cell(35_000), _cell(35_000)], ignore_index=True)
    )
    assert shared.iloc[0].equals(shared.iloc[1])


def test_operating_costs_ignore_everything_except_cell_and_price(
    operating_model: OperatingCostModel,
):
    base = _cell(35_000)
    decorated = base.assign(
        make="Somebody", model="Whatever", vehicle_id="epa-1", drive="All-Wheel Drive"
    )
    pd.testing.assert_frame_equal(
        operating_model.predict(base), operating_model.predict(decorated)
    )


def test_operating_costs_are_real_catalog_wide(
    operating_model: OperatingCostModel, candidates: pd.DataFrame, price_model
):
    prices = price_model.predict(candidates)
    costs = operating_model.predict(
        candidates.assign(purchase_price_usd=prices["purchase_price_usd"])
    )
    priced = prices["purchase_price_usd"].notna()
    for target in OPERATING_TARGETS:
        assert costs.loc[priced, target].notna().all()
        assert (costs.loc[priced, target] > 0).all()

    rate = costs.loc[priced, "annual_depreciation_rate"]
    assert rate.between(*DEPRECIATION_BOUNDS).all()


def test_costs_that_scale_with_price_do_so_monotonically(
    operating_model: OperatingCostModel,
):
    cheap = operating_model.predict(_cell(25_000)).iloc[0]
    dear = operating_model.predict(_cell(100_000)).iloc[0]
    assert dear["annual_insurance_usd"] > cheap["annual_insurance_usd"]
    assert dear["annual_registration_fees_usd"] > cheap["annual_registration_fees_usd"]


def test_no_cost_falls_as_a_vehicle_gets_more_expensive(
    operating_model: OperatingCostModel,
):
    for target in OPERATING_TARGETS:
        assert operating_model.elasticity_[target] >= 0.0


def test_electric_vehicles_depreciate_faster_than_hybrids(
    operating_model: OperatingCostModel,
):
    electric = operating_model.predict(_cell(45_000, powertrain="BEV")).iloc[0]
    hybrid = operating_model.predict(_cell(45_000, powertrain="HEV")).iloc[0]
    assert (
        electric["annual_depreciation_rate"] > hybrid["annual_depreciation_rate"]
    )


def test_plug_in_hybrids_sit_between_hybrid_and_electric(
    operating_model: OperatingCostModel,
):
    rates = {
        powertrain: operating_model.predict(
            _cell(45_000, powertrain=powertrain)
        ).iloc[0]["annual_depreciation_rate"]
        for powertrain in ("HEV", "PHEV", "BEV")
    }
    assert rates["HEV"] <= rates["PHEV"] <= rates["BEV"]


def test_registration_excludes_the_sales_tax_aaa_bundles_in(
    operating_model: OperatingCostModel,
):
    # The app charges sales tax separately from the ZIP code, so the derived
    # registration figure must be below AAA's combined fees-and-taxes line.
    training = operating_model.training_frame_
    assert (
        training["annual_registration_fees_usd"]
        < training["license_registration_taxes_usd_per_year"]
    ).all()


def test_the_recovered_price_basis_matches_the_published_average(
    operating_model: OperatingCostModel,
):
    # Inverting the finance line must land back on prices a buyer would recognise.
    basis = operating_model.training_frame_["price_basis_usd"]
    assert basis.between(15_000, 120_000).all()
    assert 20_000 < basis.median() < 60_000


def test_unknown_cells_fall_back_without_raising(
    operating_model: OperatingCostModel,
):
    predicted = operating_model.predict(_cell(30_000, category="Nonexistent"))
    assert np.isfinite(predicted.iloc[0]["annual_insurance_usd"])
