from dataclasses import replace

import pytest

from tco.engine import (
    MAINTENANCE_REFERENCE_MILES,
    GlobalAssumptions,
    VehicleAssumptions,
    components_reconcile,
    compute_tco,
    scale_maintenance,
)
from tco.finance import acquisition_reconciles

GLOBALS = GlobalAssumptions(
    gasoline_price=3.30,
    electricity_price=0.17,
    annual_miles=12_000,
    ownership_years=5,
    down_payment=5_000,
    apr=0.069,
    loan_term_months=60,
    sales_tax_rate=0.065,
    purchase_fees=600,
)

GASOLINE_CAR = VehicleAssumptions(
    vehicle_id="gas-1",
    label="Gasoline car",
    powertrain="Gasoline",
    purchase_price=30_000,
    annual_depreciation_rate=0.15,
    annual_maintenance=900,
    annual_insurance=1_800,
    annual_registration_fees=160,
    mpg=32.0,
)

BEV_CAR = VehicleAssumptions(
    vehicle_id="bev-1",
    label="Electric car",
    powertrain="BEV",
    purchase_price=42_000,
    annual_depreciation_rate=0.18,
    annual_maintenance=600,
    annual_insurance=2_100,
    annual_registration_fees=200,
    kwh_per_100mi=28.0,
)


def test_components_sum_to_the_reported_total():
    result = compute_tco(GASOLINE_CAR, GLOBALS)
    assert components_reconcile(result)
    assert result.annual_cost == pytest.approx(result.total_cost / 5)
    assert result.monthly_cost == pytest.approx(result.total_cost / 60)
    assert result.cost_per_mile == pytest.approx(result.total_cost / 60_000)


def test_total_equals_acquisition_plus_interest_and_operating_minus_resale():
    result = compute_tco(BEV_CAR, GLOBALS)
    expected = (
        result.acquisition_cost
        + result.total_financing_interest
        + result.total_energy_cost
        + result.total_maintenance_cost
        + result.total_insurance_cost
        + result.total_registration_cost
        - result.ending_resale_value
    )
    assert result.total_cost == pytest.approx(expected, abs=0.01)


def test_principal_is_not_counted_as_an_economic_cost():
    result = compute_tco(GASOLINE_CAR, GLOBALS)
    assert acquisition_reconciles(result.financing)
    assert result.financing.principal_paid > 0
    assert result.total_cost < (
        result.financing.down_payment
        + result.financing.total_payments_made
        + result.total_energy_cost
        + result.total_maintenance_cost
        + result.total_insurance_cost
        + result.total_registration_cost
    )


def test_cash_flows_reconcile_with_the_loan_and_operating_costs():
    result = compute_tco(GASOLINE_CAR, GLOBALS)
    cash = result.cash_flows
    total_out = cash["net_cash_outflow"].sum()
    expected = (
        result.financing.down_payment
        + result.financing.total_payments_made
        + result.financing.remaining_principal
        + (
            result.total_energy_cost
            + result.total_maintenance_cost
            + result.total_insurance_cost
            + result.total_registration_cost
        )
        - result.ending_resale_value
    )
    assert total_out == pytest.approx(expected, abs=0.02)
    assert cash["cumulative_cash_outflow"].iloc[-1] == pytest.approx(
        total_out, abs=0.02
    )


def test_yearly_cumulative_cost_lands_on_the_total():
    result = compute_tco(BEV_CAR, GLOBALS)
    assert result.yearly["cumulative_cost"].iloc[-1] == pytest.approx(
        result.total_cost, abs=0.01
    )
    assert len(result.yearly) == GLOBALS.ownership_years


def test_resale_override_is_used_without_double_counting():
    vehicle = VehicleAssumptions(**{**GASOLINE_CAR.__dict__, "resale_override": 9_000})
    result = compute_tco(vehicle, GLOBALS)
    assert result.ending_resale_value == 9_000
    assert result.depreciation == pytest.approx(21_000)
    assert components_reconcile(result)
    assert result.yearly["vehicle_value"].iloc[-1] == pytest.approx(9_000)


def test_sold_before_the_loan_ends_includes_the_payoff_in_cash_flow():
    assumptions = GlobalAssumptions(
        **{**GLOBALS.__dict__, "ownership_years": 3, "loan_term_months": 72}
    )
    result = compute_tco(GASOLINE_CAR, assumptions)
    assert result.remaining_loan_balance > 0
    assert result.cash_flows["loan_payoff"].iloc[-1] == pytest.approx(
        result.remaining_loan_balance
    )


def test_zero_apr_produces_no_interest():
    assumptions = GlobalAssumptions(**{**GLOBALS.__dict__, "apr": 0.0})
    result = compute_tco(GASOLINE_CAR, assumptions)
    assert result.total_financing_interest == pytest.approx(0.0, abs=0.01)
    assert components_reconcile(result)


def test_zero_annual_miles_has_no_cost_per_mile():
    assumptions = GlobalAssumptions(**{**GLOBALS.__dict__, "annual_miles": 0})
    result = compute_tco(GASOLINE_CAR, assumptions)
    assert result.total_energy_cost == 0
    assert result.cost_per_mile is None


def test_global_insurance_override_replaces_both_recurring_lines():
    assumptions = GlobalAssumptions(
        **{**GLOBALS.__dict__, "insurance_and_fees_override": 1_500}
    )
    result = compute_tco(GASOLINE_CAR, assumptions)
    assert result.total_insurance_cost == pytest.approx(7_500)
    assert result.total_registration_cost == 0
    assert components_reconcile(result)


def test_phev_share_override_changes_the_energy_split():
    phev = VehicleAssumptions(
        vehicle_id="phev-1",
        label="Plug-in hybrid",
        powertrain="PHEV",
        purchase_price=45_000,
        annual_depreciation_rate=0.15,
        annual_maintenance=900,
        annual_insurance=2_100,
        annual_registration_fees=200,
        mpg=38.0,
        kwh_per_100mi=36.0,
        electric_share=0.30,
    )
    baseline = compute_tco(phev, GLOBALS)
    overridden = compute_tco(
        phev,
        GlobalAssumptions(**{**GLOBALS.__dict__, "phev_electric_share_override": 0.9}),
    )
    assert overridden.total_gasoline_cost < baseline.total_gasoline_cost
    assert overridden.total_electricity_cost > baseline.total_electricity_cost


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ownership_years": 0},
        {"ownership_years": 2.5},
        {"annual_miles": -5},
        {"apr": -0.01},
        {"loan_term_months": -1},
        {"down_payment": -100},
        {"phev_electric_share_override": 1.4},
    ],
)
def test_invalid_global_assumptions_are_rejected(kwargs):
    with pytest.raises(ValueError):
        GlobalAssumptions(**{**GLOBALS.__dict__, **kwargs})


def test_down_payment_above_the_purchase_price_is_rejected():
    assumptions = GlobalAssumptions(**{**GLOBALS.__dict__, "down_payment": 90_000})
    with pytest.raises(ValueError):
        compute_tco(GASOLINE_CAR, assumptions)


# --------------------------------------------------------------------------- #
# Maintenance scales with mileage
# --------------------------------------------------------------------------- #
def test_maintenance_is_unchanged_at_the_reference_mileage():
    assert scale_maintenance(1_000, MAINTENANCE_REFERENCE_MILES) == pytest.approx(1_000)


def test_only_the_variable_half_moves_with_mileage():
    # Twice the reference mileage lifts maintenance by half, not by double.
    doubled = scale_maintenance(1_000, MAINTENANCE_REFERENCE_MILES * 2)
    assert doubled == pytest.approx(1_500)
    # A vehicle that never moves still carries the time-based half.
    assert scale_maintenance(1_000, 0) == pytest.approx(500)


def test_maintenance_scaling_is_linear_in_mileage():
    low = scale_maintenance(1_000, 10_000)
    mid = scale_maintenance(1_000, 20_000)
    high = scale_maintenance(1_000, 30_000)
    assert mid - low == pytest.approx(high - mid)


def test_driving_more_raises_the_maintenance_a_comparison_charges():
    fewer = compute_tco(GASOLINE_CAR, GLOBALS)
    more = compute_tco(
        GASOLINE_CAR, replace(GLOBALS, annual_miles=GLOBALS.annual_miles * 2)
    )
    assert more.total_maintenance_cost > fewer.total_maintenance_cost
    # Mileage must not leak into costs that do not depend on it.
    assert more.total_insurance_cost == fewer.total_insurance_cost
    assert more.total_registration_cost == fewer.total_registration_cost
    assert more.depreciation == pytest.approx(fewer.depreciation)


def test_the_yearly_table_reports_the_scaled_maintenance():
    result = compute_tco(GASOLINE_CAR, replace(GLOBALS, annual_miles=30_000))
    expected = scale_maintenance(GASOLINE_CAR.annual_maintenance, 30_000)
    assert result.yearly["maintenance"].eq(expected).all()
    assert result.total_maintenance_cost == pytest.approx(
        expected * GLOBALS.ownership_years
    )
    assert components_reconcile(result)
