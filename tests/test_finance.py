import pytest

from tco.finance import (
    acquisition_reconciles,
    compute_financing,
    monthly_payment,
    remaining_balance,
)


def test_positive_apr_amortization_matches_closed_form():
    payment = monthly_payment(30_000, 0.06, 60)
    assert payment == pytest.approx(579.98, abs=0.01)


def test_zero_apr_divides_principal_evenly():
    assert monthly_payment(24_000, 0.0, 48) == pytest.approx(500.0)
    assert remaining_balance(24_000, 0.0, 48, 12) == pytest.approx(18_000.0)


def test_remaining_balance_is_zero_after_the_final_payment():
    assert remaining_balance(30_000, 0.06, 60, 60) == pytest.approx(0.0)
    assert remaining_balance(30_000, 0.06, 60, 72) == pytest.approx(0.0)


def test_financed_taxes_and_fees_are_included_in_the_amount_financed():
    result = compute_financing(
        purchase_price=40_000,
        sales_tax=2_600,
        purchase_fees=600,
        down_payment=5_000,
        apr=0.069,
        loan_term_months=60,
        ownership_months=72,
    )
    assert result.acquisition_cost == pytest.approx(43_200)
    assert result.amount_financed == pytest.approx(38_200)
    assert acquisition_reconciles(result)


def test_cash_taxes_and_fees_stay_out_of_the_loan():
    result = compute_financing(
        purchase_price=40_000,
        sales_tax=2_600,
        purchase_fees=600,
        down_payment=5_000,
        apr=0.05,
        loan_term_months=60,
        ownership_months=60,
        finance_taxes_and_fees=False,
    )
    assert result.amount_financed == pytest.approx(35_000)
    assert acquisition_reconciles(result, cash_taxes_and_fees=3_200)


def test_cash_purchase_has_no_interest():
    result = compute_financing(
        purchase_price=30_000,
        sales_tax=1_950,
        purchase_fees=500,
        down_payment=30_000,
        apr=0.07,
        loan_term_months=0,
        ownership_months=60,
    )
    assert result.amount_financed == 0
    assert result.interest_paid == 0
    assert result.monthly_payment == 0


def test_ownership_shorter_than_the_loan_leaves_a_balance_to_pay_off():
    result = compute_financing(
        purchase_price=40_000,
        sales_tax=0,
        purchase_fees=0,
        down_payment=4_000,
        apr=0.06,
        loan_term_months=72,
        ownership_months=36,
    )
    assert result.payments_made == 36
    assert result.remaining_principal > 0
    assert acquisition_reconciles(result)


def test_ownership_longer_than_the_loan_repays_everything():
    result = compute_financing(
        purchase_price=40_000,
        sales_tax=0,
        purchase_fees=0,
        down_payment=4_000,
        apr=0.06,
        loan_term_months=48,
        ownership_months=96,
    )
    assert result.payments_made == 48
    assert result.remaining_principal == pytest.approx(0.0, abs=0.01)
    assert result.principal_paid == pytest.approx(36_000, abs=0.01)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"down_payment": -1},
        {"down_payment": 50_000},
        {"apr": -0.01},
        {"loan_term_months": -12},
        {"ownership_months": 0},
        {"purchase_price": -1},
    ],
)
def test_invalid_financing_inputs_are_rejected(kwargs):
    base = dict(
        purchase_price=40_000,
        sales_tax=0,
        purchase_fees=0,
        down_payment=1_000,
        apr=0.05,
        loan_term_months=60,
        ownership_months=60,
    )
    base.update(kwargs)
    with pytest.raises(ValueError):
        compute_financing(**base)
