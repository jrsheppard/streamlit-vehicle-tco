import pytest

from tco.depreciation import declining_balance_value, resale_value


def test_declining_balance_matches_the_documented_formula():
    assert declining_balance_value(40_000, 0.15, 5) == pytest.approx(
        40_000 * 0.85**5
    )


def test_depreciation_is_price_minus_ending_value():
    result = resale_value(40_000, 0.15, 5)
    assert result.is_override is False
    assert result.depreciation == pytest.approx(40_000 - result.ending_value)


def test_override_replaces_the_formula_without_double_counting():
    result = resale_value(40_000, 0.15, 5, override=18_000)
    assert result.is_override is True
    assert result.ending_value == 18_000
    assert result.depreciation == pytest.approx(22_000)


def test_override_is_capped_to_the_purchase_price():
    high = resale_value(40_000, 0.15, 5, override=90_000)
    assert high.ending_value == 40_000
    assert high.was_capped is True
    low = resale_value(40_000, 0.15, 5, override=-5_000)
    assert low.ending_value == 0
    assert low.was_capped is True


@pytest.mark.parametrize("rate", [-0.1, 1.5])
def test_invalid_depreciation_rate_is_rejected(rate):
    with pytest.raises(ValueError):
        declining_balance_value(40_000, rate, 5)
