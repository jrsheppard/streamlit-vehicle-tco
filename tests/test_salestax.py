import pandas as pd
import pytest

from tco.salestax import (
    STATUS_INVALID,
    STATUS_MULTI_STATE,
    STATUS_OK,
    STATUS_UNKNOWN,
    lookup_zip_state,
    normalize_zip,
    rates_for_state,
    resolve_sales_tax,
)

ZIPS = pd.DataFrame(
    {
        "zip_code": pd.Series(["98101", "42223", "99999"], dtype="string"),
        "primary_state": pd.Series(["WA", "TN", "XX"], dtype="string"),
        "states": pd.Series(["WA", "KY|TN", "XX"], dtype="string"),
        "state_count": [1, 2, 1],
    }
)

RATES = pd.DataFrame(
    {
        "state": pd.Series(["WA", "TN"], dtype="string"),
        "state_name": pd.Series(["Washington", "Tennessee"], dtype="string"),
        "state_rate": [0.065, 0.07],
        "avg_local_rate": [0.0301, 0.0261],
        "max_local_rate": [0.041, 0.0275],
        "combined_rate": [0.0951, 0.0961],
        "source_name": ["Tax Foundation"] * 2,
        "source_url": ["https://taxfoundation.org/"] * 2,
        "effective_date": ["January 1, 2026"] * 2,
    }
)


def test_normalize_zip_trims_plus_four():
    assert normalize_zip(" 98101-1234 ") == "98101"


def test_valid_zip_maps_to_one_state():
    lookup = lookup_zip_state("98101", ZIPS)
    assert lookup.status == STATUS_OK
    assert lookup.state == "WA"


def test_multi_jurisdiction_zip_is_flagged_and_explained():
    lookup = lookup_zip_state("42223", ZIPS)
    assert lookup.status == STATUS_MULTI_STATE
    assert lookup.states == ("KY", "TN")
    assert "spans" in lookup.message


def test_unknown_and_invalid_zips_are_distinguished():
    assert lookup_zip_state("00000", ZIPS).status == STATUS_UNKNOWN
    assert lookup_zip_state("abcde", ZIPS).status == STATUS_INVALID
    assert lookup_zip_state("981", ZIPS).status == STATUS_INVALID


def test_state_rate_plus_local_estimate_is_the_default_assumption():
    active = resolve_sales_tax("98101", ZIPS, RATES)
    assert active.state == "WA"
    assert active.state_rate == pytest.approx(0.065)
    assert active.local_rate == pytest.approx(0.0301)
    assert active.rate == pytest.approx(0.0951)
    assert active.is_override is False


def test_local_estimate_can_be_excluded():
    active = resolve_sales_tax("98101", ZIPS, RATES, include_local_estimate=False)
    assert active.rate == pytest.approx(0.065)
    assert active.local_rate == 0.0


def test_user_override_wins_and_is_labeled():
    active = resolve_sales_tax("98101", ZIPS, RATES, override_rate=0.03)
    assert active.rate == pytest.approx(0.03)
    assert active.is_override is True
    assert active.state_rate == pytest.approx(0.065)
    assert any("override" in note for note in active.notes)


def test_unmapped_state_falls_back_and_explains_itself():
    active = resolve_sales_tax("99999", ZIPS, RATES, fallback_rate=0.05)
    assert active.rates is None
    assert active.rate == pytest.approx(0.05)
    assert any("XX" in note for note in active.notes)


def test_unknown_zip_falls_back_to_the_supplied_rate():
    active = resolve_sales_tax("00000", ZIPS, RATES, fallback_rate=0.04)
    assert active.rate == pytest.approx(0.04)
    assert active.state is None


def test_rates_for_a_missing_state_return_none():
    assert rates_for_state("ZZ", RATES) is None
