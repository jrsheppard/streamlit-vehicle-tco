"""Declining-balance depreciation and ending resale value."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResaleResult:
    """Ending resale value and the depreciation it implies."""

    ending_value: float
    depreciation: float
    is_override: bool
    was_capped: bool


def declining_balance_value(
    purchase_price: float, annual_rate: float, years: float
) -> float:
    """``purchase_price * (1 - annual_rate) ** years``."""
    if purchase_price < 0:
        raise ValueError("Purchase price must be zero or greater.")
    if not 0.0 <= annual_rate <= 1.0:
        raise ValueError("Annual depreciation rate must be between 0 and 1.")
    if years < 0:
        raise ValueError("Ownership period must be zero or greater.")
    return purchase_price * (1.0 - annual_rate) ** years


def resale_value(
    purchase_price: float,
    annual_rate: float,
    years: float,
    override: float | None = None,
) -> ResaleResult:
    """Ending resale value, from the user's override when one is supplied.

    The override is capped to the range ``[0, purchase_price]``; the app does not
    model appreciation. Depreciation is always ``purchase_price - ending value``,
    so resale value is never subtracted a second time.
    """
    if override is None or override != override:
        ending = declining_balance_value(purchase_price, annual_rate, years)
        return ResaleResult(
            ending_value=ending,
            depreciation=purchase_price - ending,
            is_override=False,
            was_capped=False,
        )

    capped = min(max(float(override), 0.0), float(purchase_price))
    return ResaleResult(
        ending_value=capped,
        depreciation=purchase_price - capped,
        is_override=True,
        was_capped=capped != float(override),
    )
