"""Financing math: amortization, interest, and remaining principal.

All functions are pure and use full float precision. Rounding happens only in the
presentation layer.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Dollar tolerance used when reconciling acquisition cost against cash flows.
RECONCILIATION_TOLERANCE = 0.01


@dataclass(frozen=True)
class FinancingResult:
    """Outcome of financing one purchase over an ownership period."""

    acquisition_cost: float
    down_payment: float
    amount_financed: float
    monthly_payment: float
    scheduled_payments: int
    payments_made: int
    principal_paid: float
    interest_paid: float
    remaining_principal: float
    total_payments_made: float

    @property
    def financing_interest(self) -> float:
        """Interest actually incurred during the ownership period."""
        return self.interest_paid


def monthly_payment(principal: float, apr: float, term_months: int) -> float:
    """Standard amortized monthly payment.

    ``apr`` is an annual rate expressed as a fraction (0.059 for 5.9%). A zero
    APR divides the principal evenly across the term.
    """
    if term_months <= 0:
        return 0.0
    if principal <= 0:
        return 0.0
    if apr == 0:
        return principal / term_months
    monthly_rate = apr / 12.0
    growth = (1.0 + monthly_rate) ** term_months
    return principal * monthly_rate * growth / (growth - 1.0)


def remaining_balance(
    principal: float, apr: float, term_months: int, payments_made: int
) -> float:
    """Outstanding principal after ``payments_made`` scheduled payments."""
    if principal <= 0 or term_months <= 0:
        return 0.0
    payments_made = max(0, min(int(payments_made), int(term_months)))
    if payments_made >= term_months:
        return 0.0
    if apr == 0:
        payment = principal / term_months
        return max(principal - payment * payments_made, 0.0)
    monthly_rate = apr / 12.0
    payment = monthly_payment(principal, apr, term_months)
    growth = (1.0 + monthly_rate) ** payments_made
    balance = principal * growth - payment * (growth - 1.0) / monthly_rate
    return max(balance, 0.0)


def compute_financing(
    purchase_price: float,
    sales_tax: float,
    purchase_fees: float,
    down_payment: float,
    apr: float,
    loan_term_months: int,
    ownership_months: int,
    finance_taxes_and_fees: bool = True,
) -> FinancingResult:
    """Model the loan for one vehicle over the ownership period.

    Sales tax and purchase fees are financed by default; set
    ``finance_taxes_and_fees`` to ``False`` to pay them in cash at purchase.
    A zero-month term (or a down payment covering the financed amount) is a cash
    purchase and incurs no interest.
    """
    if purchase_price < 0:
        raise ValueError("Purchase price must be zero or greater.")
    if down_payment < 0:
        raise ValueError("Down payment must be zero or greater.")
    if down_payment > purchase_price:
        raise ValueError("Down payment cannot exceed the purchase price.")
    if apr < 0:
        raise ValueError("APR must be zero or greater.")
    if loan_term_months < 0:
        raise ValueError("Loan term must be zero or greater.")
    if ownership_months <= 0:
        raise ValueError("Ownership period must be greater than zero months.")

    acquisition_cost = purchase_price + sales_tax + purchase_fees
    financeable = (
        acquisition_cost if finance_taxes_and_fees else float(purchase_price)
    )
    amount_financed = max(financeable - down_payment, 0.0)

    if loan_term_months == 0 or amount_financed == 0:
        return FinancingResult(
            acquisition_cost=acquisition_cost,
            down_payment=min(down_payment, acquisition_cost),
            amount_financed=0.0,
            monthly_payment=0.0,
            scheduled_payments=0,
            payments_made=0,
            principal_paid=0.0,
            interest_paid=0.0,
            remaining_principal=0.0,
            total_payments_made=0.0,
        )

    payment = monthly_payment(amount_financed, apr, loan_term_months)
    payments_made = min(int(loan_term_months), int(ownership_months))
    remaining = remaining_balance(amount_financed, apr, loan_term_months, payments_made)
    principal_paid = amount_financed - remaining
    total_paid = payment * payments_made
    interest_paid = total_paid - principal_paid

    return FinancingResult(
        acquisition_cost=acquisition_cost,
        down_payment=down_payment,
        amount_financed=amount_financed,
        monthly_payment=payment,
        scheduled_payments=int(loan_term_months),
        payments_made=payments_made,
        principal_paid=principal_paid,
        interest_paid=interest_paid,
        remaining_principal=remaining,
        total_payments_made=total_paid,
    )


def acquisition_reconciles(
    result: FinancingResult, cash_taxes_and_fees: float = 0.0
) -> bool:
    """Check ``down payment + principal paid + payoff + cash items == acquisition``."""
    total = (
        result.down_payment
        + result.principal_paid
        + result.remaining_principal
        + cash_taxes_and_fees
    )
    return abs(total - result.acquisition_cost) <= RECONCILIATION_TOLERANCE
