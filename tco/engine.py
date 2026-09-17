"""The total-cost-of-ownership calculation engine.

Two views of the same purchase are produced and kept apart on purpose:

* **Economic TCO** - depreciation, transaction taxes and fees, financing
  interest, energy, maintenance, insurance, and registration. Resale value is
  already inside depreciation, so it is never subtracted again.
* **Cash outflow** - the down payment, scheduled loan payments, any remaining
  loan payoff at disposal, cash-paid taxes and fees, and operating expenses.
  Principal is a cash-flow timing detail and is never added to economic cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from tco.depreciation import ResaleResult, resale_value
from tco.energy import EnergyResult, annual_energy_cost
from tco.epa_catalog import PHEV
from tco.finance import FinancingResult, compute_financing, monthly_payment

TOLERANCE = 0.01


@dataclass(frozen=True)
class GlobalAssumptions:
    """Assumptions that apply to every vehicle in a comparison."""

    gasoline_price: float = 3.30
    electricity_price: float = 0.17
    annual_miles: float = 12000.0
    ownership_years: int = 6
    down_payment: float = 5000.0
    apr: float = 0.069
    loan_term_months: int = 60
    sales_tax_rate: float = 0.065
    purchase_fees: float = 600.0
    finance_taxes_and_fees: bool = True
    phev_electric_share_override: float | None = None
    insurance_and_fees_override: float | None = None

    def __post_init__(self) -> None:
        if self.ownership_years <= 0:
            raise ValueError("Ownership period must be at least one whole year.")
        if int(self.ownership_years) != self.ownership_years:
            raise ValueError("Ownership period is measured in whole years.")
        if self.annual_miles < 0:
            raise ValueError("Annual miles must be zero or greater.")
        if self.apr < 0:
            raise ValueError("APR must be zero or greater.")
        if self.loan_term_months < 0:
            raise ValueError("Loan term must be zero or greater.")
        if self.down_payment < 0:
            raise ValueError("Down payment must be zero or greater.")
        if self.sales_tax_rate < 0:
            raise ValueError("Sales-tax rate must be zero or greater.")
        if self.purchase_fees < 0:
            raise ValueError("Purchase fees must be zero or greater.")
        if self.phev_electric_share_override is not None and not (
            0.0 <= self.phev_electric_share_override <= 1.0
        ):
            raise ValueError("Electric-driving share must be between 0 and 1.")

    @property
    def ownership_months(self) -> int:
        return int(self.ownership_years) * 12

    @property
    def lifetime_miles(self) -> float:
        return self.annual_miles * self.ownership_years


@dataclass(frozen=True)
class VehicleAssumptions:
    """Per-vehicle cost and efficiency assumptions for one comparison."""

    vehicle_id: str
    label: str
    powertrain: str
    purchase_price: float
    annual_depreciation_rate: float
    annual_maintenance: float
    annual_insurance: float
    annual_registration_fees: float
    mpg: float | None = None
    kwh_per_100mi: float | None = None
    electric_share: float | None = None
    purchase_fees: float | None = None
    resale_override: float | None = None


@dataclass(frozen=True)
class TcoResult:
    """Every figure the UI shows for one vehicle, at full precision."""

    vehicle_id: str
    label: str
    powertrain: str
    purchase_price: float
    sales_tax: float
    purchase_fees: float
    acquisition_cost: float
    financing: FinancingResult
    energy: EnergyResult
    resale: ResaleResult
    ownership_years: int
    annual_miles: float
    total_energy_cost: float
    total_electricity_cost: float
    total_gasoline_cost: float
    total_maintenance_cost: float
    total_insurance_cost: float
    total_registration_cost: float
    total_financing_interest: float
    depreciation: float
    ending_resale_value: float
    remaining_loan_balance: float
    total_cost: float
    annual_cost: float
    monthly_cost: float
    cost_per_mile: float | None
    yearly: pd.DataFrame
    cash_flows: pd.DataFrame
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def components(self) -> dict[str, float]:
        """Economic cost components. They sum to ``total_cost``."""
        return {
            "Depreciation": self.depreciation,
            "Energy": self.total_energy_cost,
            "Maintenance": self.total_maintenance_cost,
            "Insurance": self.total_insurance_cost,
            "Registration and fees": self.total_registration_cost,
            "Financing interest": self.total_financing_interest,
            "Sales tax": self.sales_tax,
            "Purchase fees": self.purchase_fees,
        }


def _interest_schedule(
    principal: float, apr: float, term_months: int, months: int
) -> list[tuple[float, float, float]]:
    """Per-month ``(interest, principal, ending balance)`` for ``months`` months."""
    if principal <= 0 or term_months <= 0:
        return [(0.0, 0.0, 0.0) for _ in range(months)]
    payment = monthly_payment(principal, apr, term_months)
    monthly_rate = apr / 12.0
    balance = principal
    rows: list[tuple[float, float, float]] = []
    for month in range(1, months + 1):
        if month > term_months or balance <= 0:
            rows.append((0.0, 0.0, max(balance, 0.0)))
            continue
        interest = balance * monthly_rate
        principal_part = min(payment - interest, balance)
        balance = max(balance - principal_part, 0.0)
        rows.append((interest, principal_part, balance))
    return rows


def _value_path(
    purchase_price: float, ending_value: float, years: int
) -> list[float]:
    """Vehicle value at the end of each ownership year.

    The path is geometric between the purchase price and the ending value, so it
    matches plain declining-balance depreciation when no resale override is set
    and still lands exactly on an override when one is.
    """
    if purchase_price <= 0:
        return [0.0] * years
    ratio = max(ending_value, 0.0) / purchase_price
    return [purchase_price * ratio ** (year / years) for year in range(1, years + 1)]


def compute_tco(
    vehicle: VehicleAssumptions, assumptions: GlobalAssumptions
) -> TcoResult:
    """Full economic and cash-flow model for one vehicle."""
    years = int(assumptions.ownership_years)
    notes: list[str] = []

    electric_share = vehicle.electric_share
    if vehicle.powertrain == PHEV and assumptions.phev_electric_share_override is not None:
        electric_share = assumptions.phev_electric_share_override
        notes.append("Electric-driving share uses the global override.")

    annual_insurance = vehicle.annual_insurance
    annual_registration = vehicle.annual_registration_fees
    if assumptions.insurance_and_fees_override is not None:
        total_override = float(assumptions.insurance_and_fees_override)
        if total_override < 0:
            raise ValueError("Insurance and fee override must be zero or greater.")
        annual_insurance = total_override
        annual_registration = 0.0
        notes.append(
            "Insurance and registration use the global override as one combined amount."
        )

    energy = annual_energy_cost(
        powertrain=vehicle.powertrain,
        annual_miles=assumptions.annual_miles,
        electricity_price=assumptions.electricity_price,
        gasoline_price=assumptions.gasoline_price,
        kwh_per_100mi=vehicle.kwh_per_100mi,
        mpg=vehicle.mpg,
        electric_share=electric_share,
    )

    sales_tax = vehicle.purchase_price * assumptions.sales_tax_rate
    purchase_fees = (
        assumptions.purchase_fees
        if vehicle.purchase_fees is None
        else float(vehicle.purchase_fees)
    )

    financing = compute_financing(
        purchase_price=vehicle.purchase_price,
        sales_tax=sales_tax,
        purchase_fees=purchase_fees,
        down_payment=assumptions.down_payment,
        apr=assumptions.apr,
        loan_term_months=assumptions.loan_term_months,
        ownership_months=assumptions.ownership_months,
        finance_taxes_and_fees=assumptions.finance_taxes_and_fees,
    )

    resale = resale_value(
        purchase_price=vehicle.purchase_price,
        annual_rate=vehicle.annual_depreciation_rate,
        years=years,
        override=vehicle.resale_override,
    )
    if resale.is_override:
        notes.append("Ending resale value is a user override.")
    if resale.was_capped:
        notes.append(
            "The resale override was capped to the range $0 to the purchase price."
        )

    total_energy = energy.total_cost * years
    total_maintenance = vehicle.annual_maintenance * years
    total_insurance = annual_insurance * years
    total_registration = annual_registration * years

    total_cost = (
        resale.depreciation
        + sales_tax
        + purchase_fees
        + financing.interest_paid
        + total_energy
        + total_maintenance
        + total_insurance
        + total_registration
    )
    lifetime_miles = assumptions.lifetime_miles
    cost_per_mile = total_cost / lifetime_miles if lifetime_miles > 0 else None

    schedule = _interest_schedule(
        financing.amount_financed,
        assumptions.apr,
        assumptions.loan_term_months,
        assumptions.ownership_months,
    )
    values = _value_path(vehicle.purchase_price, resale.ending_value, years)

    cash_upfront = financing.down_payment
    if not assumptions.finance_taxes_and_fees:
        cash_upfront += sales_tax + purchase_fees

    yearly_rows: list[dict[str, float]] = []
    cash_rows: list[dict[str, float]] = []
    previous_value = vehicle.purchase_price
    cumulative = 0.0
    cumulative_cash = 0.0

    cash_rows.append(
        {
            "year": 0,
            "down_payment": financing.down_payment,
            "loan_payments": 0.0,
            "cash_taxes_and_fees": cash_upfront - financing.down_payment,
            "operating_costs": 0.0,
            "loan_payoff": 0.0,
            "resale_proceeds": 0.0,
            "net_cash_outflow": cash_upfront,
            "cumulative_cash_outflow": cash_upfront,
        }
    )
    cumulative_cash = cash_upfront

    annual_operating = (
        energy.total_cost
        + vehicle.annual_maintenance
        + annual_insurance
        + annual_registration
    )

    for year in range(1, years + 1):
        months = schedule[(year - 1) * 12 : year * 12]
        interest_year = sum(row[0] for row in months)
        principal_year = sum(row[1] for row in months)
        depreciation_year = previous_value - values[year - 1]
        previous_value = values[year - 1]

        one_time = sales_tax + purchase_fees if year == 1 else 0.0
        economic_year = (
            depreciation_year + interest_year + annual_operating + one_time
        )
        cumulative += economic_year
        yearly_rows.append(
            {
                "year": year,
                "depreciation": depreciation_year,
                "financing_interest": interest_year,
                "energy": energy.total_cost,
                "maintenance": vehicle.annual_maintenance,
                "insurance": annual_insurance,
                "registration_and_fees": annual_registration,
                "sales_tax_and_purchase_fees": one_time,
                "annual_cost": economic_year,
                "cumulative_cost": cumulative,
                "vehicle_value": values[year - 1],
            }
        )

        payoff = financing.remaining_principal if year == years else 0.0
        proceeds = resale.ending_value if year == years else 0.0
        loan_payments = interest_year + principal_year
        net_cash = loan_payments + annual_operating + payoff - proceeds
        cumulative_cash += net_cash
        cash_rows.append(
            {
                "year": year,
                "down_payment": 0.0,
                "loan_payments": loan_payments,
                "cash_taxes_and_fees": 0.0,
                "operating_costs": annual_operating,
                "loan_payoff": payoff,
                "resale_proceeds": proceeds,
                "net_cash_outflow": net_cash,
                "cumulative_cash_outflow": cumulative_cash,
            }
        )

    return TcoResult(
        vehicle_id=vehicle.vehicle_id,
        label=vehicle.label,
        powertrain=vehicle.powertrain,
        purchase_price=vehicle.purchase_price,
        sales_tax=sales_tax,
        purchase_fees=purchase_fees,
        acquisition_cost=financing.acquisition_cost,
        financing=financing,
        energy=energy,
        resale=resale,
        ownership_years=years,
        annual_miles=assumptions.annual_miles,
        total_energy_cost=total_energy,
        total_electricity_cost=energy.electricity_cost * years,
        total_gasoline_cost=energy.gasoline_cost * years,
        total_maintenance_cost=total_maintenance,
        total_insurance_cost=total_insurance,
        total_registration_cost=total_registration,
        total_financing_interest=financing.interest_paid,
        depreciation=resale.depreciation,
        ending_resale_value=resale.ending_value,
        remaining_loan_balance=financing.remaining_principal,
        total_cost=total_cost,
        annual_cost=total_cost / years,
        monthly_cost=total_cost / (years * 12),
        cost_per_mile=cost_per_mile,
        yearly=pd.DataFrame(yearly_rows),
        cash_flows=pd.DataFrame(cash_rows),
        notes=tuple(notes),
    )


def components_reconcile(result: TcoResult) -> bool:
    """Confirm the displayed components add up to the reported total."""
    return abs(sum(result.components.values()) - result.total_cost) <= TOLERANCE


def results_to_frame(results: list[TcoResult]) -> pd.DataFrame:
    """Flatten results into the ranked comparison table."""
    rows = []
    for result in results:
        rows.append(
            {
                "vehicle_id": result.vehicle_id,
                "label": result.label,
                "powertrain": result.powertrain,
                "total_cost": result.total_cost,
                "annual_cost": result.annual_cost,
                "monthly_cost": result.monthly_cost,
                "cost_per_mile": result.cost_per_mile,
                "purchase_price": result.purchase_price,
                "sales_tax": result.sales_tax,
                "purchase_fees": result.purchase_fees,
                "depreciation": result.depreciation,
                "energy_cost": result.total_energy_cost,
                "electricity_cost": result.total_electricity_cost,
                "gasoline_cost": result.total_gasoline_cost,
                "maintenance_cost": result.total_maintenance_cost,
                "insurance_cost": result.total_insurance_cost,
                "registration_cost": result.total_registration_cost,
                "financing_interest": result.total_financing_interest,
                "monthly_payment": result.financing.monthly_payment,
                "ending_resale_value": result.ending_resale_value,
                "remaining_loan_balance": result.remaining_loan_balance,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("total_cost").reset_index(drop=True)
