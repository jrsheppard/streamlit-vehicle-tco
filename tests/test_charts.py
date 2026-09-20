from dataclasses import replace

import pytest

from tco.charts import ranked_cost_frame
from tco.engine import GlobalAssumptions, VehicleAssumptions, compute_tco


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


def result(vehicle_id: str, label: str, purchase_price: float):
    vehicle = VehicleAssumptions(
        vehicle_id=vehicle_id,
        label=label,
        powertrain="Gasoline",
        purchase_price=purchase_price,
        annual_depreciation_rate=0.15,
        annual_maintenance=900,
        annual_insurance=1_800,
        annual_registration_fees=160,
        mpg=32.0,
    )
    return compute_tco(vehicle, GLOBALS)


def test_ranked_cost_frame_reconciles_capital_and_operating_costs():
    computed = result("car-1", "Car one", 30_000)
    frame = ranked_cost_frame(
        [computed],
        most_expensive=True,
        include_capital_costs=True,
    )

    amounts = frame.set_index("category")["amount"]
    assert amounts["Capital costs"] == pytest.approx(
        computed.depreciation
        + computed.total_financing_interest
        + computed.sales_tax
        + computed.purchase_fees
    )
    assert amounts["Operating costs"] == pytest.approx(
        computed.total_energy_cost
        + computed.total_maintenance_cost
        + computed.total_insurance_cost
        + computed.total_registration_cost
    )
    assert amounts.sum() == pytest.approx(computed.total_cost)
    assert frame["share"].sum() == pytest.approx(1.0)


def test_excluding_capital_costs_leaves_only_operating_costs():
    computed = result("car-1", "Car one", 30_000)
    included = ranked_cost_frame(
        [computed],
        most_expensive=True,
        include_capital_costs=True,
    ).set_index("category")
    excluded = ranked_cost_frame(
        [computed],
        most_expensive=True,
        include_capital_costs=False,
    ).set_index("category")

    assert excluded.at["Capital costs", "amount"] == 0
    assert excluded.at["Operating costs", "amount"] == pytest.approx(
        included.at["Operating costs", "amount"]
    )


def test_ranked_cost_frame_selects_and_orders_each_extreme():
    results = [result(f"car-{index}", f"Car {index}", 20_000 + index * 1_000) for index in range(12)]

    expensive = ranked_cost_frame(
        results,
        most_expensive=True,
        include_capital_costs=True,
    )
    inexpensive = ranked_cost_frame(
        results,
        most_expensive=False,
        include_capital_costs=True,
    )

    expensive_ids = expensive.drop_duplicates("vehicle_id")["vehicle_id"].tolist()
    inexpensive_ids = inexpensive.drop_duplicates("vehicle_id")["vehicle_id"].tolist()
    assert expensive_ids == [f"car-{index}" for index in range(11, 1, -1)]
    assert inexpensive_ids == [f"car-{index}" for index in range(10)]


def test_ranked_cost_frame_breaks_ties_by_label_then_id():
    first = result("car-b", "Alpha", 30_000)
    second = replace(first, vehicle_id="car-a")
    third = replace(first, vehicle_id="car-c", label="Beta")

    frame = ranked_cost_frame(
        [third, first, second],
        most_expensive=True,
        include_capital_costs=True,
    )

    assert frame.drop_duplicates("vehicle_id")["vehicle_id"].tolist() == [
        "car-a",
        "car-b",
        "car-c",
    ]