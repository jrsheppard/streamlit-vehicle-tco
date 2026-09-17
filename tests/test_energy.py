import pytest

from tco.energy import EnergyInputError, annual_energy_cost


def test_bev_uses_electricity_only():
    result = annual_energy_cost(
        "BEV",
        annual_miles=12_000,
        electricity_price=0.17,
        gasoline_price=3.30,
        kwh_per_100mi=25.0,
    )
    assert result.kwh_used == pytest.approx(3_000)
    assert result.electricity_cost == pytest.approx(510.0)
    assert result.gasoline_cost == 0
    assert result.total_cost == pytest.approx(510.0)


@pytest.mark.parametrize("powertrain", ["HEV", "Gasoline"])
def test_combustion_vehicles_use_gasoline_only(powertrain):
    result = annual_energy_cost(
        powertrain,
        annual_miles=12_000,
        electricity_price=0.17,
        gasoline_price=3.50,
        mpg=40.0,
    )
    assert result.gallons_used == pytest.approx(300.0)
    assert result.gasoline_cost == pytest.approx(1_050.0)
    assert result.electricity_cost == 0


def test_phev_splits_miles_by_the_electric_share():
    result = annual_energy_cost(
        "PHEV",
        annual_miles=10_000,
        electricity_price=0.20,
        gasoline_price=4.00,
        kwh_per_100mi=30.0,
        mpg=40.0,
        electric_share=0.60,
    )
    assert result.electric_miles == pytest.approx(6_000)
    assert result.gasoline_miles == pytest.approx(4_000)
    assert result.electricity_cost == pytest.approx(360.0)
    assert result.gasoline_cost == pytest.approx(400.0)
    assert result.total_cost == pytest.approx(760.0)


def test_zero_annual_miles_costs_nothing():
    result = annual_energy_cost(
        "Gasoline",
        annual_miles=0,
        electricity_price=0.17,
        gasoline_price=3.30,
        mpg=30.0,
    )
    assert result.total_cost == 0


def test_missing_efficiency_is_an_error_rather_than_a_misleading_zero():
    with pytest.raises(EnergyInputError):
        annual_energy_cost(
            "BEV", annual_miles=10_000, electricity_price=0.17, gasoline_price=3.3
        )
    with pytest.raises(EnergyInputError):
        annual_energy_cost(
            "Gasoline", annual_miles=10_000, electricity_price=0.17, gasoline_price=3.3
        )


@pytest.mark.parametrize("share", [-0.1, 1.1, None])
def test_invalid_phev_share_is_rejected(share):
    with pytest.raises(EnergyInputError):
        annual_energy_cost(
            "PHEV",
            annual_miles=10_000,
            electricity_price=0.17,
            gasoline_price=3.3,
            kwh_per_100mi=30.0,
            mpg=40.0,
            electric_share=share,
        )


def test_unsupported_powertrain_is_rejected():
    with pytest.raises(EnergyInputError):
        annual_energy_cost(
            "Other", annual_miles=1_000, electricity_price=0.17, gasoline_price=3.3
        )


def test_negative_miles_are_rejected():
    with pytest.raises(EnergyInputError):
        annual_energy_cost(
            "Gasoline",
            annual_miles=-1,
            electricity_price=0.17,
            gasoline_price=3.3,
            mpg=30,
        )
