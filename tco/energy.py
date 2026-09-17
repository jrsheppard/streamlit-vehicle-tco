"""Energy cost: electricity for electric miles, gasoline for combustion miles."""

from __future__ import annotations

from dataclasses import dataclass

from tco.epa_catalog import BEV, GASOLINE, HEV, PHEV


@dataclass(frozen=True)
class EnergyResult:
    """Annual energy use and cost for one vehicle."""

    electric_miles: float
    gasoline_miles: float
    kwh_used: float
    gallons_used: float
    electricity_cost: float
    gasoline_cost: float

    @property
    def total_cost(self) -> float:
        return self.electricity_cost + self.gasoline_cost


class EnergyInputError(ValueError):
    """A required efficiency value is missing, invalid, or incompatible."""


def _require_positive(value: float | None, name: str, powertrain: str) -> float:
    if value is None or value != value or value <= 0:
        raise EnergyInputError(
            f"{powertrain} vehicles need a positive {name}; got {value!r}."
        )
    return float(value)


def annual_energy_cost(
    powertrain: str,
    annual_miles: float,
    electricity_price: float,
    gasoline_price: float,
    kwh_per_100mi: float | None = None,
    mpg: float | None = None,
    electric_share: float | None = None,
) -> EnergyResult:
    """Annual electricity and gasoline cost for one vehicle.

    ``electric_share`` is the fraction of miles driven on electricity and applies
    to PHEVs only. It must be between 0 and 1 inclusive.
    """
    if annual_miles < 0:
        raise EnergyInputError("Annual miles must be zero or greater.")
    if electricity_price < 0 or gasoline_price < 0:
        raise EnergyInputError("Energy prices must be zero or greater.")

    if powertrain == BEV:
        consumption = _require_positive(kwh_per_100mi, "kWh/100 mi", powertrain)
        kwh = annual_miles * consumption / 100.0
        return EnergyResult(
            electric_miles=annual_miles,
            gasoline_miles=0.0,
            kwh_used=kwh,
            gallons_used=0.0,
            electricity_cost=kwh * electricity_price,
            gasoline_cost=0.0,
        )

    if powertrain in (HEV, GASOLINE):
        efficiency = _require_positive(mpg, "MPG", powertrain)
        gallons = annual_miles / efficiency
        return EnergyResult(
            electric_miles=0.0,
            gasoline_miles=annual_miles,
            kwh_used=0.0,
            gallons_used=gallons,
            electricity_cost=0.0,
            gasoline_cost=gallons * gasoline_price,
        )

    if powertrain == PHEV:
        if electric_share is None or electric_share != electric_share:
            raise EnergyInputError(
                "PHEV vehicles need an electric-driving share between 0 and 1."
            )
        if not 0.0 <= electric_share <= 1.0:
            raise EnergyInputError(
                f"Electric-driving share must be between 0 and 1; got {electric_share}."
            )
        consumption = _require_positive(kwh_per_100mi, "kWh/100 mi", powertrain)
        efficiency = _require_positive(mpg, "MPG", powertrain)
        electric_miles = annual_miles * electric_share
        gasoline_miles = annual_miles - electric_miles
        kwh = electric_miles * consumption / 100.0
        gallons = gasoline_miles / efficiency
        return EnergyResult(
            electric_miles=electric_miles,
            gasoline_miles=gasoline_miles,
            kwh_used=kwh,
            gallons_used=gallons,
            electricity_cost=kwh * electricity_price,
            gasoline_cost=gallons * gasoline_price,
        )

    raise EnergyInputError(
        f"The calculation engine does not price the {powertrain!r} powertrain."
    )
