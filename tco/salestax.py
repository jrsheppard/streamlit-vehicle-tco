"""ZIP-to-state mapping and editable sales-tax assumptions.

The rates come from the Tax Foundation's general state and local sales-tax
tables. They are a *general* sales-tax estimate, not an authoritative
motor-vehicle tax determination: the source does not encode motor-vehicle
specific rates, exemptions, trade-in credits, caps, or fees, and its local figure
is a population-weighted state average rather than a ZIP-exact rate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from tco.paths import SALES_TAX_PATH, ZIP_TO_STATE_PATH

ZIP_PATTERN = re.compile(r"^\d{5}$")

STATUS_OK = "ok"
STATUS_MULTI_STATE = "multi_state"
STATUS_UNKNOWN = "unknown"
STATUS_INVALID = "invalid"


@dataclass(frozen=True)
class ZipLookup:
    """Result of mapping a five-digit ZIP code to a state."""

    zip_code: str
    status: str
    state: str | None = None
    states: tuple[str, ...] = ()
    message: str = ""

    @property
    def is_resolved(self) -> bool:
        return self.status in (STATUS_OK, STATUS_MULTI_STATE) and bool(self.state)


@dataclass(frozen=True)
class StateSalesTax:
    """Published state and average-local sales-tax rates for one state."""

    state: str
    state_name: str
    state_rate: float
    avg_local_rate: float
    max_local_rate: float
    combined_rate: float
    source_name: str
    source_url: str
    effective_date: str


@dataclass(frozen=True)
class ActiveSalesTax:
    """The sales-tax assumption actually used in the calculation."""

    rate: float
    state: str | None
    state_rate: float | None
    local_rate: float | None
    published_combined_rate: float | None
    is_override: bool
    includes_local_estimate: bool
    zip_lookup: ZipLookup | None = None
    rates: StateSalesTax | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)


def normalize_zip(value: object) -> str:
    """Trim a user-entered ZIP code down to its five-digit form."""
    text = "" if value is None else str(value).strip()
    text = text.split("-", 1)[0].strip()
    return text


def read_zip_to_state(path: Path | str = ZIP_TO_STATE_PATH) -> pd.DataFrame:
    """Read the local ZIP-to-state table produced by the refresh script."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"ZIP-to-state table not found at {path}. "
            "Run `venv/bin/python -m scripts.refresh_data` to create it."
        )
    frame = pd.read_csv(
        path,
        dtype={"zip_code": "string", "primary_state": "string", "states": "string"},
    )
    required = {"zip_code", "primary_state", "states", "state_count"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"ZIP-to-state table is missing columns: {missing}")
    return frame


def lookup_zip_state(zip_code: object, zip_table: pd.DataFrame) -> ZipLookup:
    """Map a ZIP code to a state, flagging unknown and multi-state cases."""
    text = normalize_zip(zip_code)
    if not ZIP_PATTERN.match(text):
        return ZipLookup(
            zip_code=text,
            status=STATUS_INVALID,
            message="Enter a five-digit ZIP code.",
        )

    matches = zip_table.loc[zip_table["zip_code"] == text]
    if matches.empty:
        return ZipLookup(
            zip_code=text,
            status=STATUS_UNKNOWN,
            message=(
                "This ZIP code has no ZIP Code Tabulation Area in the local "
                "dataset. Choose a state manually or edit the tax rate."
            ),
        )

    row = matches.iloc[0]
    states = tuple(str(row["states"]).split("|"))
    primary = str(row["primary_state"])
    if int(row["state_count"]) > 1:
        return ZipLookup(
            zip_code=text,
            status=STATUS_MULTI_STATE,
            state=primary,
            states=states,
            message=(
                f"ZIP {text} spans {', '.join(states)}. "
                f"{primary} covers the largest land area and is used by default."
            ),
        )
    return ZipLookup(zip_code=text, status=STATUS_OK, state=primary, states=states)


def read_sales_tax_rates(path: Path | str = SALES_TAX_PATH) -> pd.DataFrame:
    """Read the local Tax Foundation sales-tax table."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Sales-tax table not found at {path}. "
            "Run `venv/bin/python -m scripts.refresh_data` to create it."
        )
    frame = pd.read_csv(path, dtype={"state": "string", "state_name": "string"})
    required = {
        "state",
        "state_name",
        "state_rate",
        "avg_local_rate",
        "max_local_rate",
        "combined_rate",
        "source_name",
        "source_url",
        "effective_date",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Sales-tax table is missing columns: {missing}")
    return frame


def rates_for_state(state: str, rates: pd.DataFrame) -> StateSalesTax | None:
    """Published rates for a two-letter state code, or ``None`` if absent."""
    matches = rates.loc[rates["state"] == state]
    if matches.empty:
        return None
    row = matches.iloc[0]
    return StateSalesTax(
        state=str(row["state"]),
        state_name=str(row["state_name"]),
        state_rate=float(row["state_rate"]),
        avg_local_rate=float(row["avg_local_rate"]),
        max_local_rate=float(row["max_local_rate"]),
        combined_rate=float(row["combined_rate"]),
        source_name=str(row["source_name"]),
        source_url=str(row["source_url"]),
        effective_date=str(row["effective_date"]),
    )


def resolve_sales_tax(
    zip_code: object,
    zip_table: pd.DataFrame,
    rates: pd.DataFrame,
    include_local_estimate: bool = True,
    override_rate: float | None = None,
    fallback_rate: float = 0.0,
) -> ActiveSalesTax:
    """Decide which sales-tax rate the calculation should use.

    The user override always wins. Otherwise the mapped state's rate is used,
    optionally plus the Tax Foundation's population-weighted average local rate.
    When the ZIP code cannot be mapped, ``fallback_rate`` is used and the reason
    is reported in ``notes``.
    """
    lookup = lookup_zip_state(zip_code, zip_table)
    state_rates = (
        rates_for_state(lookup.state, rates) if lookup.is_resolved else None
    )
    notes: list[str] = []
    if lookup.message:
        notes.append(lookup.message)

    if state_rates is None:
        if lookup.is_resolved:
            notes.append(
                f"No published sales-tax rate is available for {lookup.state}."
            )
        base_rate = float(fallback_rate)
        state_rate = None
        local_rate = None
        published_combined = None
    else:
        state_rate = state_rates.state_rate
        local_rate = state_rates.avg_local_rate if include_local_estimate else 0.0
        published_combined = state_rates.combined_rate
        base_rate = state_rate + local_rate

    if override_rate is not None and override_rate == override_rate:
        notes.append("A user override replaces the published estimate.")
        return ActiveSalesTax(
            rate=float(override_rate),
            state=lookup.state,
            state_rate=state_rate,
            local_rate=local_rate,
            published_combined_rate=published_combined,
            is_override=True,
            includes_local_estimate=include_local_estimate,
            zip_lookup=lookup,
            rates=state_rates,
            notes=tuple(notes),
        )

    return ActiveSalesTax(
        rate=base_rate,
        state=lookup.state,
        state_rate=state_rate,
        local_rate=local_rate,
        published_combined_rate=published_combined,
        is_override=False,
        includes_local_estimate=include_local_estimate,
        zip_lookup=lookup,
        rates=state_rates,
        notes=tuple(notes),
    )
