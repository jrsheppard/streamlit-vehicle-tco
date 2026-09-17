"""Altair chart builders.

Colors come from the Okabe-Ito color-blind-safe palette and are keyed on the
vehicle label so the same vehicle keeps the same color in every chart. Status is
never communicated by color alone; the tables carry text labels for that.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

#: Up to six vehicles can be compared at once.
VEHICLE_PALETTE: tuple[str, ...] = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#D55E00",
)

COMPONENT_ORDER: tuple[str, ...] = (
    "Depreciation",
    "Energy",
    "Maintenance",
    "Insurance",
    "Registration and fees",
    "Financing interest",
    "Sales tax",
    "Purchase fees",
)

COMPONENT_PALETTE: tuple[str, ...] = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#D55E00",
    "#8C8C8C",
    "#332288",
)

CURRENCY_FORMAT = "$,.0f"


def vehicle_color(labels: list[str]) -> alt.Scale:
    """Stable color scale shared by every chart in one comparison."""
    return alt.Scale(domain=list(labels), range=list(VEHICLE_PALETTE[: len(labels)]))


def cost_breakdown_chart(
    breakdown: pd.DataFrame, order: list[str], ownership_years: int
) -> alt.Chart:
    """Stacked economic cost by component, one bar per vehicle."""
    return (
        alt.Chart(breakdown)
        .mark_bar()
        .encode(
            x=alt.X(
                "label:N",
                title="Vehicle",
                sort=order,
                axis=alt.Axis(labelAngle=-20, labelLimit=160),
            ),
            y=alt.Y(
                "amount:Q",
                title=f"Cost over {ownership_years} years (USD)",
                stack="zero",
                scale=alt.Scale(zero=True),
                axis=alt.Axis(format=CURRENCY_FORMAT),
            ),
            color=alt.Color(
                "component:N",
                title="Cost component",
                sort=list(COMPONENT_ORDER),
                scale=alt.Scale(
                    domain=list(COMPONENT_ORDER), range=list(COMPONENT_PALETTE)
                ),
            ),
            order=alt.Order("component_rank:Q"),
            tooltip=[
                alt.Tooltip("label:N", title="Vehicle"),
                alt.Tooltip("component:N", title="Component"),
                alt.Tooltip("amount:Q", title="Cost", format="$,.0f"),
                alt.Tooltip("share:Q", title="Share of total", format=".1%"),
            ],
        )
        .properties(height=340)
    )


def cumulative_cost_chart(
    cumulative: pd.DataFrame,
    order: list[str],
    value_field: str,
    y_title: str,
) -> alt.Chart:
    """Cumulative cost by ownership year, one line per vehicle."""
    return (
        alt.Chart(cumulative)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "year:O",
                title="Ownership year",
                sort=None,
            ),
            y=alt.Y(
                f"{value_field}:Q",
                title=y_title,
                scale=alt.Scale(zero=True),
                axis=alt.Axis(format=CURRENCY_FORMAT),
            ),
            color=alt.Color(
                "label:N", title="Vehicle", sort=order, scale=vehicle_color(order)
            ),
            tooltip=[
                alt.Tooltip("label:N", title="Vehicle"),
                alt.Tooltip("year:O", title="Year"),
                alt.Tooltip(f"{value_field}:Q", title=y_title, format="$,.0f"),
            ],
        )
        .properties(height=340)
    )


def model_count_chart(frame: pd.DataFrame, dimension: str, title: str) -> alt.Chart:
    """Distinct Washington-registered model configurations, never registrations."""
    return (
        alt.Chart(frame)
        .mark_bar(color=VEHICLE_PALETTE[0])
        .encode(
            x=alt.X(
                f"{dimension}:N",
                title=title,
                sort="-y",
                axis=alt.Axis(labelAngle=-35, labelLimit=140),
            ),
            y=alt.Y(
                "model_configurations:Q",
                title="Distinct model configurations",
                scale=alt.Scale(zero=True),
                axis=alt.Axis(format=",.0f"),
            ),
            tooltip=[
                alt.Tooltip(f"{dimension}:N", title=title),
                alt.Tooltip(
                    "model_configurations:Q", title="Model configurations", format=",.0f"
                ),
            ],
        )
        .properties(height=300)
    )


def breakdown_frame(results: list) -> pd.DataFrame:
    """Long-form component table for :func:`cost_breakdown_chart`."""
    rows: list[dict[str, object]] = []
    ranks = {name: index for index, name in enumerate(COMPONENT_ORDER)}
    for result in results:
        total = result.total_cost or 1.0
        for component, amount in result.components.items():
            rows.append(
                {
                    "label": result.label,
                    "component": component,
                    "component_rank": ranks.get(component, len(ranks)),
                    "amount": amount,
                    "share": amount / total,
                }
            )
    return pd.DataFrame(rows)


def cumulative_frame(results: list, source: str = "economic") -> pd.DataFrame:
    """Long-form cumulative cost table for :func:`cumulative_cost_chart`."""
    rows: list[pd.DataFrame] = []
    for result in results:
        if source == "economic":
            frame = result.yearly.loc[:, ["year", "cumulative_cost"]].copy()
        else:
            frame = result.cash_flows.loc[
                :, ["year", "cumulative_cash_outflow"]
            ].copy()
        frame["label"] = result.label
        rows.append(frame)
    if not rows:
        return pd.DataFrame(columns=["year", "label"])
    return pd.concat(rows, ignore_index=True)
