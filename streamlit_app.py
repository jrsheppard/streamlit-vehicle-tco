"""Vehicle total-cost-of-ownership comparison.

Entry point for `streamlit run streamlit_app.py`. Data loading, validation, and
every calculation live in the `tco` package; this script only wires them to the
user interface.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from tco import charts, formatting, loaders
from tco.assumptions import (
    OVERRIDE_FIELDS,
    VALUE_STATUSES,
    build_field_provenance,
    build_vehicle_assumptions,
    comparison_readiness,
    merge_with_epa,
    parse_uploaded_catalog,
    validate_catalog,
)
from tco.energy import EnergyInputError
from tco.engine import (
    MAINTENANCE_FIXED_SHARE,
    MAINTENANCE_REFERENCE_MILES,
    GlobalAssumptions,
    components_reconcile,
    compute_tco,
    results_to_frame,
)
from tco.epa_catalog import SUPPORTED_POWERTRAINS
from tco.salestax import resolve_sales_tax
from tco.taxonomy import UNCLASSIFIED, assign_brand_group, brand_group_map

st.set_page_config(
    page_title="Vehicle total cost of ownership",
    page_icon=":material/directions_car:",
    layout="wide",
)

MAX_SHORTLIST = 6
DEFAULT_SHORTLIST_SIZE = 4
SEED_CATALOG_LABEL = "Seed catalog (illustrative defaults)"

# Session state is initialized in exactly one place.
DEFAULT_STATE: dict[str, object] = {
    "active_catalog": None,
    "catalog_label": SEED_CATALOG_LABEL,
    "vehicle_overrides": {},
    "shortlist": [],
    "shortlist_initialized": False,
    "filter_powertrain": [],
    "filter_segment": [],
    "filter_brand_group": [],
    "filter_make": [],
    "filter_year": [],
    "gasoline_price": 3.30,
    "electricity_price": 0.17,
    "annual_miles": 12000,
    "ownership_years": 6,
    "down_payment": 5000.0,
    "apr_percent": 6.9,
    "loan_term_months": 60,
    "purchase_fees": 600.0,
    "finance_taxes_and_fees": True,
    "zip_code": "98101",
    "include_local_tax": True,
    "use_tax_override": False,
    "tax_override_percent": 6.5,
    "use_phev_share_override": False,
    "phev_share_override": 0.60,
    "use_insurance_override": False,
    "insurance_override": 2000.0,
    "cost_view": "Economic cost",
    "epa_matches": None,
}
for key, value in DEFAULT_STATE.items():
    st.session_state.setdefault(key, value)

st.title("Vehicle total cost of ownership")
st.caption(
    "Compare battery electric, plug-in hybrid, conventional hybrid, and gasoline "
    "vehicles on transparent, editable assumptions. Specifications come from the "
    "U.S. DOE and EPA FuelEconomy.gov catalog. Purchase prices and operating "
    "costs are statistical estimates, not quotes: prices come from a model fitted "
    "to published manufacturer MSRPs, and depreciation, maintenance, insurance "
    "and registration come from AAA's published category averages. Every figure "
    "is editable and carries its source in the provenance tab. This is a "
    "decision-support calculator, not financial advice."
)

# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
fingerprints = loaders.fingerprints()
try:
    epa_catalog_frame = loaders.load_epa_catalog(fingerprints["epa"])
    seed_catalog = loaders.load_seed_cost_assumptions(fingerprints["cost_assumptions"])
    zip_table = loaders.load_zip_table(fingerprints["zip"])
    tax_rates = loaders.load_tax_rates(fingerprints["tax"])
    body_segments = loaders.load_body_segments(fingerprints["body_segments"])
    brand_groups = loaders.load_brand_groups(fingerprints["brand_groups"])
    provenance_overrides = loaders.load_field_provenance_overrides(
        fingerprints["field_provenance"]
    )
    source_provenance = loaders.load_source_provenance(fingerprints["provenance"])
except (FileNotFoundError, ValueError) as error:
    st.error(
        f"{error}\n\nRun `venv/bin/python -m scripts.refresh_data` to rebuild the "
        "local data files, then reload this page.",
        icon=":material/error:",
    )
    st.stop()

if st.session_state.active_catalog is None:
    st.session_state.active_catalog = seed_catalog.copy()

catalog = st.session_state.active_catalog
group_lookup = brand_group_map(brand_groups)

merged = comparison_readiness(merge_with_epa(catalog, epa_catalog_frame))
merged["brand_group"] = [
    group
    if isinstance(group, str) and group.strip()
    else assign_brand_group(make, group_lookup)
    for group, make in zip(merged["brand_group"], merged["make"])
]
merged["body_segment"] = merged["body_segment"].fillna(UNCLASSIFIED)
label_by_id = dict(zip(merged["vehicle_id"], merged["vehicle_label"]))


def sanitize_selection(key: str, options: list) -> list:
    """Drop stored selections that the current options no longer offer."""
    current = [value for value in st.session_state.get(key, []) if value in options]
    st.session_state[key] = current
    return current


def reset_filters() -> None:
    for key in (
        "filter_powertrain",
        "filter_segment",
        "filter_brand_group",
        "filter_make",
        "filter_year",
    ):
        st.session_state[key] = []


# --------------------------------------------------------------------------- #
# Sidebar: filters, global assumptions, and the tax assumption
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Filters", icon=":material/filter_list:")

    powertrain_options = [
        value for value in SUPPORTED_POWERTRAINS if value in set(merged["powertrain"])
    ]
    sanitize_selection("filter_powertrain", powertrain_options)
    selected_powertrains = st.pills(
        "Powertrain",
        powertrain_options,
        selection_mode="multi",
        key="filter_powertrain",
        help="Leave empty to include every powertrain.",
    )

    filtered = (
        merged
        if not selected_powertrains
        else merged[merged["powertrain"].isin(selected_powertrains)]
    )

    segment_options = sorted(filtered["body_segment"].dropna().unique().tolist())
    sanitize_selection("filter_segment", segment_options)
    selected_segments = st.multiselect(
        "Body segment", segment_options, key="filter_segment"
    )
    if selected_segments:
        filtered = filtered[filtered["body_segment"].isin(selected_segments)]

    group_options = sorted(filtered["brand_group"].dropna().unique().tolist())
    sanitize_selection("filter_brand_group", group_options)
    selected_groups = st.multiselect(
        "Brand group", group_options, key="filter_brand_group"
    )
    if selected_groups:
        filtered = filtered[filtered["brand_group"].isin(selected_groups)]

    make_options = sorted(filtered["make"].dropna().unique().tolist())
    sanitize_selection("filter_make", make_options)
    selected_makes = st.multiselect("Brand", make_options, key="filter_make")
    if selected_makes:
        filtered = filtered[filtered["make"].isin(selected_makes)]

    year_options = sorted(
        int(year) for year in filtered["model_year"].dropna().unique().tolist()
    )
    sanitize_selection("filter_year", year_options)
    selected_years = st.multiselect("Model year", year_options, key="filter_year")
    if selected_years:
        filtered = filtered[filtered["model_year"].isin(selected_years)]

    st.button(
        "Reset filters",
        icon=":material/restart_alt:",
        on_click=reset_filters,
        width="stretch",
    )

    st.header("Assumptions", icon=":material/tune:")
    st.number_input(
        "Gasoline price ($/gal)",
        min_value=0.0,
        max_value=15.0,
        step=0.05,
        key="gasoline_price",
    )
    st.number_input(
        "Electricity price ($/kWh)",
        min_value=0.0,
        max_value=2.0,
        step=0.01,
        format="%.3f",
        key="electricity_price",
    )
    st.number_input(
        "Miles driven per year",
        min_value=0,
        max_value=100_000,
        step=500,
        key="annual_miles",
    )
    st.slider(
        "Ownership period (whole years)",
        min_value=1,
        max_value=15,
        key="ownership_years",
    )

    with st.popover(
        "Financing terms", icon=":material/account_balance:", width="stretch"
    ):
        st.number_input(
            "Down payment ($)",
            min_value=0.0,
            max_value=500_000.0,
            step=250.0,
            key="down_payment",
        )
        st.number_input(
            "APR (%)", min_value=0.0, max_value=40.0, step=0.1, key="apr_percent"
        )
        st.number_input(
            "Loan term (months)",
            min_value=0,
            max_value=120,
            step=6,
            key="loan_term_months",
            help="Use 0 months for a cash purchase.",
        )
        st.number_input(
            "Purchase fees ($, one time)",
            min_value=0.0,
            max_value=20_000.0,
            step=50.0,
            key="purchase_fees",
            help="Documentation and title fees paid at purchase.",
        )
        st.toggle(
            "Finance sales tax and purchase fees",
            key="finance_taxes_and_fees",
            help="Turn this off to pay taxes and fees in cash at delivery.",
        )

    with st.popover(
        "Vehicle-wide overrides", icon=":material/rule_settings:", width="stretch"
    ):
        st.toggle(
            "Override the plug-in hybrid electric-driving share",
            key="use_phev_share_override",
            help="Off keeps each model's EPA combined utility factor.",
        )
        st.slider(
            "Electric-driving share",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            key="phev_share_override",
            disabled=not st.session_state.use_phev_share_override,
        )
        st.toggle(
            "Override annual insurance and registration fees",
            key="use_insurance_override",
            help="Off keeps each model's catalog values.",
        )
        st.number_input(
            "Annual insurance and fees ($)",
            min_value=0.0,
            max_value=50_000.0,
            step=50.0,
            key="insurance_override",
            disabled=not st.session_state.use_insurance_override,
        )

    st.header("Sales tax", icon=":material/receipt_long:")
    st.text_input("ZIP code", max_chars=10, key="zip_code")
    st.toggle("Include the estimated average local rate", key="include_local_tax")
    st.toggle("Override the sales-tax rate", key="use_tax_override")
    st.number_input(
        "Sales-tax rate (%)",
        min_value=0.0,
        max_value=25.0,
        step=0.05,
        key="tax_override_percent",
        disabled=not st.session_state.use_tax_override,
    )

    active_tax = resolve_sales_tax(
        zip_code=st.session_state.zip_code,
        zip_table=zip_table,
        rates=tax_rates,
        include_local_estimate=st.session_state.include_local_tax,
        override_rate=(
            st.session_state.tax_override_percent / 100.0
            if st.session_state.use_tax_override
            else None
        ),
    )

    if active_tax.rates is not None:
        st.caption(
            f"{active_tax.rates.state_name}: state "
            f"{formatting.percent(active_tax.state_rate)} plus estimated local "
            f"{formatting.percent(active_tax.local_rate)}. Published combined "
            f"{formatting.percent(active_tax.published_combined_rate)}. Active rate "
            f"{formatting.percent(active_tax.rate)}"
            f"{' (user override)' if active_tax.is_override else ''}."
        )
        st.caption(
            f"Source: {active_tax.rates.source_name}, effective "
            f"{active_tax.rates.effective_date}. The local figure is a "
            "population-weighted state average, not a ZIP-exact rate, and general "
            "sales tax is not a motor-vehicle tax determination."
        )
    else:
        st.warning(
            "No state rate is available for this ZIP code. Override the rate to "
            f"continue. Active rate {formatting.percent(active_tax.rate)}.",
            icon=":material/warning:",
        )
    for note in active_tax.notes:
        st.caption(note)

# --------------------------------------------------------------------------- #
# Shortlist
# --------------------------------------------------------------------------- #
st.subheader("Compare vehicles")

if filtered.empty:
    st.warning(
        "No vehicles match the current filters.", icon=":material/filter_alt_off:"
    )
    st.button(
        "Clear all filters", icon=":material/restart_alt:", on_click=reset_filters
    )
    st.stop()

ready = filtered[filtered["comparison_ready"]]
available_ids = ready["vehicle_id"].tolist()

previous = list(st.session_state.shortlist)
dropped = [item for item in previous if item not in available_ids]
if dropped:
    st.session_state.shortlist = [item for item in previous if item in available_ids]
    st.info(
        "Removed from the comparison because the current filters exclude them: "
        + ", ".join(label_by_id.get(item, item) for item in dropped)
        + ". No substitute was added.",
        icon=":material/info:",
    )

if not st.session_state.shortlist_initialized and available_ids:
    default: list[str] = []
    for powertrain in SUPPORTED_POWERTRAINS:
        if len(default) >= DEFAULT_SHORTLIST_SIZE:
            break
        group = ready[ready["powertrain"] == powertrain]
        if group.empty:
            continue
        # Prefer a mainstream brand so the opening comparison is relatable, and
        # within it the median-priced model rather than whichever make happens
        # to sort first alphabetically.
        mainstream = group[group["brand_group"].str.startswith("Mainstream")]
        if not mainstream.empty:
            group = mainstream
        distance = (
            group["purchase_price_usd"] - group["purchase_price_usd"].median()
        ).abs()
        default.append(str(group.loc[distance.idxmin(), "vehicle_id"]))
    for vehicle_id in available_ids:
        if len(default) >= DEFAULT_SHORTLIST_SIZE:
            break
        if vehicle_id not in default:
            default.append(vehicle_id)
    st.session_state.shortlist = default[:DEFAULT_SHORTLIST_SIZE]
    st.session_state.shortlist_initialized = True

shortlist = st.multiselect(
    f"Shortlist (up to {MAX_SHORTLIST} comparison-ready vehicles)",
    options=available_ids,
    format_func=lambda item: label_by_id.get(item, item),
    max_selections=MAX_SHORTLIST,
    key="shortlist",
    help=f"{len(available_ids):,} vehicles match the current filters. Type to search.",
)

incomplete = filtered[~filtered["comparison_ready"]]
if not incomplete.empty:
    st.caption(
        f"{len(incomplete):,} matching vehicle(s) are not comparison-ready: "
        + "; ".join(
            f"{row.vehicle_label} ({row.readiness_note})"
            for row in incomplete.head(5).itertuples()
        )
        + ". Most are vehicles the price model declined to estimate because too "
        "few comparable MSRPs were available. Enter the missing values in the "
        "assumptions workspace to compare them."
    )

if not shortlist:
    st.info(
        "Select at least one vehicle to see total cost of ownership.",
        icon=":material/directions_car:",
    )
    st.stop()

# --------------------------------------------------------------------------- #
# Calculations
# --------------------------------------------------------------------------- #
try:
    global_assumptions = GlobalAssumptions(
        gasoline_price=float(st.session_state.gasoline_price),
        electricity_price=float(st.session_state.electricity_price),
        annual_miles=float(st.session_state.annual_miles),
        ownership_years=int(st.session_state.ownership_years),
        down_payment=float(st.session_state.down_payment),
        apr=float(st.session_state.apr_percent) / 100.0,
        loan_term_months=int(st.session_state.loan_term_months),
        sales_tax_rate=float(active_tax.rate),
        purchase_fees=float(st.session_state.purchase_fees),
        finance_taxes_and_fees=bool(st.session_state.finance_taxes_and_fees),
        phev_electric_share_override=(
            float(st.session_state.phev_share_override)
            if st.session_state.use_phev_share_override
            else None
        ),
        insurance_and_fees_override=(
            float(st.session_state.insurance_override)
            if st.session_state.use_insurance_override
            else None
        ),
    )
except ValueError as error:
    st.error(f"Check the global assumptions: {error}", icon=":material/error:")
    st.stop()

overrides_by_vehicle: dict[str, dict] = st.session_state.vehicle_overrides
results = []
calculation_errors: list[tuple[str, str]] = []
for vehicle_id in shortlist:
    rows = merged.loc[merged["vehicle_id"] == vehicle_id]
    if rows.empty:
        continue
    row = rows.iloc[0]
    try:
        vehicle = build_vehicle_assumptions(row, overrides_by_vehicle.get(vehicle_id))
        results.append(compute_tco(vehicle, global_assumptions))
    except (EnergyInputError, ValueError) as error:
        calculation_errors.append((str(row["vehicle_label"]), str(error)))

for label, message in calculation_errors:
    st.error(
        f"{label} cannot be costed: {message} Fix its assumptions in the "
        "assumptions workspace, or adjust the global inputs.",
        icon=":material/error:",
    )

if not results:
    st.stop()

results.sort(key=lambda item: item.total_cost)
order = [result.label for result in results]
comparison = results_to_frame(results)
cheapest, priciest = results[0], results[-1]

# --------------------------------------------------------------------------- #
# Headline metrics
# --------------------------------------------------------------------------- #
with st.container(horizontal=True):
    st.metric(
        "Lowest total cost",
        cheapest.label,
        border=True,
        help=f"Lowest economic cost over {global_assumptions.ownership_years} years.",
    )
    st.metric(
        f"Total cost ({global_assumptions.ownership_years} yr)",
        formatting.currency(cheapest.total_cost),
        border=True,
    )
    st.metric(
        "Average annual cost", formatting.currency(cheapest.annual_cost), border=True
    )
    st.metric(
        "Cost per mile",
        formatting.per_mile(cheapest.cost_per_mile),
        border=True,
        help="Total economic cost divided by miles driven over the period.",
    )
    st.metric(
        "Saving vs. most expensive",
        formatting.currency(priciest.total_cost - cheapest.total_cost),
        delta=priciest.label,
        delta_arrow="off",
        border=True,
    )

# --------------------------------------------------------------------------- #
# Ranked comparison table
# --------------------------------------------------------------------------- #
indexed = merged.set_index("vehicle_id")
table = comparison.copy()
table["segment"] = [indexed.at[item, "body_segment"] for item in table["vehicle_id"]]
table["brand_group"] = [
    indexed.at[item, "brand_group"] for item in table["vehicle_id"]
]
table["value_status"] = [
    "User override"
    if overrides_by_vehicle.get(item)
    else indexed.at[item, "value_status"]
    for item in table["vehicle_id"]
]

with st.container(border=True):
    st.markdown("**Ranked comparison**")
    st.dataframe(
        table,
        key="comparison_table",
        hide_index=True,
        column_config={
            "vehicle_id": None,
            "label": st.column_config.TextColumn("Vehicle", pinned=True),
            "powertrain": st.column_config.TextColumn("Powertrain"),
            "segment": st.column_config.TextColumn("Segment"),
            "brand_group": st.column_config.TextColumn("Brand group"),
            "value_status": st.column_config.TextColumn(
                "Cost assumptions",
                help="Status of this vehicle's cost values.",
            ),
            "total_cost": st.column_config.NumberColumn("Total cost", format="$%.0f"),
            "annual_cost": st.column_config.NumberColumn(
                "Annual cost", format="$%.0f"
            ),
            "monthly_cost": st.column_config.NumberColumn(
                "Monthly equivalent", format="$%.0f"
            ),
            "cost_per_mile": st.column_config.NumberColumn(
                "Cost per mile", format="$%.3f"
            ),
            "purchase_price": st.column_config.NumberColumn(
                "Purchase price", format="$%.0f"
            ),
            "sales_tax": st.column_config.NumberColumn("Sales tax", format="$%.0f"),
            "purchase_fees": st.column_config.NumberColumn(
                "Purchase fees", format="$%.0f"
            ),
            "depreciation": st.column_config.NumberColumn(
                "Depreciation", format="$%.0f"
            ),
            "energy_cost": st.column_config.NumberColumn("Energy", format="$%.0f"),
            "electricity_cost": st.column_config.NumberColumn(
                "Electricity", format="$%.0f"
            ),
            "gasoline_cost": st.column_config.NumberColumn("Gasoline", format="$%.0f"),
            "maintenance_cost": st.column_config.NumberColumn(
                "Maintenance", format="$%.0f"
            ),
            "insurance_cost": st.column_config.NumberColumn(
                "Insurance", format="$%.0f"
            ),
            "registration_cost": st.column_config.NumberColumn(
                "Registration and fees", format="$%.0f"
            ),
            "financing_interest": st.column_config.NumberColumn(
                "Financing interest", format="$%.0f"
            ),
            "monthly_payment": st.column_config.NumberColumn(
                "Loan payment", format="$%.0f"
            ),
            "ending_resale_value": st.column_config.NumberColumn(
                "Ending resale value", format="$%.0f"
            ),
            "remaining_loan_balance": st.column_config.NumberColumn(
                "Loan balance at sale", format="$%.0f"
            ),
        },
    )
    st.caption(
        "Sorted by total economic cost. Resale value is already inside "
        "depreciation and is never subtracted twice."
    )

# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
breakdown_column, trend_column = st.columns(2)
with breakdown_column:
    with st.container(border=True):
        st.markdown("**Cost breakdown by component**")
        st.altair_chart(
            charts.cost_breakdown_chart(
                charts.breakdown_frame(results),
                order,
                global_assumptions.ownership_years,
            )
        )
        st.caption(
            "Economic cost components only. Down payment, loan principal, and "
            "loan payoff are cash-flow timing and are not stacked here."
        )

with trend_column:
    with st.container(border=True):
        st.markdown("**Cost over time**")
        view = st.segmented_control(
            "Cost view",
            ["Economic cost", "Cash outflow"],
            key="cost_view",
            label_visibility="collapsed",
        )
        if view == "Cash outflow":
            st.altair_chart(
                charts.cumulative_cost_chart(
                    charts.cumulative_frame(results, "cash"),
                    order,
                    "cumulative_cash_outflow",
                    "Cumulative cash outflow (USD)",
                )
            )
            st.caption(
                "Actual cash paid: down payment, loan payments, operating costs, "
                "and any loan payoff, less resale proceeds in the final year."
            )
        else:
            st.altair_chart(
                charts.cumulative_cost_chart(
                    charts.cumulative_frame(results, "economic"),
                    order,
                    "cumulative_cost",
                    "Cumulative economic cost (USD)",
                )
            )
            st.caption(
                "Cumulative economic cost. Crossing lines show where a more "
                "expensive purchase catches up through lower running costs."
            )

# --------------------------------------------------------------------------- #
# Specifications
# --------------------------------------------------------------------------- #
spec_rows = []
for result in results:
    row = indexed.loc[result.vehicle_id]
    spec_rows.append(
        {
            "Vehicle": result.label,
            "Model year": None
            if pd.isna(row["model_year"])
            else int(row["model_year"]),
            "Powertrain": row["powertrain"],
            "Segment": row["body_segment"],
            "EPA class": row.get("epa_vclass"),
            "Drive": row.get("drive"),
            "MPG": row["effective_mpg"],
            "MPG source": row["mpg_origin"],
            "kWh/100 mi": row["effective_kwh_per_100mi"],
            "Electric source": row["kwh_origin"],
            "Electric range (mi)": row["effective_electric_range_mi"],
            "Electric share": row["effective_electric_share"],
            "Cost assumptions": "User override"
            if overrides_by_vehicle.get(result.vehicle_id)
            else row["value_status"],
            "EPA record": None
            if pd.isna(row["source_vehicle_id"])
            else int(row["source_vehicle_id"]),
        }
    )

with st.container(border=True):
    st.markdown("**Specifications and provenance**")
    st.dataframe(
        pd.DataFrame(spec_rows),
        hide_index=True,
        column_config={
            "MPG": st.column_config.NumberColumn(format="%.1f"),
            "kWh/100 mi": st.column_config.NumberColumn(format="%.1f"),
            "Electric range (mi)": st.column_config.NumberColumn(format="%.0f"),
            "Electric share": st.column_config.NumberColumn(format="percent"),
        },
    )
    st.caption(
        "Efficiency and range are EPA values unless the source column says "
        "otherwise. An unknown electric range stays unknown and is never shown as "
        "zero. FuelEconomy.gov supplies no price, maintenance, insurance, "
        "depreciation, or resale data."
    )

# --------------------------------------------------------------------------- #
# Per-vehicle overrides
# --------------------------------------------------------------------------- #
override_panel = st.expander(
    "Per-vehicle overrides", icon=":material/edit_note:", on_change="rerun"
)
if override_panel.open:
    with override_panel:
        st.caption(
            "Leave a cell empty to keep the catalog or EPA value. Changes are "
            "pending until you select Apply overrides."
        )
        override_rows = [
            {
                "vehicle_id": result.vehicle_id,
                "Vehicle": result.label,
                **{
                    name: overrides_by_vehicle.get(result.vehicle_id, {}).get(name)
                    for name in OVERRIDE_FIELDS
                },
            }
            for result in results
        ]
        with st.form("overrides_form", border=False):
            edited_overrides = st.data_editor(
                pd.DataFrame(override_rows),
                key="overrides_editor",
                hide_index=True,
                disabled=["vehicle_id", "Vehicle"],
                column_config={
                    "vehicle_id": None,
                    "Vehicle": st.column_config.TextColumn(pinned=True),
                    "purchase_price_usd": st.column_config.NumberColumn(
                        "Purchase price ($)", min_value=0.0, format="$%.0f"
                    ),
                    "mpg": st.column_config.NumberColumn(
                        "MPG", min_value=0.0, format="%.1f"
                    ),
                    "kwh_per_100mi": st.column_config.NumberColumn(
                        "kWh/100 mi", min_value=0.0, format="%.1f"
                    ),
                    "phev_electric_share": st.column_config.NumberColumn(
                        "Electric share (0-1)", min_value=0.0, max_value=1.0
                    ),
                    "annual_depreciation_rate": st.column_config.NumberColumn(
                        "Depreciation rate (0-1)", min_value=0.0, max_value=1.0
                    ),
                    "annual_maintenance_usd": st.column_config.NumberColumn(
                        "Maintenance ($/yr at 15,000 mi)",
                        min_value=0.0,
                        format="$%.0f",
                        help=(
                            "Stated at 15,000 mi/yr. Half of it is rescaled to the "
                            "annual mileage in the sidebar."
                        ),
                    ),
                    "annual_insurance_usd": st.column_config.NumberColumn(
                        "Insurance ($/yr)", min_value=0.0, format="$%.0f"
                    ),
                    "annual_registration_fees_usd": st.column_config.NumberColumn(
                        "Registration ($/yr)", min_value=0.0, format="$%.0f"
                    ),
                    "resale_value_usd": st.column_config.NumberColumn(
                        "Ending resale value ($)", min_value=0.0, format="$%.0f"
                    ),
                },
            )
            override_actions = st.container(horizontal=True)
            applied = override_actions.form_submit_button(
                "Apply overrides", icon=":material/check:", type="primary"
            )
            cleared = override_actions.form_submit_button(
                "Clear all overrides", icon=":material/undo:"
            )
        if cleared:
            st.session_state.vehicle_overrides = {}
            st.rerun()
        if applied:
            updated: dict[str, dict] = {}
            for record in edited_overrides.to_dict("records"):
                values = {
                    name: record[name]
                    for name in OVERRIDE_FIELDS
                    if record.get(name) is not None and pd.notna(record.get(name))
                }
                if values:
                    updated[str(record["vehicle_id"])] = values
            st.session_state.vehicle_overrides = updated
            st.rerun()

# --------------------------------------------------------------------------- #
# Assumptions workspace
# --------------------------------------------------------------------------- #
workspace = st.expander(
    "Assumptions workspace", icon=":material/dataset:", on_change="rerun"
)
if workspace.open:
    with workspace:
        st.caption(f"Active catalog: {st.session_state.catalog_label}")
        catalog_tab, upload_tab, epa_tab, taxonomy_tab, provenance_tab = st.tabs(
            [
                "Catalog assumptions",
                "Upload and download",
                "Add EPA vehicles",
                "Taxonomy",
                "Provenance",
            ],
            on_change="rerun",
        )

        if catalog_tab.open:
            with catalog_tab:
                with st.form("catalog_form", border=False):
                    edited_catalog = st.data_editor(
                        catalog,
                        key="catalog_editor",
                        hide_index=True,
                        num_rows="dynamic",
                        column_config={
                            "powertrain": st.column_config.SelectboxColumn(
                                options=list(SUPPORTED_POWERTRAINS)
                            ),
                            "body_segment": st.column_config.SelectboxColumn(
                                options=body_segments["body_segment"].tolist()
                            ),
                            "value_status": st.column_config.SelectboxColumn(
                                options=list(VALUE_STATUSES)
                            ),
                            "purchase_price_usd": st.column_config.NumberColumn(
                                min_value=0.0, format="$%.0f"
                            ),
                            "annual_depreciation_rate": st.column_config.NumberColumn(
                                min_value=0.0, max_value=1.0
                            ),
                            "phev_electric_share": st.column_config.NumberColumn(
                                min_value=0.0, max_value=1.0
                            ),
                        },
                    )
                    catalog_actions = st.container(horizontal=True)
                    save_catalog = catalog_actions.form_submit_button(
                        "Save catalog changes", icon=":material/save:", type="primary"
                    )
                    reset_catalog = catalog_actions.form_submit_button(
                        "Reset to seed catalog", icon=":material/restart_alt:"
                    )
                if reset_catalog:
                    st.session_state.active_catalog = seed_catalog.copy()
                    st.session_state.catalog_label = SEED_CATALOG_LABEL
                    st.session_state.vehicle_overrides = {}
                    st.rerun()
                if save_catalog:
                    report = validate_catalog(edited_catalog)
                    if report.is_valid:
                        st.session_state.active_catalog = report.frame
                        st.session_state.catalog_label = "Edited in the app"
                        st.rerun()
                    else:
                        st.error(
                            "The edits were not applied. Fix the problems below and "
                            "save again.",
                            icon=":material/error:",
                        )
                        st.dataframe(report.errors, hide_index=True)

        if upload_tab.open:
            with upload_tab:
                st.markdown(
                    "Upload a CSV with the same columns to replace the active "
                    "catalog. The current catalog stays in place if validation fails."
                )
                with st.form("upload_form", border=False):
                    uploaded = st.file_uploader(
                        "Catalog CSV", type=["csv"], key="catalog_upload"
                    )
                    submitted_upload = st.form_submit_button(
                        "Validate and apply", icon=":material/upload:", type="primary"
                    )
                if submitted_upload:
                    if uploaded is None:
                        st.warning(
                            "Choose a CSV file first.", icon=":material/warning:"
                        )
                    else:
                        try:
                            report = parse_uploaded_catalog(uploaded.getvalue())
                        except ValueError as error:
                            st.error(
                                f"The upload was rejected: {error} The active "
                                "catalog is unchanged.",
                                icon=":material/error:",
                            )
                        else:
                            if report.is_valid:
                                st.session_state.active_catalog = report.frame
                                st.session_state.catalog_label = (
                                    f"Uploaded: {uploaded.name}"
                                )
                                st.session_state.vehicle_overrides = {}
                                st.session_state.shortlist = []
                                st.success(
                                    f"Applied {len(report.frame)} rows from "
                                    f"{uploaded.name}.",
                                    icon=":material/check_circle:",
                                )
                            else:
                                st.error(
                                    "The upload was rejected. The active catalog is "
                                    "unchanged.",
                                    icon=":material/error:",
                                )
                                if report.missing_columns:
                                    st.write(
                                        "Missing columns: "
                                        + ", ".join(report.missing_columns)
                                    )
                                st.dataframe(report.errors, hide_index=True)

                download_row = st.container(horizontal=True)
                download_row.download_button(
                    "Download active assumptions",
                    data=catalog.to_csv(index=False),
                    file_name="vehicle_cost_assumptions.csv",
                    mime="text/csv",
                    icon=":material/download:",
                )
                download_row.download_button(
                    "Download comparison results",
                    data=comparison.to_csv(index=False),
                    file_name="tco_comparison_results.csv",
                    mime="text/csv",
                    icon=":material/download:",
                )

        if epa_tab.open:
            with epa_tab:
                st.caption(
                    f"{len(epa_catalog_frame):,} EPA configurations are available. "
                    "Added vehicles arrive without cost assumptions and stay "
                    "labeled incomplete until you fill them in."
                )
                with st.form("epa_search_form", border=False):
                    search_row = st.container(horizontal=True)
                    search_year = search_row.number_input(
                        "Model year",
                        min_value=int(epa_catalog_frame["year"].min()),
                        max_value=int(epa_catalog_frame["year"].max()),
                        value=int(epa_catalog_frame["year"].max()) - 1,
                        key="epa_search_year",
                    )
                    search_text = search_row.text_input(
                        "Make or model contains", key="epa_search_text"
                    )
                    searched = st.form_submit_button(
                        "Search", icon=":material/search:", type="primary"
                    )
                if searched:
                    matches = epa_catalog_frame[
                        (epa_catalog_frame["year"] == search_year)
                        & epa_catalog_frame["powertrain"].isin(SUPPORTED_POWERTRAINS)
                    ]
                    if search_text:
                        needle = search_text.strip().lower()
                        matches = matches[
                            matches["make"].str.lower().str.contains(needle, na=False)
                            | matches["model"]
                            .str.lower()
                            .str.contains(needle, na=False)
                        ]
                    st.session_state.epa_matches = matches.head(200)

                matches = st.session_state.epa_matches
                if matches is not None and not matches.empty:
                    labels = dict(
                        zip(matches["source_vehicle_id"], matches["vehicle_label"])
                    )
                    with st.form("epa_add_form", border=False):
                        chosen = st.multiselect(
                            "Vehicles to add",
                            options=matches["source_vehicle_id"].tolist(),
                            format_func=lambda item: labels.get(item, str(item)),
                            key="epa_add_choice",
                        )
                        added = st.form_submit_button(
                            "Add to the catalog",
                            icon=":material/add:",
                            type="primary",
                        )
                    if added and chosen:
                        existing_ids = set(catalog["vehicle_id"])
                        new_rows = []
                        for source_id in chosen:
                            source = matches.loc[
                                matches["source_vehicle_id"] == source_id
                            ].iloc[0]
                            vehicle_id = f"epa-{int(source_id)}"
                            if vehicle_id in existing_ids:
                                continue
                            new_rows.append(
                                {
                                    "vehicle_id": vehicle_id,
                                    "source_vehicle_id": int(source_id),
                                    "model_year": int(source["year"]),
                                    "make": source["make"],
                                    "model": source["model"],
                                    "trim": source["drive"],
                                    "powertrain": source["powertrain"],
                                    "body_segment": UNCLASSIFIED,
                                    "brand_group": assign_brand_group(
                                        source["make"], group_lookup
                                    ),
                                    "data_source": (
                                        "FuelEconomy.gov specification, cost "
                                        "assumptions not yet entered"
                                    ),
                                    "value_status": "Sourced",
                                    "notes": "Cost assumptions still required.",
                                }
                            )
                        if new_rows:
                            st.session_state.active_catalog = pd.concat(
                                [catalog, pd.DataFrame(new_rows)], ignore_index=True
                            )
                            st.session_state.catalog_label = "Edited in the app"
                            st.rerun()
                elif matches is not None:
                    st.info("No EPA records matched.", icon=":material/search_off:")

        if taxonomy_tab.open:
            with taxonomy_tab:
                st.markdown("**Brand groups**")
                st.caption(
                    "Membership is data, not code. Unmapped makes stay in "
                    f"'{UNCLASSIFIED}'."
                )
                st.dataframe(brand_groups, hide_index=True)
                st.markdown("**Body segments**")
                st.dataframe(body_segments, hide_index=True)
                st.caption(
                    "The EPA size class is a separate source field and is shown in "
                    "the specifications table; it is not this segment taxonomy."
                )

        if provenance_tab.open:
            with provenance_tab:
                st.markdown("**Per-field provenance for the shortlist**")
                st.dataframe(
                    build_field_provenance(
                        merged.loc[merged["vehicle_id"].isin(shortlist)],
                        source_provenance.get("epa_vehicles", {}),
                        provenance_overrides,
                    ),
                    hide_index=True,
                )
                st.markdown("**Source datasets**")
                st.json(source_provenance, expanded=False)

# --------------------------------------------------------------------------- #
# Washington model context
# --------------------------------------------------------------------------- #
registry_panel = st.expander(
    "Washington registered models (optional)",
    icon=":material/map:",
    on_change="rerun",
)
if registry_panel.open:
    with registry_panel:
        st.caption(
            "Unique battery electric and plug-in hybrid model configurations "
            "registered in Washington State, derived from the Department of "
            "Licensing Electric Vehicle Population Data. This shows which models "
            "are registered, never how many: per-vehicle records, counts, and "
            "locations are removed, so nothing here is sales volume or market "
            "share. The file carries no price or efficiency data."
        )
        try:
            context = loaders.load_registered_models(fingerprints["wa_models"])
        except (FileNotFoundError, ValueError) as error:
            st.error(str(error), icon=":material/error:")
        else:
            span = (
                f" spanning model years {context.model_year_range[0]}"
                f"\u2013{context.model_year_range[1]}"
                if context.model_year_range
                else ""
            )
            st.caption(
                f"{context.total_models:,} distinct model configurations{span}. "
                f"{context.unknown_range_models:,} have an unknown electric range; "
                "a zero in the source means 'not researched', not a zero-mile range."
            )
            make_column, type_column = st.columns(2)
            with make_column:
                st.altair_chart(
                    charts.model_count_chart(
                        context.models_by_make.head(15), "make", "Make"
                    )
                )
            with type_column:
                st.altair_chart(
                    charts.model_count_chart(
                        context.models_by_ev_type, "ev_type", "Electric vehicle type"
                    )
                )
            st.dataframe(
                context.models,
                hide_index=True,
                column_config={
                    "model_year": st.column_config.NumberColumn(
                        "Model year", format="%d"
                    ),
                    "make": st.column_config.TextColumn("Make", pinned=True),
                    "model": st.column_config.TextColumn("Model"),
                    "ev_type": st.column_config.TextColumn("Electric vehicle type"),
                    "cafv_eligibility": st.column_config.TextColumn(
                        "CAFV eligibility"
                    ),
                    "electric_range_mi": st.column_config.NumberColumn(
                        "Electric range (mi)", format="%d"
                    ),
                },
            )

# --------------------------------------------------------------------------- #
# Methodology
# --------------------------------------------------------------------------- #
methodology = st.expander(
    "Methodology, formulas, and caveats",
    icon=":material/functions:",
    on_change="rerun",
)
if methodology.open:
    with methodology:
        st.markdown(
            f"""
**Economic total cost of ownership**

```text
TCO = depreciation + sales tax + purchase fees + financing interest
      + energy + maintenance + insurance + registration and recurring fees
```

Down payment, loan principal, and any remaining loan payoff are cash-flow timing
and are never added on top of depreciation, taxes, or fees.

**Financing**

```text
sales tax        = purchase price x {formatting.percent(global_assumptions.sales_tax_rate)}
acquisition cost = purchase price + sales tax + purchase fees
amount financed  = max(acquisition cost - down payment, 0)
monthly payment  = P x r / (1 - (1 + r)^-n),  r = APR / 12
```

Zero APR divides the financed amount evenly across the term. A zero-month term
is a cash purchase with no interest. If the vehicle is sold before the loan ends,
the remaining principal is paid off at disposal.

**Energy**

```text
BEV       electricity = miles x kWh per 100 mi / 100 x electricity price
HEV, gas  gasoline    = miles / MPG x gasoline price
PHEV      electric miles = miles x electric share; remaining miles use gasoline
```

**Depreciation and resale**

```text
ending resale value = purchase price x (1 - annual rate) ^ years
depreciation        = purchase price - ending resale value
```

A resale override replaces the formula result and is capped between $0 and the
purchase price. Resale value sits inside depreciation and is never subtracted a
second time.

**Maintenance**

```text
annual maintenance = catalog value x ({MAINTENANCE_FIXED_SHARE:.2f} + {1 - MAINTENANCE_FIXED_SHARE:.2f} x annual miles / {MAINTENANCE_REFERENCE_MILES:,.0f})
```

Catalog maintenance is stated at the {MAINTENANCE_REFERENCE_MILES:,.0f} mi/yr basis AAA
publishes. Service intervals are partly time-based and partly distance-based, so
half of the figure is held fixed and half scales with the miles actually driven.
Insurance, registration, and depreciation do not scale with mileage.

**Fractional years** - the ownership period is constrained to whole years, so
every annual figure covers a complete year.

**Caveats**

- Purchase prices are modelled estimates from a regression fitted to published
  manufacturer MSRPs, not quotes. Each carries an 80% prediction interval shown
  in the provenance tab, and a vehicle whose interval was too wide is published
  with no price rather than an invented one.
- Depreciation, maintenance, insurance, and registration come from AAA's
  published category averages. They depend on body segment, powertrain, and
  price only, so every model in a segment shares them; they are not
  model-specific figures.
- The sales-tax rate is a general sales-tax estimate for the mapped state. It is
  not a motor-vehicle tax determination and not a ZIP-exact combined rate.
- Every derived value is editable, and overriding one replaces the model for
  that vehicle.
"""
        )
        reconciled = all(components_reconcile(result) for result in results)
        st.caption(
            "Component reconciliation: "
            + (
                "every displayed component sums to the reported total."
                if reconciled
                else "a mismatch was detected; check the assumptions."
            )
        )
        active_overrides = {
            label_by_id.get(key, key): value
            for key, value in overrides_by_vehicle.items()
            if value
        }
        if active_overrides:
            st.markdown("**Active per-vehicle overrides**")
            st.json(active_overrides, expanded=False)