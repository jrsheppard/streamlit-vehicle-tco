# Build a vehicle total-cost-of-ownership comparison app

You are a senior Python and Streamlit engineer. Replace the existing basic CSV
viewer app with a polished, tested vehicle total-cost-of-ownership (TCO) comparison
app. Work autonomously through implementation, validation, and a local smoke
test. Do not stop after proposing a plan.

## Required workflow

1. Inspect the repository before editing. Keep `streamlit_app.py` as the
	 Streamlit entry point and treat `static/extract.csv` as read-only source data.
2. Locate the Streamlit environment used by this project. It may be named
	 `venv` or `.venv`. Before changing app code, read the version-matched skill:
	 `lib/python3.11/site-packages/streamlit/.agents/skills/developing-with-streamlit/SKILL.md`.
3. Follow that routing skill and read these references before implementation:
	 - `references/best-practices.md`
	 - `references/dashboards.md`
	 - `references/data-display.md`
	 - `references/performance.md`
	 - `references/design.md`
	 - `references/selection-widgets.md`
	 - `references/code-organization.md`
	 - `references/api-reference.md`
	 - `references/testing.md`
4. Use the project interpreter for all commands. Consult local API docs with
	 `streamlit docs st.<command>` when an API signature is uncertain or recent.
5. Inspect the local CSV programmatically instead of assuming its exact schema,
	 data types, row count, completeness, or value ranges. The file is larger than
	 50 MB, so use memory-conscious loading and profiling.
6. Make the smallest coherent set of files needed for a maintainable app. Add
	 dependencies only when necessary and record them in the project's existing
	 dependency mechanism, creating one if none exists.

## Product goal

Build a self-contained **Python Streamlit application** that enables users to compare the total cost of ownership (TCO) of selected battery electric vehicles (BEVs), plug-in hybrid electric vehicles (PHEVs), conventional hybrid electric vehicles (HEVs), and gasoline vehicles sold across the United States.

## 1. Application Framework

* Use Python and Streamlit as the primary application framework.
* Build a modular, maintainable application with a clear separation between data ingestion, statistical modeling, cost calculations, and the Streamlit user interface.
* Store data locally in CSV files or other appropriate local data formats.
* The app should run independently with all required dependencies documented in `requirements.txt`.
* Do not require a paid data source, API key, or live network request to use the app. Download public source data during a documented refresh step and run the app from normalized local files.

## 2. User Inputs and Customization

Users must be able to dynamically adjust the assumptions that affect vehicle ownership costs, including:

* Electricity and gasoline prices.
* Annual mileage and driving behavior, including the proportion of electric versus gasoline driving for PHEVs.
* ZIP code.
* Vehicle purchase price and other purchase assumptions.
* Financing terms, including APR (%), loan term, down payment, and financing amount.
* Ownership period.
* Other relevant vehicle-specific and ownership assumptions.

Changes to inexpensive global assumptions and filters should immediately update
the calculated total and annualized ownership costs. Use explicit apply/save
actions for catalog editing, uploads, taxonomy changes, and other multi-field
workflows where partially entered values would be invalid or expensive to
process. Clearly indicate when changes are pending rather than active.

## 3. Sales Tax Data

Use the Tax Foundation's publicly available state sales tax data:

https://taxfoundation.org/data/all/state/sales-tax-rates/

Download and store the relevant sales tax rates in a CSV file that integrates with the financing and total cost of ownership calculations.

The application should:

* Validate the user's five-digit ZIP code and map it to a state using a
	documented local ZIP-to-state dataset with its own provenance and refresh
	process. Handle unknown and multi-jurisdiction ZIP codes explicitly.
* Use the mapped state's Tax Foundation rate as an editable general-sales-tax
	estimate, not as an authoritative motor-vehicle tax determination.
* Apply the active editable sales-tax assumption to vehicle purchase
	assumptions.
* Clearly distinguish state sales tax from local taxes, fees, and other applicable charges where data is available.
* Display the source and effective date of the sales tax data.

Do not assume that a state-level sales tax rate represents the exact combined rate applicable to every ZIP code. Where local rates are not available, disclose the limitation and allow the user to edit the assumption.
Tax Foundation's local figures are population-weighted state averages rather
than ZIP-exact rates, and its general sales-tax table does not encode every
state's motor-vehicle-specific tax, exemption, trade-in credit, cap, or fee.
Never label the calculated tax as exact. Show the state rate, any estimated
local component, the active combined assumption, and the user's override as
separate values where applicable.

## 4. Nationwide vehicle data

Use the official FuelEconomy.gov vehicle dataset as the primary nationwide
vehicle catalog:

- Download page: `https://www.fueleconomy.gov/feg/download.shtml`
- Preferred machine-readable file:
  `https://www.fueleconomy.gov/feg/epadata/vehicles.csv.zip`
- Field documentation:
  `https://www.fueleconomy.gov/feg/ws/index.shtml#vehicle`

The source is maintained by the U.S. Department of Energy and U.S.
Environmental Protection Agency. It requires no API key and covers model years
1984 through the current/preliminary model year. At the time this prompt was
written, the download contained about 50,000 vehicle configurations and was
about 22 MB uncompressed. Treat those counts as observations, not invariants.

Create a repeatable data-refresh command or script that:

1. Downloads the zipped CSV to a temporary location.
2. Validates the HTTP response, ZIP member, required columns, and nonempty row
	count before replacing existing local data.
3. Records the source URL, download timestamp, HTTP `Last-Modified` and `ETag`
	values when available, row count, and a file checksum in provenance metadata.
4. Normalizes only the fields needed by the app into a local CSV or Parquet
	file. An interrupted or invalid refresh must not destroy the last valid copy.
5. Is documented and testable. The Streamlit app itself must use the local
	normalized file and must not depend on network availability.

Preserve the EPA vehicle record `id` as `source_vehicle_id`. Normalize these
source fields where present:

| App concept | FuelEconomy.gov field |
| --- | --- |
| Model year | `year` |
| Make | `make` |
| Model/configuration | `model` |
| Base model | `baseModel` |
| Fuel and technology classification | `fuelType`, `fuelType1`, `fuelType2`, `atvType` |
| EPA vehicle class | `VClass` |
| Drive and transmission | `drive`, `trany` |
| Combined gasoline efficiency | `comb08` or unrounded `comb08U` |
| Combined electric consumption | `combE` in kWh/100 miles |
| PHEV electric-driving utility factor | `combinedUF` |
| Electric or charge-depleting range | `range`, with `rangeA` as applicable |
| Charging information | `charge120`, `charge240`, `evMotor` |
| Source timestamps | `createdOn`, `modifiedOn` |

Do not assume every field is populated. Follow the official field documentation
when interpreting dual-fuel records. In particular, `comb08` represents MPGe
for electric vehicles, not gasoline MPG; use `combE` for BEV electric energy
cost. Do not use FuelEconomy.gov's precomputed annual fuel-cost fields for TCO,
because this app must recalculate energy cost from the user's mileage and energy
prices.

Implement and unit-test one deterministic powertrain classifier with this
precedence:

1. `BEV` when `atvType` is `EV` or the primary fuel is electricity.
2. `PHEV` when `atvType` is `Plug-in Hybrid`.
3. `HEV` when `atvType` is `Hybrid` and the vehicle is not a PHEV.
4. `Gasoline` for supported regular, midgrade, or premium gasoline records not
	classified above.
5. `Other` for diesel, flex-fuel, CNG, hydrogen, or ambiguous records unless the
	calculation engine explicitly supports that fuel.

These classifications can overlap in the raw fields, so do not derive totals by
adding independent field counts. Keep `Other` records available for auditing but
exclude them from default comparisons.

FuelEconomy.gov does **not** supply dependable MSRP, current market price,
maintenance, insurance, depreciation, resale value, VIN, or a general-purpose
body segment. `VClass` is an EPA size class and must be retained separately from
the app's editable body-segment taxonomy.

### Optional NHTSA enrichment

NHTSA vPIC (`https://vpic.nhtsa.dot.gov/api/`) may be used to enrich make/model,
vehicle type, body class, manufacturer, or VIN-decoded specifications. It is
optional for the first working version and must not block delivery. If used:

- cache normalized responses locally;
- respect automated traffic controls and the 50-VIN batch limit;
- retain NHTSA identifiers and provenance;
- never use fuzzy matching without recording match confidence and allowing
  manual correction; and
- do not expect vPIC to provide price, fuel economy, maintenance, depreciation,
  or insurance values.

DOE AFDC is not needed for the initial light-duty catalog because its light-duty
vehicle search directs users to FuelEconomy.gov. Consider it only if medium- or
heavy-duty alternative-fuel vehicles become in scope.

### Sources that require explicit approval

Do not scrape Consumer Reports, Edmunds, J.D. Power, manufacturer sites, dealer
sites, or marketplace listings. Do not imply these are open datasets:

- Edmunds retired its open API; its vehicle, TMV, and TCO APIs require an API key
  and an approved dealership or strategic-partner relationship.
- Consumer Reports publishes useful research, often at brand/category level,
  but does not provide a general public model-level ownership-cost feed for this
  app.
- J.D. Power valuation and automotive data should be treated as commercially
  licensed unless access is explicitly supplied.
- AAA ownership-cost reports can inform documented category-level assumptions,
  but are not a complete model-level vehicle catalog.

Only integrate one of these sources after the user supplies approval, access,
and acceptable license terms. Never bypass access controls or copy values from
web pages into a bulk local dataset.

Before commercial redistribution, review and document the current source terms.
FuelEconomy.gov's published disclaimer permits free distribution and use for
noncommercial scientific and educational purposes but does not by itself grant
unrestricted commercial redistribution rights.

## 5. Vehicle cost assumptions and modeling

Separate sourced vehicle specifications from cost assumptions. Use two linked
data layers:

1. A read-only normalized vehicle-specification catalog keyed by
	`source_vehicle_id`, sourced primarily from FuelEconomy.gov.
2. An editable cost-assumption table keyed by `source_vehicle_id` or a stable
	internal comparison ID, containing purchase price, depreciation/residual
	value, maintenance, insurance, taxes/fees, and source notes.

The app may display every supported EPA vehicle in search and filters, but a
vehicle is comparison-ready only when all cost and efficiency fields required
by its powertrain are present. Clearly label incomplete vehicles and provide an
efficient way for users to enter the missing assumptions.

Seed a small representative set of comparison-ready vehicles across BEV, PHEV,
HEV, and gasoline powertrains. EPA efficiency and specification values must
retain EPA provenance. Unsupported cost values must be labeled `Illustrative
default` unless a verified source and effective date are recorded.

Do not train a depreciation, maintenance, insurance, or resale model merely
because the prompt mentions predictive analysis. Fit a statistical model only
when an actual licensed dataset with enough observations, a defined target, and
documented methodology is available. Otherwise use transparent editable rules
or category-level defaults and label them as assumptions, not predictions.

Store source name, source URL, effective or retrieval date, units, and value
status (`Sourced`, `Derived`, `Illustrative default`, or `User override`) for
every material specification or cost assumption. Use an explicit per-field
provenance schema, such as a long-form table keyed by vehicle ID and field name,
instead of relying on one row-level source label for values from different
origins.

## 6. Total Cost of Ownership Calculations

Implement a transparent cost calculation engine that calculates the following for each selected vehicle:

* Initial purchase price.
* Sales tax and applicable purchase fees.
* Financing interest and other financing costs.
* Fuel and electricity costs.
* Maintenance and repair costs.
* Insurance costs.
* Registration fees.
* Depreciation.
* Total cost of ownership over the selected period.
* Annualized cost of ownership.
* Monthly equivalent cost, where useful.

The model should account for the timing of cash flows, financing payments, and residual vehicle value.

Clearly distinguish between:

1. **Economic TCO / net ownership cost:** Depreciation, transaction taxes and
	fees, interest, energy, maintenance, insurance, registration, and other
	ownership expenses. Depreciation already accounts for estimated resale value.
2. **Cash outflow:** Down payment, loan payments, remaining-loan payoff,
	transaction taxes or fees paid outside the loan, operating expenses, and
	other actual cash payments by period.

Avoid double-counting principal payments as an economic cost when calculating total cost of ownership.

## 7. Streamlit User Interface

Build an intuitive Streamlit interface that allows users to:

* Select and compare multiple vehicles.
* Modify assumptions using Streamlit input widgets.
* Immediately view updated total and annualized costs.
* View useful vehicle specifications alongside ownership costs.
* See a breakdown of costs by category.
* Compare vehicles using tables and visualizations.
* Understand how changing an assumption affects the results.

The UI should be responsive, readable, and usable without requiring users to understand the underlying Python code.

## 8. Transparency and Auditability

This is a decision-support calculator, not financial advice.

Every calculated result must be traceable to a visible formula and an editable assumption.

The application should provide:

* A visible breakdown of each cost component.
* The formula used to calculate each component.
* The source of each sourced assumption.
* The date and units associated with sourced data.
* Editable assumptions for all material estimates.
* Clear labeling of sourced data, statistical estimates, and user inputs.
* A methodology or assumptions section explaining the calculation engine.

Users should be able to understand exactly how the app arrives at a total cost of ownership and how changing any assumption affects the result.

## Supplied dataset and its limits

`static/extract.csv` is the Washington State Department of Licensing Electric
Vehicle Population Data. It represents currently registered battery electric
vehicles (BEVs) and plug-in hybrid electric vehicles (PHEVs), not all vehicles
available for sale. The current official schema is expected to include fields
such as:

- `VIN (1-10)`
- `County`, `City`, `State`, and `Postal Code`
- `Model Year`, `Make`, and `Model`
- `Electric Vehicle Type`
- `Clean Alternative Fuel Vehicle (CAFV) Eligibility`
- `Electric Range`
- `Legislative District`
- `DOL Vehicle ID`
- `Vehicle Location`
- `Electric Utility`
- `2020 GEOID`

Confirm the local headers rather than requiring this expected list verbatim.
Normalize only what the app uses, preserve source meaning, and handle missing or
changed fields gracefully.

Apply these source caveats throughout the implementation:

- The file contains BEVs and PHEVs only. It contains no conventional gasoline
	vehicle catalog.
- It does not provide reliable current purchase price, energy efficiency,
	depreciation, maintenance, insurance, fees, or financing data.
- The official source stopped publishing Base MSRP in December 2025. Do not
	depend on an older `Base MSRP` field if one happens to exist in a snapshot.
- An `Electric Range` value of `0` can mean the range was not researched. Treat
	zero as unknown for specifications and filtering, not as a true zero-mile
	range.
- Registration counts are neither sales volume nor market share and must not be
	presented as either.
- A make/model can span materially different trims, batteries, engines, and
	efficiencies. Do not convert registry frequency into model-level cost facts.

Use this dataset only for an optional Washington registration-context view, such
as counts by county or locally registered make/model. Do not use it as the
nationwide comparison catalog, do not require a Washington match before showing
an EPA vehicle, and do not let it overwrite EPA efficiency specifications.
Never silently infer missing cost or efficiency values from it.

## Editable comparison catalog

Create an editable cost-assumption layer joined to the normalized EPA vehicle
catalog. It must support BEVs, PHEVs, conventional HEVs, and gasoline vehicles.
Do not add fictional values and do not label illustrative values as
authoritative facts.

The catalog must support one row per comparable vehicle variant, with a stable
vehicle ID and at least these fields:

- model year, make, model, and optional trim
- powertrain: `BEV`, `PHEV`, `HEV`, or `Gasoline`
- body segment
- brand group
- purchase price in dollars
- electric efficiency in kWh per 100 miles, when applicable
- gasoline efficiency in MPG, when applicable
- PHEV electric-driving share from 0 through 1, when applicable
- usable electric range in miles, when known
- annual depreciation rate from 0 through 1
- annual maintenance cost in dollars
- annual insurance and registration/fee cost in dollars
- data source and notes

Seed a small but useful comparison-ready set spanning all four powertrains,
multiple segments, and multiple brand groups. If verified local or authoritative
data is unavailable, clearly label seed values as `Illustrative default` and
show that label in the app. Prefer a small transparent seed catalog over a large
set of invented values.

Users must be able to edit catalog assumptions in the app and upload a
replacement or supplemental CSV. They must also be able to download the active
assumptions and comparison results. Validate uploaded data before replacing the
active catalog and leave the current data intact when validation fails.

Validate and explain:

- required columns and stable vehicle IDs
- duplicate vehicle IDs
- numeric conversion failures and missing required values
- nonnegative prices and costs
- depreciation and PHEV electric share bounded from 0 through 1
- positive MPG for HEV, gasoline, and the combustion portion of PHEV records,
  plus positive kWh/100-mile values for BEV and PHEV records
- powertrain-incompatible fields
- unknown specifications, without converting unknowns to zero

## Taxonomy and filtering

Provide filters for body segment, brand, named brand group, powertrain, model
year, and a manageable shortlist of vehicles. Filters should narrow one another
where practical. Never render every registry record as a TCO candidate.

Start with a clear, editable taxonomy and an `Other / unclassified` fallback.
Include useful body segments such as:

- Small sedan
- Midsize sedan
- Large sedan
- Hatchback
- Small SUV / crossover
- Midsize SUV
- Large SUV
- Pickup
- Minivan
- Sports / performance

Include at least these initial brand groups, implemented as maintainable data
rather than scattered conditionals:

- Mainstream Japanese
- Mainstream American
- Mainstream Korean
- Mainstream European
- Luxury German
- Luxury Japanese
- Other Luxury
- Other / unclassified

Make the exact brand membership visible and editable. Do not force a vehicle
into a group when its mapping is unknown.

## Global user inputs

Expose clear, validated controls for:

- gasoline price in dollars per gallon
- electricity price in dollars per kWh
- miles driven per year
- PHEV electric-driving share from 0 through 1, with the option to retain each
	model's EPA `combinedUF` or catalog default
- ownership period in years
- down payment in dollars
- annual percentage rate (APR)
- loan term in months
- annual insurance and registration/fee override, with an option to retain each
	model's catalog default
- optional per-model purchase-price, efficiency, maintenance, depreciation, and
	insurance/fee overrides
- optional per-model ending resale-value override

Use sensible illustrative defaults, label units everywhere, and allow zero APR
and a cash purchase. Do not allow a down payment greater than purchase price.
Changing assumptions must update all dependent results consistently.

## Calculation contract

Put calculation and validation logic in pure Python functions outside the page
rendering code. Use full precision internally and round only for display.

For each selected vehicle, calculate costs over the ownership period as follows.

### Financing

- Calculate `sales_tax` from the taxable purchase-price assumption and active
	sales-tax rate. Keep purchase fees separate from recurring registration fees.
- Define `acquisition_cost = purchase_price + sales_tax + purchase_fees`.
- For financed purchases, assume sales tax and purchase fees are financed and
	calculate `amount_financed = max(acquisition_cost - down_payment, 0)`. Label
	this assumption and keep it editable if an alternate treatment is supported.
- For a financed purchase with positive APR, calculate the standard amortized
	monthly payment using monthly rate `APR / 12` and the selected loan term.
- For zero APR, monthly payment is `amount_financed / loan_term_months`.
- For a cash purchase or zero-month loan, acquisition cash flow is purchase
	`acquisition_cost` and financing interest is zero.
- Calculate the number of scheduled payments made during ownership as the lesser
	of the loan term and the ownership period in months.
- If the vehicle is sold before the loan ends, calculate the remaining principal
	at disposal and include its payoff. Do not pretend the remaining debt
	disappears.
- Report principal and financing interest separately. Ensure the identity
	`down payment + principal paid + remaining principal payoff = acquisition_cost`
	holds within a small numerical tolerance.

### Energy

- BEV electric miles equal annual miles. Annual electric cost is
	`annual_miles * kWh_per_100_miles / 100 * electricity_price`.
- HEV and gasoline vehicle gas miles equal annual miles. Annual gas cost is
	`annual_miles / MPG * gasoline_price`.
- For a PHEV, electric miles are `annual_miles * electric_share`; gas miles are
	the remainder. Calculate electric and gasoline costs independently using the
	applicable efficiencies, then add them.
- Do not calculate a misleading result when a required efficiency is missing or
	invalid. Flag the affected model and explain how to fix its assumptions.

### Depreciation and resale

- By default, estimate ending resale value with declining-balance depreciation:
	`purchase_price * (1 - annual_depreciation_rate) ** ownership_years`.
- If the user supplies an ending resale-value override, use it instead and label
	it as an override.
- Cap validated resale value between zero and purchase price unless the UI
	explicitly supports and explains appreciation.
- Depreciation is `purchase_price - ending_resale_value`.
- Never subtract both depreciation and resale value as separate benefits. The
	TCO equation must use one consistent representation.

### Total cost

Calculate:

`TCO = depreciation + sales_tax + purchase_fees + financing_interest + energy_cost + maintenance_cost + insurance_cost + registration_and_recurring_fees`

This is equivalent to acquisition cost plus financing interest and operating
costs minus ending resale value. Down payment, principal payments, and remaining
loan payoff are cash-flow timing details and must not be added again on top of
depreciation, taxes, or purchase fees.

Also calculate:

- total and annual energy cost, split into electricity and gasoline
- total maintenance cost
- total insurance cost
- sales tax, purchase fees, and recurring registration/fee costs as separate
	components
- total financing interest
- ending resale value and remaining loan balance at disposal
- total TCO
- average annual TCO
- TCO per mile
- cumulative cost by ownership year for charting

Clearly define how fractional ownership years are handled, or constrain the UI
to whole years and test that behavior.

## App experience

Create a focused, data-heavy comparison tool, not a landing page. Use a wide
layout and render useful controls before expensive work.

The primary experience must include:

1. A concise title and caption describing the calculator, FuelEconomy.gov as the
	 nationwide specification source, and the Washington registry as optional
	 regional context.
2. Filters and global assumptions in the sidebar, a popover, or forms where
	 batching changes prevents unnecessary reruns.
3. A responsive metric row showing the cheapest selected vehicle, its TCO,
	 annual cost, cost per mile, and savings versus the most expensive selection.
4. A ranked comparison table with total cost, annual cost, cost per mile, energy
	 cost, maintenance, insurance/fees, financing interest, resale value, and key
	 specifications.
5. A stacked cost-breakdown chart comparing selected models.
6. An annual or cumulative ownership-cost chart that exposes crossover points.
7. A vehicle specification table showing model year, make/model/trim,
	 powertrain, segment, efficiency, electric range, and assumption provenance.
8. An assumptions workspace for editing, uploading, validating, resetting, and
	 downloading the comparison catalog and taxonomy.
9. A lightweight methodology section that shows formulas, caveats, and active
	 overrides without overwhelming the main workflow.
10. Clear loading, empty-filter, invalid-input, missing-file, malformed-file,
		and no-valid-model states.

Limit one comparison to a manageable shortlist of four vehicles by default and
no more than six. When a filter change makes a selected vehicle unavailable,
remove it from the active shortlist, explain the change, and never silently
substitute another vehicle. Filters should expose only options valid under the
other active filters where practical, with a clear reset action for zero-result
states.

Keep the primary comparison focused. Put advanced financing controls,
per-vehicle overrides, detailed provenance, taxonomy editing, and Washington
registration context in clearly labeled popovers, dialogs, or dynamically
guarded secondary sections. Separate catalog assumptions, taxonomy, tax
assumptions, and provenance into focused editing surfaces rather than one large
data editor.

Apply these visualization conventions:

- use the consistent colors across related charts and visuals
- use a color-blind-safe palette and do not communicate status by
	color alone
- use values for y-axis, and dimensions for x-axis
- start quantitative bar-chart axes at zero, sort ranked comparisons by the
	selected cost metric, and do not use dual axes
- format currency, percentages, distances, energy, and cost-per-mile units in
	axes, tooltips, tables, and legends
- visibly mark incomplete, illustrative, estimated, and user-overridden values
- do not allow outliers to distort charts or graphs
- chart economic TCO and cash outflow in separate views; never stack or sum
	principal cash flows with economic cost components

Use native Streamlit capabilities and the installed version's conventions:

- Call `st.set_page_config` first and use sentence casing.
- Use Material Symbols such as `:material/directions_car:` instead of decorative
	emoji.
- Prefer responsive horizontal containers for metric rows.
- Prefer native Vega-based charts or Altair. Do not add Plotly merely for basic
	charts.
- Use `st.dataframe` and meaningful `column_config` for result tables; hide
	technical columns and indexes where appropriate.
- Never add deprecated `use_container_width`; use the current `width` API or its
	stretch default.
- Do not use `st.components.v1`.
- Do not inject custom CSS for routine polish. Use native layout and theme
	configuration unless a requirement cannot be met otherwise.
- Use visible labels or `label_visibility` rather than empty widget labels.
- Preserve stable widget keys and initialize session state in one clear place.

## Architecture and performance

- Keep `streamlit_app.py` as the direct page/entry script. Do not wrap the whole
	page body in a function.
- Move data loading, schema normalization, taxonomy, validation, financing,
	energy, depreciation, and TCO calculations into focused modules with type
	hints and testable pure functions.
- Cache the large static registry load with `st.cache_data`. Because it is a
	static local source, avoid an arbitrary short TTL; use a file fingerprint or
	function argument that invalidates the cache when the file changes.
- Cache expensive source loading, then apply cheap interactive filters outside
	the cached loader.
- Load only necessary registry columns where practical and normalize dtypes to
	control memory use.
- Aggregate the registry before display. Never send the complete 50+ MB dataset
	to the browser.
- Use `st.fragment` only for genuinely independent sections and `st.form` where
	users benefit from applying related changes together.
- Do not put expensive unguarded work in hidden tabs or expanders.
- Resolve paths relative to the project files, not the caller's current working
	directory.
- Handle missing or malformed registry and catalog files with actionable UI
	errors rather than raw tracebacks.

## Testing and verification

Add focused automated tests. At minimum, cover pure calculation functions for:

- positive-APR amortization
- zero-APR financing
- cash purchase
- financed sales tax and purchase fees, including acquisition-cost
	reconciliation
- ownership shorter than the loan and remaining-balance payoff
- ownership longer than the loan
- BEV, PHEV, HEV, and gasoline energy costs
- zero annual miles
- depreciation-derived resale value
- explicit resale override without double-counting
- total-cost component reconciliation
- valid, invalid, unknown, and multi-jurisdiction ZIP mappings; state-rate
	fallback; and user tax-rate override
- economic TCO and cash-outflow reconciliation without counting principal as an
	economic cost
- invalid down payment, APR, loan term, efficiency, depreciation, and share
- unknown electric range and missing model assumptions
- duplicate catalog IDs and malformed uploads

Use `st.testing.v1.AppTest` for the app's important behavior, including:

- successful default render
- filtering by segment, brand group, brand, and powertrain
- selecting a shortlist of models
- changing gas price, electricity price, mileage, APR, and ownership period and
	observing updated results
- invalid input and empty-result states
- malformed catalog upload behavior
- assumptions editor/reset behavior where AppTest supports it

Do not claim AppTest coverage for interactions it cannot simulate, including
dataframe or chart selections and browser layout behavior. Add a focused
browser-level smoke test for desktop and narrow viewports that checks for
overlap, clipped controls, unreadable tables or legends, and broken chart
rendering. Use browser testing only for these rendered behaviors; keep business
logic and ordinary widget interaction in pytest and AppTest.

Run the narrow tests first, then the full project test suite. Start the app with
the project environment using the correct command form, for example:

`venv/bin/python -m streamlit run streamlit_app.py`

Do not use `python streamlit run ...`. Confirm that startup succeeds and perform
a smoke check of the rendered app. Stop any temporary server when verification
is complete unless the user explicitly wants it left running.

## Acceptance criteria

The work is complete only when:

- The app compares selected BEV, PHEV, HEV, and gasoline vehicles across
	transparent, editable assumptions sourced from the nationwide EPA catalog.
- Segment, brand, brand-group, powertrain, year, and vehicle-shortlist filters
	work together.
- All requested user inputs affect the correct calculations.
- Financing, energy, depreciation/resale, maintenance, insurance/fees, and TCO
	reconcile and pass automated tests.
- The app never treats registry counts, missing electric range, removed MSRP,
	or unsupported third-party cost estimates as facts they are not.
- ZIP-derived tax is presented as an editable estimate with visible source,
	effective date, state rate, local-rate limitation, and active override.
- The large CSV is loaded efficiently and is never transmitted wholesale to the
	frontend.
- The default experience is useful on desktop and remains coherent at narrow
	viewport widths using Streamlit's responsive native layouts, as verified by
	the browser smoke test.
- Tests pass and the Streamlit app starts without an exception.

When finished, report the files changed, the commands and tests run, the local
app URL if left running, and any assumptions that remain illustrative. Do not
claim that seed cost data is authoritative unless its source is recorded and
verified.