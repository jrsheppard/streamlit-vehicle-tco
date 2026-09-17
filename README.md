# Vehicle total cost of ownership

A Streamlit calculator that compares the total cost of ownership (TCO) of battery
electric (BEV), plug-in hybrid (PHEV), conventional hybrid (HEV), and gasoline
vehicles sold in the United States.

It is a decision-support calculator, not financial advice. Every number it shows
traces back to a visible formula and an editable assumption.

## Run it

```bash
venv/bin/python -m pip install -r requirements.txt
venv/bin/python -m streamlit run streamlit_app.py
```

The app reads only local files. It never makes a network request at runtime.

## Project layout

| Path | Purpose |
| --- | --- |
| `streamlit_app.py` | Page entry script; UI wiring only |
| `tco/` | Data loading, taxonomy, validation, and the calculation engine |
| `scripts/refresh_data.py` | Repeatable download and normalization of source data |
| `scripts/reduce_registry.py` | Reduces the raw Washington snapshot to unique models |
| `data/` | Normalized local data plus `provenance.json` |
| `static/wa_registered_models.csv` | Unique Washington-registered models (optional context) |
| `tests/` | pytest unit tests, AppTest behavior tests, browser smoke test |

## Data sources and the refresh step

```bash
venv/bin/python -m scripts.refresh_data             # all sources
venv/bin/python -m scripts.refresh_data --source epa
```

Each source is downloaded to a temporary file, validated (HTTP status, archive
member, required columns, row-count threshold), normalized, and only then moved
into `data/`. An interrupted or invalid refresh leaves the previous valid copy in
place. Provenance - source URL, retrieval timestamp, HTTP `Last-Modified` and
`ETag`, row counts, and a SHA-256 checksum - is written to `data/provenance.json`
and surfaced in the app.

| Source | File | Notes |
| --- | --- | --- |
| [FuelEconomy.gov vehicles](https://www.fueleconomy.gov/feg/download.shtml) (U.S. DOE / EPA) | `data/epa_vehicles.csv` | Nationwide specification catalog. `id` is preserved as `source_vehicle_id`. No API key. |
| [Tax Foundation state and local sales tax rates](https://taxfoundation.org/data/all/state/sales-tax-rates/) | `data/sales_tax_rates.csv` | State rate, population-weighted average local rate, max local rate, combined rate, effective date. |
| [Census 2020 ZCTA-to-county relationship file](https://www2.census.gov/geo/docs/maps-data/data/rel2020/zcta520/tab20_zcta520_county20_natl.txt) | `data/zip_to_state.csv` | ZIP-to-state mapping. Multi-state ZCTAs keep every state and are flagged in the UI. |

Files maintained by hand rather than downloaded:
- `data/vehicle_cost_assumptions.csv` - the seed cost-assumption catalog. Cost
  values are labeled `Illustrative default`; they are editable placeholders, not
  verified market data.
- `data/taxonomy_body_segments.csv` and `data/taxonomy_brand_groups.csv` - the
  editable segment and brand-group taxonomy.
- `data/field_provenance.csv` - optional per-field provenance overrides. Rows
  here replace the provenance the app derives automatically.

## What the sources do and do not provide

- FuelEconomy.gov supplies specifications and efficiency only. It provides no
  MSRP, market price, maintenance, insurance, depreciation, resale value, VIN, or
  general-purpose body segment. Its `VClass` is an EPA size class and is kept
  separate from the app's editable body-segment taxonomy.
- For a BEV, `comb08` is MPGe, not gasoline MPG; electric energy cost uses
  `combE` (kWh/100 mi). FuelEconomy.gov's precomputed annual fuel-cost fields are
  not used, because energy cost is recalculated from the user's mileage and
  prices.
- Tax Foundation rates are general sales-tax rates. They do not encode
  motor-vehicle-specific rates, exemptions, trade-in credits, caps, or fees, and
  the local figure is a population-weighted state average rather than a ZIP-exact
  rate. The app shows the state rate, the estimated local component, the active
  combined assumption, and any user override separately, and never labels the
  result as exact.
- `static/wa_registered_models.csv` holds the 823 unique BEV and PHEV model
  configurations registered in Washington State, derived from the Department of
  Licensing Electric Vehicle Population Data by `scripts/reduce_registry.py`.
  The raw 80 MB snapshot has one row per registered vehicle and is not committed
  or needed at runtime. The reduction drops every per-vehicle identifier,
  registration count, and location field, so the app can show *which* models are
  registered in Washington but never *how many* - it is not sales volume and not
  market share. An `Electric Range` of `0` means "not researched" and is stored
  as unknown, never as a zero-mile range. It is never used to infer cost or
  efficiency values and never overwrites EPA specifications.
- Consumer Reports, Edmunds, J.D. Power, AAA, manufacturer sites, dealer sites,
  and marketplace listings are not used. They require licensing or approval.

Before any commercial redistribution, review the current source terms.
FuelEconomy.gov's disclaimer permits free distribution for noncommercial
scientific and educational purposes; it does not by itself grant unrestricted
commercial redistribution rights.

## Calculation summary

```text
TCO = depreciation + sales tax + purchase fees + financing interest
      + energy + maintenance + insurance + registration and recurring fees
```

Economic cost and cash outflow are reported separately. Down payment, loan
principal, and remaining-loan payoff are cash-flow timing and are never added on
top of depreciation, taxes, or fees. Resale value sits inside depreciation and is
never subtracted twice. The ownership period is constrained to whole years.

No depreciation, maintenance, insurance, or resale model is fitted to data. Every
estimate is a transparent, editable rule.

## Tests

```bash
venv/bin/python -m pip install -r requirements-dev.txt
venv/bin/python -m pytest tests -q
```

- `tests/test_finance.py`, `test_energy.py`, `test_depreciation.py`,
  `test_engine.py`, `test_salestax.py`, `test_catalog.py` cover the pure
  calculation, classification, and validation functions.
- `tests/test_app.py` uses `st.testing.v1.AppTest` for app behavior: rendering,
  filtering, shortlist selection, and input changes.
- `tests/test_browser_smoke.py` checks rendered layout on desktop and narrow
  viewports. It is skipped unless Playwright and Chromium are installed:

  ```bash
  venv/bin/python -m playwright install chromium
  ```

AppTest cannot simulate dataframe or chart selections, file uploads, or browser
layout; those are covered by unit tests and the browser smoke test instead.
