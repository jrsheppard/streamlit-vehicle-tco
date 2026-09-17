"""Refresh the local source data the app reads.

Run it with the project interpreter::

    venv/bin/python -m scripts.refresh_data            # everything
    venv/bin/python -m scripts.refresh_data --source epa

Each source is downloaded to a temporary file, validated, normalized, and only
then moved into ``data/``. An interrupted or invalid refresh leaves the previous
valid copy untouched. Provenance for every source is recorded in
``data/provenance.json``.

The Streamlit app itself never uses the network; it reads only the normalized
local files this script produces.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tco.epa_catalog import SOURCE_FIELD_MAP, normalize_epa_vehicles  # noqa: E402
from tco.paths import (  # noqa: E402
    DATA_DIR,
    EPA_CATALOG_PATH,
    PROVENANCE_PATH,
    SALES_TAX_PATH,
    ZIP_TO_STATE_PATH,
)

EPA_DOWNLOAD_URL = "https://www.fueleconomy.gov/feg/epadata/vehicles.csv.zip"
EPA_DOCS_URL = "https://www.fueleconomy.gov/feg/ws/index.shtml#vehicle"
CENSUS_ZCTA_URL = (
    "https://www2.census.gov/geo/docs/maps-data/data/rel2020/zcta520/"
    "tab20_zcta520_county20_natl.txt"
)
TAX_FOUNDATION_URL = "https://taxfoundation.org/data/all/state/sales-tax-rates/"

USER_AGENT = "vehicle-tco-calculator/1.0 (local research tool)"
MIN_EPA_ROWS = 10_000
MIN_ZIP_ROWS = 20_000
MIN_TAX_ROWS = 45

STATE_FIPS_TO_USPS = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO",
    "09": "CT", "10": "DE", "11": "DC", "12": "FL", "13": "GA", "15": "HI",
    "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY",
    "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN",
    "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
    "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD",
    "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA",
    "54": "WV", "55": "WI", "56": "WY", "60": "AS", "66": "GU", "69": "MP",
    "72": "PR", "78": "VI",
}

STATE_NAME_TO_USPS = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN",
    "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
    "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI",
    "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
    "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH",
    "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH",
    "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
    "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD",
    "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
    "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY",
}


class RefreshError(RuntimeError):
    """A source could not be refreshed. The previous local copy is kept."""


@dataclass
class Download:
    """A validated HTTP response body plus its caching metadata."""

    url: str
    content: bytes
    last_modified: str | None
    etag: str | None
    retrieved_at: str

    @property
    def checksum(self) -> str:
        return hashlib.sha256(self.content).hexdigest()


def fetch(url: str, timeout: int = 120) -> Download:
    """GET a URL and fail loudly on anything other than a usable response."""
    try:
        response = requests.get(
            url, timeout=timeout, headers={"User-Agent": USER_AGENT}
        )
    except requests.RequestException as error:
        raise RefreshError(f"Request to {url} failed: {error}") from error
    if response.status_code != 200:
        raise RefreshError(f"{url} returned HTTP {response.status_code}.")
    if not response.content:
        raise RefreshError(f"{url} returned an empty body.")
    return Download(
        url=url,
        content=response.content,
        last_modified=response.headers.get("Last-Modified"),
        etag=response.headers.get("ETag"),
        retrieved_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def write_csv_atomically(frame: pd.DataFrame, destination: Path) -> None:
    """Write a CSV via a temporary file so a failure cannot destroy the old one."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    os.close(handle)
    temporary_path = Path(temporary)
    try:
        frame.to_csv(temporary_path, index=False)
        temporary_path.chmod(0o644)
        os.replace(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def record_provenance(key: str, payload: dict[str, object]) -> None:
    """Merge one source's provenance into ``data/provenance.json``."""
    PROVENANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, object] = {}
    if PROVENANCE_PATH.exists():
        try:
            existing = json.loads(PROVENANCE_PATH.read_text())
        except json.JSONDecodeError:
            existing = {}
    existing[key] = payload
    PROVENANCE_PATH.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")


def refresh_epa(timeout: int = 180) -> dict[str, object]:
    """Download, validate, and normalize the FuelEconomy.gov vehicle catalog."""
    download = fetch(EPA_DOWNLOAD_URL, timeout=timeout)
    try:
        archive = zipfile.ZipFile(io.BytesIO(download.content))
    except zipfile.BadZipFile as error:
        raise RefreshError("The download is not a valid ZIP archive.") from error

    members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
    if "vehicles.csv" not in members:
        raise RefreshError(
            f"The archive does not contain vehicles.csv. Found: {archive.namelist()}"
        )

    with archive.open("vehicles.csv") as handle:
        raw = pd.read_csv(handle, low_memory=False)

    missing = sorted(
        source for source in SOURCE_FIELD_MAP.values() if source not in raw.columns
    )
    if missing:
        raise RefreshError(f"vehicles.csv is missing required columns: {missing}")
    if len(raw) < MIN_EPA_ROWS:
        raise RefreshError(
            f"vehicles.csv has only {len(raw)} rows, which is below the "
            f"{MIN_EPA_ROWS} row sanity threshold."
        )

    normalized = normalize_epa_vehicles(raw)
    if normalized.empty:
        raise RefreshError("Normalization produced zero rows.")

    write_csv_atomically(normalized, EPA_CATALOG_PATH)
    payload = {
        "source_name": "U.S. DOE / EPA FuelEconomy.gov vehicle dataset",
        "source_url": EPA_DOWNLOAD_URL,
        "field_documentation_url": EPA_DOCS_URL,
        "retrieved_at": download.retrieved_at,
        "http_last_modified": download.last_modified,
        "http_etag": download.etag,
        "download_sha256": download.checksum,
        "source_rows": int(len(raw)),
        "normalized_rows": int(len(normalized)),
        "normalized_file": str(EPA_CATALOG_PATH.relative_to(DATA_DIR.parent)),
        "powertrain_counts": {
            str(key): int(value)
            for key, value in normalized["powertrain"].value_counts().items()
        },
        "license_note": (
            "FuelEconomy.gov content may be freely distributed for noncommercial "
            "scientific and educational purposes. Review the current source terms "
            "before any commercial redistribution."
        ),
    }
    record_provenance("epa_vehicles", payload)
    return payload


def refresh_zip_to_state(timeout: int = 180) -> dict[str, object]:
    """Build the ZIP-to-state table from the Census ZCTA-county relationship file."""
    download = fetch(CENSUS_ZCTA_URL, timeout=timeout)
    frame = pd.read_csv(
        io.BytesIO(download.content),
        sep="|",
        dtype="string",
        usecols=["GEOID_ZCTA5_20", "GEOID_COUNTY_20", "AREALAND_PART"],
    )
    frame = frame.dropna(subset=["GEOID_ZCTA5_20", "GEOID_COUNTY_20"])
    frame["zip_code"] = frame["GEOID_ZCTA5_20"].str.strip().str.zfill(5)
    frame["state"] = frame["GEOID_COUNTY_20"].str.strip().str[:2].map(
        STATE_FIPS_TO_USPS
    )
    frame["area_land"] = pd.to_numeric(frame["AREALAND_PART"], errors="coerce").fillna(0)
    frame = frame.dropna(subset=["state"])
    if len(frame) < MIN_ZIP_ROWS:
        raise RefreshError(
            f"The ZCTA relationship file produced only {len(frame)} usable rows."
        )

    by_state = (
        frame.groupby(["zip_code", "state"], as_index=False)["area_land"].sum()
        .sort_values(["zip_code", "area_land"], ascending=[True, False])
    )
    grouped = by_state.groupby("zip_code")
    table = pd.DataFrame(
        {
            "zip_code": grouped["state"].first().index,
            "primary_state": grouped["state"].first().to_numpy(),
            "states": grouped["state"]
            .apply(lambda values: "|".join(sorted(set(values))))
            .to_numpy(),
            "state_count": grouped["state"].nunique().to_numpy(),
        }
    ).sort_values("zip_code").reset_index(drop=True)

    write_csv_atomically(table, ZIP_TO_STATE_PATH)
    payload = {
        "source_name": "U.S. Census Bureau 2020 ZCTA-to-county relationship file",
        "source_url": CENSUS_ZCTA_URL,
        "retrieved_at": download.retrieved_at,
        "http_last_modified": download.last_modified,
        "http_etag": download.etag,
        "download_sha256": download.checksum,
        "normalized_rows": int(len(table)),
        "multi_state_zips": int((table["state_count"] > 1).sum()),
        "normalized_file": str(ZIP_TO_STATE_PATH.relative_to(DATA_DIR.parent)),
        "limitation_note": (
            "ZIP Code Tabulation Areas approximate USPS ZIP codes. ZIP codes with "
            "no residential delivery area have no ZCTA and cannot be mapped."
        ),
    }
    record_provenance("zip_to_state", payload)
    return payload


class _TableParser(HTMLParser):
    """Collect the cells of every HTML table on a page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[str]]] = []
        self._table: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self._table = []
        elif tag == "tr" and self._table is not None:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append("".join(self._cell).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None and self._table is not None:
            if self._row:
                self._table.append(self._row)
            self._row = None
        elif tag == "table" and self._table is not None:
            self.tables.append(self._table)
            self._table = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def _parse_rate(text: str) -> float:
    cleaned = text.replace("%", "").replace(",", "").strip()
    if not cleaned:
        raise ValueError("empty rate")
    return float(cleaned) / 100.0


def refresh_sales_tax(timeout: int = 120) -> dict[str, object]:
    """Parse the Tax Foundation state and local sales-tax table."""
    download = fetch(TAX_FOUNDATION_URL, timeout=timeout)
    html = download.content.decode("utf-8", errors="replace")

    effective_date = "Unknown"
    match = re.search(r"Sales Tax Rates as of ([A-Z][a-z]+ \d{1,2}, \d{4})", html)
    if match:
        effective_date = match.group(1)

    parser = _TableParser()
    parser.feed(html)

    records: list[dict[str, object]] = []
    for table in parser.tables:
        candidate: list[dict[str, object]] = []
        for row in table:
            if len(row) < 6:
                continue
            name = re.sub(r"\s*\([a-z]\)\s*$", "", row[0]).strip()
            state = STATE_NAME_TO_USPS.get(name)
            if state is None:
                continue
            try:
                candidate.append(
                    {
                        "state": state,
                        "state_name": name,
                        "state_rate": _parse_rate(row[1]),
                        "avg_local_rate": _parse_rate(row[3]),
                        "max_local_rate": _parse_rate(row[4]),
                        "combined_rate": _parse_rate(row[5]),
                    }
                )
            except ValueError:
                continue
        if len(candidate) > len(records):
            records = candidate

    if len(records) < MIN_TAX_ROWS:
        raise RefreshError(
            f"Only {len(records)} state rows were parsed from {TAX_FOUNDATION_URL}. "
            "The page layout may have changed; the existing local file was kept."
        )

    table = pd.DataFrame(records).drop_duplicates(subset=["state"])
    table["source_name"] = "Tax Foundation, State and Local Sales Tax Rates"
    table["source_url"] = TAX_FOUNDATION_URL
    table["effective_date"] = effective_date
    table["rate_type"] = "General sales tax"
    table["coverage_note"] = (
        "State statutory rate plus a population-weighted average local rate. "
        "Not a ZIP-exact rate and not a motor-vehicle-specific tax determination."
    )
    table = table.sort_values("state").reset_index(drop=True)

    write_csv_atomically(table, SALES_TAX_PATH)
    payload = {
        "source_name": "Tax Foundation, State and Local Sales Tax Rates",
        "source_url": TAX_FOUNDATION_URL,
        "effective_date": effective_date,
        "retrieved_at": download.retrieved_at,
        "http_last_modified": download.last_modified,
        "http_etag": download.etag,
        "download_sha256": download.checksum,
        "normalized_rows": int(len(table)),
        "normalized_file": str(SALES_TAX_PATH.relative_to(DATA_DIR.parent)),
        "limitation_note": (
            "General sales-tax rates. Motor-vehicle rates, trade-in credits, caps, "
            "exemptions, and local ZIP-level rates are not encoded and must be "
            "reviewed against the buyer's jurisdiction."
        ),
    }
    record_provenance("sales_tax_rates", payload)
    return payload


REFRESHERS = {
    "epa": refresh_epa,
    "zip": refresh_zip_to_state,
    "tax": refresh_sales_tax,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=("all", *REFRESHERS),
        default="all",
        help="Which source to refresh (default: all).",
    )
    parser.add_argument(
        "--timeout", type=int, default=180, help="HTTP timeout in seconds."
    )
    args = parser.parse_args(argv)

    selected = list(REFRESHERS) if args.source == "all" else [args.source]
    failures = 0
    for name in selected:
        print(f"Refreshing {name}...", flush=True)
        try:
            payload = REFRESHERS[name](timeout=args.timeout)
        except RefreshError as error:
            failures += 1
            print(f"  FAILED: {error}", file=sys.stderr)
            continue
        print(
            f"  wrote {payload['normalized_rows']} rows to "
            f"{payload['normalized_file']}"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
