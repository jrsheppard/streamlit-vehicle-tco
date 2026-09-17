"""Cached access to the local data files.

Every loader takes a file fingerprint argument so the cache invalidates when the
file on disk changes. These are static local files, so no TTL is used.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from tco import assumptions as assumptions_module
from tco import epa_catalog, registry, salestax, taxonomy
from tco.paths import (
    BODY_SEGMENTS_PATH,
    BRAND_GROUPS_PATH,
    COST_ASSUMPTIONS_PATH,
    EPA_CATALOG_PATH,
    FIELD_PROVENANCE_PATH,
    PROVENANCE_PATH,
    SALES_TAX_PATH,
    WA_MODELS_PATH,
    ZIP_TO_STATE_PATH,
    file_fingerprint,
)


@st.cache_data(show_spinner="Loading the EPA vehicle catalog…")
def load_epa_catalog(fingerprint: tuple[str, int, int]) -> pd.DataFrame:
    return epa_catalog.read_vehicle_catalog(EPA_CATALOG_PATH)


@st.cache_data
def load_seed_cost_assumptions(fingerprint: tuple[str, int, int]) -> pd.DataFrame:
    return assumptions_module.read_cost_assumptions(COST_ASSUMPTIONS_PATH)


@st.cache_data
def load_zip_table(fingerprint: tuple[str, int, int]) -> pd.DataFrame:
    return salestax.read_zip_to_state(ZIP_TO_STATE_PATH)


@st.cache_data
def load_tax_rates(fingerprint: tuple[str, int, int]) -> pd.DataFrame:
    return salestax.read_sales_tax_rates(SALES_TAX_PATH)


@st.cache_data
def load_body_segments(fingerprint: tuple[str, int, int]) -> pd.DataFrame:
    return taxonomy.read_body_segments(BODY_SEGMENTS_PATH)


@st.cache_data
def load_brand_groups(fingerprint: tuple[str, int, int]) -> pd.DataFrame:
    return taxonomy.read_brand_groups(BRAND_GROUPS_PATH)


@st.cache_data
def load_field_provenance_overrides(
    fingerprint: tuple[str, int, int],
) -> pd.DataFrame:
    return assumptions_module.read_field_provenance(FIELD_PROVENANCE_PATH)


@st.cache_data
def load_source_provenance(fingerprint: tuple[str, int, int]) -> dict[str, dict]:
    if not PROVENANCE_PATH.exists():
        return {}
    try:
        return json.loads(PROVENANCE_PATH.read_text())
    except json.JSONDecodeError:
        return {}


@st.cache_data
def load_registered_models(
    fingerprint: tuple[str, int, int],
) -> registry.ModelContext:
    return registry.summarize_models(registry.read_registered_models(WA_MODELS_PATH))


def fingerprints() -> dict[str, tuple[str, int, int]]:
    """Current fingerprints for every data file the app reads."""
    return {
        "epa": file_fingerprint(EPA_CATALOG_PATH),
        "cost_assumptions": file_fingerprint(COST_ASSUMPTIONS_PATH),
        "zip": file_fingerprint(ZIP_TO_STATE_PATH),
        "tax": file_fingerprint(SALES_TAX_PATH),
        "body_segments": file_fingerprint(BODY_SEGMENTS_PATH),
        "brand_groups": file_fingerprint(BRAND_GROUPS_PATH),
        "field_provenance": file_fingerprint(FIELD_PROVENANCE_PATH),
        "provenance": file_fingerprint(PROVENANCE_PATH),
        "wa_models": file_fingerprint(WA_MODELS_PATH),
    }
