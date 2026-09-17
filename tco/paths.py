"""Project-relative file locations.

Paths resolve from this file, never from the caller's working directory, so the
app and the tests behave the same regardless of where they are started.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
STATIC_DIR = PROJECT_ROOT / "static"

EPA_CATALOG_PATH = DATA_DIR / "epa_vehicles.csv"
COST_ASSUMPTIONS_PATH = DATA_DIR / "vehicle_cost_assumptions.csv"
FIELD_PROVENANCE_PATH = DATA_DIR / "field_provenance.csv"
BODY_SEGMENTS_PATH = DATA_DIR / "taxonomy_body_segments.csv"
BRAND_GROUPS_PATH = DATA_DIR / "taxonomy_brand_groups.csv"
SALES_TAX_PATH = DATA_DIR / "sales_tax_rates.csv"
ZIP_TO_STATE_PATH = DATA_DIR / "zip_to_state.csv"
PROVENANCE_PATH = DATA_DIR / "provenance.json"

#: Unique Washington-registered model configurations, derived by
#: ``scripts/reduce_registry.py`` from the raw snapshot.
WA_MODELS_PATH = STATIC_DIR / "wa_registered_models.csv"


def file_fingerprint(path: Path) -> tuple[str, int, int]:
    """Identity of a file's current contents, for cache invalidation.

    Returns a tuple of (path, size in bytes, modification time in nanoseconds).
    Missing files return a zeroed fingerprint so a later appearance of the file
    invalidates the cache.
    """
    try:
        stat = path.stat()
    except OSError:
        return (str(path), 0, 0)
    return (str(path), stat.st_size, stat.st_mtime_ns)
