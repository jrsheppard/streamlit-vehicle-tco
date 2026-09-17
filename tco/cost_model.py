"""Statistical estimation of the editable cost layer from published studies.

Two models live here and they are deliberately different in resolution.

:class:`PriceModel` is a hedonic ridge regression on curated MSRP anchors. It is
allowed to resolve an individual model because the anchors are themselves
per-model observations.

:class:`OperatingCostModel` estimates depreciation, maintenance, insurance and
registration from AAA's published category tables. It is a function of body
segment, powertrain and vehicle price only. It never sees the model name, so two
different models that land in the same cell always receive identical operating
costs by construction. That is a requirement, not an implementation detail: the
underlying study publishes category averages, so per-model operating figures
would be fabricated precision.

Neither class touches Streamlit, the network, or the filesystem.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 20260917

PRICE_CATEGORICAL: tuple[str, ...] = (
    "body_segment",
    "powertrain",
    "brand_group",
    "brand_tier",
    "drive",
    "transmission",
    "segment_powertrain",
)
PRICE_NUMERIC: tuple[str, ...] = (
    "model_year",
    "epa_comb_mpg",
    "epa_kwh_per_100mi",
    "epa_electric_range_mi",
)
PRICE_FEATURES: tuple[str, ...] = PRICE_CATEGORICAL + PRICE_NUMERIC

#: Built by the model rather than supplied by the caller.
PRICE_DERIVED: tuple[str, ...] = ("segment_powertrain",)
PRICE_INPUTS: tuple[str, ...] = tuple(
    column for column in PRICE_FEATURES if column not in PRICE_DERIVED
)

#: Widest relative half-width we are willing to publish a price for. Beyond this
#: the interval spans most of the segment and the estimate stops being useful.
MAX_RELATIVE_HALF_WIDTH = 0.45
#: Pseudo-observations that inflate the interval when a cell is thinly anchored.
#: At 8.0 a cell with no anchors always exceeds the publication limit, which is
#: what keeps unanchored segment/brand combinations from being given a price.
SUPPORT_PRIOR = 8.0
#: Predictions this far outside the anchor range are treated as extrapolation.
EXTRAPOLATION_LOW = 0.40
EXTRAPOLATION_HIGH = 2.50
#: Residuals needed before a brand tier gets its own interval width.
MIN_TIER_RESIDUALS = 40
GLOBAL_TIER = "__global__"

ALPHAS = np.logspace(-3, 3, 25)

OPERATING_TARGETS: tuple[str, ...] = (
    "annual_depreciation_rate",
    "annual_maintenance_usd",
    "annual_insurance_usd",
    "annual_registration_fees_usd",
)

DEPRECIATION_BOUNDS = (0.03, 0.35)
REGISTRATION_FLOOR_USD = 60.0

BEV, PHEV, HEV, GASOLINE = "BEV", "PHEV", "HEV", "Gasoline"


# --------------------------------------------------------------------------- #
# Loan arithmetic, used to recover the price basis behind AAA's finance line
# --------------------------------------------------------------------------- #
def loan_total_interest(
    price: float, apr: float, down_fraction: float, term_months: int
) -> float:
    """Total interest paid on an amortizing loan for ``price``."""
    principal = price * (1.0 - down_fraction)
    if principal <= 0 or term_months <= 0:
        return 0.0
    if apr <= 0:
        return 0.0
    monthly = apr / 12.0
    payment = principal * monthly / (1.0 - (1.0 + monthly) ** -term_months)
    return payment * term_months - principal


def _bisect(func, target: float, lo: float, hi: float, iterations: int = 200) -> float:
    for _ in range(iterations):
        mid = (lo + hi) / 2.0
        if func(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def solve_apr(
    price: float, interest_total: float, down_fraction: float, term_months: int
) -> float:
    """The APR that reproduces ``interest_total`` for a known price."""
    return _bisect(
        lambda apr: loan_total_interest(price, apr, down_fraction, term_months),
        interest_total,
        1e-6,
        0.50,
    )


def solve_price(
    interest_total: float, apr: float, down_fraction: float, term_months: int
) -> float:
    """The vehicle price implied by a published finance charge."""
    return _bisect(
        lambda price: loan_total_interest(price, apr, down_fraction, term_months),
        interest_total,
        500.0,
        2_000_000.0,
    )


# --------------------------------------------------------------------------- #
# Purchase price
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PriceDiagnostics:
    """Out-of-sample quality of the hedonic price fit."""

    n_anchors: int
    n_folds: int
    n_repeats: int
    cv_mape: float
    cv_median_ape: float
    cv_log_rmse: float
    interval_coverage: float
    ridge_alpha: float
    log_residual_low: float
    log_residual_high: float
    anchor_price_min: float
    anchor_price_max: float
    tier_intervals: dict[str, tuple[float, float]]

    def as_dict(self) -> dict[str, object]:
        return {
            "anchors": self.n_anchors,
            "cv_folds": self.n_folds,
            "cv_repeats": self.n_repeats,
            "cv_mape": round(self.cv_mape, 5),
            "cv_median_absolute_percentage_error": round(self.cv_median_ape, 5),
            "cv_log_rmse": round(self.cv_log_rmse, 5),
            "cv_interval_coverage": round(self.interval_coverage, 5),
            "ridge_alpha": round(self.ridge_alpha, 5),
            "log_residual_low": round(self.log_residual_low, 5),
            "log_residual_high": round(self.log_residual_high, 5),
            "anchor_price_min_usd": round(self.anchor_price_min, 2),
            "anchor_price_max_usd": round(self.anchor_price_max, 2),
            "interval_by_brand_tier": {
                tier: [round(low, 5), round(high, 5)]
                for tier, (low, high) in sorted(self.tier_intervals.items())
            },
        }


class PriceModel:
    """Hedonic log-price ridge regression fitted to curated MSRP anchors.

    The interval is an empirical out-of-sample residual interval, widened where
    a vehicle's segment/brand cell carries few anchors. Rows whose interval is
    wider than ``max_relative_half_width`` get no price at all, so an unknown
    price stays unknown instead of becoming an invented number.
    """

    def __init__(
        self,
        interval: float = 0.80,
        max_relative_half_width: float = MAX_RELATIVE_HALF_WIDTH,
        support_prior: float = SUPPORT_PRIOR,
        n_splits: int = 5,
        n_repeats: int = 4,
    ) -> None:
        self.interval = interval
        self.max_relative_half_width = max_relative_half_width
        self.support_prior = support_prior
        self.n_splits = n_splits
        self.n_repeats = n_repeats

    @staticmethod
    def _pipeline() -> Pipeline:
        categorical = OneHotEncoder(handle_unknown="ignore")
        numeric = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
            ]
        )
        return Pipeline(
            [
                (
                    "features",
                    ColumnTransformer(
                        [
                            ("categorical", categorical, list(PRICE_CATEGORICAL)),
                            ("numeric", numeric, list(PRICE_NUMERIC)),
                        ]
                    ),
                ),
                ("ridge", RidgeCV(alphas=ALPHAS)),
            ]
        )

    @staticmethod
    def _augment(frame: pd.DataFrame) -> pd.DataFrame:
        """Add the interaction the additive terms cannot express on their own."""
        out = frame.loc[:, list(PRICE_INPUTS)].copy()
        out["segment_powertrain"] = (
            out["body_segment"].astype(str) + " | " + out["powertrain"].astype(str)
        )
        return out.loc[:, list(PRICE_FEATURES)]

    def fit(self, anchors: pd.DataFrame) -> "PriceModel":
        missing = [c for c in PRICE_INPUTS if c not in anchors.columns]
        if missing:
            raise ValueError(f"Price anchors are missing feature columns: {missing}")
        frame = anchors.loc[anchors["msrp_usd"].notna()].reset_index(drop=True)
        if len(frame) < self.n_splits * 3:
            raise ValueError(
                f"Need at least {self.n_splits * 3} price anchors to fit and "
                f"cross-validate; got {len(frame)}."
            )

        features = self._augment(frame)
        target = np.log(frame["msrp_usd"].to_numpy(dtype=float))

        residuals: list[np.ndarray] = []
        tiers: list[np.ndarray] = []
        for repeat in range(self.n_repeats):
            folds = KFold(
                n_splits=self.n_splits, shuffle=True, random_state=RANDOM_STATE + repeat
            )
            for train_idx, test_idx in folds.split(features):
                fold = self._pipeline().fit(features.iloc[train_idx], target[train_idx])
                predicted = fold.predict(features.iloc[test_idx])
                residuals.append(target[test_idx] - predicted)
                tiers.append(frame["brand_tier"].to_numpy()[test_idx])
        residual = np.concatenate(residuals)
        residual_tier = np.concatenate(tiers)

        # A supercar is far harder to predict than a compact sedan. One global
        # interval would let that difficulty widen every mainstream car's
        # interval, so each brand tier with enough residuals gets its own.
        tail = (1.0 - self.interval) / 2.0
        self.tier_intervals_: dict[str, tuple[float, float]] = {
            GLOBAL_TIER: (
                float(np.quantile(residual, tail)),
                float(np.quantile(residual, 1.0 - tail)),
            )
        }
        for tier in np.unique(residual_tier):
            subset = residual[residual_tier == tier]
            if len(subset) >= MIN_TIER_RESIDUALS:
                self.tier_intervals_[str(tier)] = (
                    float(np.quantile(subset, tail)),
                    float(np.quantile(subset, 1.0 - tail)),
                )
        self.log_residual_low_, self.log_residual_high_ = self.tier_intervals_[
            GLOBAL_TIER
        ]

        self.pipeline_ = self._pipeline().fit(features, target)
        self.anchor_prices_ = frame["msrp_usd"].to_numpy(dtype=float)
        self.segment_support_ = (
            frame.groupby(["body_segment", "powertrain"]).size().to_dict()
        )
        self.brand_support_ = frame.groupby("brand_group").size().to_dict()

        ape = np.abs(np.expm1(residual))
        bounds = np.array(
            [self.tier_intervals_.get(str(t), self.tier_intervals_[GLOBAL_TIER]) for t in residual_tier]
        )
        covered = (residual >= bounds[:, 0]) & (residual <= bounds[:, 1])
        self.diagnostics_ = PriceDiagnostics(
            n_anchors=len(frame),
            n_folds=self.n_splits,
            n_repeats=self.n_repeats,
            cv_mape=float(ape.mean()),
            cv_median_ape=float(np.median(ape)),
            cv_log_rmse=float(np.sqrt((residual**2).mean())),
            interval_coverage=float(covered.mean()),
            ridge_alpha=float(self.pipeline_.named_steps["ridge"].alpha_),
            anchor_price_min=float(self.anchor_prices_.min()),
            anchor_price_max=float(self.anchor_prices_.max()),
            log_residual_low=self.log_residual_low_,
            log_residual_high=self.log_residual_high_,
            tier_intervals=dict(self.tier_intervals_),
        )
        return self

    def _support(self, frame: pd.DataFrame) -> np.ndarray:
        segment = [
            self.segment_support_.get((seg, pt), 0)
            for seg, pt in zip(frame["body_segment"], frame["powertrain"])
        ]
        brand = [self.brand_support_.get(bg, 0) for bg in frame["brand_group"]]
        return np.minimum(np.array(segment, dtype=float), np.array(brand, dtype=float))

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Predict price with an interval, blanking rows that are too uncertain."""
        features = self._augment(frame)
        point = np.exp(self.pipeline_.predict(features))

        bounds = np.array(
            [
                self.tier_intervals_.get(str(t), self.tier_intervals_[GLOBAL_TIER])
                for t in frame["brand_tier"]
            ]
        )
        support = self._support(frame)
        inflation = np.sqrt(1.0 + self.support_prior / (support + 1.0))
        low = point * np.exp(bounds[:, 0] * inflation)
        high = point * np.exp(bounds[:, 1] * inflation)
        half_width = (high - low) / (2.0 * point)

        lo_bound = self.anchor_prices_.min() * EXTRAPOLATION_LOW
        hi_bound = self.anchor_prices_.max() * EXTRAPOLATION_HIGH

        out = pd.DataFrame(
            {
                "purchase_price_usd": point,
                "price_low_usd": low,
                "price_high_usd": high,
                "price_relative_half_width": half_width,
                "price_anchor_support": support,
                "price_note": "",
            },
            index=frame.index,
        )

        too_wide = half_width > self.max_relative_half_width
        out.loc[too_wide, "price_note"] = [
            f"No price published: the {hw:.0%} prediction interval exceeds the "
            f"{self.max_relative_half_width:.0%} limit ({int(s)} anchors in this "
            "segment and brand cell)."
            for hw, s in zip(half_width[too_wide], support[too_wide])
        ]
        outside = (point < lo_bound) | (point > hi_bound)
        out.loc[outside & ~too_wide, "price_note"] = (
            "No price published: the estimate falls outside the curated anchor range."
        )
        blank = too_wide | outside
        out.loc[blank, ["purchase_price_usd", "price_low_usd", "price_high_usd"]] = (
            np.nan
        )
        return out


# --------------------------------------------------------------------------- #
# Operating costs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OperatingDiagnostics:
    """What the operating-cost fit learned, and from how many observations."""

    n_observations: int
    implied_apr: float
    national_sales_tax_rate: float
    reference_miles_per_year: int
    shrinkage: float
    elasticity: dict[str, float]
    intercept: dict[str, float]
    r_squared: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        return {
            "observations": self.n_observations,
            "implied_study_apr": round(self.implied_apr, 5),
            "national_sales_tax_rate": round(self.national_sales_tax_rate, 5),
            "reference_miles_per_year": self.reference_miles_per_year,
            "shrinkage_prior": self.shrinkage,
            "price_elasticity": {k: round(v, 5) for k, v in self.elasticity.items()},
            "log_intercept": {k: round(v, 5) for k, v in self.intercept.items()},
            "r_squared": {k: round(v, 5) for k, v in self.r_squared.items()},
        }


class OperatingCostModel:
    """Depreciation, maintenance, insurance and registration by cell and price.

    Each target is modelled as a global price elasticity fitted in log space
    across AAA's published cells, times a multiplier for the vehicle's
    ``(aaa_category, powertrain)`` cell that is shrunk toward the powertrain
    marginal. Inputs are the body segment's AAA category, the powertrain, and
    the vehicle's price. Nothing model-specific enters.
    """

    def __init__(self, shrinkage: float = 0.5) -> None:
        self.shrinkage = shrinkage

    def fit(
        self,
        observations: pd.DataFrame,
        study: dict[str, float],
        national_sales_tax_rate: float,
    ) -> "OperatingCostModel":
        down = float(study["loan_down_payment_fraction"])
        term = int(round(float(study["loan_term_years"]) * 12))
        years = float(study["ownership_years"])
        self.reference_miles_ = int(float(study["miles_per_year"]))
        self.national_sales_tax_rate_ = national_sales_tax_rate

        # The study never publishes a price per category, but it does publish one
        # finance charge per category on fixed loan terms. Calibrating the rate
        # against the one category whose price is published lets every other
        # category's price basis be recovered by inverting the same loan.
        self.implied_apr_ = solve_apr(
            float(study["average_sales_weighted_msrp_usd"]),
            float(study["weighted_average_finance_usd_per_year"]) * years,
            down,
            term,
        )

        frame = observations.reset_index(drop=True).copy()
        frame["price_basis_usd"] = [
            solve_price(row.finance_usd_per_year * years, self.implied_apr_, down, term)
            for row in frame.itertuples()
        ]

        retained = np.clip(
            1.0
            - (frame["depreciation_usd_per_year"] * years) / frame["price_basis_usd"],
            1e-4,
            1.0,
        )
        frame["annual_depreciation_rate"] = 1.0 - retained ** (1.0 / years)
        frame["annual_maintenance_usd"] = frame["maintenance_usd_per_year"]
        frame["annual_insurance_usd"] = frame["insurance_usd_per_year"]
        # AAA bundles purchase taxes into its fees line; the app charges sales tax
        # separately from the ZIP code, so remove it here or it is counted twice.
        frame["annual_registration_fees_usd"] = np.maximum(
            (
                frame["license_registration_taxes_usd_per_year"] * years
                - national_sales_tax_rate * frame["price_basis_usd"]
            )
            / years,
            REGISTRATION_FLOOR_USD,
        )

        log_price = np.log(frame["price_basis_usd"].to_numpy(dtype=float))
        design = np.column_stack([np.ones_like(log_price), log_price])

        self.elasticity_: dict[str, float] = {}
        self.intercept_: dict[str, float] = {}
        self.r_squared_: dict[str, float] = {}
        self.cell_multiplier_: dict[str, dict[tuple[str, str], float]] = {}
        self.powertrain_multiplier_: dict[str, dict[str, float]] = {}

        for target in OPERATING_TARGETS:
            observed = np.log(frame[target].to_numpy(dtype=float))
            coefficients, *_ = np.linalg.lstsq(design, observed, rcond=None)
            # None of these costs can fall as a vehicle gets more expensive, so a
            # negative slope is noise in 15 observations rather than a finding.
            if coefficients[1] < 0.0:
                coefficients = np.array([observed.mean(), 0.0])
            fitted = design @ coefficients
            total = ((observed - observed.mean()) ** 2).sum()
            self.intercept_[target] = float(coefficients[0])
            self.elasticity_[target] = float(coefficients[1])
            self.r_squared_[target] = (
                float(1.0 - ((observed - fitted) ** 2).sum() / total) if total else 0.0
            )

            ratio = np.exp(observed - fitted)
            by_powertrain: dict[str, float] = {}
            for powertrain, index in frame.groupby("powertrain").groups.items():
                by_powertrain[str(powertrain)] = float(
                    np.exp(np.log(ratio[list(index)]).mean())
                )
            self.powertrain_multiplier_[target] = by_powertrain

            cells: dict[tuple[str, str], float] = {}
            for position, row in enumerate(frame.itertuples()):
                parent = by_powertrain.get(row.powertrain, 1.0)
                # One published observation per cell, shrunk toward its parent.
                shrunk = np.exp(
                    (np.log(ratio[position]) + self.shrinkage * np.log(parent))
                    / (1.0 + self.shrinkage)
                )
                cells[(row.aaa_category, row.powertrain)] = float(shrunk)
            self.cell_multiplier_[target] = cells

        self.diagnostics_ = OperatingDiagnostics(
            n_observations=len(frame),
            implied_apr=self.implied_apr_,
            national_sales_tax_rate=national_sales_tax_rate,
            reference_miles_per_year=self.reference_miles_,
            shrinkage=self.shrinkage,
            elasticity=self.elasticity_,
            intercept=self.intercept_,
            r_squared=self.r_squared_,
        )
        self.training_frame_ = frame
        return self

    def _multiplier(self, target: str, category: str, powertrain: str) -> float:
        cells = self.cell_multiplier_[target]
        marginals = self.powertrain_multiplier_[target]
        if (category, powertrain) in cells:
            return cells[(category, powertrain)]
        if powertrain == PHEV:
            # The study publishes no plug-in hybrid cell, so sit it between the
            # hybrid and battery-electric cells it is mechanically between.
            blended = [
                self._multiplier(target, category, alternative)
                for alternative in (HEV, BEV)
            ]
            return float(np.exp(np.mean(np.log(blended))))
        return float(marginals.get(powertrain, 1.0))

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Estimate the four operating-cost fields for each row."""
        price = frame["purchase_price_usd"].to_numpy(dtype=float)
        log_price = np.log(np.where(np.isfinite(price) & (price > 0), price, np.nan))

        out = pd.DataFrame(index=frame.index)
        for target in OPERATING_TARGETS:
            multiplier = np.array(
                [
                    self._multiplier(target, category, powertrain)
                    for category, powertrain in zip(
                        frame["aaa_category"], frame["powertrain"]
                    )
                ],
                dtype=float,
            )
            base = np.exp(
                self.intercept_[target] + self.elasticity_[target] * log_price
            )
            out[target] = base * multiplier

        out["annual_depreciation_rate"] = out["annual_depreciation_rate"].clip(
            *DEPRECIATION_BOUNDS
        )
        out["annual_registration_fees_usd"] = out["annual_registration_fees_usd"].clip(
            lower=REGISTRATION_FLOOR_USD
        )
        return out
