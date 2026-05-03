import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum


class RiskLevel(Enum):
    NO_DRIFT = "no_drift"
    SLIGHT_DRIFT = "slight_drift"
    MODERATE_DRIFT = "moderate_drift"
    SEVERE_DRIFT = "severe_drift"


PSI_THRESHOLDS = {
    RiskLevel.NO_DRIFT: 0.1,
    RiskLevel.SLIGHT_DRIFT: 0.25,
    RiskLevel.MODERATE_DRIFT: 0.5,
}

KS_THRESHOLDS = {
    RiskLevel.NO_DRIFT: 0.2,
    RiskLevel.SLIGHT_DRIFT: 0.3,
    RiskLevel.MODERATE_DRIFT: 0.5,
}

CHISQ_PVALUE_THRESHOLDS = {
    RiskLevel.SEVERE_DRIFT: 0.001,
    RiskLevel.MODERATE_DRIFT: 0.01,
    RiskLevel.SLIGHT_DRIFT: 0.05,
}


@dataclass
class DriftResult:
    column_name: str
    column_type: str
    metric_name: str
    metric_value: float
    risk_level: RiskLevel
    additional_info: Optional[Dict[str, Any]] = None


class PSICalculator:
    def __init__(self, n_bins: int = 10, bucket_type: str = "equal_width"):
        self.n_bins = n_bins
        self.bucket_type = bucket_type

    def calculate_for_numerical(
        self, expected: np.ndarray, actual: np.ndarray, epsilon: float = 1e-10
    ) -> Tuple[float, List[Dict[str, Any]]]:
        expected = np.asarray(expected).flatten()
        actual = np.asarray(actual).flatten()

        expected_clean = expected[~np.isnan(expected)]
        actual_clean = actual[~np.isnan(actual)]

        if len(expected_clean) == 0 or len(actual_clean) == 0:
            raise ValueError("Both expected and actual arrays must contain non-null values")

        if self.bucket_type == "equal_width":
            bins = self._equal_width_bins(expected_clean)
        elif self.bucket_type == "equal_freq":
            bins = self._equal_freq_bins(expected_clean)
        else:
            raise ValueError(f"Unknown bucket type: {self.bucket_type}")

        expected_counts, _ = np.histogram(expected_clean, bins=bins)
        actual_counts, _ = np.histogram(actual_clean, bins=bins)

        expected_proportions = expected_counts / len(expected_clean)
        actual_proportions = actual_counts / len(actual_clean)

        expected_proportions = np.clip(expected_proportions, epsilon, 1 - epsilon)
        actual_proportions = np.clip(actual_proportions, epsilon, 1 - epsilon)

        psi_values = (actual_proportions - expected_proportions) * np.log(
            actual_proportions / expected_proportions
        )
        total_psi = np.sum(psi_values)

        bucket_details = []
        for i in range(len(bins) - 1):
            bucket_details.append({
                "bucket_lower": bins[i],
                "bucket_upper": bins[i + 1],
                "expected_count": int(expected_counts[i]),
                "actual_count": int(actual_counts[i]),
                "expected_pct": float(expected_proportions[i]),
                "actual_pct": float(actual_proportions[i]),
                "psi_contribution": float(psi_values[i]),
            })

        return total_psi, bucket_details

    def calculate_for_categorical(
        self, expected: pd.Series, actual: pd.Series, epsilon: float = 1e-10
    ) -> Tuple[float, List[Dict[str, Any]]]:
        expected_clean = expected.dropna()
        actual_clean = actual.dropna()

        if len(expected_clean) == 0 or len(actual_clean) == 0:
            raise ValueError("Both expected and actual series must contain non-null values")

        all_categories = np.union1d(expected_clean.unique(), actual_clean.unique())

        expected_counts = expected_clean.value_counts().reindex(all_categories, fill_value=0)
        actual_counts = actual_clean.value_counts().reindex(all_categories, fill_value=0)

        expected_proportions = expected_counts / len(expected_clean)
        actual_proportions = actual_counts / len(actual_clean)

        expected_proportions = expected_proportions.clip(lower=epsilon, upper=1 - epsilon)
        actual_proportions = actual_proportions.clip(lower=epsilon, upper=1 - epsilon)

        psi_values = (actual_proportions - expected_proportions) * np.log(
            actual_proportions / expected_proportions
        )
        total_psi = psi_values.sum()

        bucket_details = []
        for cat in all_categories:
            bucket_details.append({
                "category": str(cat),
                "expected_count": int(expected_counts[cat]),
                "actual_count": int(actual_counts[cat]),
                "expected_pct": float(expected_proportions[cat]),
                "actual_pct": float(actual_proportions[cat]),
                "psi_contribution": float(psi_values[cat]),
            })

        return total_psi, bucket_details

    def _equal_width_bins(self, data: np.ndarray) -> np.ndarray:
        min_val = np.min(data)
        max_val = np.max(data)
        if min_val == max_val:
            return np.array([min_val - 0.5, min_val + 0.5])
        bins = np.linspace(min_val, max_val, self.n_bins + 1)
        bins[0] = -np.inf
        bins[-1] = np.inf
        return bins

    def _equal_freq_bins(self, data: np.ndarray) -> np.ndarray:
        quantiles = np.linspace(0, 1, self.n_bins + 1)
        bins = np.quantile(data, quantiles)
        bins = np.unique(bins)
        if len(bins) < 2:
            return np.array([-np.inf, np.inf])
        bins[0] = -np.inf
        bins[-1] = np.inf
        return bins


class KSCalculator:
    def calculate(self, expected: np.ndarray, actual: np.ndarray) -> Tuple[float, float]:
        expected = np.asarray(expected).flatten()
        actual = np.asarray(actual).flatten()

        expected_clean = expected[~np.isnan(expected)]
        actual_clean = actual[~np.isnan(actual)]

        if len(expected_clean) == 0 or len(actual_clean) == 0:
            raise ValueError("Both expected and actual arrays must contain non-null values")

        statistic, p_value = stats.ks_2samp(expected_clean, actual_clean)
        return statistic, p_value


class ChiSquareCalculator:
    def calculate(
        self, expected: pd.Series, actual: pd.Series
    ) -> Tuple[float, float, List[Dict[str, Any]]]:
        expected_clean = expected.dropna()
        actual_clean = actual.dropna()

        if len(expected_clean) == 0 or len(actual_clean) == 0:
            raise ValueError("Both expected and actual series must contain non-null values")

        all_categories = np.union1d(expected_clean.unique(), actual_clean.unique())

        expected_counts = expected_clean.value_counts().reindex(all_categories, fill_value=0)
        actual_counts = actual_clean.value_counts().reindex(all_categories, fill_value=0)

        contingency_table = np.array([expected_counts.values, actual_counts.values])

        if np.all(contingency_table.sum(axis=1) > 0):
            try:
                chi2, p_value, dof, expected_freq = stats.chi2_contingency(
                    contingency_table, correction=False
                )
            except ValueError:
                chi2, p_value = 0.0, 1.0
                dof = 0
        else:
            chi2, p_value = 0.0, 1.0
            dof = 0

        category_details = []
        for i, cat in enumerate(all_categories):
            category_details.append({
                "category": str(cat),
                "expected_count": int(expected_counts.values[i]),
                "actual_count": int(actual_counts.values[i]),
            })

        return chi2, p_value, category_details


def classify_psi_risk(psi_value: float) -> RiskLevel:
    if psi_value < PSI_THRESHOLDS[RiskLevel.NO_DRIFT]:
        return RiskLevel.NO_DRIFT
    elif psi_value < PSI_THRESHOLDS[RiskLevel.SLIGHT_DRIFT]:
        return RiskLevel.SLIGHT_DRIFT
    elif psi_value < PSI_THRESHOLDS[RiskLevel.MODERATE_DRIFT]:
        return RiskLevel.MODERATE_DRIFT
    else:
        return RiskLevel.SEVERE_DRIFT


def classify_ks_risk(ks_statistic: float) -> RiskLevel:
    if ks_statistic < KS_THRESHOLDS[RiskLevel.NO_DRIFT]:
        return RiskLevel.NO_DRIFT
    elif ks_statistic < KS_THRESHOLDS[RiskLevel.SLIGHT_DRIFT]:
        return RiskLevel.SLIGHT_DRIFT
    elif ks_statistic < KS_THRESHOLDS[RiskLevel.MODERATE_DRIFT]:
        return RiskLevel.MODERATE_DRIFT
    else:
        return RiskLevel.SEVERE_DRIFT


def classify_chisq_risk(p_value: float) -> RiskLevel:
    if p_value < CHISQ_PVALUE_THRESHOLDS[RiskLevel.SEVERE_DRIFT]:
        return RiskLevel.SEVERE_DRIFT
    elif p_value < CHISQ_PVALUE_THRESHOLDS[RiskLevel.MODERATE_DRIFT]:
        return RiskLevel.MODERATE_DRIFT
    elif p_value < CHISQ_PVALUE_THRESHOLDS[RiskLevel.SLIGHT_DRIFT]:
        return RiskLevel.SLIGHT_DRIFT
    else:
        return RiskLevel.NO_DRIFT
