import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from .data_loader import DataLoader
from .metrics import (
    PSICalculator,
    KSCalculator,
    ChiSquareCalculator,
    DriftResult,
    RiskLevel,
    classify_psi_risk,
    classify_ks_risk,
    classify_chisq_risk,
)


@dataclass
class ColumnReport:
    column_name: str
    column_type: str
    drift_results: List[DriftResult]
    overall_risk: RiskLevel
    summary: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftReport:
    expected_metadata: Dict[str, Any]
    actual_metadata: Dict[str, Any]
    column_reports: List[ColumnReport]
    summary: Dict[str, Any]
    config: Dict[str, Any]


class DriftDetector:
    def __init__(
        self,
        n_bins: int = 10,
        bucket_type: str = "equal_width",
        categorical_threshold: int = 10,
        enable_psi: bool = True,
        enable_ks: bool = True,
        enable_chisq: bool = True,
    ):
        self.n_bins = n_bins
        self.bucket_type = bucket_type
        self.categorical_threshold = categorical_threshold
        self.enable_psi = enable_psi
        self.enable_ks = enable_ks
        self.enable_chisq = enable_chisq

        self.psi_calculator = PSICalculator(n_bins=n_bins, bucket_type=bucket_type)
        self.ks_calculator = KSCalculator()
        self.chisq_calculator = ChiSquareCalculator()
        self.data_loader = DataLoader()

    def detect_from_files(
        self,
        expected_path: str,
        actual_path: str,
        columns: Optional[List[str]] = None,
        expected_kwargs: Optional[Dict[str, Any]] = None,
        actual_kwargs: Optional[Dict[str, Any]] = None,
    ) -> DriftReport:
        expected_kwargs = expected_kwargs or {}
        actual_kwargs = actual_kwargs or {}

        expected_loader = DataLoader()
        expected_df = expected_loader.load(expected_path, **expected_kwargs)
        expected_metadata = expected_loader.get_metadata()

        actual_loader = DataLoader()
        actual_df = actual_loader.load(actual_path, **actual_kwargs)
        actual_metadata = actual_loader.get_metadata()

        return self.detect(
            expected_df,
            actual_df,
            columns=columns,
            expected_metadata=expected_metadata,
            actual_metadata=actual_metadata,
        )

    def detect(
        self,
        expected_df: pd.DataFrame,
        actual_df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        expected_metadata: Optional[Dict[str, Any]] = None,
        actual_metadata: Optional[Dict[str, Any]] = None,
    ) -> DriftReport:
        expected_columns = set(expected_df.columns)
        actual_columns = set(actual_df.columns)

        common_columns = expected_columns & actual_columns
        if columns:
            columns = [c for c in columns if c in common_columns]
        else:
            columns = [c for c in expected_df.columns if c in common_columns]

        if not columns:
            raise ValueError("No common columns found between expected and actual datasets")

        expected_types = self.data_loader.detect_column_types(
            self._select_columns(expected_df, columns),
            categorical_threshold=self.categorical_threshold,
        )
        actual_types = self.data_loader.detect_column_types(
            self._select_columns(actual_df, columns),
            categorical_threshold=self.categorical_threshold,
        )

        column_reports: List[ColumnReport] = []

        for col in columns:
            col_type = expected_types.get(col, "numerical")
            if col_type != actual_types.get(col, col_type):
                col_type = self._resolve_type_conflict(col_type, actual_types.get(col))

            expected_series = expected_df[col]
            actual_series = actual_df[col]

            drift_results = self._calculate_drift_metrics(
                expected_series, actual_series, col, col_type
            )

            overall_risk = self._determine_overall_risk(drift_results)

            summary = self._generate_column_summary(
                expected_series, actual_series, drift_results, col_type
            )

            column_report = ColumnReport(
                column_name=col,
                column_type=col_type,
                drift_results=drift_results,
                overall_risk=overall_risk,
                summary=summary,
            )
            column_reports.append(column_report)

        overall_summary = self._generate_overall_summary(column_reports)

        config = {
            "n_bins": self.n_bins,
            "bucket_type": self.bucket_type,
            "categorical_threshold": self.categorical_threshold,
            "enable_psi": self.enable_psi,
            "enable_ks": self.enable_ks,
            "enable_chisq": self.enable_chisq,
        }

        return DriftReport(
            expected_metadata=expected_metadata or {"rows": len(expected_df), "columns": list(expected_df.columns)},
            actual_metadata=actual_metadata or {"rows": len(actual_df), "columns": list(actual_df.columns)},
            column_reports=column_reports,
            summary=overall_summary,
            config=config,
        )

    def _select_columns(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        return df[[c for c in columns if c in df.columns]]

    def _resolve_type_conflict(self, type1: str, type2: Optional[str]) -> str:
        if type2 is None:
            return type1
        if type1 == "other" or type2 == "other":
            return "other"
        if type1 != type2:
            return "categorical"
        return type1

    def _calculate_drift_metrics(
        self,
        expected: pd.Series,
        actual: pd.Series,
        column_name: str,
        column_type: str,
    ) -> List[DriftResult]:
        results: List[DriftResult] = []

        if column_type == "numerical":
            if self.enable_psi:
                psi_value, psi_details = self.psi_calculator.calculate_for_numerical(
                    expected.values, actual.values
                )
                results.append(
                    DriftResult(
                        column_name=column_name,
                        column_type=column_type,
                        metric_name="PSI",
                        metric_value=psi_value,
                        risk_level=classify_psi_risk(psi_value),
                        additional_info={"bucket_details": psi_details},
                    )
                )

            if self.enable_ks:
                try:
                    ks_stat, ks_pvalue = self.ks_calculator.calculate(
                        expected.values, actual.values
                    )
                    results.append(
                        DriftResult(
                            column_name=column_name,
                            column_type=column_type,
                            metric_name="KS",
                            metric_value=ks_stat,
                            risk_level=classify_ks_risk(ks_stat),
                            additional_info={"p_value": ks_pvalue},
                        )
                    )
                except Exception:
                    pass

        elif column_type == "categorical":
            if self.enable_psi:
                psi_value, psi_details = self.psi_calculator.calculate_for_categorical(
                    expected, actual
                )
                results.append(
                    DriftResult(
                        column_name=column_name,
                        column_type=column_type,
                        metric_name="PSI",
                        metric_value=psi_value,
                        risk_level=classify_psi_risk(psi_value),
                        additional_info={"category_details": psi_details},
                    )
                )

            if self.enable_chisq:
                try:
                    chi2_stat, chi2_pvalue, chi2_details = self.chisq_calculator.calculate(
                        expected, actual
                    )
                    results.append(
                        DriftResult(
                            column_name=column_name,
                            column_type=column_type,
                            metric_name="ChiSquare",
                            metric_value=chi2_stat,
                            risk_level=classify_chisq_risk(chi2_pvalue),
                            additional_info={"p_value": chi2_pvalue, "category_details": chi2_details},
                        )
                    )
                except Exception:
                    pass

        return results

    def _determine_overall_risk(self, results: List[DriftResult]) -> RiskLevel:
        if not results:
            return RiskLevel.NO_DRIFT

        risk_priority = {
            RiskLevel.NO_DRIFT: 0,
            RiskLevel.SLIGHT_DRIFT: 1,
            RiskLevel.MODERATE_DRIFT: 2,
            RiskLevel.SEVERE_DRIFT: 3,
        }

        max_risk = RiskLevel.NO_DRIFT
        for result in results:
            if risk_priority[result.risk_level] > risk_priority[max_risk]:
                max_risk = result.risk_level

        return max_risk

    def _generate_column_summary(
        self,
        expected: pd.Series,
        actual: pd.Series,
        drift_results: List[DriftResult],
        column_type: str,
    ) -> Dict[str, Any]:
        summary = {
            "expected_missing_pct": float(expected.isna().mean()),
            "actual_missing_pct": float(actual.isna().mean()),
            "metrics": {},
        }

        for result in drift_results:
            summary["metrics"][result.metric_name] = {
                "value": result.metric_value,
                "risk_level": result.risk_level.value,
            }

        expected_clean = expected.dropna()
        actual_clean = actual.dropna()

        if column_type == "numerical":
            if len(expected_clean) > 0:
                summary["expected_stats"] = {
                    "mean": float(expected_clean.mean()),
                    "std": float(expected_clean.std()) if len(expected_clean) > 1 else 0.0,
                    "min": float(expected_clean.min()),
                    "max": float(expected_clean.max()),
                    "median": float(expected_clean.median()),
                }
            if len(actual_clean) > 0:
                summary["actual_stats"] = {
                    "mean": float(actual_clean.mean()),
                    "std": float(actual_clean.std()) if len(actual_clean) > 1 else 0.0,
                    "min": float(actual_clean.min()),
                    "max": float(actual_clean.max()),
                    "median": float(actual_clean.median()),
                }
        elif column_type == "categorical":
            if len(expected_clean) > 0:
                summary["expected_top_categories"] = (
                    expected_clean.value_counts().head(5).to_dict()
                )
            if len(actual_clean) > 0:
                summary["actual_top_categories"] = (
                    actual_clean.value_counts().head(5).to_dict()
                )

        return summary

    def _generate_overall_summary(self, column_reports: List[ColumnReport]) -> Dict[str, Any]:
        risk_counts = {
            RiskLevel.NO_DRIFT.value: 0,
            RiskLevel.SLIGHT_DRIFT.value: 0,
            RiskLevel.MODERATE_DRIFT.value: 0,
            RiskLevel.SEVERE_DRIFT.value: 0,
        }

        numerical_columns = []
        categorical_columns = []
        high_risk_columns = []

        for report in column_reports:
            risk_counts[report.overall_risk.value] += 1

            if report.column_type == "numerical":
                numerical_columns.append(report.column_name)
            elif report.column_type == "categorical":
                categorical_columns.append(report.column_name)

            if report.overall_risk in (RiskLevel.MODERATE_DRIFT, RiskLevel.SEVERE_DRIFT):
                high_risk_columns.append(report.column_name)

        total_columns = len(column_reports)
        if total_columns > 0:
            high_risk_percentage = (
                (risk_counts[RiskLevel.MODERATE_DRIFT.value] + risk_counts[RiskLevel.SEVERE_DRIFT.value])
                / total_columns
                * 100
            )
        else:
            high_risk_percentage = 0.0

        return {
            "total_columns": total_columns,
            "numerical_columns": len(numerical_columns),
            "categorical_columns": len(categorical_columns),
            "risk_distribution": risk_counts,
            "high_risk_columns": high_risk_columns,
            "high_risk_percentage": high_risk_percentage,
        }
