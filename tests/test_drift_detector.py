import pytest
import pandas as pd
import numpy as np

from drift_monitor.drift_detector import DriftDetector, DriftReport, ColumnReport
from drift_monitor.metrics import RiskLevel


class TestDriftDetector:
    def test_detect_no_drift_numerical(self, sample_numerical_data_no_drift):
        expected, actual = sample_numerical_data_no_drift
        detector = DriftDetector(n_bins=10)

        report = detector.detect(expected, actual)

        assert isinstance(report, DriftReport)
        assert report.summary["total_columns"] == 3
        assert report.summary["numerical_columns"] == 3
        assert all(c.overall_risk == RiskLevel.NO_DRIFT for c in report.column_reports)

    def test_detect_with_drift_numerical(self, sample_numerical_data_with_drift):
        expected, actual = sample_numerical_data_with_drift
        detector = DriftDetector(n_bins=10)

        report = detector.detect(expected, actual)

        assert report.summary["total_columns"] == 2
        assert any(c.overall_risk != RiskLevel.NO_DRIFT for c in report.column_reports)

    def test_detect_no_drift_categorical(self, sample_categorical_data_no_drift):
        expected, actual = sample_categorical_data_no_drift
        detector = DriftDetector()

        report = detector.detect(expected, actual)

        assert report.summary["total_columns"] == 2
        assert report.summary["categorical_columns"] == 2

    def test_detect_with_drift_categorical(self, sample_categorical_data_with_drift):
        expected, actual = sample_categorical_data_with_drift
        detector = DriftDetector()

        report = detector.detect(expected, actual)

        assert any(c.overall_risk != RiskLevel.NO_DRIFT for c in report.column_reports)

    def test_detect_mixed_data(self, sample_mixed_data):
        expected, actual = sample_mixed_data
        detector = DriftDetector()

        report = detector.detect(expected, actual)

        assert report.summary["total_columns"] == 4
        assert report.summary["numerical_columns"] == 2
        assert report.summary["categorical_columns"] == 2

    def test_detect_with_specific_columns(self, sample_mixed_data):
        expected, actual = sample_mixed_data
        detector = DriftDetector()

        report = detector.detect(expected, actual, columns=["age", "gender"])

        assert report.summary["total_columns"] == 2
        col_names = [c.column_name for c in report.column_reports]
        assert "age" in col_names
        assert "gender" in col_names
        assert "income" not in col_names

    def test_detect_with_missing_values(self, data_with_missing_values):
        expected, actual = data_with_missing_values
        detector = DriftDetector()

        report = detector.detect(expected, actual)

        assert report.summary["total_columns"] == 3
        for col_report in report.column_reports:
            assert "expected_missing_pct" in col_report.summary
            assert "actual_missing_pct" in col_report.summary

        age_report = [c for c in report.column_reports if c.column_name == "age"][0]
        assert age_report.summary["actual_missing_pct"] > 0

    def test_detect_from_files(self, csv_files):
        expected_path, actual_path = csv_files
        detector = DriftDetector()

        report = detector.detect_from_files(str(expected_path), str(actual_path))

        assert report.summary["total_columns"] == 4
        assert "rows" in report.expected_metadata
        assert "rows" in report.actual_metadata

    def test_detect_no_common_columns_error(self):
        detector = DriftDetector()
        expected = pd.DataFrame({"a": [1, 2, 3]})
        actual = pd.DataFrame({"b": [4, 5, 6]})

        with pytest.raises(ValueError, match="No common columns"):
            detector.detect(expected, actual)

    def test_detect_disable_metrics(self, sample_numerical_data_no_drift):
        expected, actual = sample_numerical_data_no_drift

        detector_no_psi = DriftDetector(enable_psi=False)
        report = detector_no_psi.detect(expected, actual)
        for col_report in report.column_reports:
            metric_names = [m.metric_name for m in col_report.drift_results]
            assert "PSI" not in metric_names

        detector_no_ks = DriftDetector(enable_ks=False)
        report = detector_no_ks.detect(expected, actual)
        for col_report in report.column_reports:
            metric_names = [m.metric_name for m in col_report.drift_results]
            assert "KS" not in metric_names

    def test_detect_categorical_disable_chisq(self, sample_categorical_data_no_drift):
        expected, actual = sample_categorical_data_no_drift

        detector = DriftDetector(enable_chisq=False)
        report = detector.detect(expected, actual)

        for col_report in report.column_reports:
            metric_names = [m.metric_name for m in col_report.drift_results]
            assert "ChiSquare" not in metric_names

    def test_column_report_has_metrics(self, sample_mixed_data):
        expected, actual = sample_mixed_data
        detector = DriftDetector()

        report = detector.detect(expected, actual)

        for col_report in report.column_reports:
            assert len(col_report.drift_results) > 0
            assert col_report.overall_risk in [
                RiskLevel.NO_DRIFT,
                RiskLevel.SLIGHT_DRIFT,
                RiskLevel.MODERATE_DRIFT,
                RiskLevel.SEVERE_DRIFT,
            ]

    def test_report_summary_has_high_risk_columns(self, sample_numerical_data_with_drift):
        expected, actual = sample_numerical_data_with_drift
        detector = DriftDetector()

        report = detector.detect(expected, actual)

        assert "high_risk_columns" in report.summary
        assert "high_risk_percentage" in report.summary

    def test_report_includes_config(self, sample_numerical_data_no_drift):
        expected, actual = sample_numerical_data_no_drift
        detector = DriftDetector(n_bins=20, bucket_type="equal_freq", categorical_threshold=5)

        report = detector.detect(expected, actual)

        assert report.config["n_bins"] == 20
        assert report.config["bucket_type"] == "equal_freq"
        assert report.config["categorical_threshold"] == 5
