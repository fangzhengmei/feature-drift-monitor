import pytest
import json
import csv
from io import StringIO

from drift_monitor.drift_detector import DriftDetector
from drift_monitor.report_generator import ReportGenerator


class TestReportGenerator:
    def test_generate_json_report(self, sample_mixed_data):
        expected, actual = sample_mixed_data
        detector = DriftDetector()
        report = detector.detect(expected, actual)

        generator = ReportGenerator()
        json_content = generator.generate_json_report(report)

        parsed = json.loads(json_content)

        assert "report_version" in parsed
        assert "expected_metadata" in parsed
        assert "actual_metadata" in parsed
        assert "summary" in parsed
        assert "config" in parsed
        assert "columns" in parsed
        assert len(parsed["columns"]) == 4

        for col in parsed["columns"]:
            assert "column_name" in col
            assert "column_type" in col
            assert "overall_risk" in col
            assert "metrics" in col
            assert len(col["metrics"]) > 0

    def test_generate_console_report(self, sample_mixed_data):
        expected, actual = sample_mixed_data
        detector = DriftDetector()
        report = detector.detect(expected, actual)

        generator = ReportGenerator()
        console_content = generator.generate_console_report(report)

        assert "FEATURE DRIFT MONITORING REPORT" in console_content
        assert "[Overview]" in console_content
        assert "[Risk Distribution]" in console_content
        assert "[Detailed Column Report]" in console_content
        assert "END OF REPORT" in console_content

    def test_generate_csv_report(self, sample_mixed_data):
        expected, actual = sample_mixed_data
        detector = DriftDetector()
        report = detector.detect(expected, actual)

        generator = ReportGenerator()
        csv_content = generator.generate_csv_report(report)

        reader = csv.reader(StringIO(csv_content))
        rows = list(reader)

        assert len(rows) > 1
        header = rows[0]
        assert "column_name" in header
        assert "column_type" in header
        assert "overall_risk" in header
        assert "metric_name" in header
        assert "metric_value" in header
        assert "metric_risk" in header

    def test_save_report_json(self, temp_dir, sample_mixed_data):
        expected, actual = sample_mixed_data
        detector = DriftDetector()
        report = detector.detect(expected, actual)

        generator = ReportGenerator()
        output_path = temp_dir / "report.json"

        generator.save_report(report, str(output_path), "json")

        assert output_path.exists()
        with open(output_path, "r") as f:
            content = json.load(f)
        assert "report_version" in content

    def test_save_report_csv(self, temp_dir, sample_mixed_data):
        expected, actual = sample_mixed_data
        detector = DriftDetector()
        report = detector.detect(expected, actual)

        generator = ReportGenerator()
        output_path = temp_dir / "report.csv"

        generator.save_report(report, str(output_path), "csv")

        assert output_path.exists()

    def test_save_report_txt(self, temp_dir, sample_mixed_data):
        expected, actual = sample_mixed_data
        detector = DriftDetector()
        report = detector.detect(expected, actual)

        generator = ReportGenerator()
        output_path = temp_dir / "report.txt"

        generator.save_report(report, str(output_path), "txt")

        assert output_path.exists()

    def test_save_report_invalid_format(self, temp_dir, sample_mixed_data):
        expected, actual = sample_mixed_data
        detector = DriftDetector()
        report = detector.detect(expected, actual)

        generator = ReportGenerator()
        output_path = temp_dir / "report.invalid"

        with pytest.raises(ValueError, match="Unsupported format"):
            generator.save_report(report, str(output_path), "invalid")

    def test_console_report_includes_high_risk_section(self, sample_numerical_data_with_drift):
        expected, actual = sample_numerical_data_with_drift
        detector = DriftDetector()
        report = detector.detect(expected, actual)

        generator = ReportGenerator()
        content = generator.generate_console_report(report)

        assert "[High Risk Columns]" in content or len(report.summary["high_risk_columns"]) == 0

    def test_json_report_includes_additional_info(self, sample_numerical_data_no_drift):
        expected, actual = sample_numerical_data_no_drift
        detector = DriftDetector()
        report = detector.detect(expected, actual)

        generator = ReportGenerator()
        json_content = generator.generate_json_report(report)
        parsed = json.loads(json_content)

        for col in parsed["columns"]:
            for metric in col["metrics"]:
                if metric["metric_name"] == "PSI":
                    assert "additional_info" in metric
                    assert "bucket_details" in metric["additional_info"]
                elif metric["metric_name"] == "KS":
                    assert "additional_info" in metric
                    assert "p_value" in metric["additional_info"]

    def test_console_report_shows_statistics(self, sample_mixed_data):
        expected, actual = sample_mixed_data
        detector = DriftDetector()
        report = detector.detect(expected, actual)

        generator = ReportGenerator()
        content = generator.generate_console_report(report)

        assert "[Statistics]" in content
        assert "Expected Stats" in content
        assert "Actual Stats" in content

    def test_console_report_shows_categories(self, sample_categorical_data_no_drift):
        expected, actual = sample_categorical_data_no_drift
        detector = DriftDetector()
        report = detector.detect(expected, actual)

        generator = ReportGenerator()
        content = generator.generate_console_report(report)

        assert "Expected Top Categories" in content
        assert "Actual Top Categories" in content

    def test_csv_report_rows_match_metrics(self, sample_mixed_data):
        expected, actual = sample_mixed_data
        detector = DriftDetector()
        report = detector.detect(expected, actual)

        generator = ReportGenerator()
        csv_content = generator.generate_csv_report(report)

        reader = csv.DictReader(StringIO(csv_content))
        rows = list(reader)

        total_metrics = sum(len(c.drift_results) for c in report.column_reports)
        assert len(rows) == total_metrics
