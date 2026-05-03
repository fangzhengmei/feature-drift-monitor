import pytest
import json
import csv
from io import StringIO
from click.testing import CliRunner

from drift_monitor.cli import main


class TestCLI:
    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_help_command(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Feature Drift Monitoring CLI" in result.output

    def test_detect_help(self, runner):
        result = runner.invoke(main, ["detect", "--help"])
        assert result.exit_code == 0
        assert "Detect feature drift between two datasets" in result.output

    def test_detect_basic(self, runner, csv_files):
        expected_path, actual_path = csv_files
        result = runner.invoke(
            main,
            ["detect", str(expected_path), str(actual_path)],
        )
        assert result.exit_code == 0
        assert "FEATURE DRIFT MONITORING REPORT" in result.output

    def test_detect_json_output(self, runner, csv_files):
        expected_path, actual_path = csv_files
        result = runner.invoke(
            main,
            ["detect", str(expected_path), str(actual_path), "-f", "json"],
        )
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert "report_version" in parsed
        assert "columns" in parsed

    def test_detect_csv_output(self, runner, csv_files):
        expected_path, actual_path = csv_files
        result = runner.invoke(
            main,
            ["detect", str(expected_path), str(actual_path), "-f", "csv"],
        )
        assert result.exit_code == 0
        reader = csv.reader(StringIO(result.output))
        header = next(reader)
        assert "column_name" in header
        assert "metric_name" in header

    def test_detect_specific_columns(self, runner, csv_files):
        expected_path, actual_path = csv_files
        result = runner.invoke(
            main,
            ["detect", str(expected_path), str(actual_path), "-c", "age", "-c", "gender"],
        )
        assert result.exit_code == 0
        assert "age" in result.output
        assert "gender" in result.output

    def test_detect_save_to_file(self, runner, temp_dir, csv_files):
        expected_path, actual_path = csv_files
        output_path = temp_dir / "output.json"

        result = runner.invoke(
            main,
            [
                "detect",
                str(expected_path),
                str(actual_path),
                "-f",
                "json",
                "-o",
                str(output_path),
            ],
        )
        assert result.exit_code == 0
        assert output_path.exists()

        with open(output_path, "r") as f:
            content = json.load(f)
        assert "report_version" in content

    def test_detect_with_bucket_options(self, runner, csv_files):
        expected_path, actual_path = csv_files
        result = runner.invoke(
            main,
            [
                "detect",
                str(expected_path),
                str(actual_path),
                "-b",
                "5",
                "-t",
                "equal_freq",
            ],
        )
        assert result.exit_code == 0

    def test_detect_nonexistent_file(self, runner):
        result = runner.invoke(
            main,
            ["detect", "/nonexistent/expected.csv", "/nonexistent/actual.csv"],
        )
        assert result.exit_code != 0

    def test_inspect_command(self, runner, temp_dir, csv_files):
        expected_path, actual_path = csv_files
        output_path = temp_dir / "report.json"

        runner.invoke(
            main,
            [
                "detect",
                str(expected_path),
                str(actual_path),
                "-f",
                "json",
                "-o",
                str(output_path),
            ],
        )

        result = runner.invoke(main, ["inspect", str(output_path)])
        assert result.exit_code == 0
        assert "DRIFT REPORT INSPECTION" in result.output

    def test_inspect_nonexistent_file(self, runner):
        result = runner.invoke(main, ["inspect", "/nonexistent/report.json"])
        assert result.exit_code != 0

    def test_detect_disable_metrics(self, runner, csv_files):
        expected_path, actual_path = csv_files
        result = runner.invoke(
            main,
            [
                "detect",
                str(expected_path),
                str(actual_path),
                "--disable-psi",
                "--disable-ks",
                "--disable-chisq",
            ],
        )
        assert result.exit_code == 0

    def test_detect_with_categorical_threshold(self, runner, csv_files):
        expected_path, actual_path = csv_files
        result = runner.invoke(
            main,
            [
                "detect",
                str(expected_path),
                str(actual_path),
                "--categorical-threshold",
                "20",
            ],
        )
        assert result.exit_code == 0

    def test_version_command(self, runner):
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output
