import click
import sys
from typing import Optional, List
from pathlib import Path

from .drift_detector import DriftDetector
from .report_generator import ReportGenerator
from .metrics import RiskLevel


@click.group()
@click.version_option(version="0.1.0")
def main():
    """Feature Drift Monitoring CLI

    Monitor feature drift between training samples and production samples.

    \b
    Supported file formats:
    - CSV (.csv)
    - Parquet (.parquet)
    - Excel (.xlsx, .xls)

    \b
    Drift Metrics:
    - PSI (Population Stability Index): For both numerical and categorical
    - KS (Kolmogorov-Smirnov) test: For numerical features
    - Chi-Square test: For categorical features

    \b
    Risk Levels:
    - no_drift: No significant drift detected
    - slight_drift: Minor drift detected
    - moderate_drift: Moderate drift detected, review recommended
    - severe_drift: Severe drift detected, action required
    """
    pass


@main.command()
@click.argument("expected_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("actual_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--columns",
    "-c",
    multiple=True,
    help="Specific columns to monitor (can be specified multiple times)",
)
@click.option(
    "--n-bins",
    "-b",
    default=10,
    type=int,
    help="Number of bins for PSI calculation (default: 10)",
)
@click.option(
    "--bucket-type",
    "-t",
    default="equal_width",
    type=click.Choice(["equal_width", "equal_freq"]),
    help="Bucket type for numerical features (default: equal_width)",
)
@click.option(
    "--categorical-threshold",
    default=10,
    type=int,
    help="Threshold for auto-detecting categorical numerical features (default: 10)",
)
@click.option(
    "--disable-psi",
    is_flag=True,
    help="Disable PSI metric calculation",
)
@click.option(
    "--disable-ks",
    is_flag=True,
    help="Disable KS test calculation (for numerical features)",
)
@click.option(
    "--disable-chisq",
    is_flag=True,
    help="Disable Chi-Square test calculation (for categorical features)",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    default="console",
    type=click.Choice(["console", "json", "csv", "txt"]),
    help="Output format (default: console)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False),
    help="Output file path (if not specified, prints to stdout)",
)
@click.option(
    "--exit-on-severe",
    is_flag=True,
    help="Exit with code 1 if severe drift detected",
)
@click.option(
    "--exit-on-high",
    is_flag=True,
    help="Exit with code 1 if moderate or severe drift detected",
)
def detect(
    expected_file: str,
    actual_file: str,
    columns: Optional[List[str]],
    n_bins: int,
    bucket_type: str,
    categorical_threshold: int,
    disable_psi: bool,
    disable_ks: bool,
    disable_chisq: bool,
    output_format: str,
    output: Optional[str],
    exit_on_severe: bool,
    exit_on_high: bool,
):
    """Detect feature drift between two datasets.

    EXPECTED_FILE: Path to the reference/training dataset.
    ACTUAL_FILE: Path to the current/production dataset to compare.

    \b
    Examples:
      # Basic usage
      drift-monitor detect train.csv production.csv

      # Monitor specific columns
      drift-monitor detect train.csv prod.csv -c age -c income

      # Save JSON report to file
      drift-monitor detect train.csv prod.csv -f json -o report.json

      # Exit with error code if severe drift detected
      drift-monitor detect train.csv prod.csv --exit-on-severe

      # Use equal-frequency binning
      drift-monitor detect train.csv prod.csv -t equal_freq -b 20
    """
    try:
        detector = DriftDetector(
            n_bins=n_bins,
            bucket_type=bucket_type,
            categorical_threshold=categorical_threshold,
            enable_psi=not disable_psi,
            enable_ks=not disable_ks,
            enable_chisq=not disable_chisq,
        )

        columns_list = list(columns) if columns else None

        report = detector.detect_from_files(
            expected_path=expected_file,
            actual_path=actual_file,
            columns=columns_list,
        )

        generator = ReportGenerator()

        if output_format == "json":
            content = generator.generate_json_report(report)
        elif output_format == "csv":
            content = generator.generate_csv_report(report)
        elif output_format == "txt":
            content = generator.generate_console_report(report)
        else:
            content = generator.generate_console_report(report)

        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            generator.save_report(report, output, output_format)
            click.echo(f"Report saved to: {output}")
        else:
            click.echo(content)

        has_severe = any(
            col.overall_risk == RiskLevel.SEVERE_DRIFT for col in report.column_reports
        )
        has_moderate = any(
            col.overall_risk in (RiskLevel.MODERATE_DRIFT, RiskLevel.SEVERE_DRIFT)
            for col in report.column_reports
        )

        if exit_on_severe and has_severe:
            click.echo("Exiting: Severe drift detected", err=True)
            sys.exit(1)

        if exit_on_high and has_moderate:
            click.echo("Exiting: High risk drift detected", err=True)
            sys.exit(1)

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@main.command()
@click.argument("report_file", type=click.Path(exists=True, dir_okay=False))
def inspect(report_file: str):
    """Inspect a saved drift report.

    \b
    This command reads a previously saved JSON report file and displays
    a summary in a human-readable format.

    \b
    Example:
      drift-monitor inspect report.json
    """
    import json

    try:
        with open(report_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        click.echo("=" * 80)
        click.echo("                    DRIFT REPORT INSPECTION")
        click.echo("=" * 80)
        click.echo()

        click.echo("[Expected Dataset]")
        expected_meta = data.get("expected_metadata", {})
        click.echo(f"  Rows:    {expected_meta.get('rows', 'N/A')}")
        click.echo(f"  Columns: {len(expected_meta.get('columns', []))}")
        click.echo()

        click.echo("[Actual Dataset]")
        actual_meta = data.get("actual_metadata", {})
        click.echo(f"  Rows:    {actual_meta.get('rows', 'N/A')}")
        click.echo(f"  Columns: {len(actual_meta.get('columns', []))}")
        click.echo()

        summary = data.get("summary", {})
        risk_dist = summary.get("risk_distribution", {})
        click.echo("[Risk Distribution]")
        click.echo(f"  No Drift:       {risk_dist.get('no_drift', 0)}")
        click.echo(f"  Slight Drift:   {risk_dist.get('slight_drift', 0)}")
        click.echo(f"  Moderate Drift: {risk_dist.get('moderate_drift', 0)}")
        click.echo(f"  Severe Drift:   {risk_dist.get('severe_drift', 0)}")
        click.echo()

        high_risk = summary.get("high_risk_columns", [])
        if high_risk:
            click.echo("[High Risk Columns]")
            for col in high_risk:
                click.echo(f"  - {col}")
            click.echo()

        columns = data.get("columns", [])
        if columns:
            click.echo("[All Columns]")
            for col in columns:
                risk = col.get("overall_risk", "unknown")
                col_type = col.get("column_type", "unknown")
                metrics = ", ".join(
                    f"{m['metric_name']}={m['metric_value']:.4f}"
                    for m in col.get("metrics", [])
                )
                click.echo(f"  {col['column_name']} ({col_type}) [{risk}]: {metrics}")

        click.echo()
        click.echo("=" * 80)

    except Exception as e:
        click.echo(f"Error reading report: {str(e)}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
