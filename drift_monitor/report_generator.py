import json
import csv
from typing import Dict, Any, List, Optional
from io import StringIO

from .drift_detector import DriftReport, ColumnReport
from .metrics import RiskLevel


class ReportGenerator:
    RISK_COLORS = {
        RiskLevel.NO_DRIFT: "GREEN",
        RiskLevel.SLIGHT_DRIFT: "YELLOW",
        RiskLevel.MODERATE_DRIFT: "ORANGE",
        RiskLevel.SEVERE_DRIFT: "RED",
    }

    RISK_ICONS = {
        RiskLevel.NO_DRIFT: "✓",
        RiskLevel.SLIGHT_DRIFT: "⚠",
        RiskLevel.MODERATE_DRIFT: "⚡",
        RiskLevel.SEVERE_DRIFT: "✗",
    }

    def generate_console_report(self, report: DriftReport) -> str:
        output = StringIO()

        output.write("\n" + "=" * 80 + "\n")
        output.write("                    FEATURE DRIFT MONITORING REPORT\n")
        output.write("=" * 80 + "\n\n")

        output.write("[Overview]\n")
        output.write(f"  Expected Dataset: {report.expected_metadata.get('rows', 'N/A')} rows\n")
        output.write(f"  Actual Dataset:   {report.actual_metadata.get('rows', 'N/A')} rows\n")
        output.write(f"  Total Columns:    {report.summary['total_columns']}\n")
        output.write(f"  Numerical:        {report.summary['numerical_columns']}\n")
        output.write(f"  Categorical:      {report.summary['categorical_columns']}\n\n")

        risk_dist = report.summary["risk_distribution"]
        output.write("[Risk Distribution]\n")
        output.write(f"  {self.RISK_ICONS[RiskLevel.NO_DRIFT]} No Drift:       {risk_dist[RiskLevel.NO_DRIFT.value]} columns\n")
        output.write(f"  {self.RISK_ICONS[RiskLevel.SLIGHT_DRIFT]} Slight Drift:   {risk_dist[RiskLevel.SLIGHT_DRIFT.value]} columns\n")
        output.write(f"  {self.RISK_ICONS[RiskLevel.MODERATE_DRIFT]} Moderate Drift: {risk_dist[RiskLevel.MODERATE_DRIFT.value]} columns\n")
        output.write(f"  {self.RISK_ICONS[RiskLevel.SEVERE_DRIFT]} Severe Drift:   {risk_dist[RiskLevel.SEVERE_DRIFT.value]} columns\n")
        output.write(f"\n  High Risk Percentage: {report.summary['high_risk_percentage']:.1f}%\n\n")

        if report.summary["high_risk_columns"]:
            output.write("[High Risk Columns]\n")
            for col in report.summary["high_risk_columns"]:
                col_report = self._find_column_report(report, col)
                if col_report:
                    icon = self.RISK_ICONS[col_report.overall_risk]
                    metrics_str = ", ".join(
                        f"{r.metric_name}={r.metric_value:.4f}"
                        for r in col_report.drift_results
                    )
                    output.write(f"  {icon} {col} ({col_report.column_type}): {metrics_str}\n")
            output.write("\n")

        output.write("[Detailed Column Report]\n")
        output.write("-" * 80 + "\n")

        sorted_reports = sorted(
            report.column_reports,
            key=lambda x: self._risk_priority(x.overall_risk),
            reverse=True,
        )

        for col_report in sorted_reports:
            icon = self.RISK_ICONS[col_report.overall_risk]
            risk_name = col_report.overall_risk.value.replace("_", " ").title()

            output.write(f"\n  {icon} Column: {col_report.column_name}\n")
            output.write(f"     Type: {col_report.column_type}\n")
            output.write(f"     Overall Risk: {risk_name}\n\n")

            output.write("     [Drift Metrics]\n")
            for result in col_report.drift_results:
                metric_icon = self.RISK_ICONS[result.risk_level]
                metric_risk = result.risk_level.value.replace("_", " ").title()
                output.write(f"       {metric_icon} {result.metric_name}: {result.metric_value:.6f} ({metric_risk})\n")

            output.write("\n     [Statistics]\n")
            summary = col_report.summary
            missing_expected = summary["expected_missing_pct"] * 100
            missing_actual = summary["actual_missing_pct"] * 100
            output.write(f"       Missing (Expected): {missing_expected:.2f}%\n")
            output.write(f"       Missing (Actual):   {missing_actual:.2f}%\n")

            if col_report.column_type == "numerical":
                if "expected_stats" in summary:
                    es = summary["expected_stats"]
                    output.write(f"\n       Expected Stats:\n")
                    output.write(f"         Mean:   {es['mean']:.4f}\n")
                    output.write(f"         Std:    {es['std']:.4f}\n")
                    output.write(f"         Min:    {es['min']:.4f}\n")
                    output.write(f"         Max:    {es['max']:.4f}\n")
                    output.write(f"         Median: {es['median']:.4f}\n")
                if "actual_stats" in summary:
                    as_ = summary["actual_stats"]
                    output.write(f"\n       Actual Stats:\n")
                    output.write(f"         Mean:   {as_['mean']:.4f}\n")
                    output.write(f"         Std:    {as_['std']:.4f}\n")
                    output.write(f"         Min:    {as_['min']:.4f}\n")
                    output.write(f"         Max:    {as_['max']:.4f}\n")
                    output.write(f"         Median: {as_['median']:.4f}\n")

            elif col_report.column_type == "categorical":
                if "expected_top_categories" in summary:
                    output.write(f"\n       Expected Top Categories:\n")
                    for cat, count in summary["expected_top_categories"].items():
                        output.write(f"         {cat}: {count}\n")
                if "actual_top_categories" in summary:
                    output.write(f"\n       Actual Top Categories:\n")
                    for cat, count in summary["actual_top_categories"].items():
                        output.write(f"         {cat}: {count}\n")

        output.write("\n" + "=" * 80 + "\n")
        output.write("                          END OF REPORT\n")
        output.write("=" * 80 + "\n")

        return output.getvalue()

    def generate_json_report(self, report: DriftReport) -> str:
        json_data = {
            "report_version": "1.0",
            "expected_metadata": report.expected_metadata,
            "actual_metadata": report.actual_metadata,
            "summary": report.summary,
            "config": report.config,
            "columns": [],
        }

        for col_report in report.column_reports:
            col_data = {
                "column_name": col_report.column_name,
                "column_type": col_report.column_type,
                "overall_risk": col_report.overall_risk.value,
                "summary": col_report.summary,
                "metrics": [],
            }

            for result in col_report.drift_results:
                metric_data = {
                    "metric_name": result.metric_name,
                    "metric_value": result.metric_value,
                    "risk_level": result.risk_level.value,
                }
                if result.additional_info:
                    metric_data["additional_info"] = result.additional_info
                col_data["metrics"].append(metric_data)

            json_data["columns"].append(col_data)

        return json.dumps(json_data, indent=2, default=str)

    def generate_csv_report(self, report: DriftReport) -> str:
        output = StringIO()
        writer = csv.writer(output)

        header = [
            "column_name",
            "column_type",
            "overall_risk",
            "metric_name",
            "metric_value",
            "metric_risk",
            "expected_missing_pct",
            "actual_missing_pct",
        ]
        writer.writerow(header)

        for col_report in report.column_reports:
            for result in col_report.drift_results:
                row = [
                    col_report.column_name,
                    col_report.column_type,
                    col_report.overall_risk.value,
                    result.metric_name,
                    f"{result.metric_value:.6f}",
                    result.risk_level.value,
                    f"{col_report.summary['expected_missing_pct']:.4f}",
                    f"{col_report.summary['actual_missing_pct']:.4f}",
                ]
                writer.writerow(row)

        return output.getvalue()

    def save_report(self, report: DriftReport, file_path: str, format: str = "json") -> None:
        if format == "json":
            content = self.generate_json_report(report)
        elif format == "csv":
            content = self.generate_csv_report(report)
        elif format == "txt":
            content = self.generate_console_report(report)
        else:
            raise ValueError(f"Unsupported format: {format}")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

    def _find_column_report(self, report: DriftReport, column_name: str) -> Optional[ColumnReport]:
        for col_report in report.column_reports:
            if col_report.column_name == column_name:
                return col_report
        return None

    def _risk_priority(self, risk: RiskLevel) -> int:
        priorities = {
            RiskLevel.NO_DRIFT: 0,
            RiskLevel.SLIGHT_DRIFT: 1,
            RiskLevel.MODERATE_DRIFT: 2,
            RiskLevel.SEVERE_DRIFT: 3,
        }
        return priorities[risk]
