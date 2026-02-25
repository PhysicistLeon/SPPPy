"""Convert pytest-benchmark JSON output into a concise CSV summary."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CSV_COLUMNS = [
    "name",
    "scenario",
    "mean_sec",
    "stddev_sec",
    "median_sec",
    "iqr_sec",
    "rounds",
    "curves_count",
    "curves_per_sec",
    "theta_start_deg",
    "theta_stop_deg",
    "theta_step_deg",
    "lambda_start_nm",
    "lambda_stop_nm",
    "lambda_step_nm",
    "thickness_start_nm",
    "thickness_stop_nm",
    "thickness_step_nm",
]


def _safe_get(mapping: dict, *keys, default=""):
    current = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def convert(benchmark_json: Path, csv_output: Path) -> None:
    """Read pytest-benchmark JSON report and write condensed CSV rows."""
    with benchmark_json.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    entries = raw.get("benchmarks", [])

    csv_output.parent.mkdir(parents=True, exist_ok=True)
    with csv_output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for entry in entries:
            stats = entry.get("stats", {})
            extra = entry.get("extra_info", {})
            grid = extra.get("grid", {})
            writer.writerow(
                {
                    "name": entry.get("name", ""),
                    "scenario": extra.get("scenario", ""),
                    "mean_sec": stats.get("mean", ""),
                    "stddev_sec": stats.get("stddev", ""),
                    "median_sec": stats.get("median", ""),
                    "iqr_sec": stats.get("iqr", ""),
                    "rounds": stats.get("rounds", ""),
                    "curves_count": extra.get("curves_count", ""),
                    "curves_per_sec": extra.get("curves_per_sec", ""),
                    "theta_start_deg": grid.get("theta_start_deg", ""),
                    "theta_stop_deg": grid.get("theta_stop_deg", ""),
                    "theta_step_deg": grid.get("theta_step_deg", ""),
                    "lambda_start_nm": grid.get("lambda_start_nm", ""),
                    "lambda_stop_nm": grid.get("lambda_stop_nm", ""),
                    "lambda_step_nm": grid.get("lambda_step_nm", ""),
                    "thickness_start_nm": grid.get("thickness_start_nm", ""),
                    "thickness_stop_nm": grid.get("thickness_stop_nm", ""),
                    "thickness_step_nm": grid.get("thickness_step_nm", ""),
                }
            )


def main() -> None:
    """CLI entrypoint for benchmark JSON -> CSV conversion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-json", required=True, type=Path)
    parser.add_argument("--csv-output", required=True, type=Path)
    args = parser.parse_args()

    convert(args.benchmark_json, args.csv_output)


if __name__ == "__main__":
    main()
