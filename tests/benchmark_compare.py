"""Build comparative speedup summary from pytest-benchmark JSON output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PAIRINGS = [
    ("A", "A_FAST"),
    ("B", "B_FAST"),
    ("C", "C_FAST"),
]


def _extract_mean_by_scenario(payload: dict) -> dict[str, float]:
    means: dict[str, float] = {}
    for entry in payload.get("benchmarks", []):
        scenario = entry.get("extra_info", {}).get("scenario")
        if not scenario:
            continue
        stats = entry.get("stats", {})
        mean = stats.get("mean")
        if mean is not None:
            means[str(scenario)] = float(mean)
    return means


def build_summary(benchmark_json: Path) -> dict:
    """Return normalized summary dict with baseline/fast speedups."""
    with benchmark_json.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    means = _extract_mean_by_scenario(payload)
    rows = []

    for base_name, fast_name in PAIRINGS:
        base_mean = means.get(base_name)
        fast_mean = means.get(fast_name)
        if base_mean is None or fast_mean is None:
            continue

        speedup = base_mean / fast_mean if fast_mean > 0 else None
        reduction_pct = (1.0 - fast_mean / base_mean) * 100.0 if base_mean > 0 else None

        rows.append(
            {
                "scenario": base_name,
                "baseline_mean_sec": base_mean,
                "fast_mean_sec": fast_mean,
                "speedup_x": speedup,
                "time_reduction_pct": reduction_pct,
            }
        )

    return {"pairings": rows}


def to_markdown(summary: dict) -> str:
    """Render a markdown table for GitHub step summary."""
    rows = summary.get("pairings", [])
    if not rows:
        return "No comparable baseline/fast scenario pairs found."

    lines = [
        "### Performance speedup summary (baseline vs optimized)",
        "",
        "| Scenario | Baseline mean (s) | Fast mean (s) | Speedup (x) | Time reduction (%) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {scenario} | {baseline:.6f} | {fast:.6f} | {speedup:.2f} | {reduction:.2f} |".format(
                scenario=row["scenario"],
                baseline=row["baseline_mean_sec"],
                fast=row["fast_mean_sec"],
                speedup=row["speedup_x"],
                reduction=row["time_reduction_pct"],
            )
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-json", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--summary-md", required=True, type=Path)
    args = parser.parse_args()

    summary = build_summary(args.benchmark_json)

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)

    args.summary_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    args.summary_md.write_text(to_markdown(summary) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
