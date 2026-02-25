import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark_compare import build_summary, to_markdown


def test_build_summary_and_markdown(tmp_path: Path):
    data = {
        "benchmarks": [
            {"extra_info": {"scenario": "A"}, "stats": {"mean": 2.0}},
            {"extra_info": {"scenario": "A_FAST"}, "stats": {"mean": 1.0}},
            {"extra_info": {"scenario": "B"}, "stats": {"mean": 10.0}},
            {"extra_info": {"scenario": "B_FAST"}, "stats": {"mean": 5.0}},
        ]
    }
    src = tmp_path / "bench.json"
    src.write_text(json.dumps(data), encoding="utf-8")

    summary = build_summary(src)
    assert len(summary["pairings"]) == 2

    row_a = next(item for item in summary["pairings"] if item["scenario"] == "A")
    assert abs(row_a["speedup_x"] - 2.0) < 1e-12
    assert abs(row_a["time_reduction_pct"] - 50.0) < 1e-12

    md = to_markdown(summary)
    assert "| A |" in md
    assert "Speedup" in md
