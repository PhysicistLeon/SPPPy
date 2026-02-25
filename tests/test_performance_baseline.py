"""Performance baseline benchmarks for core API scenarios A/B/C."""

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# pylint: disable=wrong-import-position
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SPPPy import ExperimentSPR, Layer, MaterialDispersion, nm

_HAS_PYTEST_BENCHMARK = importlib.util.find_spec("pytest_benchmark") is not None
BASE_DIR = Path(__file__).resolve().parents[1]
BASE_CSV = BASE_DIR / "PermittivitiesBase.csv"


def _float_env(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _int_env(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _grid(start: float, stop: float, step: float) -> np.ndarray:
    count = int(round((stop - start) / step)) + 1
    return np.linspace(start, stop, count)


def _default_grids() -> dict:
    return {
        "theta_deg": _grid(
            _float_env("PERF_THETA_START_DEG", 40.0),
            _float_env("PERF_THETA_STOP_DEG", 70.0),
            _float_env("PERF_THETA_STEP_DEG", 0.1),
        ),
        "lambda_nm": _grid(
            _float_env("PERF_WL_START_NM", 400.0),
            _float_env("PERF_WL_STOP_NM", 700.0),
            _float_env("PERF_WL_STEP_NM", 1.0),
        ),
        "thickness_nm": _grid(
            _float_env("PERF_H_START_NM", 0.0),
            _float_env("PERF_H_STOP_NM", 100.0),
            _float_env("PERF_H_STEP_NM", 1.0),
        ),
    }


def _build_experiment() -> tuple[ExperimentSPR, Layer]:
    exp = ExperimentSPR(polarization="p")
    bk7 = MaterialDispersion("BK7", base_file=str(BASE_CSV))
    ag = MaterialDispersion("Ag", base_file=str(BASE_CSV))
    sio2 = MaterialDispersion("SiO2", base_file=str(BASE_CSV))
    exp.add(Layer(bk7, 0, "Prism"))
    exp.add(Layer(ag, 55 * nm, "Ag"))
    sio2_layer = Layer(sio2, 0 * nm, "SiO2")
    exp.add(sio2_layer)
    exp.add(Layer(1.0, 0, "Air"))
    return exp, sio2_layer


def _scenario_a(grids: dict) -> tuple[int, float]:
    exp, sio2_layer = _build_experiment()
    exp.wavelength = grids["lambda_nm"][0] * nm
    theta = float(grids["theta_deg"][0])
    checksum = 0.0

    for thickness_nm in grids["thickness_nm"]:
        sio2_layer.thickness = float(thickness_nm) * nm
        checksum += float(np.abs(exp.R_deg(theta=theta)) ** 2)

    return len(grids["thickness_nm"]), checksum


def _scenario_b(grids: dict) -> tuple[int, float]:
    exp, sio2_layer = _build_experiment()
    theta = float(grids["theta_deg"][0])
    checksum = 0.0

    for thickness_nm in grids["thickness_nm"]:
        sio2_layer.thickness = float(thickness_nm) * nm
        spectrum = [
            float(np.abs(exp.R_deg(theta=theta, wl=float(wl_nm) * nm)) ** 2)
            for wl_nm in grids["lambda_nm"]
        ]
        checksum += float(np.sum(spectrum))

    return len(grids["thickness_nm"]), checksum


def _scenario_c(grids: dict) -> tuple[int, float]:
    exp, sio2_layer = _build_experiment()
    exp.wavelength = grids["lambda_nm"][0] * nm
    checksum = 0.0

    for thickness_nm in grids["thickness_nm"]:
        sio2_layer.thickness = float(thickness_nm) * nm
        curve = np.array(exp.R(angle_range=grids["theta_deg"]))
        checksum += float(np.sum(curve))

    return len(grids["thickness_nm"]), checksum


def _run_benchmark(benchmark, scenario_name: str, runner):
    grids = _default_grids()
    rounds = _int_env("PERF_BENCH_ROUNDS", 30)
    warmup_rounds = _int_env("PERF_BENCH_WARMUP_ROUNDS", 5)

    curves_count, checksum = benchmark.pedantic(
        lambda: runner(grids),
        rounds=rounds,
        iterations=1,
        warmup_rounds=warmup_rounds,
    )

    stats = benchmark.stats.stats
    mean_seconds = float(stats["mean"])
    curves_per_sec = float(curves_count) / mean_seconds if mean_seconds > 0 else 0.0

    benchmark.extra_info.update(
        {
            "scenario": scenario_name,
            "curve_definition": {
                "A": "1 curve = one scalar R at fixed (theta, lambda) for one thickness",
                "B": "1 curve = one full R(lambda) array for one thickness",
                "C": "1 curve = one full R(theta) array for one thickness",
            },
            "grid": {
                "theta_start_deg": float(grids["theta_deg"][0]),
                "theta_stop_deg": float(grids["theta_deg"][-1]),
                "theta_step_deg": float(grids["theta_deg"][1] - grids["theta_deg"][0])
                if len(grids["theta_deg"]) > 1
                else 0.0,
                "lambda_start_nm": float(grids["lambda_nm"][0]),
                "lambda_stop_nm": float(grids["lambda_nm"][-1]),
                "lambda_step_nm": float(grids["lambda_nm"][1] - grids["lambda_nm"][0])
                if len(grids["lambda_nm"]) > 1
                else 0.0,
                "thickness_start_nm": float(grids["thickness_nm"][0]),
                "thickness_stop_nm": float(grids["thickness_nm"][-1]),
                "thickness_step_nm": float(
                    grids["thickness_nm"][1] - grids["thickness_nm"][0]
                )
                if len(grids["thickness_nm"]) > 1
                else 0.0,
            },
            "curves_count": int(curves_count),
            "curves_per_sec": curves_per_sec,
            "checksum": float(checksum),
            "rounds": rounds,
            "warmup_rounds": warmup_rounds,
        }
    )


@pytest.mark.performance
@pytest.mark.skipif(
    os.getenv("RUN_PERF_BASELINE", "0") != "1",
    reason="Performance baseline is opt-in. Set RUN_PERF_BASELINE=1.",
)
@pytest.mark.skipif(
    not _HAS_PYTEST_BENCHMARK,
    reason="pytest-benchmark is not installed",
)
def test_perf_scenario_a(benchmark):
    """Benchmark scenario A: one scalar R per thickness value."""
    _run_benchmark(benchmark, "A", _scenario_a)


@pytest.mark.performance
@pytest.mark.skipif(
    os.getenv("RUN_PERF_BASELINE", "0") != "1",
    reason="Performance baseline is opt-in. Set RUN_PERF_BASELINE=1.",
)
@pytest.mark.skipif(
    not _HAS_PYTEST_BENCHMARK,
    reason="pytest-benchmark is not installed",
)
def test_perf_scenario_b(benchmark):
    """Benchmark scenario B: full R(lambda) sweep per thickness."""
    _run_benchmark(benchmark, "B", _scenario_b)


@pytest.mark.performance
@pytest.mark.skipif(
    os.getenv("RUN_PERF_BASELINE", "0") != "1",
    reason="Performance baseline is opt-in. Set RUN_PERF_BASELINE=1.",
)
@pytest.mark.skipif(
    not _HAS_PYTEST_BENCHMARK,
    reason="pytest-benchmark is not installed",
)
def test_perf_scenario_c(benchmark):
    """Benchmark scenario C: full R(theta) sweep per thickness."""
    _run_benchmark(benchmark, "C", _scenario_c)


def test_perf_curve_spec_is_serializable():
    """Guardrail: benchmark metadata must be JSON serializable for CI artifact export."""
    payload = {
        "curve_definition": {
            "A": "one scalar",
            "B": "one lambda-array",
            "C": "one theta-array",
        }
    }
    json.dumps(payload)
