"""Two-stage profiling for baseline SPR API workloads.

Stage 1: cProfile + pstats (top cumulative)
Stage 2: optional line_profiler for hot paths
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from SPPPy import ExperimentSPR, Layer, MaterialDispersion, nm

BASE_DIR = Path(__file__).resolve().parents[1]
BASE_CSV = BASE_DIR / "PermittivitiesBase.csv"


def _grid(start: float, stop: float, step: float) -> np.ndarray:
    count = int(round((stop - start) / step)) + 1
    return np.linspace(start, stop, count)


def _build_experiment() -> tuple[ExperimentSPR, Layer]:
    exp = ExperimentSPR(polarization="p")
    exp.add(Layer(MaterialDispersion("BK7", base_file=str(BASE_CSV)), 0, "Prism"))
    exp.add(Layer(MaterialDispersion("Ag", base_file=str(BASE_CSV)), 55 * nm, "Ag"))
    sio2_layer = Layer(MaterialDispersion("SiO2", base_file=str(BASE_CSV)), 0, "SiO2")
    exp.add(sio2_layer)
    exp.add(Layer(1.0, 0, "Air"))
    return exp, sio2_layer


def run_baseline_workload() -> float:
    """Execute representative single-thread baseline workload and return checksum."""
    exp, sio2_layer = _build_experiment()
    theta_deg = _grid(40.0, 70.0, 0.2)
    lambda_nm = _grid(450.0, 650.0, 2.0)
    thickness_nm = _grid(0.0, 100.0, 2.0)

    checksum = 0.0
    for h_nm in thickness_nm:
        sio2_layer.thickness = float(h_nm) * nm
        # mixed workload across B + C flavors
        curve_theta = np.array(exp.R(angle_range=theta_deg, wl_range=None, angle=None))
        checksum += float(np.sum(curve_theta))

        spectrum = [
            float(np.abs(exp.R_deg(theta=float(theta_deg[0]), wl=float(wl) * nm)) ** 2)
            for wl in lambda_nm
        ]
        checksum += float(np.sum(spectrum))

    return checksum


def run_cprofile(output_dir: Path, top_n: int = 25) -> None:
    """Profile baseline workload with cProfile and save pstats artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = output_dir / "baseline.cprofile"
    txt_path = output_dir / "baseline_pstats.txt"

    prof = cProfile.Profile()
    prof.enable()
    checksum = run_baseline_workload()
    prof.disable()
    prof.dump_stats(str(profile_path))

    stream = io.StringIO()
    stats = pstats.Stats(prof, stream=stream).sort_stats("cumulative")
    stats.print_stats(top_n)
    stream.write(f"\nchecksum={checksum}\n")
    txt_path.write_text(stream.getvalue(), encoding="utf-8")


def run_line_profiler(output_dir: Path) -> bool:
    """Profile hot path with line_profiler when dependency is available."""
    try:
        # pylint: disable=import-outside-toplevel
        from line_profiler import LineProfiler
    except ImportError:
        return False

    # pylint: disable=import-outside-toplevel
    from SPPPy.experiment import ExperimentSPR as ExperimentSPRImpl
    from SPPPy.materials import DispersionABS
    from SPPPy.materials import Layer as LayerImpl

    output_dir.mkdir(parents=True, exist_ok=True)
    txt_path = output_dir / "baseline_line_profiler.txt"

    lp = LineProfiler()
    lp.add_function(ExperimentSPRImpl.Transfer_matrix)
    lp.add_function(LayerImpl.S_matrix)
    lp.add_function(DispersionABS.CRI)

    wrapped = lp(run_baseline_workload)
    checksum = wrapped()

    with txt_path.open("w", encoding="utf-8") as fh:
        lp.print_stats(stream=fh)
        fh.write(f"\nchecksum={checksum}\n")

    return True


def main() -> None:
    """CLI entrypoint for two-stage profiling."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/profiling"),
        help="Directory where profiling artifacts will be written",
    )
    args = parser.parse_args()

    run_cprofile(args.output_dir)
    has_line_profiler = run_line_profiler(args.output_dir)
    if not has_line_profiler:
        print("line_profiler is not installed; skipped stage 2.")


if __name__ == "__main__":
    main()
