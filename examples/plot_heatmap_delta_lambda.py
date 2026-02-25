# heatmap_delta_lambda_vs_theta_and_h_masked.py
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ------------------------------------------------------------
# Paths (SPPPy/ is in parent directory)
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
PROJECT_ROOT = BASE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from SPPPy import ExperimentSPR, Layer, MaterialDispersion, nm  # noqa: E402
from find_minima import find_resonance_wavelength  # noqa: E402

# ------------------------------------------------------------
# User parameters
# ------------------------------------------------------------
POLARIZATION = "p"
AG_THICKNESS_NM = 55.0

THETA_DEG = np.arange(40.0, 70.0 + 1e-9, 0.5)  # 40..70°, step 0.5°
H_NM = np.arange(0.0, 30.0 + 1e-9, 1.0)  # 0..30 nm, step 1 nm

WL_MIN = 400 * nm
WL_MAX = 900 * nm
WL_STEP = 1 * nm

USE_TRACKING_WITH_GUESS = True  # branch tracking по h


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def get_layer_by_name(exp: ExperimentSPR, layer_name: str) -> Layer:
    matches = [
        layer for layer in exp.layers if getattr(layer, "name", None) == layer_name
    ]
    if not matches:
        names = [
            getattr(layer, "name", f"layer_{i}") for i, layer in enumerate(exp.layers)
        ]
        raise ValueError(f'Слой "{layer_name}" не найден. Доступные слои: {names}')
    if len(matches) > 1:
        raise ValueError(
            f'Найдено несколько слоёв с именем "{layer_name}". Сделай имена уникальными.'
        )
    return matches[0]


def build_default_kretschmann_exp(base_csv_path: Path) -> ExperimentSPR:
    """BK7 / Ag(55 nm) / SiO2 / Air"""
    bk7 = MaterialDispersion("BK7", base_file=str(base_csv_path))
    ag = MaterialDispersion("Ag", base_file=str(base_csv_path))
    sio2 = MaterialDispersion("SiO2", base_file=str(base_csv_path))
    air_n = 1.0

    exp = ExperimentSPR(polarization=POLARIZATION)
    exp.add(Layer(bk7, 0, "BK7"))
    exp.add(Layer(ag, AG_THICKNESS_NM * nm, "Ag"))
    exp.add(Layer(sio2, 0 * nm, "SiO2"))  # variable thickness
    exp.add(Layer(air_n, 0, "Air"))
    return exp


def is_edge_minimum(lambda_res_nm: float, used_range_nm, wl_step_nm: float) -> bool:
    lo_nm, hi_nm = used_range_nm
    tol_nm = max(0.51 * wl_step_nm, 1e-6)  # ~ half step
    return (abs(lambda_res_nm - lo_nm) <= tol_nm) or (
        abs(lambda_res_nm - hi_nm) <= tol_nm
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    base_csv = PROJECT_ROOT / "PermittivitiesBase.csv"
    exp = build_default_kretschmann_exp(base_csv)
    sio2_layer = get_layer_by_name(exp, "SiO2")
    SIO2_LAYER_INDEX = 2  # Индекс слоя SiO2 в структуре BK7/Ag/SiO2/Air

    n_h = len(H_NM)
    n_theta = len(THETA_DEG)

    baseline_lambda0_nm_by_theta = np.full(n_theta, np.nan, dtype=float)
    delta_lambda_nm_map = np.full((n_h, n_theta), np.nan, dtype=float)
    rmin_map = np.full((n_h, n_theta), np.nan, dtype=float)

    # Маска невалидных точек (warning edge minimum) -> будут чёрные
    warning_mask = np.zeros((n_h, n_theta), dtype=bool)

    # Если baseline (h=0) невалидный, вся колонка невалидна
    baseline_invalid_col = np.zeros(n_theta, dtype=bool)

    total_edge_warnings = 0

    print("\n=== Heatmap build log: Δλ(theta, h) with edge masking ===")
    print(
        f"Angle scan: {THETA_DEG[0]:.1f}..{THETA_DEG[-1]:.1f} deg (step {THETA_DEG[1]-THETA_DEG[0]:.1f})"
    )
    print(f"h scan    : {H_NM[0]:.1f}..{H_NM[-1]:.1f} nm (step {H_NM[1]-H_NM[0]:.1f})")
    print(f"Search λ  : {WL_MIN/nm:.1f}..{WL_MAX/nm:.1f} nm (step {WL_STEP/nm:.1f})")
    print("-" * 124)
    print(
        f"{'theta (deg)':>10} | {'lambda_res(0) (nm)':>17} | {'Δλ(h=30) (nm)':>12} | {'Δλ min..max (nm)':>24} | {'edge warn pts':>11} | {'status':<18}"
    )
    print("-" * 124)

    for j, theta_deg in enumerate(THETA_DEG):
        lambda_guess = None  # branch tracking внутри колонки
        lambda0_nm = None
        edge_warn_this_theta = 0
        status = "ok"

        # Временное хранилище для колонки (чтобы можно было целиком отбросить при invalid baseline)
        col_lambda_res_nm = np.full(n_h, np.nan, dtype=float)
        col_rmin = np.full(n_h, np.nan, dtype=float)
        col_edge = np.zeros(n_h, dtype=bool)

        for i, h_nm in enumerate(H_NM):
            sio2_layer.thickness = float(h_nm) * nm

            kwargs = dict(
                exp=exp,
                angle_deg=float(theta_deg),
                wl_min=WL_MIN,
                wl_max=WL_MAX,
                step=WL_STEP,
                verbose=False,
                layer_index=SIO2_LAYER_INDEX,
            )
            if USE_TRACKING_WITH_GUESS and (lambda_guess is not None):
                kwargs["lambda_guess"] = lambda_guess

            lambda_res_m, R_min, info = find_resonance_wavelength(**kwargs)
            lambda_res_nm = lambda_res_m / nm
            used_range_nm = info["used_range_nm"]

            edge = is_edge_minimum(lambda_res_nm, used_range_nm, WL_STEP / nm)
            col_edge[i] = edge

            if edge:
                edge_warn_this_theta += 1
                total_edge_warnings += 1
                print(
                    f"WARNING edge minimum: theta={theta_deg:.1f}°, h={h_nm:.0f} nm, "
                    f"lambda_res={lambda_res_nm:.2f} nm, range={used_range_nm[0]:.1f}-{used_range_nm[1]:.1f} nm"
                )

            # baseline point
            if i == 0:
                if edge:
                    baseline_invalid_col[j] = True
                    status = "invalid baseline"
                    # baseline невалиден -> колонку дальше считаем для логов, но в карту не включаем
                else:
                    lambda0_nm = lambda_res_nm
                    baseline_lambda0_nm_by_theta[j] = lambda0_nm

            # Сохраняем промежуточно (даже edge) — потом маскируем
            col_lambda_res_nm[i] = lambda_res_nm
            col_rmin[i] = R_min

            # Branch tracking: не обновляем guess, если точка edge (чтобы не утащить трекинг к границе)
            if not edge:
                lambda_guess = lambda_res_m

        # Собираем колонку в итоговые массивы
        if baseline_invalid_col[j]:
            # Вся колонка невалидна (чёрная)
            warning_mask[:, j] = True
            delta_lambda_nm_map[:, j] = np.nan
            rmin_map[:, j] = np.nan
            baseline_lambda0_nm_by_theta[j] = np.nan

            dmin = np.nan
            dmax = np.nan
            dlast = np.nan
        else:
            # baseline валиден: считаем Δλ, но edge-точки маскируем
            lambda0_nm = baseline_lambda0_nm_by_theta[j]
            delta_col = col_lambda_res_nm - lambda0_nm

            # Все warning-точки (edge minima) -> black/NaN
            warning_mask[:, j] = col_edge
            delta_col[col_edge] = np.nan
            col_rmin[col_edge] = np.nan

            delta_lambda_nm_map[:, j] = delta_col
            rmin_map[:, j] = col_rmin

            # статистика для лога по валидным точкам
            if np.all(np.isnan(delta_col)):
                dmin = np.nan
                dmax = np.nan
                dlast = np.nan
                status = "all points masked"
            else:
                dmin = float(np.nanmin(delta_col))
                dmax = float(np.nanmax(delta_col))
                dlast = float(delta_col[-1]) if np.isfinite(delta_col[-1]) else np.nan
                if edge_warn_this_theta > 0:
                    status = "partial masked"

        lambda0_str = (
            f"{baseline_lambda0_nm_by_theta[j]:.3f}"
            if np.isfinite(baseline_lambda0_nm_by_theta[j])
            else "NaN"
        )
        dlast_str = f"{dlast:12.3f}" if np.isfinite(dlast) else f"{'NaN':>12}"
        dmm_str = (
            f"[{dmin:8.3f}, {dmax:8.3f}]"
            if (np.isfinite(dmin) and np.isfinite(dmax))
            else f"[{'NaN':>8}, {'NaN':>8}]"
        )

        print(
            f"{theta_deg:10.1f} | {lambda0_str:>17} | {dlast_str} | {dmm_str:>24} | {edge_warn_this_theta:11d} | {status:<18}"
        )

    print("-" * 124)
    print(f"Done. Total edge warnings masked (black): {total_edge_warnings}")

    # ------------------------------------------------------------
    # Plot heatmap: X = theta, Y = h, color = Δλ (nm)
    # warning/invalid points -> black
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)

    # Маскируем warning-точки и NaN
    data_plot = delta_lambda_nm_map.copy()
    data_plot[warning_mask] = np.nan
    data_ma = np.ma.masked_invalid(data_plot)

    valid_abs_max = np.nanmax(np.abs(data_plot))
    vmax_abs = (
        float(valid_abs_max)
        if np.isfinite(valid_abs_max) and valid_abs_max > 0
        else 1e-6
    )
    norm = TwoSlopeNorm(vmin=-vmax_abs, vcenter=0.0, vmax=vmax_abs)

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="black")  # warning/invalid points -> black

    mesh = ax.pcolormesh(
        THETA_DEG,
        H_NM,
        data_ma,
        shading="auto",
        cmap=cmap,
        norm=norm,
    )

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(r"$\Delta \lambda$ (nm)")

    ax.set_xlabel(r"Incidence angle $\theta$ (deg)")
    ax.set_ylabel(r"SiO$_2$ thickness $h$ (nm)")
    ax.set_title(
        r"Heatmap of $\Delta\lambda(h,\theta)=\lambda_{res}(h,\theta)-\lambda_{res}(0,\theta)$"
        + f"\nBK7 / Ag({AG_THICKNESS_NM:.0f} nm) / SiO$_2$ / Air, p-pol"
    )
    ax.set_xlim(float(THETA_DEG.min()), float(THETA_DEG.max()))
    ax.set_ylim(float(H_NM.min()), float(H_NM.max()))

    # Верхняя ось: baseline λ_res(0,θ). Невалидные baseline показываем как "—"
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())

    top_tick_angles = np.arange(40.0, 70.0 + 1e-9, 4.0)
    step_theta = THETA_DEG[1] - THETA_DEG[0]
    top_tick_indices = [
        int(round((ang - THETA_DEG[0]) / step_theta)) for ang in top_tick_angles
    ]

    top_tick_labels = []
    for idx in top_tick_indices:
        val = baseline_lambda0_nm_by_theta[idx]
        top_tick_labels.append(f"{val:.1f}" if np.isfinite(val) else "—")

    ax_top.set_xticks(top_tick_angles)
    ax_top.set_xticklabels(top_tick_labels)
    ax_top.set_xlabel(r"Baseline resonance wavelength $\lambda_{res}(0,\theta)$ (nm)")

    ax.grid(True, alpha=0.15)
    plt.show()

    # Данные доступны для дальнейшего сохранения/анализа:
    # THETA_DEG, H_NM, baseline_lambda0_nm_by_theta, delta_lambda_nm_map, rmin_map, warning_mask


if __name__ == "__main__":
    main()
