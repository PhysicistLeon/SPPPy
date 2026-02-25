# delta_lambda_vs_sio2_thickness.py
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Paths (SPPPy/ is in parent directory)
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
PROJECT_ROOT = BASE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from SPPPy import ExperimentSPR, Layer, MaterialDispersion, nm  # noqa: E402
from find_minima import find_resonance_wavelength  # noqa: E402

# ------------------------------------------------------------
# User parameters (можно менять)
# ------------------------------------------------------------
ANGLE_DEG = 45.0  # фиксированный угол
POLARIZATION = "p"  # SPR -> p-поляризация
AG_THICKNESS_NM = 55.0  # серебро
H_RANGE_NM = np.arange(0, 31, 1)  # h = 0..30 нм, шаг 1 нм

WL_MIN = 400 * nm
WL_MAX = 900 * nm
WL_STEP = 1 * nm

# Если helper поддерживает lambda_guess (как в предыдущем коде),
# это поможет отслеживать ту же ветвь резонанса без "перепрыгивания".
USE_TRACKING_WITH_GUESS = True


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def get_layer_by_name(exp: ExperimentSPR, layer_name: str) -> Layer:
    """Найти слой по имени (точное совпадение)."""
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
    """
    Собирает структуру BK7 / Ag(55 nm) / SiO2 / Air.
    Если у тебя exp уже создаётся в другом месте — можно не использовать эту функцию.
    """
    bk7 = MaterialDispersion("BK7", base_file=str(base_csv_path))
    ag = MaterialDispersion("Ag", base_file=str(base_csv_path))
    sio2 = MaterialDispersion("SiO2", base_file=str(base_csv_path))
    air_n = 1.0

    exp = ExperimentSPR(polarization=POLARIZATION)
    exp.add(Layer(bk7, 0, "BK7"))  # полубесконечная призма
    exp.add(Layer(ag, AG_THICKNESS_NM * nm, "Ag"))  # 55 нм
    exp.add(Layer(sio2, 0 * nm, "SiO2"))  # переменный слой
    exp.add(Layer(air_n, 0, "Air"))  # полубесконечная среда

    return exp


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    base_csv = PROJECT_ROOT / "PermittivitiesBase.csv"

    # Если у тебя exp уже собирается отдельно — подставь его вместо этой строки.
    exp = build_default_kretschmann_exp(base_csv)

    # Слой, толщину которого меняем
    sio2_layer = get_layer_by_name(exp, "SiO2")

    # Массивы результатов
    lambda_res_nm_list = []
    rmin_list = []
    delta_lambda_nm_list = []
    warnings_count = 0

    # Для трекинга ветви резонанса
    lambda_guess = None

    # Шапка лога
    print("\n=== Resonance tracking log ===")
    print(
        f"Angle = {ANGLE_DEG:.3f} deg, search range = {WL_MIN/nm:.1f}-{WL_MAX/nm:.1f} nm"
    )
    print(
        f"{'h (nm)':>6} | {'lambda_res (nm)':>15} | {'Delta lambda (nm)':>16} | {'R_min':>10} | {'note':<20}"
    )
    print("-" * 80)

    lambda0_nm = None

    for h_nm in H_RANGE_NM:
        # Меняем толщину SiO2
        sio2_layer.thickness = float(h_nm) * nm

        # Вызов helper-функции
        kwargs = dict(
            exp=exp,
            angle_deg=ANGLE_DEG,
            wl_min=WL_MIN,
            wl_max=WL_MAX,
            step=WL_STEP,
            verbose=False,
        )
        if USE_TRACKING_WITH_GUESS and lambda_guess is not None:
            # branch tracking: искать минимум в окрестности предыдущего резонанса
            kwargs["lambda_guess"] = lambda_guess

        lambda_res_m, R_min, info = find_resonance_wavelength(**kwargs)
        lambda_res_nm = lambda_res_m / nm

        # Базовая длина волны для Δλ
        if lambda0_nm is None:
            lambda0_nm = lambda_res_nm

        delta_lambda_nm = lambda_res_nm - lambda0_nm

        # Предупреждение, если минимум на границе диапазона
        used_lo_nm, used_hi_nm = info["used_range_nm"]
        edge_note = ""
        tol_nm = max(0.51 * (WL_STEP / nm), 1e-6)  # примерно половина шага
        if (
            abs(lambda_res_nm - used_lo_nm) <= tol_nm
            or abs(lambda_res_nm - used_hi_nm) <= tol_nm
        ):
            edge_note = "WARNING: edge min"
            warnings_count += 1

        # Лог
        print(
            f"{h_nm:6.1f} | {lambda_res_nm:15.4f} | {delta_lambda_nm:16.4f} | {R_min:10.6f} | {edge_note:<20}"
        )

        # Сохраняем
        lambda_res_nm_list.append(lambda_res_nm)
        rmin_list.append(R_min)
        delta_lambda_nm_list.append(delta_lambda_nm)

        # Обновляем guess для следующего h
        lambda_guess = lambda_res_m

    # В numpy
    h_nm_arr = H_RANGE_NM.astype(float)
    delta_lambda_nm_arr = np.asarray(delta_lambda_nm_list, dtype=float)

    # Итого
    print("-" * 80)
    print(f"Done. Baseline lambda_res(0) = {lambda0_nm:.4f} nm")
    if warnings_count > 0:
        print(
            f"Warnings: {warnings_count} point(s) had minimum at search-range edge (kept on plot)."
        )

    # График Δλ(h)
    fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    ax.plot(h_nm_arr, delta_lambda_nm_arr, marker="o", linewidth=1.5, markersize=4)

    ax.set_xlabel("SiO$_2$ thickness h (nm)")
    ax.set_ylabel(r"$\Delta \lambda$ (nm)")
    ax.set_title(
        r"$\Delta\lambda(h)=\lambda_{{res}}(h)-\lambda_{{res}}(0)$"
        + f"\nBK7 / Ag({AG_THICKNESS_NM:.0f} nm) / SiO$_2$ / Air, "
        f"$\\theta$ = {ANGLE_DEG:.1f}°, p-pol"
    )

    ax.set_xlim(0, 30)
    ax.grid(True, alpha=0.3)

    plt.show()

    # Если захочешь потом сохранять данные — вот готовые массивы:
    # h_nm_arr, lambda_res_nm_arr, delta_lambda_nm_arr, rmin_arr


if __name__ == "__main__":
    main()
