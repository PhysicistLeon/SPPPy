import warnings
from typing import Optional, Dict, Any, Tuple

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

# ------------------------------------------------------------
# Paths (SPPPy/ is in parent directory)
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
PROJECT_ROOT = BASE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from SPPPy import ExperimentSPR, Layer, MaterialDispersion, nm  # noqa: E402


def find_resonance_wavelength(
    exp,
    angle_deg: float,
    wl_min: float = 400 * nm,
    wl_max: float = 900 * nm,
    step: float = 1 * nm,
    lambda_guess: Optional[float] = None,
    guess_window: float = 40 * nm,
    refine: bool = True,
    verbose: bool = True,
    layer_index: Optional[int] = None,
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Найти резонансную длину волны как argmin R(lambda) для готового объекта ExperimentSPR
    при фиксированном угле падения.

    Параметры
    ---------
    exp : SPPPy.ExperimentSPR
        Уже собранный объект эксперимента (слои/материалы заданы).
    angle_deg : float
        Угол падения в градусах.
    wl_min, wl_max : float
        Запрошенный диапазон поиска в метрах (используй nm из SPPPy, например 400*nm).
    step : float
        Шаг грубого скана в метрах (например 1*nm).
    lambda_guess : float or None
        Если задано (в метрах), выбирается минимум в локальном окне вокруг guess.
        Удобно, если в спектре несколько провалов.
    guess_window : float
        Полуширина окна для lambda_guess (в метрах).
    refine : bool
        Если True, выполняется локальная дооптимизация (bounded minimize) вокруг грубого минимума.
    verbose : bool
        Печатать диагностические сообщения (обрезка диапазона, результат).
    layer_index : int or None
        Индекс слоя с переменной толщиной. Если передан, используется оптимизированный
        метод R_lambda_vs_thickness для ускорения грубого скана.

    Возвращает
    ----------
    lambda_res_m : float
        Резонансная длина волны (в метрах).
    R_min : float
        Минимальное значение отражения R в этой точке.
    info : dict
        Служебная информация:
        {
            "lambda_res_nm": ...,
            "angle_deg": ...,
            "requested_range_nm": (..., ...),
            "used_range_nm": (..., ...),
            "range_clipped": bool,
            "selection_mode": "global" | "near_guess",
            "coarse_step_nm": ...,
            "coarse_lambda_nm": np.ndarray,
            "coarse_R": np.ndarray,
            "builtin_pointSPR_available": bool,
            "materials_limits_nm": [(layer_idx, layer_name, lmin_nm, lmax_nm), ...]
        }

    Примечание
    ----------
    В SPPPy есть exp.pointSPR(wl_range=...), но этот helper:
      - безопаснее для мультимодальных спектров (грубый глобальный поиск + локальная дооптимизация)
      - автоматически учитывает диапазоны дисперсий материалов
      - не меняет состояние exp
    """

    # -------------------- валидация --------------------
    if wl_max <= wl_min:
        raise ValueError("wl_max должен быть больше wl_min")
    if step <= 0:
        raise ValueError("step должен быть > 0")
    if lambda_guess is not None and not (wl_min <= lambda_guess <= wl_max):
        warnings.warn(
            "lambda_guess вне запрошенного диапазона; будет всё равно учтён после пересечения диапазонов."
        )

    # -------------------- пересечение диапазонов материалов --------------------
    materials_limits = []
    mat_lo = -np.inf
    mat_hi = np.inf

    # В exp.layers могут быть как дисперсионные материалы (с lambda_min/max), так и константы (Air=1.0)
    for idx, layer in enumerate(getattr(exp, "layers", [])):
        layer_name = getattr(layer, "name", None) or f"layer_{idx}"
        n_obj = getattr(layer, "n", None)

        if hasattr(n_obj, "lambda_min") and hasattr(n_obj, "lambda_max"):
            lmin = float(n_obj.lambda_min)
            lmax = float(n_obj.lambda_max)
            materials_limits.append((idx, layer_name, lmin / nm, lmax / nm))
            mat_lo = max(mat_lo, lmin)
            mat_hi = min(mat_hi, lmax)

    # Если дисперсионных материалов нет — используем пользовательский диапазон как есть
    eff_lo = wl_min if not np.isfinite(mat_lo) else max(wl_min, mat_lo)
    eff_hi = wl_max if not np.isfinite(mat_hi) else min(wl_max, mat_hi)

    range_clipped = (abs(eff_lo - wl_min) > 1e-18) or (abs(eff_hi - wl_max) > 1e-18)

    if eff_hi <= eff_lo:
        raise ValueError(
            "Нет пересечения между запрошенным диапазоном и диапазонами материалов. "
            f"Запрошено: [{wl_min/nm:.1f}, {wl_max/nm:.1f}] нм, "
            f"доступно по материалам: [{mat_lo/nm:.1f}, {mat_hi/nm:.1f}] нм."
        )

    if range_clipped and verbose:
        print(
            "[find_resonance_wavelength] Диапазон поиска ограничен диапазонами материалов: "
            f"{wl_min/nm:.1f}–{wl_max/nm:.1f} нм -> {eff_lo/nm:.1f}–{eff_hi/nm:.1f} нм"
        )

    # -------------------- грубый спектральный скан --------------------
    # Включаем правую границу (если попадает в сетку) и при необходимости добавим её вручную
    wl_grid = np.arange(eff_lo, eff_hi + 0.5 * step, step, dtype=float)
    if wl_grid.size == 0:
        wl_grid = np.array([eff_lo, eff_hi], dtype=float)
    elif wl_grid[-1] < eff_hi - 1e-15:
        wl_grid = np.append(wl_grid, eff_hi)

    # Используем оптимизированный метод R_lambda_vs_thickness если передан layer_index
    if layer_index is not None:
        cur_thickness = exp.layers[layer_index].thickness
        R_curves = exp.R_lambda_vs_thickness(
            layer_index,
            [cur_thickness],
            wl_grid,
            theta=angle_deg,
        )
        R_grid = np.asarray(R_curves[0], dtype=float)
    else:
        # exp.R(...) возвращает R=|r|^2 при wl_range и фиксированном угле
        R_grid = np.asarray(exp.R(wl_range=wl_grid, angle=angle_deg), dtype=float)

    # Численная защита
    R_grid = np.clip(R_grid, 0.0, 1.0)

    # -------------------- выбор минимума --------------------
    if lambda_guess is None:
        idx0 = int(np.argmin(R_grid))
        selection_mode = "global"
    else:
        lo_g = max(eff_lo, lambda_guess - guess_window)
        hi_g = min(eff_hi, lambda_guess + guess_window)
        mask = (wl_grid >= lo_g) & (wl_grid <= hi_g)

        if not np.any(mask):
            raise ValueError(
                f"Окно around lambda_guess не пересекается с допустимым диапазоном. "
                f"guess={lambda_guess/nm:.1f} нм, окно ±{guess_window/nm:.1f} нм, "
                f"допустимо {eff_lo/nm:.1f}–{eff_hi/nm:.1f} нм."
            )

        idx_candidates = np.where(mask)[0]
        idx0 = idx_candidates[int(np.argmin(R_grid[mask]))]
        selection_mode = "near_guess"

    lam0 = float(wl_grid[idx0])

    # -------------------- локальная дооптимизация --------------------
    def objective(lam_m: float) -> float:
        # R_deg возвращает комплексный коэффициент отражения r
        # R = |r|^2
        return float(np.abs(exp.R_deg(theta=angle_deg, wl=lam_m)) ** 2)

    if refine and wl_grid.size >= 2:
        i_left = max(0, idx0 - 1)
        i_right = min(len(wl_grid) - 1, idx0 + 1)
        a = float(wl_grid[i_left])
        b = float(wl_grid[i_right])

        # Если минимум на краю сетки, расширим/оставим границу как есть
        if a == b:
            lambda_res = lam0
            R_min = float(R_grid[idx0])
        else:
            # xatol ~ 1e-3 шага, но не грубее 1e-4 нм
            xatol = max(step * 1e-3, 1e-4 * nm)

            res = minimize_scalar(
                objective,
                bounds=(a, b),
                method="bounded",
                options={"xatol": xatol},
            )

            if res.success:
                lambda_res = float(res.x)
                R_min = float(res.fun)
            else:
                lambda_res = lam0
                R_min = float(R_grid[idx0])
                if verbose:
                    print(
                        "[find_resonance_wavelength] refine step failed, fallback to coarse minimum."
                    )
    else:
        lambda_res = lam0
        R_min = float(R_grid[idx0])

    # Финальный clamp по численной погрешности
    R_min = float(np.clip(R_min, 0.0, 1.0))

    # -------------------- служебная информация --------------------
    info: Dict[str, Any] = {
        "lambda_res_nm": lambda_res / nm,
        "angle_deg": float(angle_deg),
        "requested_range_nm": (wl_min / nm, wl_max / nm),
        "used_range_nm": (eff_lo / nm, eff_hi / nm),
        "range_clipped": bool(range_clipped),
        "selection_mode": selection_mode,
        "coarse_step_nm": step / nm,
        "coarse_lambda_nm": wl_grid / nm,
        "coarse_R": R_grid,
        "builtin_pointSPR_available": callable(getattr(exp, "pointSPR", None)),
        "materials_limits_nm": materials_limits,
    }

    if verbose:
        print(
            f"[find_resonance_wavelength] θ={angle_deg:.3f}°, "
            f"λ_res={lambda_res/nm:.3f} нм, R_min={R_min:.6f}, "
            f"mode={selection_mode}"
        )

    return lambda_res, R_min, info


def main():
    BASE_CSV = PROJECT_ROOT / "PermittivitiesBase.csv"

    # ------------------------------------------------------------
    # User parameters (easy to tweak)
    # ------------------------------------------------------------
    ANGLE_DEG = 45.0  # fixed incidence angle (you can adjust)
    POLARIZATION = "p"  # SPR => p-polarization
    AG_THICKNESS_NM = 55.0  # Ag thickness

    # ------------------------------------------------------------
    # Materials from your database
    # ------------------------------------------------------------
    bk7 = MaterialDispersion("BK7", base_file=str(BASE_CSV))
    ag = MaterialDispersion("Ag", base_file=str(BASE_CSV))
    sio2 = MaterialDispersion("SiO2", base_file=str(BASE_CSV))
    air_n = 1.0  # Air as constant refractive index

    # ------------------------------------------------------------
    # Build Kretschmann structure: BK7 / Ag / SiO2 / Air
    # ------------------------------------------------------------
    exp = ExperimentSPR(polarization=POLARIZATION)

    prism_layer = Layer(bk7, 0, "BK7 prism")  # semi-infinite incident medium
    ag_layer = Layer(ag, AG_THICKNESS_NM * nm, "Ag")  # 55 nm silver
    sio2_layer = Layer(sio2, 0 * nm, "SiO2")  # variable thickness
    air_layer = Layer(air_n, 0, "Air")  # semi-infinite output medium

    exp.add(prism_layer)
    exp.add(ag_layer)
    exp.add(sio2_layer)
    exp.add(air_layer)

    lambda_res, R_min, info = find_resonance_wavelength(
        exp,
        angle_deg=ANGLE_DEG,
        wl_min=400 * nm,
        wl_max=900 * nm,
        step=1 * nm,
        # lambda_guess=650*nm,  # опционально, если хочешь выбрать конкретный провал
        verbose=True,
    )

    print("Resonance wavelength:", lambda_res / nm, "nm")
    print("R_min:", R_min)


if __name__ == "__main__":
    main()
