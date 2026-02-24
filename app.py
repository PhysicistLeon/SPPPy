# -*- coding: utf-8 -*-
"""
app.py  (корень проекта)

Главный файл приложения:
- Главное окно (QMainWindow)
- MenuBar + popup-меню Scheme/Plots/Functions (заглушки)
- Центральный QSplitter: слева вкладки/отрисовка, справа панель слоёв

Важно:
- Реальную интеграцию с SPPy (расчёт) пока не пишем, но каркас для "обмена" заложим:
  MainWindow будет местом, где связываются:
  - состояние эксперимента (слои + настройки вкладок),
  - загрузка/сохранение,
  - запуск расчёта и обновление отрисовки.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from scipy.interpolate import PchipInterpolator
import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from gui import (
    LayerState,
    LayerWidget,
    LayersPanel,
    PlotTabsWidget,
    PlotDisplaySettingsDialog,
)

from SPPPy import (
    ExperimentSPR, Layer, nm, Anisotropic, MaterialDispersion,
    CauchyDispersion, LorentzDrudeDispersion
    )



LayerBuilder = Callable[[LayerState], Any]  # Any -> чтобы не упираться в типизацию SPPPy
APP_SETTINGS_PATH = Path("gui") / "spr_app_settings.json"


class LayerFactory:
    def __init__(self):
        self._builders: Dict[str, LayerBuilder] = {}

    # поведение "как словарь"
    def __setitem__(self, layer_type: str, builder: LayerBuilder) -> None:
        self._builders[str(layer_type)] = builder

    def __getitem__(self, layer_type: str) -> LayerBuilder:
        return self._builders[str(layer_type)]

    def __contains__(self, layer_type: str) -> bool:
        return str(layer_type) in self._builders

    # удобная регистрация декоратором (опционально)
    def register(self, layer_type: str):
        def deco(fn: LayerBuilder) -> LayerBuilder:
            self[str(layer_type)] = fn
            return fn
        return deco

    # сборка одного слоя
    def build_one(self, st: LayerState):
        if st.type not in self._builders:
            raise ValueError(f"Unsupported layer type: {st.type}")
        return self._builders[st.type](st)

    # сборка списка слоёв (в нужном порядке)
    def build_all(self, states: List[LayerState]):
        return [self.build_one(st) for st in states]
    

def layer_states_from_export(data: Dict[str, Any]) -> List[LayerState]:
    layers_ui = data.get("layers", [])
    if not isinstance(layers_ui, list):
        raise TypeError("export_layers(): data['layers'] must be a list")

    tmp = LayerWidget(None)
    out: List[LayerState] = []

    for ui_st in layers_ui:
        if not isinstance(ui_st, dict):
            raise TypeError("export_layers(): each layer state must be a dict")
        tmp.set_ui_state(ui_st)     # восстановили UI слоя
        out.append(tmp.get_state()) # получили физику (SI)

    return out

factory = LayerFactory()


@factory.register("Dielectric")
def build_dielectric(st: LayerState):
    p = st.params or {}
    n = float(p.get("n", 1.0))
    d = float(p.get("d", 0.0))

    # Диэлектрик: не создаём искусственный 0j
    # (если SPPPy Layer принимает float n — будет чисто)
    return Layer(n, d, name=st.type)


@factory.register("Metal")
def build_metal(st: LayerState):
    p = st.params or {}
    n = float(p.get("n", 0.2))
    k = float(p.get("k", 3.0))
    d = float(p.get("d", 0.0))

    return Layer(complex(n, k), d, name=st.type)


@factory.register("Anisotropic")
def build_anisotropic(st: LayerState):

    p = st.params or {}
    n0 = float(p.get("n0", 1.2))
    n1 = float(p.get("n1", 1.4))
    theta_deg = float(p.get("theta_deg", 45.0))
    d = float(p.get("d", 0.0))

    ani = Anisotropic(n0=n0, n1=n1, anisotropic_angle=theta_deg)
    return Layer(ani, d, name=st.type)

# Cauchy
@factory.register("Cauchy")
def build_cauchy(st: LayerState):
    print(st)
    p = st.params or {}
    
    disp_model = CauchyDispersion(
        A=float(p.get("A", 1.3)),
        B=float(p.get("B", 0.02)), 
        C=float(p.get("C", 0.001))
    )
    
    # Если Layer требует Material, то: disp = Material(dispersion=disp_model)
    # Но исходя из build_dispersion, передаем объект напрямую:
    return Layer(
        disp_model, 
        float(p.get("d", 0.0)),  # Позиционный аргумент
        name=st.type
    )

# LorentzDrude (1 осциллятор)
@factory.register("LorentzDrude")
def build_lorentz_drude(st: LayerState):
    # print(st) # Можно добавить для единообразия, если нужно
    p = st.params or {}
    
    disp_model = LorentzDrudeDispersion(
        wp=float(p.get("wp", 1.2e15)),
        wt=float(p.get("wt", 0.1e15)),
        w0=float(p.get("w0", 1.0e15)),
        amplitude=float(p.get("ampl", 0.8)),
        eps_inf=float(p.get("eps_inf", 1.0))
    )

    return Layer(
        disp_model, 
        float(p.get("d", 0.0)),  # Позиционный аргумент
        name=st.type
    )


@factory.register("Dispersion")
def build_dispersion(st: LayerState):
    p = st.params or {}
    material = str(p.get("material", "BK7"))
    d = float(p.get("d", 0.0))

    disp = MaterialDispersion(material)
    return Layer(disp, d, name=st.type)



@factory.register("Gradient")
def build_gradient(st: LayerState):
    p = st.params or {}
    d = float(p.get("d", 0.0))
    prof = p.get("profile", {})
    if not isinstance(prof, dict):
        prof = {}
    
    
    x = np.asarray(prof.get("x", []), dtype=float)
    re = np.asarray(prof.get("re", []), dtype=float)
    im = np.asarray(prof.get("im", []), dtype=float) if isinstance(prof.get("im", None), list) else np.zeros_like(re)
    mode = str(prof.get("mode", "real")).lower()

    # ожидаем, что x/re/im уже отсортированы окном; но для PCHIP x должен быть монотонный [page:1]
    f_re = PchipInterpolator(x, re, extrapolate=True)
    f_im = PchipInterpolator(x, im, extrapolate=True)

    def n_callable(t):
        tt = np.asarray(t, dtype=float)
        nre = np.asarray(f_re(tt), dtype=float)
        nim = np.asarray(f_im(tt), dtype=float) if mode == "complex" else 0.0 * nre
        out = nre + 1j * nim
        return complex(out) if out.shape == () else out

    return Layer(n_callable, d, name=st.type)



# ---------------------------------------------------------------------
# Главное окно
# ---------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("SPR curves (PyQt)")

        mb = self.menuBar()
        mb.clear()

        m_exp = mb.addMenu("Эксперимент")

        act_load = m_exp.addAction("Загрузить схему…")
        act_save = m_exp.addAction("Сохранить схему…")
        act_reset = m_exp.addAction("Сбросить схему")
        m_exp.addSeparator()
        act_exit = m_exp.addAction("Закрыть приложение")

        act_load.triggered.connect(self._on_load_scheme)
        act_save.triggered.connect(self._on_save_scheme)
        act_reset.triggered.connect(self._on_reset_scheme)
        act_exit.triggered.connect(self.close)


        m_plots = mb.addMenu("Графики")
        
        act_img = m_plots.addAction("Сохранить как картинку…")
        act_data = m_plots.addAction("Сохранить как данные…")
        m_plots.addSeparator()
        act_view = m_plots.addAction("Настройки отображения…")
        
        act_img.triggered.connect(self._on_save_plot_image)
        act_data.triggered.connect(self._on_save_plot_data)
        act_view.triggered.connect(self._on_plot_display_settings)




        # --- дальше твой существующий код создания центральной части ---
        central = QWidget(self)
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        splitter = QSplitter(Qt.Horizontal, central)

        self.plots = PlotTabsWidget(splitter)
        self.layers = LayersPanel(splitter)

        splitter.addWidget(self.plots)
        splitter.addWidget(self.layers)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 4)

        root.addWidget(splitter, 1)

        self.plots.requestRender.connect(self.draw_current)
        self.layers.layersChanged.connect(self._on_layers_changed)

        self.statusBar().showMessage("Ready")
        # ВАЖНО: запоминаем состояние “как при старте”
        self._loaded_scheme_state = self._collect_scheme_state()
        self.plots.canvas.nearestPointChanged.connect(self._on_nearest_point)
        # --- app settings (stage 1) ---
        self.app_settings = self._load_app_settings()
        self._apply_app_settings_to_ui()

    def _on_nearest_point(self, x: float, y: float, idx: int) -> None:
        c = self.plots.canvas
        xl = getattr(c, "_x_label", "x")
        yl = getattr(c, "_y_label", "y")
    
        msg = f"{xl} = {x:.6g}    {yl} = {y:.6g}"
    
        p = getattr(c, "_param", None)
        pl = getattr(c, "_param_label", "")
    
        # Параметр показываем только если он не дублирует оси
        if (
            isinstance(pl, str) and pl
            and pl != xl and pl != yl
            and isinstance(p, np.ndarray) and 0 <= idx < p.size
        ):
            msg += f"    {pl} = {float(p[idx]):.6g}"
    
        self.statusBar().showMessage(msg)

    def draw_current(self) -> None:
        # 1) Слои -> Unit
        page = self.plots.current_page()
        kind = page.spec.kind
        N = self.plots.n_points_for_kind(kind)
        data = self.layers.list.export_layers()
        try:
            states = layer_states_from_export(data)
            spp_layers = factory.build_all(states)
        except Exception as e:
            QMessageBox.critical(self, "Слои", f"Не удалось собрать слои:\n{e}")
            return
    
        Unit = ExperimentSPR(polarization="p")
        for lay in spp_layers:
            Unit.add(lay)
    
        # 2) Активная вкладка
        page = self.plots.current_page()
        kind = page.spec.kind
        # N = 400
    
        # --- сканирование по углу ---
        if kind in ("re_theta", "im_theta"):
            wl = float(page.f_lambda.valueSI())   # метры
            tmin = float(page.f_tmin.valueSI())   # градусы
            tmax = float(page.f_tmax.valueSI())   # градусы
    
            if wl <= 0 or not np.isfinite(wl):
                QMessageBox.warning(self, "Параметры", "λ должна быть > 0.")
                return
            if not (np.isfinite(tmin) and np.isfinite(tmax) and tmax > tmin):
                QMessageBox.warning(self, "Параметры", "Нужно: θmax > θmin.")
                return
    
            Unit.wavelength = wl
            theta = np.linspace(tmin, tmax, N, dtype=float)
            
            Unit.show_info()
            if kind == "re_theta":
                try:
                    R = Unit.R(angle_range=theta, is_complex=False)
                except Exception as e:
                    QMessageBox.critical(self, "SPPPy", f"Ошибка Unit.R(angle_range):\n{e}")
                    return
    
                x_plot = theta
                y_plot = np.asarray(R, dtype=float)
                x_label = "θ, °"
                y_label = "R"
                title = kind
                param = theta
                param_label = "θ, °"
    
            else:  # im_theta
                try:
                    r = Unit.R(angle_range=theta, is_complex=True)
                except Exception as e:
                    QMessageBox.critical(self, "SPPPy", f"Ошибка Unit.R(angle_range):\n{e}")
                    return
    
                r = np.asarray(r)
                x_plot = np.real(r)
                y_plot = np.imag(r)
                x_label = "Re(r)"
                y_label = "Im(r)"
                title = kind
                param = theta
                param_label = "θ, °"
    
        # --- сканирование по длине волны ---
        elif kind in ("re_lambda", "im_lambda"):
            theta0 = float(page.f_theta.valueSI())  # градусы
            lmin = float(page.f_lmin.valueSI())     # метры
            lmax = float(page.f_lmax.valueSI())     # метры
    
            if not (np.isfinite(lmin) and np.isfinite(lmax) and lmax > lmin and lmin > 0):
                QMessageBox.warning(self, "Параметры", "Нужно: 0 < λmin < λmax.")
                return
    
            Unit.incidence_angle = theta0
            wl = np.linspace(lmin, lmax, N, dtype=float)
            wl_nm = wl / nm
    
            if kind == "re_lambda":
                try:
                    R = Unit.R(wl_range=wl, is_complex=False)
                except Exception as e:
                    QMessageBox.critical(self, "SPPPy", f"Ошибка Unit.R(wl_range):\n{e}")
                    return
    
                x_plot = wl_nm
                y_plot = np.asarray(R, dtype=float)
                x_label = "λ, nm"
                y_label = "R"
                title = kind
                param = wl_nm
                param_label = "λ, nm"
    
            else:  # im_lambda
                try:
                    r = Unit.R(wl_range=wl, is_complex=True)
                except Exception as e:
                    QMessageBox.critical(self, "SPPPy", f"Ошибка Unit.R(wl_range):\n{e}")
                    return
    
                r = np.asarray(r)
                x_plot = np.real(r)
                y_plot = np.imag(r)
                x_label = "Re(r)"
                y_label = "Im(r)"
                title = kind
                param = wl_nm
                param_label = "λ, nm"
    
        else:
            QMessageBox.warning(self, "Вкладка", f"Неизвестный kind: {kind}")
            return
    
        # 3) Кэш + отрисовка
        # title: человекочитаемый, без kind
        title = page.spec.title
        
        # В real-вкладках param не нужен (иначе дублируем θ/λ в статусбаре)
        if kind in ("re_theta", "re_lambda"):
            param_to_show = None
            param_label_to_show = ""
        else:
            param_to_show = param
            param_label_to_show = param_label
        
        title = ""  # ничего не рисуем внутри canvas — вкладка и так видна
        
        self.plots.cache_set(
            kind,
            x_plot,
            y_plot,
            x_label=x_label,
            y_label=y_label,
            title=title,
            param=param_to_show,
            param_label=param_label_to_show,
        )
        self.plots.show_cached_or_axes()


    def _on_layers_changed(self) -> None:
        if self.plots.autodraw_enabled():
            self.plots.requestRender.emit()
    
    
    def _collect_scheme_state(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        data["layers"] = self.layers.list.export_layers()
        data.update(self.plots.export_scan_params())
        return data


    def _apply_scheme_state(self, data: Dict[str, Any]) -> None:
        if not isinstance(data, dict):
            return

        layers_blob = data.get("layers", {})
        if isinstance(layers_blob, dict):
            self.layers.list.import_layers(layers_blob)

        # Параметры вкладок (theta_scan / lambda_scan)
        self.plots.import_scan_params(data)

        # Перерисуем (если хочешь — можно сделать условно по Autodraw)
        self.draw_current()


    def _on_save_scheme(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить схему",
            "",
            "Scheme (*.json);;All files (*.*)",
        )
        if not path:
            return

        data = self._collect_scheme_state()

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            QMessageBox.critical(self, "Сохранение", f"Не удалось сохранить файл:\n{e}")


    def _on_load_scheme(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Загрузить схему",
            "",
            "Scheme (*.json);;All files (*.*)",
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Загрузка", f"Не удалось прочитать файл:\n{e}")
            return

        if not isinstance(data, dict):
            QMessageBox.warning(self, "Загрузка", "Некорректный формат файла (ожидался JSON-объект).")
            return

        # применяем
        self._apply_scheme_state(data)

        # запоминаем “как было при загрузке” для Reset
        self._loaded_scheme_state = copy.deepcopy(data)


    def _current_canvas_curve(self):
        c = self.plots.canvas
    
        x = np.asarray(getattr(c, "_x", np.array([], dtype=float)), dtype=float)
        y = np.asarray(getattr(c, "_y", np.array([], dtype=float)), dtype=float)
    
        if x.size == 0 or y.size == 0:
            raise RuntimeError("График отсутствует: сначала нужно выполнить Draw.")
    
        if x.size != y.size:
            raise RuntimeError("Некорректные данные графика: размерности x и y не совпадают.")
    
        x_label = str(getattr(c, "_x_label", "x"))
        y_label = str(getattr(c, "_y_label", "y"))
        title = str(getattr(c, "_title", ""))
    
        return x, y, x_label, y_label, title
    
    
    def _on_save_plot_image(self) -> None:
        try:
            self._current_canvas_curve()
        except Exception as e:
            QMessageBox.warning(self, "Сохранение", str(e))
            return
    
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить график как картинку",
            "",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;TIFF (*.tif *.tiff);;All files (*.*)",
        )
        if not path:
            return
    
        pm = self.plots.canvas.grab()
        ok = pm.save(path)
        if not ok:
            QMessageBox.critical(self, "Сохранение", "Не удалось сохранить изображение.")
    
    
    def _on_save_plot_data(self) -> None:
        try:
            x, y, x_label, y_label, title = self._current_canvas_curve()
        except Exception as e:
            QMessageBox.warning(self, "Сохранение", str(e))
            return
    
        path, selected = QFileDialog.getSaveFileName(
            self,
            "Сохранить данные графика",
            "",
            "NumPy (npz) (*.npz);;NumPy (npy) (*.npy);;CSV (*.csv);;TSV (*.txt);;All files (*.*)",
        )
        if not path:
            return
    
        try:
            low = path.lower()
    
            # 1) npz: сохраняем x/y + подписи
            if low.endswith(".npz") or "npz" in selected.lower():
                np.savez(path, x=x, y=y, x_label=x_label, y_label=y_label, title=title)
                return
    
            # 2) npy: Nx2
            if low.endswith(".npy") or "npy" in selected.lower():
                arr = np.column_stack([x, y])
                np.save(path, arr)
                return
    
            # 3) CSV/TSV: 2 колонки
            if low.endswith(".csv") or "csv" in selected.lower():
                header = f"{x_label},{y_label}"
                np.savetxt(path, np.column_stack([x, y]), delimiter=",", header=header, comments="")
                return
    
            if low.endswith(".txt") or "tsv" in selected.lower():
                header = f"{x_label}\t{y_label}"
                np.savetxt(path, np.column_stack([x, y]), delimiter="\t", header=header, comments="")
                return
    
            # fallback: если расширение непонятно — сохраняем npz
            np.savez(path, x=x, y=y, x_label=x_label, y_label=y_label, title=title)
    
        except Exception as e:
            QMessageBox.critical(self, "Сохранение", f"Не удалось сохранить данные:\n{e}")
    

    def _default_app_settings(self) -> dict:
        # ВАЖНО: это дефолты "если файла нет / битый файл"
        # N пока совпадает с текущим поведением draw_current(): N=400 [file:26]
        return {
            "version": 1,
            "plots": {
                "autodraw": True,  # позже чекбокс перенесём в диалог
                "limits": {
                    "enabled": False,
                    "real": {"ymin": 0.0, "ymax": 1.0},
                    "complex": {"xmin": 0.0, "xmax": 1.0, "ymin": 0.0, "ymax": 1.0},
                },
                "resolution": {
                    "re_theta": 400,
                    "im_theta": 400,
                    "re_lambda": 400,
                    "im_lambda": 400,
                },
            },
        }
    
    def _normalize_app_settings(self, st: object) -> dict:
        # “мягкое” чтение: все отсутствующие поля заполняем дефолтами
        base = self._default_app_settings()
        if not isinstance(st, dict):
            return base
    
        def merge(dst: dict, src: dict) -> None:
            for k, v in src.items():
                if isinstance(v, dict) and isinstance(dst.get(k), dict):
                    merge(dst[k], v)
                else:
                    dst[k] = v
    
        merge(base, st)
    
        # минимальная валидация типов/границ (без фанатизма на этапе 1)
        plots = base.get("plots", {})
        plots["autodraw"] = bool(plots.get("autodraw", True))
    
        res = plots.get("resolution", {})
        for kind in ("re_theta", "im_theta", "re_lambda", "im_lambda"):
            try:
                n = int(res.get(kind, 400))
            except Exception:
                n = 400
            if n < 2:
                n = 2
            res[kind] = n
        plots["resolution"] = res
    
        lim = plots.get("limits", {})
        lim["enabled"] = bool(lim.get("enabled", False))
        plots["limits"] = lim
    
        base["plots"] = plots
        return base
    
    def _load_app_settings(self) -> dict:
        path = APP_SETTINGS_PATH
        if not path.exists():
            return self._default_app_settings()
    
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return self._default_app_settings()
    
        return self._normalize_app_settings(data)
    
    def _save_app_settings(self, st: dict) -> None:
        # На этапе 1 НЕ вызываем автоматически.
        path = APP_SETTINGS_PATH
        st = self._normalize_app_settings(st)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(st, f, ensure_ascii=False, indent=2)
    
    def _apply_app_settings_to_ui(self) -> None:
        plots = self.app_settings.get("plots", {})
        # ВАЖНО: прокидываем ВСЕ plot-настройки (autodraw, limits, resolution) в PlotTabsWidget
        self.plots.set_plot_settings(plots)


    def _on_plot_display_settings(self) -> None:
        dlg = PlotDisplaySettingsDialog(copy.deepcopy(self.app_settings), self)
    
        def apply_from_dialog(st: dict) -> None:
            if not isinstance(st, dict):
                return
    
            self.app_settings = self._normalize_app_settings(st)
    
            try:
                self._save_app_settings(self.app_settings)
            except Exception as e:
                QMessageBox.critical(self, "Настройки", f"Не удалось сохранить файл настроек:\n{e}")
                return
    
            self._apply_app_settings_to_ui()
    
            plots = self.app_settings.get("plots", {})
            if bool(plots.get("autodraw", True)):
                self.draw_current()
    
        dlg.settingsApplied.connect(apply_from_dialog)
        dlg.exec()


    

    def _on_reset_scheme(self) -> None:
        st = getattr(self, "_loaded_scheme_state", None)
        if not isinstance(st, dict):
            return
        self._apply_scheme_state(copy.deepcopy(st))

    # Пример будущего API (пока заглушка)
    def recalc_and_redraw(self) -> None:
        """Потом: собрать state -> прогнать SPPy -> обновить отрисовку активной вкладки."""
        pass


def main(argv=None) -> int:
    app = QApplication(sys.argv if argv is None else argv)
    win = MainWindow()
    win.resize(1200, 720)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
