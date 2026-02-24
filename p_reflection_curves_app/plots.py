# -*- coding: utf-8 -*-
"""
p_reflection_curves_app/plots.py

Левая часть приложения: вкладки с отрисовками.

Сейчас это каркас (без реальной графики), но с правильной архитектурой:
- QTabWidget с 5 вкладками:
  1) Real R(θ)
  2) Imaginary R(θ)
  3) Real R(λ)
  4) Imaginary R(λ)
  5) Gradient profile
- Каждая вкладка имеет:
  - верхнюю строку настроек (top bar)
  - область "canvas" (пока заглушка QFrame)
  - нижнюю строку кнопок/режимов (bottom bar)

Дальше ты сможешь:
- заменить PlotCanvasStub на matplotlib/pyqtgraph виджет,
- в каждой вкладке расширить настройки (поля/комбобоксы),
- подключить методы update_from_experiment(...) и render(...).

Файл не зависит от слоёв напрямую: связь делается в MainWindow (по сигналу layersChanged).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QSignalBlocker, QPointF, QRectF
from PyQt5.QtGui import QPainter, QPen, QColor, QPolygonF
from PyQt5.QtWidgets import (
    QWidget,
    QTabWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QFrame,
    QSizePolicy,
    QPushButton,
    QComboBox,
)

from .layer import ParamField
from .digit_number import DigitFormat



class SharedParam(QObject):
    stateChanged = pyqtSignal(dict, object)  # (state, source)

    def __init__(self, initial: dict, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._state = dict(initial)

    def state(self) -> dict:
        return dict(self._state)

    def set_state(self, st: dict, *, source=None) -> None:
        st = dict(st)
        if st == self._state:
            return
        self._state = st
        self.stateChanged.emit(self.state(), source)


def bind_paramfield(shared: SharedParam, field: ParamField) -> None:
    field.set_ui_state(shared.state())

    def on_field_changed(_v=None):
        shared.set_state(field.ui_state(), source=field)

    field.valueChanged.connect(on_field_changed)

    def on_shared_changed(st: dict, source):
        if source is field:
            return
        blockers = [QSignalBlocker(field.edit)]
        if field.unit_box is not None:
            blockers.append(QSignalBlocker(field.unit_box))
        field.set_ui_state(st)

    shared.stateChanged.connect(on_shared_changed)



# ============================================================
# Общие "мелочи" для вкладок
# ============================================================

@dataclass(frozen=True)
class PlotTabSpec:
    tab_text: str     # текст на ярлыке вкладки
    title: str        # "человеческое" название в заглушках/логах
    kind: str         # ключ/тип (на будущее для маршрутизации)


class PlotCanvasStub(QFrame):
    """
    Заглушка области отрисовки.

    Позже можно заменить на:
    - matplotlib FigureCanvasQTAgg,
    - pyqtgraph PlotWidget,
    - или OpenGL-виджет.
    """

    def __init__(self, title: str, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # Важно: рамку делаем общей на весь левый блок (PlotTabsWidget),
        # а canvas оставляем без своей рамки, чтобы не было "двойных" границ.
        self.setFrameShape(QFrame.NoFrame)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)

        lbl = QLabel(f"Область отрисовки: {title}", self)
        lbl.setAlignment(Qt.AlignCenter)
        lay.addWidget(lbl)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


# ============================================================
# Базовый класс одной вкладки
# ============================================================

class PlotTabBase(QWidget):
    """
    Страница вкладки: только верхняя панель параметров.

    kind:
      - re_theta / im_theta : λ, θ_min, θ_max
      - re_lambda / im_lambda : θ, λ_min, λ_max
    """
    requestRender = pyqtSignal()

    def __init__(
        self,
        spec: PlotTabSpec,
        *,
        model_theta: Dict[str, SharedParam],
        model_lambda: Dict[str, SharedParam],
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self.spec = spec

        # Верхняя панель
        self.top_bar = QFrame(self)
        self.top_bar.setFrameShape(QFrame.StyledPanel)
        self.top_bar.setFrameShadow(QFrame.Raised)

        top_l = QHBoxLayout(self.top_bar)
        top_l.setContentsMargins(6, 4, 6, 4)
        top_l.setSpacing(8)

        # Создаём поля по kind
        if spec.kind in ("re_theta", "im_theta"):
            fmt_wl = DigitFormat(4, 3)
            fmt_th = DigitFormat(2, 2)

            self.f_lambda = ParamField(
                "λ", fmt_wl,
                units={"nm": 1e-9, "um": 1e-6},
                default_unit="nm",
                label_width=18,
                parent=self.top_bar,
            )
            self.f_tmin = ParamField(
                "θmin", fmt_th,
                units={"deg": 1.0},
                default_unit="deg",
                label_width=32,
                parent=self.top_bar,
            )
            self.f_tmax = ParamField(
                "θmax", fmt_th,
                units={"deg": 1.0},
                default_unit="deg",
                label_width=32,
                parent=self.top_bar,
            )

            bind_paramfield(model_theta["lambda"], self.f_lambda)
            bind_paramfield(model_theta["tmin"], self.f_tmin)
            bind_paramfield(model_theta["tmax"], self.f_tmax)

            for w in (self.f_lambda, self.f_tmin, self.f_tmax):
                w.valueChanged.connect(self.requestRender.emit)
                top_l.addWidget(w)

        elif spec.kind in ("re_lambda", "im_lambda"):
            fmt_wl = DigitFormat(4, 3)
            fmt_th = DigitFormat(2, 2)

            self.f_theta = ParamField(
                "θ", fmt_th,
                units={"deg": 1.0},
                default_unit="deg",
                label_width=18,
                parent=self.top_bar,
            )
            self.f_lmin = ParamField(
                "λmin", fmt_wl,
                units={"nm": 1e-9, "um": 1e-6},
                default_unit="nm",
                label_width=32,
                parent=self.top_bar,
            )
            self.f_lmax = ParamField(
                "λmax", fmt_wl,
                units={"nm": 1e-9, "um": 1e-6},
                default_unit="nm",
                label_width=32,
                parent=self.top_bar,
            )

            bind_paramfield(model_lambda["theta"], self.f_theta)
            bind_paramfield(model_lambda["lmin"], self.f_lmin)
            bind_paramfield(model_lambda["lmax"], self.f_lmax)

            for w in (self.f_theta, self.f_lmin, self.f_lmax):
                w.valueChanged.connect(self.requestRender.emit)
                top_l.addWidget(w)

        else:
            top_l.addWidget(QLabel("Unknown tab kind", self.top_bar))

        top_l.addStretch(1)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self.top_bar)

        self._experiment_state: Optional[Dict[str, Any]] = None

    def set_experiment_state(self, state: Dict[str, Any]) -> None:
        self._experiment_state = state


# ============================================================
# Контейнер вкладок (левый блок)
# ============================================================

class PlotTabsWidget(QFrame):
    # дефолтные вкладки (если tabs=None)
    DEFAULT_TABS: List[PlotTabSpec] = [
        PlotTabSpec("Real R(θ)", "Real R(θ)", "re_theta"),
        PlotTabSpec("Imaginary R(θ)", "Imaginary R(θ)", "im_theta"),
        PlotTabSpec("Real R(λ)", "Real R(λ)", "re_lambda"),
        PlotTabSpec("Imaginary R(λ)", "Imaginary R(λ)", "im_lambda"),
    ]

    requestRender = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None, tabs: Optional[List[PlotTabSpec]] = None):
        super().__init__(parent)

        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        # ---- models (shared) ----
        self.model_theta = {
            "lambda": SharedParam({"scaled": 450000, "unit": "nm", "fmt": 4, "dec": 3}, self),
            "tmin":   SharedParam({"scaled": 0,      "unit": "deg", "fmt": 2, "dec": 2}, self),
            "tmax":   SharedParam({"scaled": 9000,   "unit": "deg", "fmt": 2, "dec": 2}, self),
        }

        self.model_lambda = {
            "theta": SharedParam({"scaled": 3000,   "unit": "deg", "fmt": 2, "dec": 2}, self),
            "lmin":  SharedParam({"scaled": 250000, "unit": "nm",  "fmt": 4, "dec": 3}, self),
            "lmax":  SharedParam({"scaled": 800000, "unit": "nm",  "fmt": 4, "dec": 3}, self),
        }

        # --- tabs ---
        self.tabs = QTabWidget(self)
        self.tabs.setDocumentMode(True)
        self.tabs.setStyleSheet("QTabWidget::pane { border: 0; }")

        self._pages: List[PlotTabBase] = []
        self._cache: Dict[str, Dict[str, object]] = {}
        self._plot_settings: Dict[str, Any] = {}

        spec = tabs if tabs is not None else list(self.DEFAULT_TABS)

        for t in spec:
            page = PlotTabBase(t, model_theta=self.model_theta, model_lambda=self.model_lambda, parent=self.tabs)
            page.requestRender.connect(self._on_page_request_render)
            self.tabs.addTab(page, t.tab_text)
            self._pages.append(page)

        # --- canvas + bottom ---
        self.canvas = PlotCanvasQt(self)

        self.bottom_bar = QFrame(self)
        self.bottom_bar.setFrameShape(QFrame.StyledPanel)
        self.bottom_bar.setFrameShadow(QFrame.Raised)

        bot_l = QHBoxLayout(self.bottom_bar)
        bot_l.setContentsMargins(6, 4, 6, 4)
        bot_l.setSpacing(10)

        self.btn_draw = QPushButton("Draw", self.bottom_bar)
        self.btn_draw.setFixedHeight(24)
        bot_l.addWidget(self.btn_draw)

        self._autodraw_enabled = True
        bot_l.addStretch(1)

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(0)

        root.addWidget(self.tabs)
        root.addWidget(self.canvas, 1)
        root.addWidget(self.bottom_bar)

        # wiring
        self.btn_draw.clicked.connect(self.requestRender.emit)
        self.tabs.currentChanged.connect(self._on_tab_changed)

        self._experiment_state: Optional[Dict[str, Any]] = None

        self.show_cached_or_axes()

    def set_plot_settings(self, plots_settings: Dict[str, Any]) -> None:
        self._plot_settings = dict(plots_settings) if isinstance(plots_settings, dict) else {}

        # autodraw
        self.set_autodraw_enabled(bool(self._plot_settings.get("autodraw", True)))

        # если на экране уже есть график — перерисуем с новыми лимитами
        self.show_cached_or_axes()

    def plot_settings(self) -> Dict[str, Any]:
        return dict(self._plot_settings)

    def n_points_for_kind(self, kind: str) -> int:
        res = self._plot_settings.get("resolution", {})
        if not isinstance(res, dict):
            return 400
        return int(res.get(kind, 400))

    def _limits_for_kind(self, kind: str):
        """
        Возвращает (xlim, ylim) или (None, None).
    
        Real вкладки: limits.real.ymin/ymax -> ylim.
        Imag (complex-plane) вкладки: limits.complex.xmin/xmax -> xlim, limits.complex.ymin/ymax -> ylim.
        """
        lim = self._plot_settings.get("limits", {})
        if not isinstance(lim, dict) or not bool(lim.get("enabled", False)):
            return None, None
    
        real = lim.get("real", {})
        cplx = lim.get("complex", {})
    
        xlim = None
        ylim = None
    
        if kind.startswith("re_") and isinstance(real, dict):
            y0 = real.get("ymin", None)
            y1 = real.get("ymax", None)
            if isinstance(y0, (int, float)) and isinstance(y1, (int, float)) and (y1 > y0):
                ylim = (float(y0), float(y1))
    
        if kind.startswith("im_") and isinstance(cplx, dict):
            # Y limits
            y0 = cplx.get("ymin", None)
            y1 = cplx.get("ymax", None)
            if isinstance(y0, (int, float)) and isinstance(y1, (int, float)) and (y1 > y0):
                ylim = (float(y0), float(y1))
    
            # X limits (ВКЛЮЧАЕМ наконец)
            x0 = cplx.get("xmin", None)
            x1 = cplx.get("xmax", None)
            if isinstance(x0, (int, float)) and isinstance(x1, (int, float)) and (x1 > x0):
                xlim = (float(x0), float(x1))
    
        return xlim, ylim


    def _on_page_request_render(self) -> None:
        if self.autodraw_enabled():
            self.requestRender.emit()

    def _on_tab_changed(self, _index: int) -> None:
        self.show_cached_or_axes()

    def set_experiment_state(self, state: Dict[str, Any]) -> None:
        self._experiment_state = state
        for p in self._pages:
            p.set_experiment_state(state)

    def current_page(self) -> PlotTabBase:
        w = self.tabs.currentWidget()
        assert isinstance(w, PlotTabBase)
        return w

    def set_autodraw_enabled(self, enabled: bool) -> None:
        self._autodraw_enabled = bool(enabled)

    def autodraw_enabled(self) -> bool:
        return bool(getattr(self, "_autodraw_enabled", True))

    def _default_axes_for_kind(self, kind: str) -> Tuple[str, str, str]:
        # (x_label, y_label, title)
        if kind == "re_theta":
            return ("θ, °", "R", "Real R(θ)")
        if kind == "im_theta":
            return ("θ, °", "Im(r)", "Imaginary R(θ)")
        if kind == "re_lambda":
            return ("λ, nm", "R", "Real R(λ)")
        if kind == "im_lambda":
            return ("λ, nm", "Im(r)", "Imaginary R(λ)")
        return ("x", "y", "")

    def cache_set(
        self,
        kind: str,
        x,
        y,
        *,
        x_label: str,
        y_label: str,
        title: str,
        param=None,
        param_label: str = "",
    ) -> None:
        self._cache[kind] = {
            "x": np.asarray(x, dtype=float),
            "y": np.asarray(y, dtype=float),
            "x_label": str(x_label),
            "y_label": str(y_label),
            "title": str(title),
            "param": None if param is None else np.asarray(param, dtype=float),
            "param_label": str(param_label),
        }

    def show_cached_or_axes(self) -> None:
        kind = self.current_page().spec.kind

        if kind in self._cache:
            c = self._cache[kind]
            xlim, ylim = self._limits_for_kind(kind)

            self.canvas.plot_xy(
                c["x"], c["y"],
                x_label=c["x_label"],
                y_label=c["y_label"],
                title=c["title"],
                grid=True,
                param=c.get("param", None),
                param_label=c.get("param_label", ""),
                xlim=xlim,
                ylim=ylim,
            )

        else:
            x_label, y_label, title = self._default_axes_for_kind(kind)
            self.canvas.show_axes(x_label=x_label, y_label=y_label, title=title, grid=True)

    def export_scan_params(self) -> Dict[str, Any]:
        return {
            "theta_scan": {
                "lambda": dict(self.model_theta["lambda"].state()),
                "tmin":   dict(self.model_theta["tmin"].state()),
                "tmax":   dict(self.model_theta["tmax"].state()),
            },
            "lambda_scan": {
                "theta": dict(self.model_lambda["theta"].state()),
                "lmin":  dict(self.model_lambda["lmin"].state()),
                "lmax":  dict(self.model_lambda["lmax"].state()),
            },
        }

    def import_scan_params(self, data: Dict[str, Any]) -> None:
        if not isinstance(data, dict):
            return

        th = data.get("theta_scan", {})
        if isinstance(th, dict):
            for key in ("lambda", "tmin", "tmax"):
                st = th.get(key, None)
                if isinstance(st, dict) and key in self.model_theta:
                    self.model_theta[key].set_state(st, source=self)

        lm = data.get("lambda_scan", {})
        if isinstance(lm, dict):
            for key in ("theta", "lmin", "lmax"):
                st = lm.get(key, None)
                if isinstance(st, dict) and key in self.model_lambda:
                    self.model_lambda[key].set_state(st, source=self)


# --- imports needed (put near top of plots.py) ---


class PlotCanvasQt(QFrame):
    """
    Простой самописный canvas:
    - grid + подписи тиков
    - кривая polyline
    - маркер ближайшей точки (по X)
    """
    nearestPointChanged = pyqtSignal(float, float, int)  # x, y, idx

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.NoFrame)
        self.setMouseTracking(True)

        self._x = np.array([], dtype=float)
        self._y = np.array([], dtype=float)

        self._param = None
        self._param_label = ""
        self._poly_i: List[int] = []

        self._x_label = "x"
        self._y_label = "y"
        self._title = ""
        self._grid = True

        self._plot_rect = QRectF()
        self._poly = QPolygonF()
        self._nearest_idx = None
        self._poly_i = []              # poly_index -> data_index
        self._nearest_poly_idx = None  # индекс в self._poly
        self._nearest_data_idx = None  # индекс в self._x/self._y
        self._nearest_data_idx = None
        self._nearest_poly_idx = None
        
        self._xmin, self._xmax = 0.0, 1.0
        self._ymin, self._ymax = 0.0, 1.0

        self._snap_px = 12.0  # радиус прилипания к точке (в пикселях)

        # поля под подписи (в px)
        self._m_left = 65.0
        self._m_right = 15.0
        self._m_top = 22.0
        self._m_bottom = 45.0

        self._target_ticks = 6

    # ---------------- public API ----------------

    def show_axes(self, *, x_label="x", y_label="y", title="", grid=True) -> None:
        self._x = np.array([], dtype=float)
        self._y = np.array([], dtype=float)
        self._x_label = str(x_label)
        self._y_label = str(y_label)
        self._title = str(title)
        self._grid = bool(grid)
    
        self._param = None
        self._param_label = ""
        self._poly_i = []
    
        self._xmin, self._xmax = 0.0, 1.0
        self._ymin, self._ymax = 0.0, 1.0
        self._nearest_idx = None
        self._nearest_data_idx = None
        self._nearest_poly_idx = None
        self._poly = QPolygonF()
    
        self._update_plot_rect()
        self.update()


    def plot_xy(self, x, y, *, x_label="x", y_label="y", title="", grid=True,
                param=None, param_label="", xlim=None, ylim=None) -> None:
        self._x = np.asarray(x, dtype=float)
        self._y = np.asarray(y, dtype=float)
        self._x_label = str(x_label)
        self._y_label = str(y_label)
        self._title = str(title)
        self._grid = bool(grid)
    
        self._param = None if param is None else np.asarray(param, dtype=float)
        self._param_label = str(param_label)
        self._poly_i = []
    
        self._nearest_idx = None
        self._nearest_data_idx = None
        self._nearest_poly_idx = None

        if self._x.size == 0:
            self._xmin, self._xmax = 0.0, 1.0
            self._ymin, self._ymax = 0.0, 1.0
            self._poly = QPolygonF()
            self._update_plot_rect()
            self.update()
            return

        xmin = float(np.nanmin(self._x))
        xmax = float(np.nanmax(self._x))
        ymin = float(np.nanmin(self._y))
        ymax = float(np.nanmax(self._y))
    
        # override limits from settings (if valid)
        if isinstance(xlim, (tuple, list)) and len(xlim) == 2:
            x0, x1 = xlim
            if isinstance(x0, (int, float)) and isinstance(x1, (int, float)) and (x1 > x0) and np.isfinite(x0) and np.isfinite(x1):
                xmin, xmax = float(x0), float(x1)
        
        if isinstance(ylim, (tuple, list)) and len(ylim) == 2:
            y0, y1 = ylim
            if isinstance(y0, (int, float)) and isinstance(y1, (int, float)) and (y1 > y0) and np.isfinite(y0) and np.isfinite(y1):
                ymin, ymax = float(y0), float(y1)
                pad = 0.0  # если лимиты заданы явно — не раздуваем
        

        if (not np.isfinite(xmin)) or (not np.isfinite(xmax)) or (xmax == xmin):
            xmin, xmax = 0.0, 1.0
        if (not np.isfinite(ymin)) or (not np.isfinite(ymax)) or (ymax == ymin):
            ymin, ymax = 0.0, 1.0
    
        pad = 0.05 * (ymax - ymin)
        ymin -= pad
        ymax += pad
    
        self._xmin, self._xmax = xmin, xmax
        self._ymin, self._ymax = ymin, ymax
    
        self._update_plot_rect()
        self._rebuild_poly()
        self.update()

    # ---------------- geometry/cache ----------------

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._update_plot_rect()
        self._rebuild_poly()

    def _update_plot_rect(self) -> None:
        w = float(max(1, self.width()))
        h = float(max(1, self.height()))
        self._plot_rect = QRectF(
            self._m_left,
            self._m_top,
            max(1.0, w - self._m_left - self._m_right),
            max(1.0, h - self._m_top - self._m_bottom),
        )

    def _to_px(self, xv: float, yv: float) -> QPointF:
        pr = self._plot_rect
        xr = (self._xmax - self._xmin)
        yr = (self._ymax - self._ymin)
        xpx = pr.left() + (xv - self._xmin) / xr * pr.width()
        ypx = pr.bottom() - (yv - self._ymin) / yr * pr.height()
        return QPointF(float(xpx), float(ypx))

    def _rebuild_poly(self) -> None:
        if self._x.size == 0:
            self._poly = QPolygonF()
            self._poly_i = []
            return
    
        poly = QPolygonF()
        idxmap: List[int] = []
    
        for i, (xv, yv) in enumerate(zip(self._x, self._y)):
            if np.isfinite(xv) and np.isfinite(yv):
                poly.append(self._to_px(float(xv), float(yv)))
                idxmap.append(i)
    
        self._poly = poly
        self._poly_i = idxmap



    # ---------------- ticks ----------------

    def _nice_step(self, vmin: float, vmax: float, target_ticks: int) -> float:
        span = float(vmax - vmin)
        if not math.isfinite(span) or span <= 0:
            return 1.0

        raw = span / max(1, int(target_ticks))
        p10 = 10.0 ** math.floor(math.log10(raw))
        m = raw / p10

        if m <= 1.0:
            nice = 1.0
        elif m <= 2.0:
            nice = 2.0
        elif m <= 5.0:
            nice = 5.0
        else:
            nice = 10.0

        return nice * p10

    def _make_ticks(self, vmin: float, vmax: float, target_ticks: int):
        step = self._nice_step(vmin, vmax, target_ticks)
        if step <= 0 or not math.isfinite(step):
            return []

        start = math.floor(vmin / step) * step
        end = math.ceil(vmax / step) * step

        ticks = []
        t = start
        # защитимся от бесконечного цикла из-за float
        for _ in range(1000):
            if t > end + 0.5 * step:
                break
            if t >= vmin - 1e-12 * abs(vmax - vmin) and t <= vmax + 1e-12 * abs(vmax - vmin):
                ticks.append(t)
            t += step
        return ticks

    def _fmt_tick(self, v: float) -> str:
        return f"{v:.6g}"

    # ---------------- painting ----------------

    def paintEvent(self, e) -> None:
        super().paintEvent(e)

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)

        # фон
        p.fillRect(self.rect(), QColor(255, 255, 255))

        pr = self._plot_rect
        if pr.width() <= 1 or pr.height() <= 1:
            self._update_plot_rect()
            pr = self._plot_rect

        # рамка области графика
        p.setPen(QPen(QColor(180, 180, 180), 1))
        p.drawRect(pr)

        # заголовок
        p.setPen(QPen(QColor(30, 30, 30), 1))

        # подписи осей (названия)
        p.drawText(int(pr.left()), int(self.height() - 10), self._x_label)
        p.save()
        p.translate(16, pr.center().y())
        p.rotate(-90)
        p.drawText(0, 0, self._y_label)
        p.restore()

        # тики
        xticks = self._make_ticks(self._xmin, self._xmax, self._target_ticks)
        yticks = self._make_ticks(self._ymin, self._ymax, self._target_ticks)

        xr = (self._xmax - self._xmin)
        yr = (self._ymax - self._ymin)

        def x_to_px(xv: float) -> float:
            return pr.left() + (xv - self._xmin) / xr * pr.width()

        def y_to_px(yv: float) -> float:
            return pr.bottom() - (yv - self._ymin) / yr * pr.height()

        fm = p.fontMetrics()

        # сетка
        if self._grid:
            p.setPen(QPen(QColor(230, 230, 230), 1))
            for xv in xticks:
                xg = x_to_px(xv)
                p.drawLine(QPointF(xg, pr.top()), QPointF(xg, pr.bottom()))
            for yv in yticks:
                yg = y_to_px(yv)
                p.drawLine(QPointF(pr.left(), yg), QPointF(pr.right(), yg))

        # короткие риски на рамке + подписи
        p.setPen(QPen(QColor(90, 90, 90), 1))

        tick_len = 4.0

        # X ticks + labels
        for xv in xticks:
            xg = x_to_px(xv)
            # риск на нижней рамке
            p.drawLine(QPointF(xg, pr.bottom()), QPointF(xg, pr.bottom() + tick_len))

            txt = self._fmt_tick(xv)
            w = fm.horizontalAdvance(txt)
            h = fm.height()
            rect = QRectF(xg - 0.5 * w - 2, pr.bottom() + tick_len + 2, w + 4, h)
            p.drawText(rect, Qt.AlignHCenter | Qt.AlignTop, txt)

        # Y ticks + labels
        for yv in yticks:
            yg = y_to_px(yv)
            # риск на левой рамке
            p.drawLine(QPointF(pr.left() - tick_len, yg), QPointF(pr.left(), yg))

            txt = self._fmt_tick(yv)
            h = fm.height()
            rect = QRectF(0, yg - 0.5 * h, pr.left() - tick_len - 6, h)
            p.drawText(rect, Qt.AlignRight | Qt.AlignVCenter, txt)

        # кривая (внутри области графика)
        p.save()
        p.setClipRect(pr)
        if self._poly.size() > 1:
            p.setPen(QPen(QColor(40, 80, 200), 2))
            p.drawPolyline(self._poly)
        p.restore()

        # маркер ближайшей точки
        if self._nearest_poly_idx is not None and 0 <= self._nearest_poly_idx < self._poly.size():
            pt = self._poly[self._nearest_poly_idx]
            p.setPen(QPen(QColor(220, 50, 50), 2))
            p.setBrush(QColor(220, 50, 50))
            p.drawEllipse(pt, 4, 4)

    # ---------------- mouse ----------------

    def mouseMoveEvent(self, ev) -> None:
        super().mouseMoveEvent(ev)
    
        if self._x.size == 0 or self._poly.size() == 0:
            return
    
        pr = self._plot_rect
        pos = ev.pos()
    
        if not pr.contains(pos):
            if self._nearest_poly_idx is not None:
                self._nearest_poly_idx = None
                self._nearest_data_idx = None
                self.update()
            return
    
        best_poly_i = None
        best_d2 = None
    
        # Ищем ближайшую вершину полилинии в пикселях (работает и для немонотонного X)
        for i in range(self._poly.size()):
            pt = self._poly[i]
            dx = float(pt.x() - pos.x())
            dy = float(pt.y() - pos.y())
            d2 = dx * dx + dy * dy
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best_poly_i = i
    
        if best_poly_i is None or best_d2 is None or best_d2 > (self._snap_px * self._snap_px):
            if self._nearest_poly_idx is not None:
                self._nearest_poly_idx = None
                self._nearest_data_idx = None
                self.update()
            return
    
        # poly_index -> data_index (если были NaN/Inf, индексы могут отличаться)
        data_i = self._poly_i[best_poly_i] if best_poly_i < len(self._poly_i) else best_poly_i
    
        if (best_poly_i != self._nearest_poly_idx) or (data_i != self._nearest_data_idx):
            self._nearest_poly_idx = best_poly_i
            self._nearest_data_idx = data_i
            self.nearestPointChanged.emit(float(self._x[data_i]), float(self._y[data_i]), int(data_i))
            self.update()

