from __future__ import annotations

from typing import Dict

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QColor, QPolygonF
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel

from SPPPy import MaterialDispersion


import copy
from typing import Optional, Any

from PyQt5.QtWidgets import (
    QHBoxLayout,
    QGroupBox,
    QGridLayout,
    QCheckBox,
    QDoubleSpinBox,
    QSpinBox,
    QPushButton,
    QMessageBox,
)


from .plots import PlotCanvasQt

from typing import Tuple

from scipy.interpolate import PchipInterpolator


_GRAD_PROFILE_WINDOWS: Dict[str, "GradientProfileDialog"] = {}

_CRI_WINDOWS: Dict[str, "CRIDialog"] = {}


def show_cri_dialog(material: str, parent=None) -> None:
    key = str(material or "").strip() or "Air"
    w = _CRI_WINDOWS.get(key)
    if w is not None:
        w.show()
        w.raise_()
        w.activateWindow()
        return
    w = CRIDialog(key, parent=parent)
    _CRI_WINDOWS[key] = w
    w.destroyed.connect(lambda _obj=None, k=key: _CRI_WINDOWS.pop(k, None))
    w.show()
    w.raise_()
    w.activateWindow()


class CRIDialog(QDialog):
    def __init__(self, material: str, parent=None):
        super().__init__(parent)

        # локально, чтобы не было циклического импорта dialogs <-> plots
        from .plots import PlotCanvasQt

        class CRICanvas(PlotCanvasQt):
            hoveredIndexChanged = pyqtSignal(int)

            def __init__(self, parent=None):
                super().__init__(parent)
                self._y2 = np.array([], dtype=float)
                self._poly2 = QPolygonF()
                self._poly2_i = []
                self._nearest_poly2_idx = None
                self._c2 = QColor(40, 140, 40)  # k(λ) зелёный

            def set_curves(
                self, x, y1, y2, *, x_label: str, title: str, ylim=None
            ) -> None:
                self._y2 = np.asarray(y2, dtype=float)

                self.plot_xy(
                    x,
                    y1,
                    x_label=x_label,
                    y_label="n, k",
                    title=title,
                    grid=True,
                    ylim=ylim,
                )
                self._rebuild_poly2()
                self.update()

            def _rebuild_poly2(self) -> None:
                self._poly2 = QPolygonF()
                self._poly2_i = []
                self._nearest_poly2_idx = None

                x = getattr(self, "_x", np.array([], dtype=float))
                if x.size == 0 or self._y2.size == 0:
                    return

                idxmap = []
                for i, (xv, yv) in enumerate(zip(x, self._y2)):
                    if np.isfinite(xv) and np.isfinite(yv):
                        self._poly2.append(self._to_px(float(xv), float(yv)))
                        idxmap.append(i)
                self._poly2_i = idxmap

            def resizeEvent(self, e) -> None:
                super().resizeEvent(e)
                self._rebuild_poly2()

            def leaveEvent(self, e) -> None:
                super().leaveEvent(e)
                self._nearest_poly_idx = None
                self._nearest_data_idx = None
                self._nearest_poly2_idx = None
                self.update()

            def paintEvent(self, e) -> None:
                super().paintEvent(e)

                pr = getattr(self, "_plot_rect", None)
                if pr is None or self._poly2.size() <= 1:
                    return

                p = QPainter(self)
                p.setRenderHint(QPainter.Antialiasing, True)
                p.save()
                p.setClipRect(pr)

                # кривая k поверх базовой n
                p.setPen(QPen(self._c2, 2))
                p.drawPolyline(self._poly2)

                # маркер для k
                if (
                    self._nearest_poly2_idx is not None
                    and 0 <= self._nearest_poly2_idx < self._poly2.size()
                ):
                    pt = self._poly2[self._nearest_poly2_idx]
                    p.setPen(QPen(QColor(220, 50, 50), 2))
                    p.setBrush(QColor(220, 50, 50))
                    p.drawEllipse(pt, 4, 4)

                p.restore()

            def mouseMoveEvent(self, ev) -> None:
                # наведение только по X (λ)
                xarr = getattr(self, "_x", np.array([], dtype=float))
                yarr = getattr(self, "_y", np.array([], dtype=float))
                if xarr.size == 0 or yarr.size == 0:
                    return

                pr = getattr(self, "_plot_rect", None)
                if pr is None or not pr.contains(ev.pos()):
                    self._nearest_poly_idx = None
                    self._nearest_data_idx = None
                    self._nearest_poly2_idx = None
                    self.update()
                    return

                xr = float(self._xmax - self._xmin)
                if pr.width() <= 1 or xr == 0:
                    return

                x_data = (
                    float(self._xmin)
                    + (float(ev.pos().x()) - float(pr.left())) / float(pr.width()) * xr
                )
                idx = int(np.argmin(np.abs(xarr - x_data)))

                # маркер n: data_idx -> poly_idx
                poly_i = getattr(self, "_poly_i", [])
                try:
                    self._nearest_poly_idx = poly_i.index(idx) if poly_i else idx
                except ValueError:
                    self._nearest_poly_idx = None
                self._nearest_data_idx = idx

                # маркер k: data_idx -> poly2_idx
                try:
                    self._nearest_poly2_idx = (
                        self._poly2_i.index(idx) if self._poly2_i else idx
                    )
                except ValueError:
                    self._nearest_poly2_idx = None

                self.hoveredIndexChanged.emit(idx)
                self.update()

        self._material = str(material or "").strip() or "Air"

        md = MaterialDispersion(self._material)
        lam = np.linspace(float(md.lambda_min), float(md.lambda_max), 500)  # meters

        self._wl_um = lam * 1e6
        cri = np.array([md.CRI(float(x)) for x in lam], dtype=np.complex128)
        self._n = np.real(cri)
        self._k = np.imag(cri)

        # Источник (если доступен в materials_list)
        src = ""
        try:
            summary = MaterialDispersion("Air").materials_list()
            if hasattr(summary, "index") and (self._material in list(summary.index)):
                if hasattr(summary, "columns") and ("Source" in list(summary.columns)):
                    val = summary.loc[self._material, "Source"]
                    if hasattr(val, "item"):
                        val = val.item()
                    src = "" if val is None else str(val).strip()
        except Exception:
            src = ""

        # Заголовок окна: материал + диапазон + источник
        title = (
            f"{self._material} | λ: {self._wl_um.min():.3g}…{self._wl_um.max():.3g} μm"
        )
        if src:
            title += f" | {src}"
        self.setWindowTitle(title)

        self.setModal(False)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.resize(900, 650)

        # общий ylim по n и k (чтобы k не обрезался)
        y_all = np.concatenate([self._n, self._k])
        y0 = float(np.nanmin(y_all))
        y1 = float(np.nanmax(y_all))
        pad = 0.05 * (y1 - y0) if (y1 > y0) else 1.0
        ylim = (y0 - pad, y1 + pad)

        self.canvas = CRICanvas(self)
        self.canvas.set_curves(
            self._wl_um,
            self._n,
            self._k,
            x_label="λ, μm",
            title="n(λ) и k(λ)",
            ylim=ylim,
        )

        self.info = QLabel("Наведение: λ = — μm, n = —, k = —", self)
        self.info.setTextInteractionFlags(Qt.TextSelectableByMouse)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)
        lay.addWidget(self.canvas, 1)
        lay.addWidget(self.info)

        self.canvas.hoveredIndexChanged.connect(self._on_hover_idx)

    def _on_hover_idx(self, idx: int) -> None:
        if not (0 <= idx < int(self._wl_um.size)):
            return
        wl = float(self._wl_um[idx])
        n = float(self._n[idx])
        k = float(self._k[idx])
        self.info.setText(f"Наведение: λ = {wl:.4g} μm, n = {n:.6g}, k = {k:.6g}")


# Если у тебя в layer.py вызывается show_cri_dialog(...), оставь этот thin-wrapper:
_CRI_WINDOWS: Dict[str, CRIDialog] = {}


def show_cri_dialog(material: str, parent=None) -> None:
    key = str(material or "").strip() or "Air"
    w = _CRI_WINDOWS.get(key)
    if w is not None:
        w.show()
        w.raise_()
        w.activateWindow()
        return
    w = CRIDialog(key, parent=parent)
    _CRI_WINDOWS[key] = w
    w.destroyed.connect(lambda _obj=None, k=key: _CRI_WINDOWS.pop(k, None))
    w.show()
    w.raise_()
    w.activateWindow()


class PlotDisplaySettingsDialog(QDialog):
    settingsApplied = pyqtSignal(dict)

    def __init__(self, app_settings: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Настройки отображения")

        self._out_settings: Optional[Dict[str, Any]] = None
        self._in_settings = (
            copy.deepcopy(app_settings) if isinstance(app_settings, dict) else {}
        )

        plots = (
            self._in_settings.get("plots", {})
            if isinstance(self._in_settings, dict)
            else {}
        )
        limits = plots.get("limits", {}) if isinstance(plots, dict) else {}
        limits_real = limits.get("real", {}) if isinstance(limits, dict) else {}
        limits_cplx = limits.get("complex", {}) if isinstance(limits, dict) else {}
        res = plots.get("resolution", {}) if isinstance(plots, dict) else {}

        root = QVBoxLayout(self)

        label_w = 120
        spin_w = 120

        def mk_lbl(text: str, parentw) -> QLabel:
            lab = QLabel(text, parentw)
            lab.setFixedWidth(label_w)
            lab.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            return lab

        def mk_dspin(parentw) -> QDoubleSpinBox:
            s = QDoubleSpinBox(parentw)
            s.setDecimals(2)
            s.setRange(-1e9, 1e9)
            s.setMinimumWidth(spin_w)
            s.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            return s

        def mk_spin(parentw, v: int) -> QSpinBox:
            s = QSpinBox(parentw)
            s.setRange(2, 200000)
            s.setValue(int(v))
            s.setMinimumWidth(spin_w)
            s.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            return s

        # --- Block 1: Limits ---
        gb_limits = QGroupBox("Границы", self)
        gb_limits_l = QVBoxLayout(gb_limits)

        self.chk_limits = QCheckBox("Учитывать границы", gb_limits)
        self.chk_limits.setChecked(bool(limits.get("enabled", False)))
        gb_limits_l.addWidget(self.chk_limits)

        grid_limits = QGridLayout()
        grid_limits.setHorizontalSpacing(10)
        grid_limits.setVerticalSpacing(6)
        grid_limits.setColumnStretch(2, 1)  # прослойка/растяжка между парами
        gb_limits_l.addLayout(grid_limits)

        # Real: Ymin/Ymax (в одну строку)
        self.real_ymin = mk_dspin(gb_limits)
        self.real_ymax = mk_dspin(gb_limits)
        self.real_ymin.setValue(float(limits_real.get("ymin", 0.0)))
        self.real_ymax.setValue(float(limits_real.get("ymax", 1.0)))

        r = 0
        grid_limits.addWidget(mk_lbl("Real: Ymin", gb_limits), r, 0)
        grid_limits.addWidget(self.real_ymin, r, 1)
        grid_limits.addWidget(mk_lbl("Real: Ymax", gb_limits), r, 3)
        grid_limits.addWidget(self.real_ymax, r, 4)

        # Complex: Xmin/Xmax
        self.c_xmin = mk_dspin(gb_limits)
        self.c_xmax = mk_dspin(gb_limits)
        self.c_xmin.setValue(float(limits_cplx.get("xmin", 0.0)))
        self.c_xmax.setValue(float(limits_cplx.get("xmax", 1.0)))

        r = 1
        grid_limits.addWidget(mk_lbl("Complex: Xmin", gb_limits), r, 0)
        grid_limits.addWidget(self.c_xmin, r, 1)
        grid_limits.addWidget(mk_lbl("Complex: Xmax", gb_limits), r, 3)
        grid_limits.addWidget(self.c_xmax, r, 4)

        # Complex: Ymin/Ymax
        self.c_ymin = mk_dspin(gb_limits)
        self.c_ymax = mk_dspin(gb_limits)
        self.c_ymin.setValue(float(limits_cplx.get("ymin", 0.0)))
        self.c_ymax.setValue(float(limits_cplx.get("ymax", 1.0)))

        r = 2
        grid_limits.addWidget(mk_lbl("Complex: Ymin", gb_limits), r, 0)
        grid_limits.addWidget(self.c_ymin, r, 1)
        grid_limits.addWidget(mk_lbl("Complex: Ymax", gb_limits), r, 3)
        grid_limits.addWidget(self.c_ymax, r, 4)

        root.addWidget(gb_limits)

        # --- Block 2: Resolution ---
        gb_res = QGroupBox("Разрешение графиков", self)
        grid_res = QGridLayout(gb_res)
        grid_res.setHorizontalSpacing(10)
        grid_res.setVerticalSpacing(6)
        grid_res.setColumnStretch(2, 1)

        self.n_re_theta = mk_spin(gb_res, res.get("re_theta", 400))
        self.n_im_theta = mk_spin(gb_res, res.get("im_theta", 400))
        self.n_re_lambda = mk_spin(gb_res, res.get("re_lambda", 400))
        self.n_im_lambda = mk_spin(gb_res, res.get("im_lambda", 400))

        r = 0
        grid_res.addWidget(mk_lbl("Real R(θ): N", gb_res), r, 0)
        grid_res.addWidget(self.n_re_theta, r, 1)
        grid_res.addWidget(mk_lbl("Imag R(θ): N", gb_res), r, 3)
        grid_res.addWidget(self.n_im_theta, r, 4)

        r = 1
        grid_res.addWidget(mk_lbl("Real R(λ): N", gb_res), r, 0)
        grid_res.addWidget(self.n_re_lambda, r, 1)
        grid_res.addWidget(mk_lbl("Imag R(λ): N", gb_res), r, 3)
        grid_res.addWidget(self.n_im_lambda, r, 4)

        root.addWidget(gb_res)

        # enable/disable limits fields
        def apply_limits_enabled(on: bool) -> None:
            for w in (
                self.real_ymin,
                self.real_ymax,
                self.c_xmin,
                self.c_xmax,
                self.c_ymin,
                self.c_ymax,
            ):
                w.setEnabled(bool(on))

        apply_limits_enabled(self.chk_limits.isChecked())
        self.chk_limits.toggled.connect(apply_limits_enabled)

        # --- bottom row: autodraw + buttons ---
        bottom = QHBoxLayout()

        self.chk_autodraw = QCheckBox("Autodraw", self)
        self.chk_autodraw.setChecked(bool(plots.get("autodraw", True)))
        bottom.addWidget(self.chk_autodraw)

        bottom.addStretch(1)

        self.btn_save = QPushButton("Сохранить", self)
        self.btn_cancel = QPushButton("Отмена", self)
        bottom.addWidget(self.btn_save)
        bottom.addWidget(self.btn_cancel)

        root.addLayout(bottom)

        self.btn_cancel.clicked.connect(self.reject)
        self.btn_save.clicked.connect(self._on_save)

    def result_settings(self) -> Optional[Dict[str, Any]]:
        return (
            copy.deepcopy(self._out_settings)
            if isinstance(self._out_settings, dict)
            else None
        )

    def _on_save(self) -> None:
        if self.chk_limits.isChecked():
            if not (self.real_ymax.value() > self.real_ymin.value()):
                QMessageBox.warning(self, "Границы", "Real: нужно Ymax > Ymin.")
                return
            if not (self.c_xmax.value() > self.c_xmin.value()):
                QMessageBox.warning(self, "Границы", "Complex: нужно Xmax > Xmin.")
                return
            if not (self.c_ymax.value() > self.c_ymin.value()):
                QMessageBox.warning(self, "Границы", "Complex: нужно Ymax > Ymin.")
                return

        st = (
            copy.deepcopy(self._in_settings)
            if isinstance(self._in_settings, dict)
            else {}
        )
        st.setdefault("version", 1)

        plots = st.get("plots")
        if not isinstance(plots, dict):
            plots = {}
            st["plots"] = plots

        plots["autodraw"] = bool(self.chk_autodraw.isChecked())
        plots["limits"] = {
            "enabled": bool(self.chk_limits.isChecked()),
            "real": {
                "ymin": float(self.real_ymin.value()),
                "ymax": float(self.real_ymax.value()),
            },
            "complex": {
                "xmin": float(self.c_xmin.value()),
                "xmax": float(self.c_xmax.value()),
                "ymin": float(self.c_ymin.value()),
                "ymax": float(self.c_ymax.value()),
            },
        }
        plots["resolution"] = {
            "re_theta": int(self.n_re_theta.value()),
            "im_theta": int(self.n_im_theta.value()),
            "re_lambda": int(self.n_re_lambda.value()),
            "im_lambda": int(self.n_im_lambda.value()),
        }

        self._out_settings = st
        self.settingsApplied.emit(copy.deepcopy(st))
        return


# --- Gradient profile editor -------------------------------------------------


def show_gradient_profile_dialog(
    key: str = "default", parent=None, init: Optional[Dict[str, Any]] = None
):
    k = str(key or "default")
    w = _GRAD_PROFILE_WINDOWS.get(k)
    if w is not None:
        # важно: если повторно открыли с новым init — обновим диалог
        if isinstance(init, dict):
            w._apply_init(copy.deepcopy(init))
            w._redraw()
        w.show()
        w.raise_()
        w.activateWindow()
        return w

    w = GradientProfileDialog(parent=parent, init=init)
    _GRAD_PROFILE_WINDOWS[k] = w
    w.destroyed.connect(lambda _obj=None, kk=k: _GRAD_PROFILE_WINDOWS.pop(kk, None))
    w.show()
    w.raise_()
    w.activateWindow()
    return w


class GradientProfileDialog(QDialog):
    profileApplied = pyqtSignal(dict)

    def __init__(self, parent=None, init: Optional[Dict[str, Any]] = None):
        super().__init__(parent)
        self.setWindowTitle("Профиль градиентного слоя")
        self.setModal(False)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.resize(900, 650)

        self._mode_complex = False
        self._ymax = 2.0
        self._npts = 5

        # control points in [0..1] x [0..ymax]
        self._x = np.linspace(0.0, 1.0, self._npts)
        self._yre = np.linspace(1.0, 1.5, self._npts)
        self._yim = np.zeros(self._npts, dtype=float)

        if isinstance(init, dict):
            self._apply_init(init)

        # --- UI
        root = QVBoxLayout(self)
        top = QHBoxLayout()

        self.chk_complex = QCheckBox("Complex (Re + Im)", self)
        self.chk_complex.setChecked(bool(self._mode_complex))
        top.addWidget(self.chk_complex)

        top.addWidget(QLabel("N points:", self))
        self.spin_n = QSpinBox(self)
        self.spin_n.setRange(3, 7)
        self.spin_n.setValue(int(self._npts))
        top.addWidget(self.spin_n)

        top.addWidget(QLabel("Y max:", self))
        self.spin_ymax = QDoubleSpinBox(self)
        self.spin_ymax.setDecimals(3)
        self.spin_ymax.setRange(1e-6, 1e9)
        self.spin_ymax.setValue(float(self._ymax))
        top.addWidget(self.spin_ymax)

        top.addStretch(1)

        self.btn_reset = QPushButton("Reset", self)
        self.btn_apply = QPushButton("Apply", self)
        self.btn_close = QPushButton("Close", self)
        top.addWidget(self.btn_reset)
        top.addWidget(self.btn_apply)
        top.addWidget(self.btn_close)

        root.addLayout(top)

        self.canvas = GradientProfileCanvas(self)
        root.addWidget(self.canvas, 1)

        self.info = QLabel("Drag points with mouse. X: 0..1, Y: 0..Ymax", self)
        self.info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(self.info)

        # --- signals
        self.btn_close.clicked.connect(self.close)
        self.btn_apply.clicked.connect(self._on_apply)
        self.btn_reset.clicked.connect(self._on_reset)

        self.chk_complex.toggled.connect(self._on_complex_toggled)
        self.spin_n.valueChanged.connect(self._on_n_changed)
        self.spin_ymax.valueChanged.connect(self._on_ymax_changed)

        self._redraw()

    def _apply_init(self, init: Dict[str, Any]) -> None:
        # ожидаем структуру типа:
        # {"mode": "real"|"complex", "ymax": float, "x": [...], "re": [...], "im": [...]}
        mode = str(init.get("mode", "real")).lower()
        self._mode_complex = mode == "complex"
        try:
            self._ymax = float(init.get("ymax", self._ymax))
        except Exception:
            pass

        x = init.get("x")
        re = init.get("re")
        im = init.get("im")
        if (
            isinstance(x, (list, tuple))
            and isinstance(re, (list, tuple))
            and len(x) == len(re)
            and 3 <= len(x) <= 7
        ):
            self._x = np.asarray(x, dtype=float)
            self._yre = np.asarray(re, dtype=float)
            if isinstance(im, (list, tuple)) and len(im) == len(x):
                self._yim = np.asarray(im, dtype=float)
            else:
                self._yim = np.zeros_like(self._yre)

            self._npts = int(self._x.size)
            self._sanitize_points()

    def export_profile(self) -> Dict[str, Any]:
        xs_s, yre_s, yim_s = self._sorted_export_arrays()
        return {
            "mode": "complex" if self._mode_complex else "real",
            "ymax": float(self._ymax),
            "x": [float(v) for v in xs_s],
            "re": [float(v) for v in yre_s],
            "im": [float(v) for v in yim_s],
        }

    def _sanitize_points(self) -> None:
        # clip
        self._x = np.clip(self._x, 0.0, 1.0)
        self._yre = np.clip(self._yre, 0.0, float(self._ymax))
        self._yim = np.clip(self._yim, 0.0, float(self._ymax))

        # фиксируем крайние X
        if int(self._x.size) >= 2:
            self._x[0] = 0.0
            self._x[-1] = 1.0

    def _interp_curve(self, x: np.ndarray, y: np.ndarray, xs: np.ndarray) -> np.ndarray:
        f = PchipInterpolator(x, y, extrapolate=True)
        return np.asarray(f(xs), dtype=float)

    def _redraw(self) -> None:
        # внутри диалога точки не сортируем (индексы стабильны)
        self._sanitize_points()

        xs_dense = np.linspace(0.0, 1.0, 400)

        xs_s, yre_s, yim_s = self._sorted_export_arrays()

        yre_dense = self._interp_curve(xs_s, yre_s, xs_dense)
        yim_dense = (
            self._interp_curve(xs_s, yim_s, xs_dense) if self._mode_complex else None
        )

        self.canvas.set_profile(
            xs=xs_dense,
            yre=yre_dense,
            yim=yim_dense,
            ctrl_x=self._x,  # как есть (не сортируем)
            ctrl_re=self._yre,
            ctrl_im=self._yim,
            ymax=float(self._ymax),
            complex_mode=bool(self._mode_complex),
        )

    def _on_complex_toggled(self, on: bool) -> None:
        self._mode_complex = bool(on)
        self._redraw()

    def _on_n_changed(self, n: int) -> None:
        n = int(n)
        n = max(3, min(7, n))
        if n == self._npts:
            return

        # берём "экспортный" (отсортированный) профиль и пересэмплируем
        xs_s, yre_s, yim_s = self._sorted_export_arrays()

        x_new = np.linspace(0.0, 1.0, n)
        re_new = self._interp_curve(xs_s, yre_s, x_new)
        im_new = self._interp_curve(xs_s, yim_s, x_new)

        self._x = x_new
        self._yre = re_new
        self._yim = im_new
        self._npts = n
        self._redraw()

    def _on_ymax_changed(self, v: float) -> None:
        try:
            self._ymax = float(v)
        except Exception:
            return
        self._redraw()

    def _on_reset(self) -> None:
        self._x = np.linspace(0.0, 1.0, int(self.spin_n.value()))
        self._yre = np.linspace(1.0, 1.5, int(self.spin_n.value()))
        self._yim = np.zeros(int(self.spin_n.value()), dtype=float)
        self._npts = int(self._x.size)
        self._redraw()

    def _on_apply(self) -> None:
        self.profileApplied.emit(self.export_profile())

    # called from canvas
    def _set_ctrl_point(self, which: str, idx: int, x: float, y: float) -> None:
        if not (0 <= idx < int(self._x.size)):
            return

        # крайние точки по X фиксированы
        if idx == 0:
            x = 0.0
        elif idx == int(self._x.size) - 1:
            x = 1.0

        self._x[idx] = float(x)
        if which == "im":
            self._yim[idx] = float(y)
        else:
            self._yre[idx] = float(y)

        self._redraw()

    def _sorted_export_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Возвращает (xs, yre, yim) ОТСОРТИРОВАННЫЕ по xs и с xs строго возрастающим.
        ВАЖНО: не меняет порядок self._x/self._yre/self._yim внутри диалога.
        Требование PCHIP: x 1D монотонно возрастающий и без дублей. [web:13]
        """
        self._sanitize_points()

        idx = np.argsort(np.asarray(self._x, dtype=float))
        xs = np.asarray(self._x, dtype=float)[idx].copy()
        yre = np.asarray(self._yre, dtype=float)[idx].copy()
        yim = np.asarray(self._yim, dtype=float)[idx].copy()

        # строго возрастающий xs (устраняем дубли)
        eps = 1e-6
        for i in range(1, int(xs.size)):
            if xs[i] <= xs[i - 1]:
                xs[i] = min(1.0, xs[i - 1] + eps)

        # фиксируем края
        if int(xs.size) >= 2:
            xs[0] = 0.0
            xs[-1] = 1.0
            if int(xs.size) >= 3 and xs[-2] >= xs[-1]:
                xs[-2] = max(0.0, xs[-1] - eps)

        return xs, yre, yim


class GradientProfileCanvas(PlotCanvasQt):
    def __init__(self, dlg: GradientProfileDialog):
        super().__init__(dlg)
        self._dlg = dlg

        self._complex = False
        self._ymax = 1.0

        self._ctrl_x = np.array([], dtype=float)
        self._ctrl_re = np.array([], dtype=float)
        self._ctrl_im = np.array([], dtype=float)

        self._xs = np.array([], dtype=float)
        self._yre = np.array([], dtype=float)
        self._yim = np.array([], dtype=float)

        self._poly_im = QPolygonF()
        self._poly_im_i = []

        self._grab = None  # ("re"/"im", idx)

        self._c_re = QColor(40, 80, 200)
        self._c_im = QColor(40, 140, 40)

    def set_profile(
        self,
        *,
        xs: np.ndarray,
        yre: np.ndarray,
        yim: Optional[np.ndarray],
        ctrl_x: np.ndarray,
        ctrl_re: np.ndarray,
        ctrl_im: np.ndarray,
        ymax: float,
        complex_mode: bool,
    ) -> None:
        self._xs = np.asarray(xs, dtype=float)
        self._yre = np.asarray(yre, dtype=float)
        self._yim = (
            np.asarray(yim, dtype=float)
            if yim is not None
            else np.array([], dtype=float)
        )

        self._ctrl_x = np.asarray(ctrl_x, dtype=float)
        self._ctrl_re = np.asarray(ctrl_re, dtype=float)
        self._ctrl_im = np.asarray(ctrl_im, dtype=float)

        self._ymax = float(ymax)
        self._complex = bool(complex_mode)

        self.plot_xy(
            self._xs,
            self._yre,
            x_label="t (0..1)",
            y_label="n(t) / k(t)",
            title="Gradient profile",
            grid=True,
            ylim=(0.0, float(self._ymax)),
        )
        self._rebuild_im_poly()
        self.update()

    def _rebuild_im_poly(self) -> None:
        self._poly_im = QPolygonF()
        self._poly_im_i = []
        if (not self._complex) or self._xs.size == 0 or self._yim.size == 0:
            return
        idxmap = []
        for i, (xv, yv) in enumerate(zip(self._xs, self._yim)):
            if np.isfinite(xv) and np.isfinite(yv):
                self._poly_im.append(self._to_px(float(xv), float(yv)))
                idxmap.append(i)
        self._poly_im_i = idxmap

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._rebuild_im_poly()

    def _px_to_data(self, px, py) -> Tuple[float, float]:
        pr = getattr(self, "_plot_rect", None)
        if pr is None or pr.width() <= 1 or pr.height() <= 1:
            return 0.0, 0.0
        xr = float(self._xmax - self._xmin)
        yr = float(self._ymax - self._ymin)
        x = float(self._xmin) + (float(px) - float(pr.left())) / float(pr.width()) * xr
        y = float(self._ymax) - (float(py) - float(pr.top())) / float(pr.height()) * yr
        return x, y

    def _nearest_ctrl(self, pos) -> Optional[Tuple[str, int]]:
        if self._ctrl_x.size == 0:
            return None
        pr = getattr(self, "_plot_rect", None)
        if pr is None or not pr.contains(pos):
            return None

        # compare distance to Re points (and Im if enabled)
        best = None
        best_d2 = None

        def consider(which: str, yarr: np.ndarray):
            nonlocal best, best_d2
            for i, (xv, yv) in enumerate(zip(self._ctrl_x, yarr)):
                pt = self._to_px(float(xv), float(yv))
                dx = float(pt.x()) - float(pos.x())
                dy = float(pt.y()) - float(pos.y())
                d2 = dx * dx + dy * dy
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
                    best = (which, int(i))

        consider("re", self._ctrl_re)
        if self._complex:
            consider("im", self._ctrl_im)

        # small picking radius
        if best_d2 is not None and best_d2 <= (12.0 * 12.0):
            return best
        return None

    def mousePressEvent(self, ev) -> None:
        if ev.button() == Qt.LeftButton:
            hit = self._nearest_ctrl(ev.pos())
            if hit is not None:
                self._grab = hit
                ev.accept()
                return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev) -> None:
        if self._grab is None:
            super().mouseMoveEvent(ev)
            return

        which, idx = self._grab
        x, y = self._px_to_data(ev.pos().x(), ev.pos().y())
        x = float(np.clip(x, 0.0, 1.0))
        y = float(np.clip(y, 0.0, float(self._ymax)))

        self._dlg._set_ctrl_point(which, idx, x, y)
        ev.accept()

    def mouseReleaseEvent(self, ev) -> None:
        if ev.button() == Qt.LeftButton and self._grab is not None:
            self._grab = None
            ev.accept()
            return
        super().mouseReleaseEvent(ev)

    def paintEvent(self, e) -> None:
        super().paintEvent(e)

        pr = getattr(self, "_plot_rect", None)
        if pr is None:
            return

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.save()
        p.setClipRect(pr)

        # draw Im curve on top
        if self._complex and self._poly_im.size() > 1:
            p.setPen(QPen(self._c_im, 2))
            p.drawPolyline(self._poly_im)

        # draw control points
        def draw_points(yarr: np.ndarray, color: QColor):
            p.setPen(QPen(QColor(20, 20, 20), 1))
            p.setBrush(color)
            for xv, yv in zip(self._ctrl_x, yarr):
                pt = self._to_px(float(xv), float(yv))
                p.drawEllipse(pt, 5, 5)

        draw_points(self._ctrl_re, self._c_re)
        if self._complex:
            draw_points(self._ctrl_im, self._c_im)

        p.restore()
