# -*- coding: utf-8 -*-
"""
p_reflection_curves_app/layer.py

"Чистый слой": редакторы параметров и LayerWidget (один слой).

Содержит:
- ParamField (DigitNumberEdit + units)
- Редакторы типов слоёв (Dielectric/Metal/Gradient/Anisotropic/Dispersion)
- LayerWidget (variant B: hide/show редакторов => корректная высота)
- get_ui_state()/set_ui_state() для точного сохранения цифр и множителей (units)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple, Type

from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QComboBox,
    QFrame,
    QPushButton,
    QSizePolicy,
)

from .digit_number import DigitNumberEdit, DigitFormat
from SPPPy import MaterialDispersion

# ============================================================
# ParamField (DigitNumberEdit + optional units)
# ============================================================

@dataclass(frozen=True)
class UnitsSpec:
    units: Dict[str, float]

    def items_sorted_by_multiplier(self) -> List[Tuple[str, float]]:
        return sorted(self.units.items(), key=lambda kv: kv[1])


class ParamField(QWidget):
    """
    label + DigitNumberEdit + optional units combobox.

    valueSI = valueDisplay * multiplier.

    Важно для "точного" save/load:
    - DigitNumberEdit хранит состояние как целое scaled (scaledValue/setScaledValue),
      это воспроизводит те же разряды при загрузке.
    - unit_box хранит выбранный множитель (nm/um, x1/x10/...).
    """
    valueChanged = pyqtSignal(float)  # emits valueSI()

    def __init__(
        self,
        name: str,
        fmt: DigitFormat,
        *,
        units: Optional[Dict[str, float]] = None,
        default_unit: Optional[str] = None,
        label_width: int = 32,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)

        self._fmt = fmt
        self._units_spec: Optional[UnitsSpec] = UnitsSpec(units) if units else None

        self.lbl = QLabel(name, self)
        self.lbl.setFixedWidth(int(label_width))

        # Совместимо с текущим DigitNumberEdit(fmt, parent, label) [file:116]
        self.edit = DigitNumberEdit(fmt, self, label="")
        self.edit.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.unit_box: Optional[QComboBox] = None
        if self._units_spec is not None:
            self.unit_box = QComboBox(self)
            self.unit_box.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)

            for key, _mult in self._units_spec.items_sorted_by_multiplier():
                self.unit_box.addItem(key)

            if default_unit and default_unit in self._units_spec.units:
                self.unit_box.setCurrentText(default_unit)
            else:
                self.unit_box.setCurrentIndex(0)

            self.unit_box.currentTextChanged.connect(lambda _t: self.valueChanged.emit(self.valueSI()))

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2)
        row.addWidget(self.lbl)
        row.addWidget(self.edit)
        if self.unit_box is not None:
            row.addWidget(self.unit_box)

        self.edit.valueChanged.connect(lambda _v: self.valueChanged.emit(self.valueSI()))

    # --- enable/disable editing (для правила верх/низ) ---
    def setEditingEnabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        self.edit.setEnabled(enabled)
        if self.unit_box is not None:
            self.unit_box.setEnabled(enabled)

    # --- physical value API ---
    def unit(self) -> Optional[str]:
        return self.unit_box.currentText() if self.unit_box is not None else None

    def multiplier(self) -> float:
        if self._units_spec is None or self.unit_box is None:
            return 1.0
        return float(self._units_spec.units[self.unit_box.currentText()])

    def valueDisplay(self) -> float:
        return float(self.edit.value())

    def valueSI(self) -> float:
        return self.valueDisplay() * self.multiplier()

    def setValueSI(self, value_si: float, *, emit_signal: bool = False) -> None:
        value_si = float(value_si)
        display = value_si / self.multiplier()
        self.edit.setValue(display, emit_signal=False)
        if emit_signal:
            self.valueChanged.emit(self.valueSI())

    # --- exact UI save/load (digits + unit) ---
    def ui_state(self) -> Dict[str, Any]:
        return {
            "scaled": int(self.edit.scaledValue()),
            "unit": self.unit(),
            "fmt": {"int": int(self._fmt.integer_digits), "dec": int(self._fmt.decimal_digits)},
        }

    def set_ui_state(self, st: Dict[str, Any]) -> None:
        # Восстанавливаем unit первым, чтобы множитель был тем же
        if self.unit_box is not None:
            u = st.get("unit", None)
            if isinstance(u, str) and u:
                self.unit_box.setCurrentText(u)

        if "scaled" in st:
            self.edit.setScaledValue(int(st["scaled"]), emit_signal=False)


# ============================================================
# Layer types (editors)
# ============================================================

@dataclass
class LayerState:
    type: str
    params: Dict[str, Any]


class LayerTypeEditor(QWidget):
    changed = pyqtSignal()
    TYPE_NAME: str = "Base"

    def state(self) -> LayerState:
        raise NotImplementedError

    def set_state(self, st: LayerState) -> None:
        raise NotImplementedError


class DielectricEditor(LayerTypeEditor):
    TYPE_NAME = "Dielectric"

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
    
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)
    
        scale_units = {"x1": 1.0, "x10": 10.0, "x100": 100.0}
        self.n = ParamField("n =", DigitFormat(4, 3), units=scale_units, default_unit="x1")
        self.d = ParamField("d =", DigitFormat(4, 3), units={"nm": 1e-9, "um": 1e-6}, default_unit="nm")
    
        v.addWidget(self.n)
        v.addWidget(self.d)
    
        self.n.valueChanged.connect(lambda _v: self.changed.emit())
        self.d.valueChanged.connect(lambda _v: self.changed.emit())
        self.set_state()





    def state(self) -> LayerState:
        return LayerState(self.TYPE_NAME, {"n": self.n.valueSI(), "d": self.d.valueSI()})

    def set_state(self, st: Optional[LayerState] = None) -> None:
        if st is None:
            st = LayerState(self.TYPE_NAME, {})
        self.n.setValueSI(st.params.get("n", 1.0))
        self.d.setValueSI(st.params.get("d", 0.0))
        self.changed.emit()


class MetalEditor(LayerTypeEditor):
    TYPE_NAME = "Metal"

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)

        scale_units = {"x1": 1.0, "x10": 10.0, "x100": 100.0}
        self.n = ParamField("n =", DigitFormat(4, 3), units=scale_units, default_unit="x1")
        self.k = ParamField("k =", DigitFormat(4, 3), units=scale_units, default_unit="x1")
        self.d = ParamField("d =", DigitFormat(4, 3), units={"nm": 1e-9, "um": 1e-6}, default_unit="nm")

        v.addWidget(self.n)
        v.addWidget(self.k)
        v.addWidget(self.d)

        self.n.valueChanged.connect(lambda _v: self.changed.emit())
        self.k.valueChanged.connect(lambda _v: self.changed.emit())
        self.d.valueChanged.connect(lambda _v: self.changed.emit())
        self.set_state()

    def state(self) -> LayerState:
        return LayerState(self.TYPE_NAME, {"n": self.n.valueSI(), "k": self.k.valueSI(), "d": self.d.valueSI()})

    def set_state(self, st: Optional[LayerState] = None) -> None:
        if st is None:
            st = LayerState(self.TYPE_NAME, {})
        self.n.setValueSI(st.params.get("n", 0.2))
        self.k.setValueSI(st.params.get("k", 3.0))
        self.d.setValueSI(st.params.get("d", 0.0))
        self.changed.emit()


class GradientEditor(LayerTypeEditor):
    TYPE_NAME = "Gradient"

    def __init__(self, parent=None):
        super().__init__(parent)
    
        from PyQt5.QtWidgets import QVBoxLayout, QPushButton, QSizePolicy
    
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)
    
        # Кнопка: сверху, на всю ширину
        self.btn_profile = QPushButton("Настроить профиль", self)
        self.btn_profile.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # растягиваем по X [web:132]
        self.btn_profile.setMinimumHeight(22)
        v.addWidget(self.btn_profile)
    
        # Толщина: ниже
        self.d = ParamField(
            "d =",
            DigitFormat(4, 3),
            units={"nm": 1e-9, "um": 1e-6},
            default_unit="nm",
        )
        v.addWidget(self.d)
    
        # Данные градиента (единственный источник истины)
        self.profile = self._default_profile()
    
        # Сигналы
        self.btn_profile.clicked.connect(self._on_edit_profile)
        self.d.valueChanged.connect(lambda _v=None: self.changed.emit())


    # -------- profile helpers --------
    def _default_profile(self) -> Dict[str, Any]:
        x = [0.0, 0.25, 0.5, 0.75, 1.0]
        re = [1.0, 1.0, 1.0, 1.0, 1.0]
        im = [0.0, 0.0, 0.0, 0.0, 0.0]
        return {"mode": "real", "ymax": 2.0, "x": x, "re": re, "im": im}

    def _profile_to_points(self, prof: Dict[str, Any], *, which: str) -> List[Tuple[float, float]]:
        x = prof.get("x", [])
        y = prof.get("im", []) if which == "im" else prof.get("re", [])
        if not (isinstance(x, list) and isinstance(y, list) and len(x) == len(y)):
            return []
        out: List[Tuple[float, float]] = []
        for xv, yv in zip(x, y):
            try:
                out.append((float(xv), float(yv)))
            except Exception:
                pass
        return out

    def _points_to_profile(
        self,
        points_re: List[Tuple[float, float]],
        points_im: List[Tuple[float, float]],
        *,
        mode: Optional[str] = None,
        ymax: Optional[float] = None,
    ) -> Dict[str, Any]:
        x = []
        re = []
        im = []

        for xv, yv in (points_re or []):
            x.append(float(xv))
            re.append(float(yv))

        # im: пытаемся сопоставить по индексу (как было раньше)
        pim = list(points_im or [])
        for i in range(len(x)):
            if i < len(pim):
                im.append(float(pim[i][1]))
            else:
                im.append(0.0)

        prof = dict(self.profile) if isinstance(self.profile, dict) else {}
        if mode is not None:
            prof["mode"] = str(mode)
        if ymax is not None:
            prof["ymax"] = float(ymax)
        prof["x"] = x
        prof["re"] = re
        prof["im"] = im
        return prof

    # -------- UI handlers --------
    def _on_edit_profile(self) -> None:
        # локальный импорт — как у DispersionEditor, чтобы не получить циклические импорты
        from .dialogs import show_gradient_profile_dialog

        init = dict(self.profile) if isinstance(self.profile, dict) else None

        dlg = show_gradient_profile_dialog(
            key=f"grad_profile_{id(self)}",
            parent=self,
            init=init,
        )

        # на всякий: чтобы не накапливать коннекты при повторном открытии
        try:
            dlg.profileApplied.disconnect(self._on_profile_applied)
        except Exception:
            pass
        dlg.profileApplied.connect(self._on_profile_applied)

    def _on_profile_applied(self, prof: Dict[str, Any]) -> None:
        if not isinstance(prof, dict):
            return
        # без лишних конверсий — просто сохраняем как есть (он уже отсортирован export_profile())
        self.profile = dict(prof)
        self.changed.emit()


    # -------- LayerTypeEditor API --------
    def state(self) -> LayerState:
        return LayerState(self.TYPE_NAME, {"d": self.d.valueSI(), "profile": dict(self.profile)})


    def set_state(self, st: Optional[LayerState] = None) -> None:
        if st is None:
            st = LayerState(self.TYPE_NAME, {})
    
        self.d.setValueSI(st.params.get("d", 0.0))
    
        prof = st.params.get("profile", None)
        if isinstance(prof, dict):
            self.profile = dict(prof)
        else:
            self.profile = self._default_profile()
    
        self.changed.emit()


class AnisotropicEditor(LayerTypeEditor):
    TYPE_NAME = "Anisotropic"

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)

        scale_units = {"x1": 1.0, "x10": 10.0, "x100": 100.0}
        self.n0 = ParamField("n0", DigitFormat(4, 3), units=scale_units, default_unit="x1", label_width=24)
        self.n1 = ParamField("n1", DigitFormat(4, 3), units=scale_units, default_unit="x1", label_width=24)
        self.theta = ParamField("θ =", DigitFormat(3, 3), units={"deg": 1.0}, default_unit="deg")
        self.d = ParamField("d =", DigitFormat(4, 3), units={"nm": 1e-9, "um": 1e-6}, default_unit="nm")

        v.addWidget(self.n0)
        v.addWidget(self.n1)
        v.addWidget(self.theta)
        v.addWidget(self.d)

        for w in (self.n0, self.n1, self.theta, self.d):
            w.valueChanged.connect(lambda _v: self.changed.emit())

        # Как у Dielectric/Metal: принудительно выставляем дефолтное состояние
        self.set_state()

    def state(self) -> LayerState:
        return LayerState(
            self.TYPE_NAME,
            {"n0": self.n0.valueSI(), "n1": self.n1.valueSI(), "theta_deg": self.theta.valueSI(), "d": self.d.valueSI()},
        )

    def set_state(self, st: Optional[LayerState] = None) -> None:
        if st is None:
            st = LayerState(self.TYPE_NAME, {})

        self.n0.setValueSI(st.params.get("n0", 1.2))
        self.n1.setValueSI(st.params.get("n1", 1.4))
        self.theta.setValueSI(st.params.get("theta_deg", 45.0))
        self.d.setValueSI(st.params.get("d", 0.0))
        self.changed.emit()


class CauchyEditor(LayerTypeEditor):
    TYPE_NAME = "Cauchy"

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)

        scale_units = {"x1": 1.0, "x10": 10.0, "x100": 100.0}
        self.A = ParamField("A =", DigitFormat(4, 3), units=scale_units, default_unit="x1")
        self.B = ParamField("B =", DigitFormat(4, 3), units=scale_units, default_unit="x1")
        self.C = ParamField("C =", DigitFormat(4, 3), units=scale_units, default_unit="x1")
        self.d = ParamField("d =", DigitFormat(4, 3), units={"nm": 1e-9, "um": 1e-6}, default_unit="nm")

        v.addWidget(self.A)
        v.addWidget(self.B)
        v.addWidget(self.C)
        v.addWidget(self.d)  # ← ВСЕГДА ПОСЛЕДНИЙ

        for f in (self.A, self.B, self.C, self.d):
            f.valueChanged.connect(lambda _: self.changed.emit())
        self.set_state()

    def state(self) -> LayerState:
        return LayerState(self.TYPE_NAME, {
            "A": self.A.valueSI(), "B": self.B.valueSI(), "C": self.C.valueSI(), "d": self.d.valueSI()
        })

    def set_state(self, st: Optional[LayerState] = None) -> None:
        if st is None:
            st = LayerState(self.TYPE_NAME, {})
        p = st.params
        self.A.setValueSI(p.get("A", 1.3))
        self.B.setValueSI(p.get("B", 0.02))
        self.C.setValueSI(p.get("C", 0.001))
        self.d.setValueSI(p.get("d", 0.0))
        self.changed.emit()


class LorentzDrudeEditor(LayerTypeEditor):
    TYPE_NAME = "LorentzDrude"

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)

        e15_units = {"e13": 1e13, "e14": 1e14, "e15": 1e15}
        self.wp = ParamField("ωp =", DigitFormat(3, 2), units=e15_units, default_unit="e15", label_width=32)
        self.wt = ParamField("γ =",  DigitFormat(3, 2), units=e15_units, default_unit="e15", label_width=32)
        self.w0 = ParamField("ω0 =", DigitFormat(3, 2), units=e15_units, default_unit="e15", label_width=32)
        self.ampl = ParamField("A =", DigitFormat(3, 2), units={"x1": 1.0, "x10": 10.0}, default_unit="x1")
        self.eps_inf = ParamField("ε∞ =", DigitFormat(3, 2), units={"x1": 1.0}, default_unit="x1")
        self.d = ParamField("d =", DigitFormat(4, 3), units={"nm": 1e-9, "um": 1e-6}, default_unit="nm")

        v.addWidget(self.wp)
        v.addWidget(self.wt)
        v.addWidget(self.w0)
        v.addWidget(self.ampl)
        v.addWidget(self.eps_inf)
        v.addWidget(self.d)  # ← ВСЕГДА ПОСЛЕДНИЙ

        for f in (self.wp, self.wt, self.w0, self.ampl, self.eps_inf, self.d):
            f.valueChanged.connect(lambda _: self.changed.emit())
        self.set_state()

    def state(self) -> LayerState:
        return LayerState(self.TYPE_NAME, {
            "wp": self.wp.valueSI(), "wt": self.wt.valueSI(), "w0": self.w0.valueSI(),
            "ampl": self.ampl.valueSI(), "eps_inf": self.eps_inf.valueSI(), "d": self.d.valueSI()
        })

    def set_state(self, st: Optional[LayerState] = None) -> None:
        if st is None:
            st = LayerState(self.TYPE_NAME, {})
        p = st.params
        self.wp.setValueSI(p.get("wp", 1.2e15))
        self.wt.setValueSI(p.get("wt", 0.1e15))
        self.w0.setValueSI(p.get("w0", 1.0e15))
        self.ampl.setValueSI(p.get("ampl", 0.8))
        self.eps_inf.setValueSI(p.get("eps_inf", 1.0))
        self.d.setValueSI(p.get("d", 0.0))
        self.changed.emit()



class DispersionEditor(LayerTypeEditor):
    TYPE_NAME = "Dispersion"

    # cache на класс (чтобы CSV читался 1 раз на приложение)
    _cached_items: Optional[List[Tuple[str, str]]] = None  # [(Element, Source), ...]
    _cached_err: str = ""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
    
        self._pending_material_key: Optional[str] = None
    
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)
    
        mat_row = QHBoxLayout()
        mat_row.setContentsMargins(0, 0, 0, 0)
        mat_row.setSpacing(4)
        
        lab = QLabel("Mat", self)
        lab.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)   # минимум, не растягивать
        mat_row.addWidget(lab, 0)
        
        self.material = QComboBox(self)
        self.material.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.material.setMinimumWidth(80)  # чтобы не схлопывался в ноль
        mat_row.addWidget(self.material, 10)  # доля остатка
        
        self.btn_cri = QPushButton("CRI", self)
        self.btn_cri.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # важно: дать право растягиваться
        self.btn_cri.setMinimumWidth(self.btn_cri.sizeHint().width())         # минимальная адекватная ширина
        self.btn_cri.setFixedHeight(22)
        self.btn_cri.clicked.connect(self._on_show_cri)
        mat_row.addWidget(self.btn_cri, 3)  # доля остатка
        
        v.addLayout(mat_row)

    
        self.d = ParamField("d =", DigitFormat(4, 3), units={"nm": 1e-9, "um": 1e-6}, default_unit="nm")
        v.addWidget(self.d)
    
        self.material.currentTextChanged.connect(lambda _t: self.changed.emit())
        self.d.valueChanged.connect(lambda _v: self.changed.emit())
    
        # загрузить из базы
        self.refresh_materials(force=False, debug=False)


    def _on_show_cri(self) -> None:
        from .dialogs import show_cri_dialog  # локально, чтобы не было циклического импорта
    
        key = self._key(self.material.currentText()) or "Air"
        show_cri_dialog(key, parent=self)


    

    def _key(self, txt: str) -> str:
        return str(txt or "").split(",", 1)[0].strip()

    @classmethod
    def _load_items_from_db(cls) -> List[Tuple[str, str]]:
        # Единственный источник правды: MaterialDispersion("").materials_list()
        

        m = MaterialDispersion("Air")           # “технический слой” без имени, как в твоём примере
        summary = m.materials_list()         # index=Element, columns: λ_min, λ_max, Source

        items: List[Tuple[str, str]] = []
        if getattr(summary, "empty", True):
            return items

        for el in list(summary.index):
            name = str(el).strip()
            if not name:
                continue

            src = ""
            if "Source" in getattr(summary, "columns", []):
                try:
                    val = summary.loc[el, "Source"]
                    if hasattr(val, "item"):
                        val = val.item()
                    src = "" if val is None else str(val).strip()
                except Exception:
                    src = ""

            items.append((name, src))

        items.sort(key=lambda x: x[0].lower())
        return items

    @classmethod
    def db_items(cls, *, force: bool = False, debug: bool = False) -> List[Tuple[str, str]]:
        if force or cls._cached_items is None:
            try:
                cls._cached_items = cls._load_items_from_db()
                cls._cached_err = ""
            except Exception as e:
                cls._cached_items = []
                cls._cached_err = repr(e)

            if debug:
                print(f"[disp] materials_list: n={len(cls._cached_items)} err={cls._cached_err}")

        return list(cls._cached_items or [])

    def refresh_materials(self, *, force: bool = False, debug: bool = False) -> None:
        items = self.db_items(force=force, debug=debug)

        # фоллбек, чтобы combo не был совсем пустой, если база не найдена/пустая
        if not items:
            items = [("Air", "")]

        self.set_material_choices(items, debug=debug)

    def set_material_choices(self, items: List[Tuple[str, str]], *, debug: bool = False) -> None:
        key = self._pending_material_key or self._key(self.material.currentText())

        self.material.clear()
        for name, src in (items or []):
            name = str(name).strip()
            if not name:
                continue
            src = "" if src is None else str(src).strip()
            self.material.addItem(f"{name}, {src}" if src else name)

        self._pending_material_key = None

        if key:
            i = self.material.findText(key, Qt.MatchStartsWith)
            if i >= 0:
                self.material.setCurrentIndex(i)
                return

        if self.material.count() > 0:
            self.material.setCurrentIndex(0)

        if debug:
            print(f"[disp] combo count={self.material.count()} current='{self.material.currentText()}'")

    def state(self) -> LayerState:
        key = self._key(self.material.currentText()) or "Air"
        return LayerState(self.TYPE_NAME, {"material": key, "d": self.d.valueSI()})

    def set_state(self, st: LayerState) -> None:
        key = self._key(st.params.get("material", "")) or "Air"

        if self.material.count() == 0:
            self._pending_material_key = key
        else:
            i = self.material.findText(key, Qt.MatchStartsWith)
            if i >= 0:
                self.material.setCurrentIndex(i)

        self.d.setValueSI(st.params.get("d", 0.0))
        self.changed.emit()


LAYER_TYPE_REGISTRY: Dict[str, Type[LayerTypeEditor]] = {
    DielectricEditor.TYPE_NAME: DielectricEditor,
    MetalEditor.TYPE_NAME: MetalEditor,
    GradientEditor.TYPE_NAME: GradientEditor,
    AnisotropicEditor.TYPE_NAME: AnisotropicEditor,
    CauchyEditor.TYPE_NAME: CauchyEditor,      # ← ДОБАВИТЬ
    LorentzDrudeEditor.TYPE_NAME: LorentzDrudeEditor,  # ← ДОБАВИТЬ
    DispersionEditor.TYPE_NAME: DispersionEditor,
}


# ============================================================
# LayerWidget (Variant B: all editors exist, only one visible)
# ============================================================

class LayerWidget(QWidget):
    """
    Один слой: выбор типа + редактор параметров.

    Variant B:
    - создаём все редакторы один раз
    - при смене типа: старый editor hide, новый show
    - высота подстраивается под активный editor
    """
    changed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None, *, allowed_types: Optional[List[str]] = None):
        super().__init__(parent)

        types = allowed_types if allowed_types is not None else list(LAYER_TYPE_REGISTRY.keys())
        if not types:
            raise ValueError("LayerWidget: no layer types available")

        self.type_box = QComboBox(self)
        self.type_box.addItems(types)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(2)
        top.addWidget(QLabel("Type:", self))
        top.addWidget(self.type_box, 1)

        self.type_frame = QFrame(self)
        self.type_frame.setFrameShape(QFrame.StyledPanel)
        self.type_frame.setFrameShadow(QFrame.Raised)

        self._frame_lay = QVBoxLayout(self.type_frame)
        self._frame_lay.setContentsMargins(4, 4, 4, 4)
        self._frame_lay.setSpacing(2)

        self._editors: Dict[str, LayerTypeEditor] = {}
        for t in types:
            ed = LAYER_TYPE_REGISTRY[t](self.type_frame)
            ed.setVisible(False)
            ed.changed.connect(self.changed)
            self._editors[t] = ed
            self._frame_lay.addWidget(ed)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)
        root.addLayout(top)
        root.addWidget(self.type_frame)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

        self._active_type: Optional[str] = None
        self.type_box.currentTextChanged.connect(self._on_type_changed)
        self._on_type_changed(self.type_box.currentText())
        
        # Синхронизация толщины между редакторами
        self._shared_d_si = 0.0
        self._sync_d_guard = False
        
        for name, ed in self._editors.items():
            d = getattr(ed, "d", None)
            if d is not None:
                d.valueChanged.connect(lambda checked, src_ed=ed: self._on_d_changed(src_ed))
        
        # self.type_box.currentTextChanged.connect(self._on_type_changed)

    def _on_d_changed(self, src_editor) -> None:
        """Любой d изменился → синхронизировать все остальные."""
        if self._sync_d_guard:
            return
        
        src_d = getattr(src_editor, "d", None)
        if src_d is None:
            return
        
        new_d_si = float(src_d.valueSI())
        if abs(new_d_si - self._shared_d_si) < 1e-12:  # без изменений
            return
        
        self._sync_d_guard = True
        try:
            self._shared_d_si = new_d_si
            # Раздать ВСЕМ кроме источника
            for ed in self._editors.values():
                if ed is src_editor:
                    continue
                target_d = getattr(ed, "d", None)
                if target_d is None:
                    continue
                target_d.blockSignals(True)
                target_d.setValueSI(new_d_si)
                target_d.blockSignals(False)
        finally:
            self._sync_d_guard = False
        
        self.changed.emit()
    
    def _sync_all_thickness(self) -> None:
        """Принудительно синхронизировать все d по текущему активному."""
        current_type = self.type_box.currentText()
        current_ed = self._editors.get(current_type)
        if current_ed is None:
            return
        
        current_d = getattr(current_ed, "d", None)
        if current_d is None:
            return
        
        self._sync_d_guard = True
        try:
            self._shared_d_si = float(current_d.valueSI())
            for ed in self._editors.values():
                target_d = getattr(ed, "d", None)
                if target_d is None:
                    continue
                target_d.blockSignals(True)
                target_d.setValueSI(self._shared_d_si)
                target_d.blockSignals(False)
        finally:
            self._sync_d_guard = False

    def _on_type_changed(self, t: str) -> None:
        if t == self._active_type:
            return
    
        if self._active_type in self._editors:
            self._editors[self._active_type].setVisible(False)
    
        self._editors[t].setVisible(True)
        self._active_type = t
    
        # помогает layout правильно пересчитать высоту
        self._frame_lay.invalidate()
        self.type_frame.updateGeometry()
        self.updateGeometry()
    
        # Синхронизация толщины при переключении типа
        self._sync_all_thickness()
        
        self.changed.emit()


    def set_thickness_enabled(self, enabled: bool) -> None:
        """Только enable/disable поля толщины d (без изменения значения)."""
        enabled = bool(enabled)
        for _t, ed in self._editors.items():
            d = getattr(ed, "d", None)
            if isinstance(d, ParamField):
                d.setEditingEnabled(enabled)

    # ---- physical state (SI) ----
    def get_state(self) -> LayerState:
        t = self.type_box.currentText()
        return self._editors[t].state()

    def set_state(self, st: LayerState) -> None:
        if st.type in self._editors:
            self.type_box.setCurrentText(st.type)
            self._editors[st.type].set_state(st)
        self.changed.emit()
        self._sync_all_thickness()

    # ---- exact UI state (digits + units + extras) ----
    def _editor_paramfields(self, ed: QWidget) -> Dict[str, ParamField]:
        out: Dict[str, ParamField] = {}
        for name, obj in vars(ed).items():
            if isinstance(obj, ParamField):
                out[name] = obj
        return out

    def get_ui_state(self) -> Dict[str, Any]:
        t = self.type_box.currentText()
        ed = self._editors[t]
    
        fields = {name: pf.ui_state() for name, pf in self._editor_paramfields(ed).items()}
    
        extra: Dict[str, Any] = {}
        mat = getattr(ed, "material", None)
        if isinstance(mat, QComboBox):
            extra["material"] = str(mat.currentText()).split(",", 1)[0].strip()
    
        # Gradient: только profile
        prof = getattr(ed, "profile", None)
        if isinstance(prof, dict):
            extra["profile"] = dict(prof)
    
        return {"type": t, "fields": fields, "extra": extra}

    def set_ui_state(self, st: Dict[str, Any]) -> None:
        t = st.get("type", None)
        if isinstance(t, str) and t in self._editors:
            self.type_box.setCurrentText(t)
        else:
            t = self.type_box.currentText()
    
        ed = self._editors[t]
    
        # extras
        extra = st.get("extra", {})
        if isinstance(extra, dict):
            mat = getattr(ed, "material", None)
            if isinstance(mat, QComboBox) and isinstance(extra.get("material", None), str):
                key = str(extra["material"]).split(",", 1)[0].strip()
                i = mat.findText(key, Qt.MatchStartsWith)
                if i >= 0:
                    mat.setCurrentIndex(i)
    
            # Gradient: только profile
            if isinstance(extra.get("profile", None), dict) and hasattr(ed, "profile"):
                setattr(ed, "profile", dict(extra["profile"]))
    
        # fields
        fields = st.get("fields", {})
        if isinstance(fields, dict):
            pf_map = self._editor_paramfields(ed)
            for name, pf_state in fields.items():
                if name in pf_map and isinstance(pf_state, dict):
                    pf_map[name].set_ui_state(pf_state)
    
        self.changed.emit()
        # Синхронизировать толщину по всем редакторам
        self._sync_all_thickness()

