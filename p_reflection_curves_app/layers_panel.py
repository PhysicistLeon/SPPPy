# -*- coding: utf-8 -*-
"""
p_reflection_curves_app/layers_panel.py

Правая часть (управление слоями):
- LayerItem: карточка (LayerWidget + x/up/dn)
- LayersListWidget: scroll + add/delete/move + boundary-rule (disable thickness for top/bottom)
- LayersPanel: заголовок + список + кнопка Add (без Save/Load кнопок — они в меню главного окна)

Save/Load (JSON) остаются методами LayersListWidget, чтобы их вызывало меню.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget,
    QFrame,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QHBoxLayout,
    QToolButton,
    QPushButton,
    QSizePolicy,
)

from .layer import LayerWidget, LayerState



class LayerItem(QFrame):
    """Одна карточка: LayerWidget слева + столбик управления справа (x/up/dn)."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # Рамка карточки через QSS — стабильно между стилями
        self.setObjectName("LayerCard")
        self.setFrameShape(QFrame.NoFrame)
        self.setStyleSheet("""
            QFrame#LayerCard {
                border: 1px solid #b5b5b5;
                border-radius: 2px;
                background: #ffffff;
            }
        """)

        self.layer = LayerWidget(self)

        self.btn_del = QToolButton(self)
        self.btn_del.setText("x")

        self.btn_up = QToolButton(self)
        self.btn_up.setText("up")

        self.btn_dn = QToolButton(self)
        self.btn_dn.setText("dn")

        for b in (self.btn_del, self.btn_up, self.btn_dn):
            b.setFixedSize(28, 24)
            b.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        ctrl_col = QVBoxLayout()
        ctrl_col.setContentsMargins(0, 0, 0, 0)
        ctrl_col.setSpacing(2)
        ctrl_col.addWidget(self.btn_del)
        ctrl_col.addStretch(1)
        ctrl_col.addWidget(self.btn_up)
        ctrl_col.addWidget(self.btn_dn)

        sep = QFrame(self)
        sep.setFrameShape(QFrame.VLine)
        sep.setFrameShadow(QFrame.Sunken)
        sep.setLineWidth(1)

        root = QHBoxLayout(self)
        root.setContentsMargins(3, 3, 3, 3)
        root.setSpacing(4)
        root.addWidget(self.layer, 1)
        root.addWidget(sep)
        root.addLayout(ctrl_col)


class LayersListWidget(QWidget):
    """
    Scroll-список слоёв.

    ВАЖНО:
    - boundary rule: толщину (d) у верхнего и нижнего слоя редактировать нельзя
    - save/load JSON сохраняют точное UI-состояние через LayerWidget.get_ui_state()
    """

    layersChanged = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._items: List[LayerItem] = []

        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)

        self.inner = QWidget(self.scroll)
        self.inner_l = QVBoxLayout(self.inner)
        self.inner_l.setContentsMargins(2, 2, 2, 2)
        self.inner_l.setSpacing(4)
        self.inner_l.addStretch(1)

        self.scroll.setWidget(self.inner)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self.scroll)

    def set_default_layers(self) -> None:
        # Очистить текущие
        for it in list(self._items):
            it.setParent(None)
            it.deleteLater()
        self._items.clear()
    
        defaults = [
            LayerState("Dielectric", {"n": 1.23, "d": 0.0}),
            LayerState("Metal",      {"n": 0.05, "k": 4.24, "d": 55e-9}),  # 55 nm
            LayerState("Dielectric", {"n": 1.0,  "d": 0.0}),
        ]
    
        for st in defaults:
            item = LayerItem(self.inner)
    
            item.btn_del.clicked.connect(lambda _=False, it=item: self.delete_item(it))
            item.btn_up.clicked.connect(lambda _=False, it=item: self.move_up(it))
            item.btn_dn.clicked.connect(lambda _=False, it=item: self.move_down(it))
            item.layer.changed.connect(self._emit_changed)
    
            self.inner_l.insertWidget(self.inner_l.count() - 1, item)
            self._items.append(item)
    
            item.layer.set_state(st)
    
        self._update_boundary_thickness_enabled()
        self._emit_changed()

    

    def _update_boundary_thickness_enabled(self) -> None:
        n = len(self._items)
        for i, it in enumerate(self._items):
            is_boundary = (i == 0) or (i == n - 1)
            it.layer.set_thickness_enabled(not is_boundary)

    def _emit_changed(self) -> None:
        self.layersChanged.emit()

    

    def add_layer(self) -> None:
        item = LayerItem(self.inner)

        item.btn_del.clicked.connect(lambda _=False, it=item: self.delete_item(it))
        item.btn_up.clicked.connect(lambda _=False, it=item: self.move_up(it))
        item.btn_dn.clicked.connect(lambda _=False, it=item: self.move_down(it))

        item.layer.changed.connect(self._emit_changed)

        self.inner_l.insertWidget(self.inner_l.count() - 1, item)
        self._items.append(item)

        self._update_boundary_thickness_enabled()
        self._emit_changed()

    def delete_item(self, item: LayerItem) -> None:
        if item not in self._items:
            return

        self._items.remove(item)
        item.setParent(None)
        item.deleteLater()

        self._update_boundary_thickness_enabled()
        self._emit_changed()

    def move_up(self, item: LayerItem) -> None:
        i = self._items.index(item)
        if i <= 0:
            return

        self._items[i - 1], self._items[i] = self._items[i], self._items[i - 1]
        self._rebuild_layout()

        self._update_boundary_thickness_enabled()
        self._emit_changed()

    def move_down(self, item: LayerItem) -> None:
        i = self._items.index(item)
        if i >= len(self._items) - 1:
            return

        self._items[i + 1], self._items[i] = self._items[i], self._items[i + 1]
        self._rebuild_layout()

        self._update_boundary_thickness_enabled()
        self._emit_changed()

    def _rebuild_layout(self) -> None:
        # оставляем последний stretch
        while self.inner_l.count() > 1:
            w = self.inner_l.itemAt(0).widget()
            self.inner_l.removeWidget(w)
            if w is not None:
                w.setParent(None)

        for it in self._items:
            self.inner_l.insertWidget(self.inner_l.count() - 1, it)

    # --- exact UI save/load ---
    def export_layers(self) -> Dict[str, Any]:
        return {"version": 1, "layers": [it.layer.get_ui_state() for it in self._items]}

    def import_layers(self, data: Dict[str, Any]) -> None:
        layers = data.get("layers", [])
        if not isinstance(layers, list):
            return

        # clear
        for it in list(self._items):
            it.setParent(None)
            it.deleteLater()
        self._items.clear()

        # recreate in exact order
        for layer_st in layers:
            if not isinstance(layer_st, dict):
                continue

            item = LayerItem(self.inner)
            item.btn_del.clicked.connect(lambda _=False, it=item: self.delete_item(it))
            item.btn_up.clicked.connect(lambda _=False, it=item: self.move_up(it))
            item.btn_dn.clicked.connect(lambda _=False, it=item: self.move_down(it))
            item.layer.changed.connect(self._emit_changed)

            self.inner_l.insertWidget(self.inner_l.count() - 1, item)
            self._items.append(item)

            item.layer.set_ui_state(layer_st)

        self._update_boundary_thickness_enabled()
        self._emit_changed()

class LayersPanel(QFrame):
    """Правая панель: заголовок + список + кнопка 'Добавить слой' (без Save/Load)."""

    layersChanged = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        title = QLabel("Последовательность слоёв", self)
        title.setAlignment(Qt.AlignCenter)

        self.list = LayersListWidget(self)
        self.list.layersChanged.connect(self.layersChanged)
        self.list.set_default_layers()
        btn_add = QPushButton("Добавить слой", self)

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)
        root.addWidget(title)
        root.addWidget(self.list, 1)
        root.addWidget(btn_add)

        btn_add.clicked.connect(self.list.add_layer)
