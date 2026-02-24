# -*- coding: utf-8 -*-
"""
p_reflection_curves_app/digit_number.py

Общий модуль для "цифрового" ввода:
- DigitFormat: количество целых/дробных разрядов.
- DigitNumberEdit: виджет числа как набор DigitCell, с сохранением точных разрядов через scaled integer.
- DigitCell: одна цифра, с up/down и вводом одной цифры.
- SelectAllLineEdit: автоселект текста при фокусе/клике.

Ключевая идея "точности":
- Значение хранится как целое scaled (например, 1.234 при dec=3 -> 1234).
- Это позволяет сохранять/восстанавливать ровно те же цифры (без float-ошибок).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

from PyQt5.QtCore import Qt, pyqtSignal, QObject, QEvent
from PyQt5.QtGui import QIntValidator, QFocusEvent, QMouseEvent
from PyQt5.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QToolButton,
    QLineEdit,
    QLabel,
    QSizePolicy,
)



class SelectAllLineEdit(QLineEdit):
    """QLineEdit, который выделяет весь текст при получении фокуса и при клике."""

    def focusInEvent(self, event: QFocusEvent) -> None:
        super().focusInEvent(event)
        self.selectAll()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        super().mousePressEvent(event)
        self.selectAll()


class DigitCell(QWidget):
    """
    Одна цифра:
    - up/down кнопки (автоповтор)
    - QLineEdit на 1 символ с валидатором 0..9
    - стрелки влево/вправо перемещают фокус между соседними цифрами
    """

    digitEdited = pyqtSignal(int, int)      # (index, digit 0..9)
    stepRequested = pyqtSignal(int, int)    # (index, direction +1/-1)
    focusMoveRequested = pyqtSignal(int)    # (+1 right, -1 left)

    def __init__(self, index: int, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._index = int(index)
        self._updating = False

        # Небольшие размеры, чтобы было похоже на оригинал.
        self._edit_w = 18
        self._btn_h = 10
        self._btn_font_px = 10

        self.btn_up = QToolButton(self)
        self.btn_up.setText("˄")
        self.btn_up.setAutoRepeat(True)
        self.btn_up.setAutoRepeatDelay(300)
        self.btn_up.setAutoRepeatInterval(60)

        self.edit = SelectAllLineEdit(self)
        self.edit.setMaxLength(1)
        self.edit.setFixedWidth(self._edit_w)
        self.edit.setAlignment(Qt.AlignCenter)
        self.edit.setValidator(QIntValidator(0, 9, self.edit))
        self.edit.setText("0")

        self.btn_dn = QToolButton(self)
        self.btn_dn.setText("˅")
        self.btn_dn.setAutoRepeat(True)
        self.btn_dn.setAutoRepeatDelay(300)
        self.btn_dn.setAutoRepeatInterval(60)

        for b in (self.btn_up, self.btn_dn):
            b.setFixedWidth(self._edit_w)
            b.setFixedHeight(self._btn_h)
            b.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            b.setFocusPolicy(Qt.NoFocus)  # фокус остаётся на edit
            b.setStyleSheet(
                "QToolButton { padding: 0px; margin: 0px; }"
                f"QToolButton {{ font-size: {self._btn_font_px}px; }}"
            )

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(1)
        lay.addWidget(self.btn_up)
        lay.addWidget(self.edit)
        lay.addWidget(self.btn_dn)

        self.btn_up.clicked.connect(lambda: self.stepRequested.emit(self._index, +1))
        self.btn_dn.clicked.connect(lambda: self.stepRequested.emit(self._index, -1))
        self.edit.textEdited.connect(self._on_text_edited)

        # Навигация стрелками и шаг вверх/вниз прямо с клавиатуры
        self.edit.installEventFilter(self)

    def setDigit(self, d: int) -> None:
        d = int(d)
        if d < 0:
            d = 0
        if d > 9:
            d = 9

        self._updating = True
        try:
            self.edit.setText(str(d))
        finally:
            self._updating = False

    def digit(self) -> int:
        t = self.edit.text()
        if t == "":
            return 0
        try:
            d = int(t)
        except ValueError:
            return 0
        return max(0, min(9, d))

    def focusAndSelect(self) -> None:
        self.edit.setFocus(Qt.TabFocusReason)
        self.edit.selectAll()

    def _on_text_edited(self, text: str) -> None:
        if self._updating:
            return

        cleaned = text or "0"
        # оставляем последнюю цифру, если вставили несколько символов
        if len(cleaned) > 1:
            cleaned = cleaned[-1]
        if not cleaned.isdigit():
            cleaned = "0"

        d = int(cleaned)
        if d > 9:
            cleaned = "0"
            d = 0

        if cleaned != self.edit.text():
            self._updating = True
            try:
                self.edit.setText(cleaned)
            finally:
                self._updating = False

        self.digitEdited.emit(self._index, d)
        # как в оригинале: после ввода сдвигаемся вправо
        self.focusMoveRequested.emit(+1)

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if obj is self.edit and event.type() == QEvent.KeyPress:
            key = event.key()
            if key == Qt.Key_Left:
                self.focusMoveRequested.emit(-1)
                return True
            if key == Qt.Key_Right:
                self.focusMoveRequested.emit(+1)
                return True
            if key == Qt.Key_Up:
                self.stepRequested.emit(self._index, +1)
                return True
            if key == Qt.Key_Down:
                self.stepRequested.emit(self._index, -1)
                return True

        return super().eventFilter(obj, event)


@dataclass(frozen=True)
class DigitFormat:
    """Сколько разрядов: целые + дробные."""
    integer_digits: int = 4
    decimal_digits: int = 3

    @property
    def total_digits(self) -> int:
        return int(self.integer_digits + self.decimal_digits)

    @property
    def scale(self) -> int:
        return 10 ** int(self.decimal_digits)

    @property
    def max_scaled(self) -> int:
        # максимум для total_digits (например, 4+3 -> 9999999)
        return (10 ** self.total_digits) - 1


class DigitNumberEdit(QWidget):
    """
    Число как набор DigitCell.

    Интерфейс:
    - value() -> float
    - setValue(float)
    - scaledValue() -> int         (точное целое состояние разрядов)
    - setScaledValue(int)          (точное восстановление разрядов)

    Гибкость:
    - можно передать fmt=DigitFormat(...)
    - или не передавать fmt, но задать integer_digits/decimal_digits.
    """

    valueChanged = pyqtSignal(float)

    def __init__(
        self,
        fmt: Optional[DigitFormat] = None,
        parent: Optional[QWidget] = None,
        label: str = "",
        *,
        integer_digits: int = 4,
        decimal_digits: int = 3,
    ):
        super().__init__(parent)

        self._fmt = fmt if fmt is not None else DigitFormat(integer_digits=integer_digits, decimal_digits=decimal_digits)
        self._scaled = 0
        self._updating = False

        self.lbl = QLabel(label, self)
        self.lbl.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)

        self._cells: List[DigitCell] = []
        self._weights: List[int] = []

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2)

        if label:
            row.addWidget(self.lbl)

        N = self._fmt.total_digits
        for i in range(N):
            weight = 10 ** (N - 1 - i)
            self._weights.append(weight)

            cell = DigitCell(i, self)
            cell.digitEdited.connect(self._on_digit_edited)
            cell.stepRequested.connect(self._on_step_requested)
            cell.focusMoveRequested.connect(lambda step, i=i: self._move_focus(i, step))
            self._cells.append(cell)

            row.addWidget(cell)

            # точка между целой и дробной частью
            if i == self._fmt.integer_digits - 1 and self._fmt.decimal_digits > 0:
                dot = QLabel(".", self)
                dot.setAlignment(Qt.AlignCenter)
                dot.setFixedWidth(6)
                row.addWidget(dot)

        self.setLayout(row)

        # стартовое значение
        self.setValue(0.0, emit_signal=False)

    @property
    def fmt(self) -> DigitFormat:
        return self._fmt

    def value(self) -> float:
        return self._scaled / self._fmt.scale

    def scaledValue(self) -> int:
        return int(self._scaled)

    def setScaledValue(self, scaled: int, *, emit_signal: bool = False) -> None:
        scaled = int(scaled)
        scaled = max(0, min(self._fmt.max_scaled, scaled))

        if scaled == self._scaled:
            return

        self._scaled = scaled
        self._render_from_scaled()

        if emit_signal:
            self.valueChanged.emit(self.value())

    def setValue(self, value: float, *, emit_signal: bool = False) -> None:
        scaled = int(round(float(value) * self._fmt.scale))
        self.setScaledValue(scaled, emit_signal=emit_signal)

    def focusFirst(self) -> None:
        if self._cells:
            self._cells[0].focusAndSelect()

    def _render_from_scaled(self) -> None:
        s = str(self._scaled).rjust(self._fmt.total_digits, "0")[-self._fmt.total_digits:]
        self._updating = True
        try:
            for i, ch in enumerate(s):
                self._cells[i].setDigit(int(ch))
        finally:
            self._updating = False

    def _recompute_scaled_from_cells(self) -> int:
        scaled = 0
        for cell, w in zip(self._cells, self._weights):
            scaled += int(cell.digit()) * w
        return max(0, min(self._fmt.max_scaled, scaled))

    def _on_digit_edited(self, _index: int, _digit: int) -> None:
        if self._updating:
            return

        new_scaled = self._recompute_scaled_from_cells()
        if new_scaled != self._scaled:
            self._scaled = new_scaled
            self.valueChanged.emit(self.value())

    def _on_step_requested(self, index: int, direction: int) -> None:
        delta = int(direction) * self._weights[index]
        self.setScaledValue(self._scaled + delta, emit_signal=True)
        self._cells[index].focusAndSelect()

    def _move_focus(self, from_index: int, step: int) -> None:
        to = int(from_index) + int(step)
        if 0 <= to < len(self._cells):
            self._cells[to].focusAndSelect()
