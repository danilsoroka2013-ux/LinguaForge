# -*- coding: utf-8 -*-
"""Виджеты Qt: график лосса на QPainter, чат, плитки метрик, слайдеры."""
from __future__ import annotations

from PySide6.QtCore import Qt, QPointF, QSize, Signal, QTimer
from PySide6.QtGui import QColor, QPainter, QPen, QFont
from PySide6.QtWidgets import (
    QWidget, QLabel, QSlider, QVBoxLayout, QHBoxLayout, QFrame, QSizePolicy,
)

from ui_qt import theme as T
from ui_qt import icons


# ----------------------------------------------------------------- карточки

def card() -> QFrame:
    f = QFrame()
    f.setObjectName("card")
    return f


def vwrap(*widgets, spacing: int = 8) -> QFrame:
    """Вертикальная укладка внутри прозрачного контейнера."""
    w = QWidget()
    w.setStyleSheet("background: transparent")
    lay = QVBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(spacing)
    for x in widgets:
        lay.addWidget(x)
    return w


def label(text: str, name: str = "", wrap: int = 0) -> QLabel:
    lbl = QLabel(text)
    if name:
        lbl.setObjectName(name)
    if wrap:
        lbl.setWordWrap(True)
        lbl.setMaximumWidth(wrap)
    return lbl


class StatRow(QWidget):
    """Строка «название — значение»."""

    def __init__(self, caption: str, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.cap = label(caption, "statLabel")
        self.val = label("—", "statValue")
        lay.addWidget(self.cap)
        lay.addStretch(1)
        lay.addWidget(self.val)

    def set(self, v: str) -> None:
        self.val.setText(v)


class MetricTile(QFrame):
    """Плитка-метрика: подпись + крупное значение."""

    def __init__(self, caption: str, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 9, 12, 9)
        lay.setSpacing(1)
        self.cap = label(caption, "metricLabel")
        self.val = label("—", "metricValue")
        lay.addWidget(self.cap)
        lay.addWidget(self.val)

    def set(self, v: str) -> None:
        self.val.setText(v)


class ParamSlider(QWidget):
    """Подпись + текущее значение + слайдер. signal changed(float)."""

    changed = Signal(float)

    def __init__(self, caption: str, lo: float, hi: float, value: float,
                 step: float = 1.0, fmt=None, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent")
        self.fmt = fmt or (lambda v: f"{v:g}")
        self.step = step
        self._suppress = False

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.addWidget(label(caption, "statLabel"))
        head.addStretch(1)
        self.val_lbl = label(self.fmt(value), "statValue")
        self.val_lbl.setStyleSheet(f"color: {T.ACCENT}; font-weight: 600; background: transparent")
        head.addWidget(self.val_lbl)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(int(round((hi - lo) / step)))
        self.slider.setValue(int(round((value - lo) / step)))
        self._lo, self._hi, self._step = lo, hi, step
        self.slider.valueChanged.connect(self._fire)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lay.addLayout(head)
        lay.addWidget(self.slider)

    def _fire(self, _=0) -> None:
        if self._suppress:
            return
        v = self.value()
        self.val_lbl.setText(self.fmt(v))
        self.changed.emit(v)

    def value(self) -> float:
        return self._lo + self.slider.value() * self._step

    def set_value(self, v: float) -> None:
        self._suppress = True
        try:
            self.slider.setValue(int(round((v - self._lo) / self._step)))
            self.val_lbl.setText(self.fmt(v))
        finally:
            self._suppress = False


# ----------------------------------------------------------------- график

class LossChart(QWidget):
    """График лосса на QPainter: сетка, две серии, последняя точка."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(220)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._train: list[tuple[float, float]] = []
        self._val: list[tuple[float, float]] = []

    def set_data(self, hist: list[dict]) -> None:
        tr, va = [], []
        for h in hist:
            step = h.get("step", 0)
            loss = h.get("loss")
            if loss is not None and loss == loss:
                tr.append((float(step), float(loss)))
            v = h.get("val")
            if v is not None and v == v:
                va.append((float(step), float(v)))
        self._train, self._val = tr, va
        self.update()

    def clear(self) -> None:
        self._train, self._val = [], []
        self.update()

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(T.CARD))
        pad_l, pad_r, pad_t, pad_b = 46, 12, 14, 22

        p.setPen(QPen(QColor(T.TEXT_FAINT)))
        p.setFont(QFont("Segoe UI", 7))
        p.drawText(8, 14, "лосс")

        pts = self._train + self._val
        if not pts:
            p.setPen(QPen(QColor(T.TEXT_FAINT)))
            p.drawText(w // 2 - 60, h // 2, "данных пока нет")
            p.end()
            return

        xs = [a for a, _ in pts]
        ys = [b for _, b in pts]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(min(ys), 0.0), max(ys) * 1.08 + 1e-6

        def X(v):  # noqa: E741
            if x1 - x0 < 1e-9:
                return pad_l + (w - pad_l - pad_r) / 2
            return pad_l + (v - x0) / (x1 - x0) * (w - pad_l - pad_r)

        def Y(v):  # noqa: E741
            return h - pad_b - (v - y0) / (y1 - y0) * (h - pad_t - pad_b)

        # сетка
        grid_pen = QPen(QColor(T.CHART_GRID), 1)
        p.setFont(QFont("Segoe UI", 7))
        for i in range(5):
            v = y0 + (y1 - y0) * i / 4
            yy = Y(v)
            p.setPen(grid_pen)
            p.drawLine(int(pad_l), int(yy), int(w - pad_r), int(yy))
            p.setPen(QPen(QColor(T.TEXT_FAINT)))
            p.drawText(4, int(yy) + 4, f"{v:.2f}")

        def draw(series: list, color: str, width: float) -> None:
            if not series:
                return
            p.setPen(QPen(QColor(color), width))
            p.setRenderHint(QPainter.Antialiasing, True)
            for i in range(1, len(series)):
                p.drawLine(QPointF(X(series[i - 1][0]), Y(series[i - 1][1])),
                           QPointF(X(series[i][0]), Y(series[i][1])))
            if len(series) == 1:
                p.drawPoint(QPointF(X(series[0][0]), Y(series[0][1])))

        draw(self._val, T.CHART_LINE2, 1.6)
        draw(self._train, T.CHART_LINE, 2.0)

        if self._train:
            lx, ly = self._train[-1]
            p.setBrush(QColor(T.CHART_LINE))
            p.setPen(QPen(QColor(T.CHART_LINE)))
            p.drawEllipse(QPointF(X(lx), Y(ly)), 3, 3)
            p.setPen(QPen(QColor(T.CHART_LINE)))
            p.drawText(int(X(lx)) - 20, int(Y(ly)) - 8, f"{ly:.3f}")

        if x1 > x0:
            p.setPen(QPen(QColor(T.TEXT_FAINT)))
            p.drawText(int(pad_l), h - 8, str(int(x0)))
            p.drawText(int(w - pad_r) - 60, h - 8, f"шаг {int(x1)}")
        p.end()


# ----------------------------------------------------------------- чат

class Bubble(QFrame):
    """Пузырь сообщения: user / model / system."""

    def __init__(self, role: str, text: str, max_width: int = 620, parent=None):
        super().__init__(parent)
        self.setObjectName({"user": "bubbleUser", "model": "bubbleModel",
                            "system": "bubbleSystem"}[role])
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 7, 12, 9)
        lay.setSpacing(2)
        author = {"user": "Вы", "model": "Модель", "system": "Система"}[role]
        lay.addWidget(label(author, "bubbleAuthor"))
        t = label(text, "bubbleText")
        t.setWordWrap(True)
        t.setTextInteractionFlags(Qt.TextSelectableByMouse)
        t.setMaximumWidth(max_width)
        lay.addWidget(t)


class ChatWidget(QWidget):
    """Скролл-лента пузырей + поле ввода + кнопка отправки."""

    send = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent")
        from PySide6.QtWidgets import QScrollArea, QLineEdit, QPushButton

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setStyleSheet("background: transparent")
        self.inner = QWidget()
        self.inner.setStyleSheet("background: transparent")
        self.lay = QVBoxLayout(self.inner)
        self.lay.setContentsMargins(8, 8, 8, 8)
        self.lay.setSpacing(8)
        self.lay.addStretch(1)
        self.scroll.setWidget(self.inner)

        row = QHBoxLayout()
        row.setContentsMargins(8, 4, 8, 8)
        self.input = QLineEdit()
        self.input.setPlaceholderText("Напишите что-нибудь — модель продолжит…")
        self.btn = QPushButton("Отправить")
        self.btn.setProperty("variant", "primary")
        self.btn.setIcon(icons.icon("spark", T.ACCENT_TXT))
        self.btn.setIconSize(QSize(16, 16))
        self.btn.clicked.connect(self._emit)
        self.input.returnPressed.connect(self._emit)
        self.btn_clear = QPushButton("Очистить")
        self.btn_clear.setToolTip("Удалить все сообщения из чата")
        self.btn_clear.setIcon(icons.icon("broom", icons.normal_color()))
        self.btn_clear.setIconSize(QSize(16, 16))
        self.btn_clear.clicked.connect(self.clear)
        row.addWidget(self.input, 1)
        row.addWidget(self.btn)
        row.addWidget(self.btn_clear)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.scroll, 1)
        lay.addLayout(row)

    def _emit(self) -> None:
        text = self.input.text().strip()
        if text:
            self.input.clear()
            self.send.emit(text)

    def add(self, role: str, text: str) -> None:
        b = Bubble(role, text)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        if role == "user":
            row.addStretch(1)
            row.addWidget(b)
        else:
            row.addWidget(b)
            row.addStretch(1)
        wrap = QWidget()
        wrap.setStyleSheet("background: transparent")
        wrap.setLayout(row)
        self.lay.insertWidget(self.lay.count() - 1, wrap)
        QTimer.singleShot(30, self._to_bottom)

    def _to_bottom(self) -> None:
        sb = self.scroll.verticalScrollBar()
        sb.setValue(sb.maximum())

    def set_busy(self, busy: bool) -> None:
        self.input.setEnabled(not busy)
        self.btn.setEnabled(not busy)
        self.btn_clear.setEnabled(not busy)
        if busy:
            self.input.setPlaceholderText("Модель думает…")
        else:
            self.input.setPlaceholderText("Напишите что-нибудь — модель продолжит…")

    def clear(self) -> None:
        while self.lay.count() > 1:
            item = self.lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
