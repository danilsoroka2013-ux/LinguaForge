# -*- coding: utf-8 -*-
"""SVG-иконки приложения: загрузка из assets/icons и перекраска под тему.

Qt умеет рендерить SVG через QtSvg. Чтобы иконки реагировали на активное
состояние навигации (неактивный — серый, активный — бирюзовый), файл
перечитывается с заменой цвета обводки/заливки на нужный.
"""
from __future__ import annotations

import os
import sys

from PySide6.QtCore import QByteArray, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

from ui_qt import theme as T

def _app_root() -> str:
    """Корень приложения: в exe — распакованный PyInstaller-каталог."""
    base = getattr(sys, "_MEIPASS", None)
    if base:
        return base
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


ICONS_DIR = os.path.join(_app_root(), "assets", "icons")

_cache: dict[tuple[str, str], QIcon] = {}


def _svg_bytes(name: str, color: str) -> QByteArray:
    path = os.path.join(ICONS_DIR, name + ".svg")
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    raw = raw.replace("#9aa7b8", color)
    return QByteArray(raw.encode("utf-8"))


def _render_icon(name: str, color: str, size: int = 22) -> QIcon:
    renderer = QSvgRenderer(_svg_bytes(name, color))
    icon = QIcon()
    for dpr in (1, 2):  # обычные и HiDPI-экраны
        px = QPixmap(size * dpr, size * dpr)
        px.fill(Qt.transparent)
        p = QPainter(px)
        p.setRenderHint(QPainter.Antialiasing)
        renderer.render(p)
        p.end()
        px.setDevicePixelRatio(dpr)
        icon.addPixmap(px)
    return icon


def icon(name: str, color: str, size: int = 22) -> QIcon:
    """Иконка по имени файла без расширения, перекрашенная в color."""
    key = (name, color)
    ic = _cache.get(key)
    if ic is None:
        ic = _render_icon(name, color, size)
        _cache[key] = ic
    return ic


# Цвета из темы — чтобы не хардкодить в окне.
def normal_color() -> str:
    return T.TEXT_DIM


def active_color() -> str:
    return T.ACCENT
