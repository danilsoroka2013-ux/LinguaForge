# -*- coding: utf-8 -*-
"""Палитра и общие константы оформления.

Тёмная, спокойная тема с бирюзовым акцентом. Всё приложение рисуется
средствами tkinter: плоские панели, аккуратные шрифты, никаких картинок.
"""
from __future__ import annotations

import tkinter.font as tkfont

# --- базовые цвета ------------------------------------------------------
BG        = "#0f1216"   # фон приложения
BG_ALT    = "#141920"   # боковая панель / вторичный фон
CARD      = "#171d26"   # карточки
CARD_HI   = "#1d2530"   # карточка при наведении
BORDER    = "#242d3a"   # тонкие линии
BORDER_HI = "#314052"

TEXT      = "#e8edf4"   # основной текст
TEXT_DIM  = "#9aa7b8"   # приглушённый текст
TEXT_FAINT= "#66738a"

ACCENT    = "#3fd0c9"   # бирюзовый акцент
ACCENT_DI = "#2aa39e"   # нажатый акцент
ACCENT_TXT= "#07211f"   # текст на акцентной кнопке

OK        = "#5fd07a"
WARN      = "#e8b64c"
ERR       = "#e5646e"

CHART_LINE = "#3fd0c9"
CHART_LINE2 = "#e8b64c"
CHART_GRID = "#232b38"

# --- шрифты --------------------------------------------------------------
FONT_STACK = ("Segoe UI", "SF Pro Text", "Helvetica Neue", "Arial")


def font(size: int, bold: bool = False, mono: bool = False, family: str = "") -> tuple:
    fam = family or ("Consolas" if mono else FONT_STACK)
    return (fam, size, "bold" if bold else "normal")


def pick_family(root) -> str:
    """Выбирает первый доступный шрифт из стека (для не-Windows платформ)."""
    try:
        names = set(tkfont.families(root))
        for cand in FONT_STACK:
            if cand in names:
                return cand
        return "TkDefaultFont"
    except Exception:
        return FONT_STACK[0]


MONO_FALLBACKS = ("Consolas", "Menlo", "DejaVu Sans Mono", "Courier New")
