# -*- coding: utf-8 -*-
"""Тема LinguaForge для Qt: палитра + полный QSS (CSS внутри Python).

QSS — это stylesheet-язык Qt, синтаксис почти 1:1 с CSS: селекторы,
свойства, псевдоклассы :hover/:pressed/:focus. Никакого HTML.
"""
from __future__ import annotations

# --- палитра (как в движке, чтобы всё было в одном стиле) ----------------
BG         = "#0f1216"
BG_ALT     = "#141920"
CARD       = "#171d26"
CARD_HI    = "#1d2530"
BORDER     = "#242d3a"
BORDER_HI  = "#314052"

TEXT       = "#e8edf4"
TEXT_DIM   = "#9aa7b8"
TEXT_FAINT = "#66738a"

ACCENT     = "#3fd0c9"
ACCENT_HI  = "#5adcd5"
ACCENT_DN  = "#2aa39e"
ACCENT_TXT = "#07211f"

OK     = "#5fd07a"
WARN   = "#e8b64c"
ERR    = "#e5646e"

CHART_GRID  = "#232b38"
CHART_LINE  = "#3fd0c9"
CHART_LINE2 = "#e8b64c"

FONT   = '"Segoe UI"'
MONO   = '"Consolas"'


def qss() -> str:
    """Полный stylesheet приложения."""
    return f"""
/* ---------------- база ---------------- */
QWidget {{
    background-color: {BG};
    color: {TEXT};
    font-family: {FONT};
    font-size: 13px;
}}
QMainWindow, #root {{ background-color: {BG}; }}

/* ---------------- сайдбар ---------------- */
#sidebar {{
    background-color: {BG_ALT};
    border-right: 1px solid {BORDER};
}}
#appTitle {{
    background: transparent; color: {TEXT};
    font-size: 19px; font-weight: 700; padding: 20px 18px 0 18px;
}}
#appSub {{
    background: transparent; color: {TEXT_FAINT};
    font-size: 11px; padding: 0 18px 14px 18px;
}}
QPushButton.nav {{
    background: transparent; color: {TEXT_DIM};
    border: none; border-radius: 8px;
    text-align: left; padding: 10px 16px;
    font-size: 14px;
}}
QPushButton.nav:hover {{ background: {CARD_HI}; color: {TEXT}; }}
QPushButton.nav:checked {{
    background: {CARD}; color: {ACCENT}; font-weight: 600;
}}
#sidebarFooter {{
    background: transparent;
    border-top: 1px solid {BORDER};
}}

/* ---------------- карточки ---------------- */
#card {{
    background-color: {CARD};
    border: 1px solid {BORDER};
    border-radius: 10px;
}}
#cardTitle {{
    background: transparent; color: {TEXT};
    font-size: 14px; font-weight: 600;
}}
#cardHint, #hint {{
    background: transparent; color: {TEXT_FAINT}; font-size: 11px;
}}
#subtitle {{ background: transparent; color: {TEXT_DIM}; font-size: 12px; }}
#h1 {{ background: transparent; color: {TEXT}; font-size: 22px; font-weight: 700; }}

/* ---------------- кнопки ---------------- */
QPushButton {{
    background-color: {CARD};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 13px;
}}
QPushButton:hover {{ background-color: {CARD_HI}; border-color: {BORDER_HI}; }}
QPushButton:pressed {{ background-color: {BG_ALT}; }}
QPushButton:disabled {{ color: {TEXT_FAINT}; border-color: {BORDER}; }}
QPushButton[variant="primary"] {{
    background-color: {ACCENT}; color: {ACCENT_TXT};
    border: 1px solid {ACCENT}; font-weight: 600;
}}
QPushButton[variant="primary"]:hover {{ background-color: {ACCENT_HI}; }}
QPushButton[variant="primary"]:pressed {{ background-color: {ACCENT_DN}; }}
QPushButton[variant="primary"]:disabled {{ background-color: {BORDER}; color: {TEXT_FAINT}; border-color: {BORDER}; }}
QPushButton[variant="danger"] {{
    background-color: {ERR}; color: #2a0d10; border: 1px solid {ERR}; font-weight: 600;
}}
QPushButton[variant="danger"]:hover {{ background-color: #ef7d86; }}
QPushButton[variant="danger"]:disabled {{ background-color: {BORDER}; color: {TEXT_FAINT}; border-color: {BORDER}; }}
QPushButton[variant="ghost"] {{ background: transparent; }}

/* ---------------- поля ввода ---------------- */
QLineEdit, QTextEdit, QPlainTextEdit {{
    background-color: {BG};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 7px 10px;
    selection-background-color: {ACCENT_DN};
}}
QLineEdit:focus, QTextEdit:focus {{ border-color: {ACCENT}; }}
QTextEdit[mono="true"] {{ font-family: {MONO}; font-size: 12px; }}

/* ---------------- радиокнопки ---------------- */
QRadioButton {{
    background: transparent; color: {TEXT_DIM};
    spacing: 7px; font-size: 13px;
}}
QRadioButton:hover {{ color: {TEXT}; }}
QRadioButton::indicator {{
    width: 15px; height: 15px;
    border: 2px solid {BORDER_HI}; border-radius: 9px;
    background: {BG};
}}
QRadioButton::indicator:hover {{ border-color: {ACCENT}; }}
QRadioButton::indicator:checked {{
    border: 4px solid {ACCENT}; background: {BG};
}}

/* ---------------- слайдеры ---------------- */
QSlider::groove:horizontal {{
    height: 4px; border-radius: 2px;
    background: {BORDER_HI};
}}
QSlider::sub-page:horizontal {{
    height: 4px; border-radius: 2px;
    background: {ACCENT};
}}
QSlider::handle:horizontal {{
    width: 15px; height: 15px; margin: -6px 0;
    border-radius: 8px;
    background: {TEXT};
}}
QSlider::handle:horizontal:hover {{ background: {ACCENT_HI}; }}
QSlider::groove:horizontal:disabled, QSlider::sub-page:horizontal:disabled {{ background: {BORDER}; }}
QSlider::handle:horizontal:disabled {{ background: {TEXT_FAINT}; }}

/* ---------------- скроллбары ---------------- */
QScrollBar:vertical {{
    background: transparent; width: 10px; margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: {BORDER_HI}; border-radius: 4px; min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{ background: {TEXT_FAINT}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {BORDER_HI}; border-radius: 4px; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

/* ---------------- чат ---------------- */
#bubbleUser, #bubbleModel, #bubbleSystem {{
    border: 1px solid {BORDER};
    border-radius: 10px;
}}
#bubbleUser  {{ background-color: {BG_ALT}; }}
#bubbleModel {{ background-color: #173034; }}
#bubbleSystem {{ background-color: #231f16; }}
#bubbleAuthor {{
    background: transparent; font-size: 10px; font-weight: 700;
    color: {TEXT_FAINT};
}}
#bubbleModel #bubbleAuthor {{ color: {ACCENT}; }}
#bubbleText {{ background: transparent; font-size: 13px; }}

/* ---------------- метрики ---------------- */
#metricValue {{ background: transparent; color: {TEXT}; font-size: 21px; font-weight: 700; }}
#metricLabel {{ background: transparent; color: {TEXT_FAINT}; font-size: 10px; }}
#statValue {{ background: transparent; color: {TEXT}; font-weight: 600; }}
#statLabel {{ background: transparent; color: {TEXT_DIM}; }}

/* ---------------- прочее ---------------- */
#statusOk    {{ color: {OK}; font-weight: 600; background: transparent; }}
#statusAccent{{ color: {ACCENT}; font-weight: 600; background: transparent; }}
#statusWarn  {{ color: {WARN}; font-weight: 600; background: transparent; }}
#statusErr   {{ color: {ERR}; font-weight: 600; background: transparent; }}
#statusDim   {{ color: {TEXT_DIM}; background: transparent; }}
QToolTip {{
    background-color: {CARD_HI}; color: {TEXT};
    border: 1px solid {BORDER_HI}; padding: 5px;
}}
"""
