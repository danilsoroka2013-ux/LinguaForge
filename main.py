# -*- coding: utf-8 -*-
"""Точка входа LinguaForge.

Запуск:  python main.py
Требования: Python 3.9+, numpy, PySide6. Всё остальное — стандартная библиотека.
"""
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    try:
        import numpy  # noqa: F401
    except ImportError:
        print("Не найден numpy. Установите: pip install numpy")
        return 1
    try:
        from PySide6 import QtWidgets  # noqa: F401
    except ImportError:
        print("Не найден PySide6. Установите: pip install PySide6")
        return 1

    from app.main_window_qt import run_app
    try:
        return run_app()
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
