# -*- coding: utf-8 -*-
"""Старый tkinter-интерфейс (запасной): python main_tk.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.main_window import MainApp

if __name__ == "__main__":
    MainApp().mainloop()
