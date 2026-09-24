# -*- coding: utf-8 -*-
"""Переиспользуемые виджеты: кнопки, поля, карточки, графики, чат, тосты.

Ничего внешнего: только tkinter/ttk, всё рисуется вручную и выглядит
единообразно в тёмной теме.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Callable, List, Optional, Sequence, Tuple

from ui import theme as T


# ----------------------------------------------------------------- базовые

class Card(tk.Frame):
    """Плоская карточка с тонкой рамкой."""

    def __init__(self, master, **kw):
        super().__init__(master, bg=T.CARD, highlightthickness=1,
                         highlightbackground=T.BORDER, **kw)


class AppButton(tk.Canvas):
    """Плоская кнопка на Canvas: hover, pressed, disabled, варианты стиля."""

    PADX = 14
    PADY = 8

    def __init__(self, master, text: str, command: Callable[[], None],
                 style: str = "primary", width: Optional[int] = None,
                 font: Optional[tuple] = None, **kw):
        super().__init__(master, highlightthickness=0, bd=0, **kw)
        self.command = command
        self.text = text
        self.style = style
        self._font = font or T.font(10)
        self._state = "normal"
        self._hover = False
        self._press = False

        self._measure(width)
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Enter>", lambda e: self._set_hover(True))
        self.bind("<Leave>", lambda e: self._set_hover(False))
        self._draw()

    def _measure(self, width: Optional[int]) -> None:
        try:
            import tkinter.font as tkfont
            m = tkfont.Font(font=self._font).measure(self.text)
        except Exception:
            m = 10 * len(self.text)
        self._bw = int(width or (m + 2 * self.PADX))
        self._bh = 34
        self.configure(width=self._bw, height=self._bh)

    def _colors(self) -> Tuple[str, str, str]:
        if self._state == "disabled":
            return T.BORDER, T.TEXT_FAINT, T.TEXT_FAINT
        if self.style == "primary":
            bg = T.ACCENT if not self._hover else "#5adcd5"
            bg = T.ACCENT_DI if self._press else bg
            return bg, T.ACCENT_TXT, T.ACCENT_TXT
        if self.style == "danger":
            bg = T.ERR if not self._hover else "#ef7d86"
            return (T.ERR if self._press else bg), "#2a0d10", "#ffd9dc"
        # ghost / secondary
        if self._press:
            return T.CARD, T.TEXT, T.BORDER_HI
        if self._hover:
            return T.CARD_HI, T.TEXT, T.BORDER_HI
        return T.CARD, T.TEXT_DIM, T.BORDER

    def _draw(self) -> None:
        bg, fg, border = self._colors()
        self.delete("all")
        w, h = self._bw, self._bh
        self.create_rectangle(1.5, 1.5, w - 1.5, h - 1.5, fill=bg, outline=border,
                              width=1)
        self.create_text(w / 2, h / 2, text=self.text, fill=fg, font=self._font)

    def _set_hover(self, on: bool) -> None:
        if self._state != "normal":
            return
        self._hover = on
        self._draw()

    def _on_press(self, _e) -> None:
        if self._state != "normal":
            return
        self._press = True
        self._draw()

    def _on_release(self, e) -> None:
        if self._state != "normal":
            return
        was = self._press
        self._press = False
        self._draw()
        if was and 0 <= e.x <= self._bw and 0 <= e.y <= self._bh:
            try:
                self.command()
            except Exception:
                import traceback
                traceback.print_exc()

    def set_enabled(self, enabled: bool) -> None:
        self._state = "normal" if enabled else "disabled"
        self._draw()

    def set_text(self, text: str) -> None:
        self.text = text
        self._measure(None)
        self._draw()


import tkinter.font as _tkfont  # noqa: E402

# ----------------------------------------------------------------- ввод

class LabeledEntry(tk.Frame):
    """Подпись + поле ввода в едином стиле."""

    def __init__(self, master, label: str, value: str = "", width: int = 12,
                 on_change: Optional[Callable[[str], None]] = None,
                 hint: str = ""):
        super().__init__(master, bg=T.CARD)
        self.on_change = on_change
        tk.Label(self, text=label, bg=T.CARD, fg=T.TEXT_DIM,
                 font=T.font(9)).pack(anchor="w")
        row = tk.Frame(self, bg=T.CARD)
        row.pack(anchor="w", fill="x")
        self.var = tk.StringVar(value=value)
        self.entry = tk.Entry(row, textvariable=self.var, width=width,
                              bg=T.BG, fg=T.TEXT, insertbackground=T.TEXT,
                              relief="flat", font=T.font(10),
                              highlightthickness=1,
                              highlightbackground=T.BORDER,
                              highlightcolor=T.ACCENT)
        self.entry.pack(side="left", ipady=5, fill="x", expand=True)
        if hint:
            tk.Label(self, text=hint, bg=T.CARD, fg=T.TEXT_FAINT,
                     font=T.font(8)).pack(anchor="w")
        if on_change:
            self.var.trace_add("write", lambda *_: self._fire())

    def _fire(self) -> None:
        if self.on_change:
            try:
                self.on_change(self.var.get())
            except Exception:
                pass

    def get(self) -> str:
        return self.var.get()

    def set(self, v: str) -> None:
        self.var.set(v)


class LabeledSlider(tk.Frame):
    """Подпись + значение + слайдер в едином стиле."""

    def __init__(self, master, label: str, from_: float, to: float,
                 value: float, step: float = 1.0, on_change=None,
                 fmt: Optional[Callable[[float], str]] = None, bg: str = T.CARD):
        super().__init__(master, bg=bg)
        self.step = step
        self.fmt = fmt or (lambda v: f"{v:g}")
        self.on_change = on_change
        self._suppress = False

        head = tk.Frame(self, bg=bg)
        head.pack(fill="x")
        tk.Label(head, text=label, bg=bg, fg=T.TEXT_DIM,
                 font=T.font(9)).pack(side="left")
        self.val_lbl = tk.Label(head, text=self.fmt(value), bg=bg, fg=T.ACCENT,
                                font=T.font(9, bold=True))
        self.val_lbl.pack(side="right")

        self.var = tk.DoubleVar(value=value)
        self.scale = tk.Scale(self, variable=self.var, from_=from_, to=to,
                              resolution=step, orient="horizontal", showvalue=False,
                              bg=bg, fg=T.TEXT_DIM, troughcolor=T.BG,
                              highlightthickness=0, bd=0, activebackground=T.ACCENT,
                              sliderrelief="flat", sliderlength=14,
                              command=lambda _v: self._fire())
        self.scale.pack(fill="x", pady=(2, 0))

    def _fire(self) -> None:
        if self._suppress:
            return
        self.val_lbl.config(text=self.fmt(self.var.get()))
        if self.on_change:
            try:
                self.on_change(float(self.var.get()))
            except Exception:
                pass

    def get(self) -> float:
        return float(self.var.get())

    def set(self, v: float) -> None:
        self._suppress = True
        try:
            self.var.set(v)
            self.val_lbl.config(text=self.fmt(v))
        finally:
            self._suppress = False


class Toggle(tk.Canvas):
    """Небольшой переключатель вкл/выкл."""

    def __init__(self, master, label: str, value: bool = False,
                 on_change: Optional[Callable[[bool], None]] = None):
        super().__init__(master, width=64, height=24, bg=master["bg"],
                         highlightthickness=0)
        self.value = bool(value)
        self.label = label
        self.on_change = on_change
        self.bind("<Button-1>", lambda e: self.toggle())
        self._draw()

    def toggle(self) -> None:
        self.value = not self.value
        self._draw()
        if self.on_change:
            try:
                self.on_change(self.value)
            except Exception:
                pass

    def _draw(self) -> None:
        self.delete("all")
        w, h = 40, 20
        on = self.value
        bg = T.ACCENT if on else T.BORDER
        kx = w - h / 2 if on else h / 2
        self.create_round_rect = None
        # корпус
        self._rr(0, 0, w, h, 10, bg)
        # кружок
        r = 7
        self.create_oval(kx - r, h / 2 - r, kx + r, h / 2 + r,
                         fill="#ffffff" if on else T.TEXT_DIM, outline="")
        self.create_text(w + 8, h / 2, text=self.label, anchor="w",
                         fill=T.TEXT_DIM, font=T.font(9))

    def _rr(self, x1, y1, x2, y2, r, color):
        # скруглённый прямоугольник из элементов
        self.create_rectangle(x1 + r, y1, x2 - r, y2, fill=color, outline="")
        self.create_rectangle(x1, y1 + r, x2, y2 - r, fill=color, outline="")
        self.create_oval(x1, y1, x1 + 2 * r, y1 + 2 * r, fill=color, outline="")
        self.create_oval(x2 - 2 * r, y1, x2, y1 + 2 * r, fill=color, outline="")
        self.create_oval(x1, y2 - 2 * r, x1 + 2 * r, y2, fill=color, outline="")
        self.create_oval(x2 - 2 * r, y2 - 2 * r, x2, y2, fill=color, outline="")


# ----------------------------------------------------------------- графики

class LossChart(tk.Canvas):
    """Простой график лосса с сеткой, значениями и легендой."""

    def __init__(self, master, height: int = 170, bg: str = T.CARD):
        super().__init__(master, height=height, bg=bg, highlightthickness=0,
                         bd=0)
        self.bg = bg
        self.series_train: List[Tuple[float, float]] = []
        self.series_val: List[Tuple[float, float]] = []
        self.bind("<Configure>", lambda e: self.redraw())

    def set_data(self, hist: Sequence[dict]) -> None:
        tr, va = [], []
        for h in hist:
            step = h.get("step", 0)
            loss = h.get("loss")
            if loss is not None and loss == loss:  # not NaN
                tr.append((float(step), float(loss)))
            v = h.get("val")
            if v is not None and v == v:
                va.append((float(step), float(v)))
        self.series_train, self.series_val = tr, va
        self.redraw()

    def clear(self) -> None:
        self.series_train = []
        self.series_val = []
        self.redraw()

    def redraw(self) -> None:
        self.delete("all")
        w = max(10, self.winfo_width())
        h = max(10, self.winfo_height())
        pad_l, pad_r, pad_t, pad_b = 44, 12, 12, 22

        self.create_text(6, 6, text="лосс", anchor="nw", fill=T.TEXT_FAINT,
                         font=T.font(8))

        pts_all = self.series_train + self.series_val
        if not pts_all:
            self.create_text(w / 2, h / 2, text="данных пока нет",
                             fill=T.TEXT_FAINT, font=T.font(9))
            return

        xs = [p[0] for p in pts_all]
        x0, x1 = min(xs), max(xs)
        ys = [p[1] for p in pts_all]
        y0, y1 = min(min(ys), 0.0), max(ys)
        if y1 - y0 < 1e-6:
            y1 = y0 + 1e-6
        span_y = (y1 - y0) * 1.08
        y1 = y0 + span_y

        def X(v):  # noqa: E741
            if x1 - x0 < 1e-9:
                return pad_l + (w - pad_l - pad_r) / 2
            return pad_l + (v - x0) / (x1 - x0) * (w - pad_l - pad_r)

        def Y(v):  # noqa: E741
            return h - pad_b - (v - y0) / (y1 - y0) * (h - pad_t - pad_b)

        # сетка: 4 горизонтальных деления
        for i in range(5):
            v = y0 + (y1 - y0) * i / 4
            yy = Y(v)
            self.create_line(pad_l, yy, w - pad_r, yy, fill=T.CHART_GRID)
            self.create_text(pad_l - 6, yy, text=f"{v:.2f}", anchor="e",
                             fill=T.TEXT_FAINT, font=T.font(8))

        # линии
        def draw_series(pts, color):
            if len(pts) < 2:
                if pts:
                    x, y = pts[0]
                    self.create_oval(X(x) - 2, Y(y) - 2, X(x) + 2, Y(y) + 2,
                                     fill=color, outline="")
                return
            coords = []
            for x, y in pts:
                coords.extend((X(x), Y(y)))
            self.create_line(*coords, fill=color, width=2, smooth=True)

        draw_series(self.series_train, T.CHART_LINE)
        draw_series(self.series_val, T.CHART_LINE2)

        # последняя точка и значение
        if self.series_train:
            lx, ly = self.series_train[-1]
            self.create_oval(X(lx) - 3, Y(ly) - 3, X(lx) + 3, Y(ly) + 3,
                             fill=T.CHART_LINE, outline="")
            self.create_text(X(lx), Y(ly) - 10, text=f"{ly:.3f}",
                             fill=T.CHART_LINE, font=T.font(8, bold=True))

        # ось X: step
        if x1 > x0:
            self.create_text(pad_l, h - 8, text=f"{int(x0)}", anchor="w",
                             fill=T.TEXT_FAINT, font=T.font(8))
            self.create_text(w - pad_r, h - 8, text=f"шаг {int(x1)}", anchor="e",
                             fill=T.TEXT_FAINT, font=T.font(8))


# ----------------------------------------------------------------- чат

class ChatView(tk.Frame):
    """Лента сообщений + поле ввода. Роли: user / model / system."""

    BUBBLE_USER = T.BG_ALT
    BUBBLE_MODEL = "#173034"

    def __init__(self, master, on_send: Callable[[str], None]):
        super().__init__(master, bg=T.BG)
        self.on_send = on_send
        self._busy = False

        self.canvas = tk.Canvas(self, bg=T.BG, highlightthickness=0, bd=0)
        self.scroll = tk.Scrollbar(self, orient="vertical",
                                   command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scroll.set)
        self.inner = tk.Frame(self.canvas, bg=T.BG)
        self._win = self.canvas.create_window((0, 0), window=self.inner,
                                              anchor="nw")

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scroll.pack(side="right", fill="y")
        self.inner.bind("<Configure>", self._on_inner_conf)
        self.canvas.bind("<Configure>", self._on_canvas_conf)
        self.canvas.bind_all("<MouseWheel>", self._on_wheel)

        bar = tk.Frame(self, bg=T.BG)
        bar.pack(side="bottom", fill="x", padx=10, pady=(4, 10))
        self.input = tk.Entry(bar, bg=T.CARD, fg=T.TEXT, relief="flat",
                              insertbackground=T.TEXT, font=T.font(11),
                              highlightthickness=1,
                              highlightbackground=T.BORDER,
                              highlightcolor=T.ACCENT, state="disabled")
        self.input.pack(side="left", fill="x", expand=True, ipady=8, padx=(0, 8))
        self.input.bind("<Return>", lambda e: self._send())
        self.send_btn = AppButton(bar, "Отправить", self._send)
        self.send_btn.pack(side="right")

    # ------------------------------------------------------------- служебное
    def _on_inner_conf(self, _e) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_conf(self, e) -> None:
        self.canvas.itemconfigure(self._win, width=e.width)

    def _on_wheel(self, e) -> None:
        try:
            self.canvas.yview_scroll(int(-e.delta / 120), "units")
        except tk.TclError:
            pass

    def _send(self) -> None:
        if self._busy:
            return
        text = self.input.get().strip()
        if not text:
            return
        self.input.delete(0, "end")
        self.on_send(text)

    # ------------------------------------------------------------- API
    def add_bubble(self, role: str, text: str) -> None:
        is_user = role == "user"
        name = "Вы" if is_user else ("Система" if role == "system" else "Модель")
        color = T.TEXT_DIM if role == "system" else (T.TEXT if is_user else T.TEXT)
        wrap = self.canvas.winfo_width() - 60

        row = tk.Frame(self.inner, bg=T.BG)
        row.pack(fill="x", padx=10, pady=4)
        bubble = tk.Frame(row, bg=self.BUBBLE_USER if is_user else
                          ("#231f16" if role == "system" else self.BUBBLE_MODEL),
                          highlightthickness=1,
                          highlightbackground=T.BORDER)
        bubble.pack(anchor="e" if is_user else "w", fill="x",
                    padx=(60 if is_user else 0, 0 if is_user else 60))
        tk.Label(bubble, text=name, bg=bubble["bg"],
                 fg=T.ACCENT if not is_user else T.TEXT_FAINT,
                 font=T.font(8, bold=True)).pack(anchor="w", padx=12, pady=(8, 0))
        lbl = tk.Label(bubble, text=text, bg=bubble["bg"], fg=color,
                       font=T.font(10), wraplength=max(240, wrap - 40),
                       justify="left")
        lbl.pack(anchor="w", padx=12, pady=(2, 10))
        self.inner.update_idletasks()
        self.canvas.yview_moveto(1.0)

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.input.config(state="normal" if not busy else "disabled")
        self.send_btn.set_enabled(not busy)

    def clear(self) -> None:
        for w in self.inner.winfo_children():
            w.destroy()


# ----------------------------------------------------------------- диалоги

def ask_open_text_file(parent) -> Optional[str]:
    return filedialog.askopenfilename(
        parent=parent, title="Выберите текстовый файл",
        filetypes=[("Текстовые файлы", "*.txt *.md *.csv *.log"),
                   ("Все файлы", "*.*")])


def ask_save_model(parent) -> Optional[str]:
    return filedialog.asksaveasfilename(
        parent=parent, title="Сохранить модель", defaultextension=".lfmodel",
        filetypes=[("LinguaForge модель", "*.lfmodel")])


def ask_open_model(parent) -> Optional[str]:
    return filedialog.askopenfilename(
        parent=parent, title="Открыть модель",
        filetypes=[("LinguaForge модель", "*.lfmodel"), ("Все файлы", "*.*")])


def info(parent, title: str, msg: str) -> None:
    messagebox.showinfo(title, msg, parent=parent)


def warn(parent, title: str, msg: str) -> None:
    messagebox.showwarning(title, msg, parent=parent)


def error(parent, title: str, msg: str) -> None:
    messagebox.showerror(title, msg, parent=parent)


def ask_yes_no(parent, title: str, msg: str) -> bool:
    return messagebox.askyesno(title, msg, parent=parent)


# ----------------------------------------------------------------- прочее

class StatRow(tk.Frame):
    """Строка «название — значение» для сводок."""

    def __init__(self, master, label: str, value: str = "—", bg: Optional[str] = None):
        try:
            bg = bg or master["bg"]
        except Exception:
            bg = bg or T.CARD
        super().__init__(master, bg=bg)
        tk.Label(self, text=label, bg=bg, fg=T.TEXT_DIM,
                 font=T.font(9)).pack(side="left")
        self.val = tk.Label(self, text=value, bg=bg, fg=T.TEXT,
                            font=T.font(9, bold=True))
        self.val.pack(side="right")

    def set(self, v: str) -> None:
        self.val.config(text=v)
