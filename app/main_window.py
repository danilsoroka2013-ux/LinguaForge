# -*- coding: utf-8 -*-
"""Главное окно LinguaForge: боковая навигация + страницы-конструктор."""
from __future__ import annotations

import os
import threading
import tkinter as tk
from typing import Optional

import numpy as np

from ui import theme as T
from ui.widgets import (AppButton, Card, ChatView, LossChart, LabeledEntry,
                        LabeledSlider, StatRow, Toggle, ask_open_model,
                        ask_open_text_file, ask_save_model, error, info)

from core.config import VERSION, APP_NAME, human_size, list_checkpoints
from core.trainer import Trainer

from app.project import ProjectState, demo_corpus_path


PAGES = [
    ("overview", "Обзор", "◈"),
    ("data",     "Данные", "▤"),
    ("tokenizer","Токенизатор", "⊞"),
    ("model",    "Модель", "⬡"),
    ("train",    "Обучение", "▶"),
    ("play",     "Песочница", "✦"),
]


class MainApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(f"{APP_NAME} — конструктор языковых моделей")
        self.geometry("1080x720")
        self.minsize(940, 640)
        self.configure(bg=T.BG)

        fam = T.pick_family(self)
        self._font_base = (fam, 10)
        self.option_add("*Font", self._font_base)

        self.state_ = ProjectState()

        self._train_thread = None
        self._gen_busy = False
        self.nav_key: Optional[str] = None
        self.presets = {
            "Крошечная (для теста)": dict(dim=48, n_layers=2, n_heads=2,
                                          block_size=64, ffn_mult=2.667),
            "Мини (рекомендуется)": dict(dim=96, n_layers=3, n_heads=3,
                                         block_size=96, ffn_mult=2.667),
            "Малая": dict(dim=128, n_layers=4, n_heads=4,
                          block_size=128, ffn_mult=2.667),
        }

        self._build_layout()
        self._build_pages()
        self.show_page("overview")
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(400, self._poll_train)

    def _on_close(self) -> None:
        tr = getattr(self, "trainer", None)
        if tr is not None and tr.is_running():
            tr.stop()
        self.destroy()

    # ================================================================ layout
    def _build_layout(self) -> None:
        # ---- сайдбар
        side = tk.Frame(self, bg=T.BG_ALT, width=190)
        side.pack(side="left", fill="y")
        side.pack_propagate(False)

        tk.Label(side, text=f"◆ {APP_NAME}", bg=T.BG_ALT, fg=T.TEXT,
                 font=T.font(13, bold=True)).pack(anchor="w", padx=18,
                                                  pady=(20, 2))
        tk.Label(side, text=f"v{VERSION} · локально · офлайн", bg=T.BG_ALT,
                 fg=T.TEXT_FAINT, font=T.font(8)).pack(anchor="w", padx=18,
                                                       pady=(0, 16))

        self.nav_buttons = {}
        for key, label, icon in PAGES:
            b = tk.Label(side, text=f"  {icon}  {label}", bg=T.BG_ALT,
                         fg=T.TEXT_DIM, font=T.font(10), anchor="w",
                         padx=14, pady=9, cursor="hand2")
            b.pack(fill="x", padx=10, pady=2)
            b.bind("<Button-1>", lambda e, k=key: self.show_page(k))
            self.nav_buttons[key] = b

        # низ сайдбара: статус проекта
        foot = tk.Frame(side, bg=T.BG_ALT)
        foot.pack(side="bottom", fill="x", padx=14, pady=14)
        self.foot_corpus = StatRow(foot, "Корпус", "не загружен", bg=T.BG_ALT)
        self.foot_corpus.pack(fill="x")
        self.foot_model = StatRow(foot, "Модель", "не собрана", bg=T.BG_ALT)
        self.foot_model.pack(fill="x", pady=(4, 0))
        self.foot_step = StatRow(foot, "Шаг", "—", bg=T.BG_ALT)
        self.foot_step.pack(fill="x", pady=(4, 0))

        # ---- контент
        self.content = tk.Frame(self, bg=T.BG)
        self.content.pack(side="right", fill="both", expand=True)

        self.pages = {}

    def _refresh_nav(self) -> None:
        for key, b in self.nav_buttons.items():
            active = (key == self.nav_key)
            b.config(bg=T.CARD if active else T.BG_ALT,
                     fg=T.ACCENT if active else T.TEXT_DIM,
                     font=T.font(10, bold=active))

    def show_page(self, key: str) -> None:
        self.nav_key = key
        self._refresh_nav()
        for k, page in self.pages.items():
            page.pack_forget()
        self.pages[key].pack(fill="both", expand=True)
        refresh = getattr(self.pages[key], "refresh", None)
        if refresh:
            refresh()
        self._refresh_footer()
        if key == "overview":
            self._refresh_overview()
        elif key == "data":
            self._refresh_data_page()
        elif key == "tokenizer":
            self.refresh_tokenizer()
        elif key == "model":
            self.refresh_model()
        elif key == "train":
            self._refresh_train_page()

    def _page(self, key: str, builder) -> None:
        page = builder(self.content)
        page.pack_forget()
        self.pages[key] = page

    # ================================================================ helpers
    def _h1(self, parent, text: str, sub: str = "") -> tk.Frame:
        head = tk.Frame(parent, bg=T.BG)
        head.pack(fill="x", padx=22, pady=(18, 10))
        tk.Label(head, text=text, bg=T.BG, fg=T.TEXT,
                 font=T.font(16, bold=True)).pack(anchor="w")
        if sub:
            tk.Label(head, text=sub, bg=T.BG, fg=T.TEXT_DIM,
                     font=T.font(9)).pack(anchor="w", pady=(2, 0))
        return head

    def _refresh_footer(self) -> None:
        st = self.state_
        if st.text:
            s = st.corpus_stats()
            self.foot_corpus.set(f"{s['chars']:,} симв.")
        else:
            self.foot_corpus.set("не загружен")
        if st.model:
            self.foot_model.set(f"{st.model.num_params():,} пар.")
        else:
            self.foot_model.set("не собрана")

    # ================================================================ страницы
    def _build_pages(self) -> None:
        self._page("overview", self._build_overview)
        self._page("data", self._build_data)
        self._page("tokenizer", self._build_tokenizer)
        self._page("model", self._build_model_page)
        self._page("train", self._build_train)
        self._page("play", self._build_play)

    # ------------------------------------------------------------ обзор
    def _build_overview(self, parent) -> tk.Frame:
        pg = tk.Frame(parent, bg=T.BG)

        self._h1(pg, "Обзор",
                 "Соберите свою языковую модель за пять шагов — полностью "
                 "локально, без интернета и GPU.")

        row = tk.Frame(pg, bg=T.BG)
        row.pack(fill="x", padx=22)

        # --- шаги
        steps = Card(row)
        steps.pack(side="left", fill="both", expand=True, pady=(0, 10))
        inner = tk.Frame(steps, bg=T.CARD)
        inner.pack(fill="both", expand=True, padx=16, pady=14)
        tk.Label(inner, text="Путь сборки", bg=T.CARD, fg=T.TEXT,
                 font=T.font(11, bold=True)).pack(anchor="w", pady=(0, 8))
        items = [
            ("1", "Данные", "загрузите текст или демо-корпус"),
            ("2", "Токенизатор", "символьный или BPE, обучается на корпусе"),
            ("3", "Модель", "размер, слои, контекст — как у больших, только малый"),
            ("4", "Обучение", "кривая лосса, чекпоинты, валидация"),
            ("5", "Песочница", "чат с вашей моделью"),
        ]
        grid = tk.Frame(inner, bg=T.CARD)
        grid.pack(fill="x")
        for i, (num, name, desc) in enumerate(items):
            c = tk.Frame(grid, bg=T.CARD)
            c.grid(row=i // 3, column=i % 3, sticky="nsew", padx=4, pady=6)
            numf = tk.Frame(c, bg=T.CARD)
            numf.pack(anchor="w")
            tk.Canvas(numf, width=22, height=22, bg=T.ACCENT,
                      highlightthickness=0).pack(side="left")
            tk.Label(numf, text=num, bg=T.CARD, fg=T.ACCENT,
                     font=T.font(10, bold=True)).place(x=7, y=3)
            tk.Label(c, text=name, bg=T.CARD, fg=T.TEXT,
                     font=T.font(10, bold=True)).pack(anchor="w", pady=(6, 0))
            tk.Label(c, text=desc, bg=T.CARD, fg=T.TEXT_DIM,
                     font=T.font(8), wraplength=190, justify="left"
                     ).pack(anchor="w")
        for i in range(3):
            grid.columnconfigure(i, weight=1)

        # --- карточка модели
        stat = Card(row, width=300)
        stat.pack(side="right", fill="y", padx=(12, 0), pady=(0, 10))
        sin = tk.Frame(stat, bg=T.CARD)
        sin.pack(fill="both", expand=True, padx=16, pady=14)
        tk.Label(sin, text="Ваш проект", bg=T.CARD, fg=T.TEXT,
                 font=T.font(11, bold=True)).pack(anchor="w", pady=(0, 8))
        self.ov_corpus = StatRow(sin, "Корпус", "—")
        self.ov_corpus.pack(fill="x", pady=2)
        self.ov_vocab = StatRow(sin, "Словарь", "—")
        self.ov_vocab.pack(fill="x", pady=2)
        self.ov_params = StatRow(sin, "Параметры", "—")
        self.ov_params.pack(fill="x", pady=2)
        self.ov_ctx = StatRow(sin, "Контекст", "—")
        self.ov_ctx.pack(fill="x", pady=2)
        self.ov_ckpt = StatRow(sin, "Чекпоинт", "—")
        self.ov_ckpt.pack(fill="x", pady=2)

        tk.Label(sin, text="", bg=T.CARD).pack()
        self.ov_open_btn = AppButton(sin, "Открыть .lfmodel", self._open_model_dialog,
                                     style="ghost")
        self.ov_open_btn.pack(anchor="w", pady=(6, 2))
        self.ov_demo_btn = AppButton(sin, "Загрузить демо-корпус", self._load_demo)
        self.ov_demo_btn.pack(anchor="w", pady=(2, 0))

        # нижняя плашка
        bar = Card(pg)
        bar.pack(fill="x", padx=22, pady=(4, 18))
        bin_ = tk.Frame(bar, bg=T.CARD)
        bin_.pack(fill="x", padx=16, pady=12)
        tk.Label(bin_, text="Всё считается на вашем Xeon в оперативной памяти. "
                            "Модель «Мини» (~0.4 млн параметров) учится за минуты, "
                            "весит меньше мегабайта и запускается где угодно.",
                 bg=T.CARD, fg=T.TEXT_DIM, font=T.font(9),
                 wraplength=900, justify="left").pack(anchor="w")
        return pg

    def _load_demo(self) -> None:
        self.state_.load_demo_corpus()
        info(self, "Демо-корпус",
             f"Загружен: {demo_corpus_path()}\n"
             f"Символов: {len(self.state_.text):,}")
        self._refresh_footer()
        self.show_page("data")

    def _open_model_dialog(self) -> None:
        path = ask_open_model(self)
        if not path:
            return
        try:
            self.state_.load_model(path)
        except Exception as e:
            error(self, "Ошибка", f"Не удалось открыть модель:\n{e}")
            return
        self._refresh_footer()
        info(self, "Модель загружена", os.path.basename(path))
        self.show_page("play")

    # ------------------------------------------------------------ данные
    def _build_data(self, parent) -> tk.Frame:
        pg = tk.Frame(parent, bg=T.BG)
        self._h1(pg, "Данные",
                 "Корпус — это учебник модели. Чем чище текст, тем связнее речь.")

        card = Card(pg)
        card.pack(fill="both", expand=True, padx=22, pady=(0, 10))
        inner = tk.Frame(card, bg=T.CARD)
        inner.pack(fill="both", expand=True, padx=16, pady=14)

        btns = tk.Frame(inner, bg=T.CARD)
        btns.pack(fill="x", pady=(0, 10))
        AppButton(btns, "Загрузить .txt", self._pick_text).pack(side="left")
        AppButton(btns, "Демо-корпус", self._load_demo, style="ghost").pack(
            side="left", padx=(8, 0))

        self.data_stats = tk.Label(inner, text="Файл не выбран", bg=T.CARD,
                                   fg=T.TEXT_DIM, font=T.font(9))
        self.data_stats.pack(anchor="w", pady=(0, 8))

        self.text_box = tk.Text(inner, height=14, bg=T.BG, fg=T.TEXT,
                                insertbackground=T.TEXT, relief="flat",
                                font=T.font(10, mono=True), wrap="word",
                                highlightthickness=1,
                                highlightbackground=T.BORDER,
                                highlightcolor=T.ACCENT)
        self.text_box.pack(fill="both", expand=True)

        val_card = Card(pg)
        val_card.pack(fill="x", padx=22, pady=(0, 18))
        vin = tk.Frame(val_card, bg=T.CARD)
        vin.pack(fill="x", padx=16, pady=12)
        tk.Label(vin, text="Доля валидации", bg=T.CARD, fg=T.TEXT_DIM,
                 font=T.font(9)).pack(anchor="w")
        self.val_slider = LabeledSlider(vin, "часть корпуса для контроля качества",
                                        0.0, 0.3, self.state_.val_frac, 0.01,
                                        fmt=lambda v: f"{v*100:.0f} %",
                                        on_change=self._set_val_frac)
        self.val_slider.pack(fill="x")
        return pg

    def _set_val_frac(self, v: float) -> None:
        self.state_.val_frac = float(v)

    def _pick_text(self) -> None:
        path = ask_open_text_file(self)
        if not path:
            return
        try:
            self.state_.load_text(path)
        except Exception as e:
            error(self, "Ошибка", f"Не удалось прочитать файл:\n{e}")
            return
        self._refresh_data_page()
        self._refresh_footer()

    def _refresh_data_page(self) -> None:
        st = self.state_
        if not st.text:
            self.data_stats.config(text="Файл не выбран")
            return
        s = st.corpus_stats()
        src = os.path.basename(st.text_path) if st.text_path else "(вставлено)"
        self.data_stats.config(
            text=f"{src}: {s['chars']:,} символов · {s['words']:,} слов · "
                 f"{s['lines']:,} строк")
        self.text_box.delete("1.0", "end")
        self.text_box.insert("1.0", st.text[:40000])

    def refresh_data(self) -> None:
        self._refresh_data_page()
        self._refresh_footer()

    # ------------------------------------------------------------ токенизатор
    def _build_tokenizer(self, parent) -> tk.Frame:
        pg = tk.Frame(parent, bg=T.BG)
        self._h1(pg, "Токенизатор",
                 "Модель читает не буквы, а номера-токены. Здесь вы решаете, "
                 "как текст превратить в числа.")

        card = Card(pg)
        card.pack(fill="x", padx=22)
        inner = tk.Frame(card, bg=T.CARD)
        inner.pack(fill="x", padx=16, pady=14)

        self.tok_kind = tk.StringVar(value=self.state_.tokenizer_kind)
        rb_row = tk.Frame(inner, bg=T.CARD)
        rb_row.pack(anchor="w")
        for val, label, hint in [
            ("char", "Символьный", "простой и надёжный, словарь маленький"),
            ("bpe", "BPE", "сливает частые пары, текст сжимается в ~2–3 раза"),
        ]:
            tk.Radiobutton(rb_row, text=label, variable=self.tok_kind, value=val,
                           command=self._tok_kind_changed, bg=T.CARD, fg=T.TEXT,
                           selectcolor=T.BG, activebackground=T.CARD,
                           activeforeground=T.TEXT, font=T.font(10),
                           highlightthickness=0).pack(side="left", padx=(0, 18))
        self.tok_hint = tk.Label(inner, text="", bg=T.CARD, fg=T.TEXT_DIM,
                                 font=T.font(8))
        self.tok_hint.pack(anchor="w", pady=(2, 8))

        self.vocab_slider = LabeledSlider(
            inner, "размер словаря (для BPE)", 64, 2048, self.state_.vocab_size, 64,
            on_change=self._vocab_changed)
        self.vocab_slider.pack(fill="x")

        btns = tk.Frame(inner, bg=T.CARD)
        btns.pack(anchor="w", pady=(10, 0))
        AppButton(btns, "Обучить токенизатор", self._train_tokenizer).pack(side="left")

        self.tok_status = tk.Label(inner, text="", bg=T.CARD, fg=T.OK,
                                   font=T.font(9, bold=True))
        self.tok_status.pack(anchor="w", pady=(8, 0))

        # просмотр
        prev = Card(pg)
        prev.pack(fill="both", expand=True, padx=22, pady=(10, 18))
        pin = tk.Frame(prev, bg=T.CARD)
        pin.pack(fill="both", expand=True, padx=16, pady=14)
        tk.Label(pin, text="Как текст видит модель", bg=T.CARD, fg=T.TEXT,
                 font=T.font(11, bold=True)).pack(anchor="w", pady=(0, 8))
        self.tok_preview = tk.Text(pin, height=8, bg=T.BG, fg=T.TEXT,
                                   insertbackground=T.TEXT, relief="flat",
                                   font=T.font(10, mono=True), wrap="word",
                                   highlightthickness=1,
                                   highlightbackground=T.BORDER)
        self.tok_preview.pack(fill="both", expand=True)
        tk.Label(pin, text="Совет: BPE удобнее для больших корпусов, символьный — "
                           "лучший выбор для первого знакомства.",
                 bg=T.CARD, fg=T.TEXT_FAINT, font=T.font(8)).pack(anchor="w",
                                                                  pady=(8, 0))
        return pg

    def _tok_kind_changed(self) -> None:
        self.state_.tokenizer_kind = self.tok_kind.get()
        self.tok_hint.config(
            text="Каждый символ — один токен. Простой и предсказуемый."
            if self.state_.tokenizer_kind == "char" else
            "Частые пары символов сливаются в один токен — текст упаковывается плотнее.")

    def _vocab_changed(self, v: float) -> None:
        self.state_.vocab_size = int(v)

    def _train_tokenizer(self) -> None:
        st = self.state_
        if not st.text:
            warn_no_corpus(self)
            return
        try:
            tok = st.build_tokenizer()
        except Exception as e:
            error(self, "Ошибка", str(e))
            return
        self.tok_status.config(
            text=f"Готово: {tok.vocab_size} токенов · "
                 f"{len(st.tokenize_corpus()):,} токенов в корпусе",
            fg=T.OK)
        self._show_tok_preview()
        self._refresh_footer()

    def _show_tok_preview(self) -> None:
        st = self.state_
        if not st.tokenizer:
            return
        sample = st.text[:220]
        ids = st.tokenizer.encode(sample)
        lines = ["текст:   " + sample.replace("\n", " ")[:110],
                 "",
                 "токены:  " + " ".join(map(str, ids[:80])),
                 "",
                 "обратно: " + st.tokenizer.decode(ids[:80]).replace("\n", " ")]
        self.tok_preview.delete("1.0", "end")
        self.tok_preview.insert("1.0", "\n".join(lines))

    def refresh_tokenizer(self) -> None:
        if self.state_.tokenizer:
            self.tok_status.config(
                text=f"Активен: {self.state_.tokenizer.name}, "
                     f"{self.state_.tokenizer.vocab_size} токенов", fg=T.OK)
            self._show_tok_preview()

    # ------------------------------------------------------------ модель
    def _build_model_page(self, parent) -> tk.Frame:
        pg = tk.Frame(parent, bg=T.BG)
        self._h1(pg, "Модель",
                 "Архитектура — миниатюрный GPT: внимание, слои, контекст. "
                 "Больше параметров = умнее, но медленнее.")

        left = Card(pg)
        left.pack(side="left", fill="both", expand=True, padx=(22, 10),
                  pady=(0, 18))
        lin = tk.Frame(left, bg=T.CARD)
        lin.pack(fill="both", expand=True, padx=16, pady=14)

        preset_row = tk.Frame(lin, bg=T.CARD)
        preset_row.pack(fill="x", pady=(0, 10))
        tk.Label(preset_row, text="Пресет", bg=T.CARD, fg=T.TEXT_DIM,
                 font=T.font(9)).pack(side="left")
        self.preset_var = tk.StringVar(value="Мини (рекомендуется)")
        for name in self.presets:
            tk.Radiobutton(preset_row, text=name, value=name,
                           variable=self.preset_var, command=self._apply_preset,
                           bg=T.CARD, fg=T.TEXT, selectcolor=T.BG,
                           activebackground=T.CARD, activeforeground=T.TEXT,
                           font=T.font(9), highlightthickness=0).pack(
                side="left", padx=(14, 0))

        self.m_dim = LabeledSlider(lin, "размерность (dim)", 32, 256,
                                   self.state_.model_cfg["dim"], 16,
                                   on_change=self._model_changed)
        self.m_dim.pack(fill="x")
        self.m_layers = LabeledSlider(lin, "слои (глубина)", 1, 8,
                                      self.state_.model_cfg["n_layers"], 1,
                                      on_change=self._model_changed)
        self.m_layers.pack(fill="x")
        self.m_heads = LabeledSlider(lin, "головы внимания", 1, 8,
                                     self.state_.model_cfg["n_heads"], 1,
                                     on_change=self._model_changed)
        self.m_heads.pack(fill="x")
        self.m_ctx = LabeledSlider(lin, "контекст (block_size)", 32, 256,
                                   self.state_.model_cfg["block_size"], 16,
                                   on_change=self._model_changed)
        self.m_ctx.pack(fill="x")

        # панель справа: сводка и сборка
        right = tk.Frame(pg, bg=T.BG)
        right.pack(side="right", fill="y", padx=(0, 22), pady=(0, 18))

        card = Card(right, width=280)
        card.pack(fill="x")
        cin = tk.Frame(card, bg=T.CARD)
        cin.pack(fill="x", padx=16, pady=14)
        tk.Label(cin, text="Сводка", bg=T.CARD, fg=T.TEXT,
                 font=T.font(11, bold=True)).pack(anchor="w", pady=(0, 8))
        self.m_params = StatRow(cin, "Параметров", "—")
        self.m_params.pack(fill="x", pady=2)
        self.m_ram = StatRow(cin, "Вес модели", "—")
        self.m_ram.pack(fill="x", pady=2)
        self.m_vocab = StatRow(cin, "Словарь", "—")
        self.m_vocab.pack(fill="x", pady=2)
        self.m_ctx_stat = StatRow(cin, "Контекст", "—")
        self.m_ctx_stat.pack(fill="x", pady=2)

        self.m_build_btn = AppButton(cin, "Собрать модель", self._build_model)
        self.m_build_btn.pack(anchor="w", pady=(12, 0))

        self.m_status = tk.Label(cin, text="", bg=T.CARD, fg=T.OK,
                                 font=T.font(9, bold=True), wraplength=230,
                                 justify="left")
        self.m_status.pack(anchor="w", pady=(8, 0))
        return pg

    def _apply_preset(self) -> None:
        p = self.presets[self.preset_var.get()]
        self.state_.model_cfg.update(p)
        self.m_dim.set(p["dim"])
        self.m_layers.set(p["n_layers"])
        self.m_heads.set(p["n_heads"])
        self.m_ctx.set(p["block_size"])
        self._update_model_summary()

    def _model_changed(self, _v=None) -> None:
        # dim кратно головам подберём при сборке
        self.state_.model_cfg["dim"] = int(self.m_dim.get())
        self.state_.model_cfg["n_layers"] = int(self.m_layers.get())
        self.state_.model_cfg["n_heads"] = int(self.m_heads.get())
        self.state_.model_cfg["block_size"] = int(self.m_ctx.get())
        self._update_model_summary()

    def _estimate_params(self, vocab: int, dim: int, layers: int, ctx: int,
                         ffn_mult: float) -> int:
        hidden = int(dim * ffn_mult)
        hidden += hidden % 2
        per_layer = 3 * dim * dim + dim * dim + 2 * dim * hidden  # qkv+proj+ffn
        return vocab * dim + layers * per_layer + layers * 2 * dim + dim

    def _update_model_summary(self) -> None:
        c = self.state_.model_cfg
        vocab = self.state_.tokenizer.vocab_size if self.state_.tokenizer else 256
        params = self._estimate_params(vocab, c["dim"], c["n_layers"],
                                       c["block_size"], c["ffn_mult"])
        self.m_params.set(f"{params:,}")
        self.m_ram.set(f"~{human_size(params * 4)}")
        self.m_vocab.set(str(vocab))
        self.m_ctx_stat.set(f"{c['block_size']} симв.")

    def _build_model(self) -> None:
        st = self.state_
        if not st.tokenizer:
            st.build_tokenizer()
        c = st.model_cfg
        # dim должен делиться на головы
        heads = c["n_heads"]
        if c["dim"] % heads:
            c["dim"] = (c["dim"] // heads) * heads
            self.m_dim.set(c["dim"])
        try:
            model = st.build_model(force=True)
        except Exception as e:
            error(self, "Ошибка", f"Не удалось собрать модель:\n{e}")
            return
        self.m_status.config(
            text=f"Модель готова: {model.num_params():,} параметров. "
                 f"Переходите к обучению.", fg=T.OK)
        self._update_model_summary()
        self._refresh_footer()

    def refresh_model(self) -> None:
        self._update_model_summary()
        if self.state_.model:
            self.m_status.config(
                text=f"Активна модель: {self.state_.model.num_params():,} параметров.",
                fg=T.OK)

    # ------------------------------------------------------------ обучение
    def _build_train(self, parent) -> tk.Frame:
        pg = tk.Frame(parent, bg=T.BG)
        self._h1(pg, "Обучение",
                 "Нажмите «Начать» и смотрите, как лосс ползёт вниз. "
                 "Чекпоинты сохраняются автоматически.")

        # панель гиперпараметров слева
        left = Card(pg, width=300)
        left.pack(side="left", fill="y", padx=(22, 10), pady=(0, 18))
        lin = tk.Frame(left, bg=T.CARD)
        lin.pack(fill="both", expand=True, padx=16, pady=14)
        tk.Label(lin, text="Гиперпараметры", bg=T.CARD, fg=T.TEXT,
                 font=T.font(11, bold=True)).pack(anchor="w", pady=(0, 8))

        tc = self.state_.train_cfg
        self.t_steps = LabeledSlider(lin, "шагов обучения", 200, 20000, tc["steps"],
                                     100, on_change=lambda v: tc.__setitem__("steps", int(v)))
        self.t_steps.pack(fill="x")
        self.t_lr = LabeledSlider(lin, "learning rate", 0.0005, 0.01, tc["lr"],
                                  0.0005, fmt=lambda v: f"{v:.4f}",
                                  on_change=lambda v: tc.__setitem__("lr", float(v)))
        self.t_lr.pack(fill="x")
        self.t_bs = LabeledSlider(lin, "batch size", 4, 48, tc["batch_size"], 4,
                                  on_change=lambda v: tc.__setitem__("batch_size", int(v)))
        self.t_bs.pack(fill="x")
        self.t_seed = LabeledEntry(lin, "seed", str(tc["seed"]), width=10,
                                   on_change=lambda s: tc.__setitem__("seed", int(s or 0)))
        self.t_seed.pack(fill="x", pady=(6, 0))

        self.t_start = AppButton(lin, "▶  Начать обучение", self._start_train)
        self.t_start.pack(fill="x", pady=(14, 6))
        self.t_stop = AppButton(lin, "■  Остановить", self._stop_train,
                                style="danger")
        self.t_stop.pack(fill="x")
        self.t_stop.set_enabled(False)

        self.train_status = tk.Label(lin, text="Готово к запуску", bg=T.CARD,
                                     fg=T.TEXT_DIM, font=T.font(9),
                                     wraplength=250, justify="left")
        self.train_status.pack(anchor="w", pady=(10, 0))

        # правая часть: график и метрики
        right = tk.Frame(pg, bg=T.BG)
        right.pack(side="right", fill="both", expand=True, padx=(0, 22),
                   pady=(0, 18))

        chart_card = Card(right)
        chart_card.pack(fill="both", expand=True)
        chin = tk.Frame(chart_card, bg=T.CARD)
        chin.pack(fill="both", expand=True, padx=14, pady=12)
        self.loss_chart = LossChart(chin, height=240)
        self.loss_chart.pack(fill="both", expand=True)

        metrics = tk.Frame(right, bg=T.BG)
        metrics.pack(fill="x", pady=(10, 0))
        self.metr_loss = MetricTile(metrics, "лосс")
        self.metr_val = MetricTile(metrics, "валидация")
        self.metr_ppl = MetricTile(metrics, "перплексия")
        self.metr_speed = MetricTile(metrics, "токенов/с")
        for i, w in enumerate([self.metr_loss, self.metr_val, self.metr_ppl,
                               self.metr_speed]):
            w.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 8, 0))
        for i in range(4):
            metrics.columnconfigure(i, weight=1)
        return pg

    # ---- управление обучением
    def _start_train(self) -> None:
        st = self.state_
        try:
            st.ensure_model_for_training()
        except Exception as e:
            error(self, "Не готово", str(e))
            return
        tc = dict(st.train_cfg)
        tc["seq_len"] = min(int(tc.get("seq_len", st.model_cfg["block_size"])),
                            st.model_cfg["block_size"])
        train_ids, val_ids = st.train_val_ids()
        if train_ids is None or train_ids.size < tc["seq_len"] + 2:
            error(self, "Мало данных",
                  "Корпус слишком короткий для выбранных настроек.")
            return

        self.trainer = Trainer(st.model, st.tokenizer, tc, train_ids, val_ids,
                               st.dir, on_done=lambda: self.after(0, self._train_done))
        st.history = []
        self.loss_chart.clear()
        self.trainer.start()
        self.t_start.set_enabled(False)
        self.t_stop.set_enabled(True)
        self.train_status.config(text="Обучение идёт…", fg=T.ACCENT)

    def _stop_train(self) -> None:
        if self.trainer:
            self.trainer.stop()

    def _poll_train(self) -> None:
        try:
            tr = getattr(self, "trainer", None)
            if tr:
                snap = tr.state.snapshot()
                self.loss_chart.set_data(snap["history"])
                self.metr_loss.set(f"{snap['loss']:.3f}" if snap["loss"] is not None else "—")
                self.metr_val.set(f"{snap['val_loss']:.3f}" if snap["val_loss"] is not None else "—")
                self.metr_ppl.set(f"{snap['val_ppl']:.1f}" if snap["val_ppl"] else "—")
                self.metr_speed.set(f"{snap['tok_per_s']:,.0f}" if snap["tok_per_s"] else "—")
                self.foot_step.set(str(snap["step"] or "—"))
                self.state_.history = snap["history"]
            self.after(400, self._poll_train)
        except (tk.TclError, RuntimeError, AttributeError):
            pass  # окно закрывается — перестаём опрашивать

    def _train_done(self) -> None:
        self.t_start.set_enabled(True)
        self.t_stop.set_enabled(False)
        tr = self.trainer
        if tr:
            snap = tr.state.snapshot()
            if snap["error"]:
                self.train_status.config(text=f"Ошибка: {snap['error']}", fg=T.ERR)
                error(self, "Обучение прервано", snap["error"])
            else:
                self.train_status.config(
                    text=f"{snap['message']} · шаг {snap['step']}",
                    fg=T.OK if snap["message"] == "Готово" else T.WARN)
                self.state_.history = snap["history"]

    # ------------------------------------------------------------ песочница
    def _build_play(self, parent) -> tk.Frame:
        pg = tk.Frame(parent, bg=T.BG)
        self._h1(pg, "Песочница",
                 "Поговорите со своей моделью. Генерация идёт локально, "
                 "на вашем процессоре.")

        card = Card(pg)
        card.pack(fill="both", expand=True, padx=22, pady=(0, 10))
        cin = tk.Frame(card, bg=T.CARD)
        cin.pack(fill="both", expand=True, padx=10, pady=10)

        self.chat = ChatView(cin, on_send=self._on_chat_send)
        self.chat.pack(fill="both", expand=True)

        # настройки генерации
        opts = Card(pg)
        opts.pack(fill="x", padx=22, pady=(0, 18))
        oin = tk.Frame(opts, bg=T.CARD)
        oin.pack(fill="x", padx=14, pady=10)
        self.g_temp = LabeledSlider(oin, "температура", 0.1, 1.5, 0.8, 0.05,
                                    fmt=lambda v: f"{v:.2f}", bg=T.CARD)
        self.g_temp.pack(side="left", fill="x", expand=True, padx=(0, 14))
        self.g_max = LabeledSlider(oin, "макс. длина", 16, 256, 96, 16, bg=T.CARD)
        self.g_max.pack(side="left", fill="x", expand=True, padx=(0, 14))
        self.g_rep = LabeledSlider(oin, "штраф повтора", 1.0, 1.6, 1.15, 0.05,
                                   fmt=lambda v: f"{v:.2f}", bg=T.CARD)
        self.g_rep.pack(side="left", fill="x", expand=True)

        self.chat.add_bubble("system",
                             "Это песочница. Загрузите или обучите модель — и "
                             "пишите в поле ниже. Модель продолжит текст.")
        self.chat.set_busy(False)
        return pg

    def _on_chat_send(self, text: str) -> None:
        st = self.state_
        if self._gen_busy:
            return
        if not st.model or not st.tokenizer:
            self.chat.add_bubble("system", "Сначала соберите модель на странице "
                                           "«Модель» (или откройте .lfmodel).")
            return
        self.chat.add_bubble("user", text)
        self.chat.set_busy(True)
        self._gen_busy = True

        temp = self.g_temp.get()
        max_new = int(self.g_max.get())
        rep = self.g_rep.get()

        def worker():
            try:
                tok = st.tokenizer
                ids = tok.encode(text)[-st.model.block_size:]
                if not ids:
                    ids = tok.encode("Кот")
                ctx = np.array([ids], dtype=np.int32)
                out = st.model.generate(
                    ctx, max_new_tokens=max_new, temperature=temp,
                    repetition_penalty=rep, top_p=0.95)
                reply = tok.decode(out[0][len(ids):])
            except Exception as e:
                reply = f"(ошибка генерации: {e})"
            try:
                self.after(0, lambda: self._chat_reply(reply))
            except (RuntimeError, tk.TclError):
                pass  # окно уже закрыто

        self._gen_thread = threading.Thread(target=worker, daemon=True)
        self._gen_thread.start()

    def _chat_reply(self, reply: str) -> None:
        self.chat.add_bubble("model", reply if reply.strip() else "(пусто)")
        self.chat.set_busy(False)
        self._gen_busy = False

    def _refresh_train_page(self) -> None:
        if self.state_.history:
            self.loss_chart.set_data(self.state_.history)
            hist = self.state_.history[-1] if self.state_.history else {}
            if hist.get("loss") is not None:
                self.metr_loss.set(f"{hist['loss']:.3f}")
            v = hist.get("val")
            if v is not None and v == v:
                self.metr_val.set(f"{v:.3f}")

    def _refresh_overview(self) -> None:
        st = self.state_
        if st.text:
            s = st.corpus_stats()
            self.ov_corpus.set(f"{s['chars']:,} симв.")
        else:
            self.ov_corpus.set("—")
        self.ov_vocab.set(str(st.tokenizer.vocab_size) if st.tokenizer else "—")
        self.ov_params.set(f"{st.model.num_params():,}" if st.model else "—")
        self.ov_ctx.set(str(st.model_cfg["block_size"]))
        ckpts = list_checkpoints(st.dir)
        if ckpts:
            name = os.path.basename(ckpts[0])
            size = human_size(os.path.getsize(ckpts[0]))
            self.ov_ckpt.set(f"{name} · {size}")
        else:
            self.ov_ckpt.set("—")

    # ---- цикл опроса статуса


def warn_no_corpus(parent) -> None:
    from ui.widgets import warn
    warn(parent, "Нужен корпус",
         "Сначала загрузите текст на странице «Данные» или возьмите демо-корпус.")


class MetricTile(Card):
    """Небольшая плитка-метрика."""

    def __init__(self, master, label: str):
        super().__init__(master)
        inner = tk.Frame(self, bg=T.CARD)
        inner.pack(fill="both", expand=True, padx=12, pady=10)
        tk.Label(inner, text=label, bg=T.CARD, fg=T.TEXT_FAINT,
                 font=T.font(8)).pack(anchor="w")
        self.val = tk.Label(inner, text="—", bg=T.CARD, fg=T.TEXT,
                            font=T.font(15, bold=True))
        self.val.pack(anchor="w")

    def set(self, v: str) -> None:
        self.val.config(text=v)


if __name__ == "__main__":  # pragma: no cover
    MainApp().mainloop()
