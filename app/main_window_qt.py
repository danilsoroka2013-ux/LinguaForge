# -*- coding: utf-8 -*-
"""Главное окно LinguaForge на Qt (PySide6) + QSS-стили."""
from __future__ import annotations

import os
import threading

import numpy as np

from PySide6.QtCore import Qt, QTimer, QSize, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QRadioButton, QButtonGroup, QLineEdit, QTextEdit, QFileDialog,
    QMessageBox, QSizePolicy, QApplication, QStackedWidget,
)

from ui_qt import theme as T
from ui_qt import icons
from ui_qt.widgets import (Bubble, ChatWidget, LossChart, MetricTile,
                           ParamSlider, StatRow, card, label, vwrap)

from core.config import VERSION, APP_NAME, human_size, list_checkpoints
from core.trainer import Trainer

from app.project import ProjectState, demo_corpus_path


PAGES = [
    ("overview", "Обзор", "overview"),
    ("data", "Данные", "data"),
    ("tokenizer", "Токенизатор", "tokenizer"),
    ("model", "Модель", "model"),
    ("train", "Обучение", "train"),
    ("play", "Песочница", "play"),
]


class Page(QWidget):
    """Базовая страница с заголовком."""

    def __init__(self, win: "MainWindow", title: str, sub: str):
        super().__init__()
        self.win = win
        self.root_lay = QVBoxLayout(self)
        self.root_lay.setContentsMargins(26, 18, 26, 18)
        self.root_lay.setSpacing(10)
        self.root_lay.addWidget(label(title, "h1"))
        if sub:
            self.root_lay.addWidget(label(sub, "subtitle"))
        self.root_lay.addStretch(1)

    def add(self, w: QWidget, stretch: int = 0) -> None:
        self.root_lay.insertWidget(self.root_lay.count() - 1, w, stretch)

    def refresh(self) -> None:
        pass


# ================================================================ страницы

class OverviewPage(Page):
    def __init__(self, win):
        super().__init__(
            win, "Обзор",
            "Соберите свою языковую модель за пять шагов — полностью локально, "
            "без интернета и GPU.")
        row = QHBoxLayout()
        row.setSpacing(12)

        card_steps = card()
        steps_lay = QVBoxLayout(card_steps)
        steps_lay.setContentsMargins(16, 14, 16, 14)
        steps_lay.addWidget(label("Путь сборки", "cardTitle"))
        items = [
            ("1 · Данные", "загрузите текст или демо-корпус"),
            ("2 · Токенизатор", "символьный или BPE, обучается за секунду"),
            ("3 · Модель", "размер, слои, контекст — как у больших, только малый"),
            ("4 · Обучение", "кривая лосса, валидация, чекпоинты"),
            ("5 · Песочница", "чат с вашей моделью"),
        ]
        for name, desc in items:
            row_i = QHBoxLayout()
            t = label(name, "cardTitle")
            t.setStyleSheet("font-size: 13px; background: transparent")
            d = label(desc, "hint")
            row_i.addWidget(t)
            row_i.addSpacing(8)
            row_i.addWidget(d, 1)
            steps_lay.addLayout(row_i)
        row.addWidget(card_steps, 3)

        side = card()
        side.setFixedWidth(300)
        side_lay = QVBoxLayout(side)
        side_lay.setContentsMargins(16, 14, 16, 14)
        side_lay.addWidget(label("Ваш проект", "cardTitle"))
        self.st_corpus = StatRow("Корпус")
        self.st_vocab = StatRow("Словарь")
        self.st_params = StatRow("Параметры")
        self.st_ctx = StatRow("Контекст")
        self.st_ckpt = StatRow("Чекпоинт")
        for s in (self.st_corpus, self.st_vocab, self.st_params, self.st_ctx,
                  self.st_ckpt):
            side_lay.addWidget(s)
        side_lay.addSpacing(8)
        self.btn_open = QPushButton("Открыть .lfmodel")
        self.btn_open.clicked.connect(self.win.open_model_dialog)
        self.btn_demo = QPushButton("Загрузить демо-корпус")
        self.btn_demo.setProperty("variant", "primary")
        self.btn_demo.clicked.connect(self.win.load_demo)
        side_lay.addWidget(self.btn_open)
        side_lay.addWidget(self.btn_demo)
        side_lay.addStretch(1)
        row.addWidget(side, 1)

        self.add(vwrap(QWidget(), spacing=0), 0)  # распорка не нужна, layout сам
        self.add(QWidget(), 1)
        wrap = QWidget()
        wrap.setLayout(row)
        wrap.setStyleSheet("background: transparent")
        # вставляем row-контейнер перед stretch
        self.root_lay.insertWidget(self.root_lay.count() - 1, wrap, 1)

    def refresh(self) -> None:
        st = self.win.state
        if st.text:
            self.st_corpus.set(f"{st.corpus_stats()['chars']:,} симв.")
        else:
            self.st_corpus.set("—")
        self.st_vocab.set(str(st.tokenizer.vocab_size) if st.tokenizer else "—")
        self.st_params.set(f"{st.model.num_params():,}" if st.model else "—")
        self.st_ctx.set(str(st.model_cfg["block_size"]))
        ckpts = list_checkpoints(st.dir)
        if ckpts:
            self.st_ckpt.set(f"{os.path.basename(ckpts[0])} · "
                             f"{human_size(os.path.getsize(ckpts[0]))}")
        else:
            self.st_ckpt.set("—")


class DataPage(Page):
    def __init__(self, win):
        super().__init__(
            win, "Данные",
            "Корпус — учебник модели. Чем чище текст, тем связнее речь.")
        c = card()
        lay = QVBoxLayout(c)
        lay.setContentsMargins(16, 14, 16, 14)

        btns = QHBoxLayout()
        b1 = QPushButton("Загрузить .txt")
        b1.clicked.connect(self.win.pick_text)
        b2 = QPushButton("Демо-корпус")
        b2.setProperty("variant", "primary")
        b2.clicked.connect(self.win.load_demo)
        btns.addWidget(b1)
        btns.addWidget(b2)
        btns.addStretch(1)
        lay.addLayout(btns)

        self.stats = label("Файл не выбран", "hint")
        lay.addWidget(self.stats)

        self.text = QTextEdit()
        self.text.setProperty("mono", True)
        self.text.setReadOnly(True)
        lay.addWidget(self.text, 1)
        self.add(c, 1)

        self.val_slider = ParamSlider(
            "часть корпуса для контроля качества (валидация)",
            0.0, 0.3, win.state.val_frac, 0.01,
            fmt=lambda v: f"{v*100:.0f} %")
        self.val_slider.changed.connect(lambda v: setattr(win.state, "val_frac", v))
        cc = card()
        cl = QVBoxLayout(cc)
        cl.setContentsMargins(16, 12, 16, 12)
        cl.addWidget(self.val_slider)
        self.add(cc, 0)

    def refresh(self) -> None:
        st = self.win.state
        if not st.text:
            self.stats.setText("Файл не выбран")
            return
        s = st.corpus_stats()
        src = os.path.basename(st.text_path) if st.text_path else "(вставлено)"
        self.stats.setText(f"{src}: {s['chars']:,} символов · {s['words']:,} слов "
                           f"· {s['lines']:,} строк")
        self.text.setPlainText(st.text[:40000])


class TokenizerPage(Page):
    def __init__(self, win):
        super().__init__(
            win, "Токенизатор",
            "Модель читает не буквы, а номера-токены. Здесь вы решаете, как "
            "текст превращается в числа.")
        c = card()
        lay = QVBoxLayout(c)
        lay.setContentsMargins(16, 14, 16, 14)

        kinds = QHBoxLayout()
        self.kind_char = QRadioButton("Символьный")
        self.kind_bpe = QRadioButton("BPE")
        (self.kind_char if win.state.tokenizer_kind == "char"
         else self.kind_bpe).setChecked(True)
        self.group = QButtonGroup(self)
        self.group.addButton(self.kind_char)
        self.group.addButton(self.kind_bpe)
        for rb in (self.kind_char, self.kind_bpe):
            kinds.addWidget(rb)
        kinds.addStretch(1)
        lay.addLayout(kinds)
        self.hint = label("", "hint")
        lay.addWidget(self.hint)

        self.vocab = ParamSlider("размер словаря (для BPE)", 64, 2048,
                                 win.state.vocab_size, 64)
        self.vocab.changed.connect(lambda v: setattr(win.state, "vocab_size", int(v)))
        lay.addWidget(self.vocab)

        self.btn_train = QPushButton("Обучить токенизатор")
        self.btn_train.setProperty("variant", "primary")
        self.btn_train.clicked.connect(self.win.train_tokenizer)
        lay.addWidget(self.btn_train)
        self.status = label("", "statusOk")
        lay.addWidget(self.status)
        self.add(c, 0)

        p = card()
        pl = QVBoxLayout(p)
        pl.setContentsMargins(16, 14, 16, 14)
        pl.addWidget(label("Как текст видит модель", "cardTitle"))
        self.preview = QTextEdit()
        self.preview.setProperty("mono", True)
        self.preview.setReadOnly(True)
        pl.addWidget(self.preview, 1)
        self.add(p, 1)
        self.kind_char.toggled.connect(self._kind_changed)
        self._kind_changed()

    def _kind_changed(self) -> None:
        self.win.state.tokenizer_kind = ("char" if self.kind_char.isChecked()
                                         else "bpe")
        self.hint.setText(
            "Каждый символ — один токен. Простой и предсказуемый." if
            self.win.state.tokenizer_kind == "char" else
            "Частые пары символов сливаются в один токен — текст упаковывается плотнее.")

    def refresh(self) -> None:
        st = self.win.state
        if st.tokenizer:
            self.status.setText(f"Активен: {st.tokenizer.name}, "
                                f"{st.tokenizer.vocab_size} токенов")
            self.status.setObjectName("statusOk")
            self._show_preview()

    def _show_preview(self) -> None:
        st = self.win.state
        if not st.tokenizer:
            return
        sample = st.text[:220]
        ids = st.tokenizer.encode(sample)
        self.preview.setPlainText(
            "текст:   " + sample.replace("\n", " ")[:110] + "\n\n" +
            "токены:  " + " ".join(map(str, ids[:80])) + "\n\n" +
            "обратно: " + st.tokenizer.decode(ids[:80]).replace("\n", " "))


class ModelPage(Page):
    def __init__(self, win):
        super().__init__(
            win, "Модель",
            "Архитектура — миниатюрный GPT: внимание, слои, контекст. "
            "Больше параметров — умнее, но медленнее.")
        row = QHBoxLayout()
        row.setSpacing(12)

        left = card()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(16, 14, 16, 14)
        presets_row = QHBoxLayout()
        presets_row.addWidget(label("Пресет", "statLabel"))
        self.preset_group = QButtonGroup(self)
        self.presets = {
            "Крошечная (для теста)": dict(dim=48, n_layers=2, n_heads=2,
                                          block_size=64, ffn_mult=2.667),
            "Мини (рекомендуется)": dict(dim=96, n_layers=3, n_heads=3,
                                         block_size=96, ffn_mult=2.667),
            "Малая": dict(dim=128, n_layers=4, n_heads=4,
                          block_size=128, ffn_mult=2.667),
        }
        self.preset_buttons = {}
        for name in self.presets:
            rb = QRadioButton(name)
            if name == "Мини (рекомендуется)":
                rb.setChecked(True)
            rb.toggled.connect(lambda on, n=name: self._preset(n, on))
            self.preset_group.addButton(rb)
            self.preset_buttons[name] = rb
            presets_row.addWidget(rb)
        presets_row.addStretch(1)
        ll.addLayout(presets_row)

        c = win.state.model_cfg
        self.s_dim = ParamSlider("размерность (dim)", 32, 256, c["dim"], 16)
        self.s_layers = ParamSlider("слои (глубина)", 1, 8, c["n_layers"], 1)
        self.s_heads = ParamSlider("головы внимания", 1, 8, c["n_heads"], 1)
        self.s_ctx = ParamSlider("контекст (block_size)", 32, 256, c["block_size"], 16)
        for s in (self.s_dim, self.s_layers, self.s_heads, self.s_ctx):
            s.changed.connect(self._update_summary)
            ll.addWidget(s)
        row.addWidget(left, 2)

        right = card()
        right.setFixedWidth(290)
        rl = QVBoxLayout(right)
        rl.setContentsMargins(16, 14, 16, 14)
        rl.addWidget(label("Сводка", "cardTitle"))
        self.st_params = StatRow("Параметров")
        self.st_ram = StatRow("Вес модели")
        self.st_vocab = StatRow("Словарь")
        self.st_ctx = StatRow("Контекст")
        for s in (self.st_params, self.st_ram, self.st_vocab, self.st_ctx):
            rl.addWidget(s)
        rl.addSpacing(8)
        self.btn_build = QPushButton("Собрать модель")
        self.btn_build.setProperty("variant", "primary")
        self.btn_build.clicked.connect(self.win.build_model)
        rl.addWidget(self.btn_build)
        self.status = label("")
        self.status.setWordWrap(True)
        rl.addWidget(self.status)
        rl.addStretch(1)
        row.addWidget(right, 1)

        wrap = QWidget()
        wrap.setLayout(row)
        wrap.setStyleSheet("background: transparent")
        self.add(wrap, 1)
        self._update_summary()

    def _preset(self, name: str, on: bool) -> None:
        if not on:
            return
        self.win.state.model_cfg.update(self.presets[name])
        p = self.presets[name]
        self.s_dim.set_value(p["dim"])
        self.s_layers.set_value(p["n_layers"])
        self.s_heads.set_value(p["n_heads"])
        self.s_ctx.set_value(p["block_size"])
        self._update_summary()

    def _update_summary(self) -> None:
        c = self.win.state.model_cfg
        c["dim"] = int(self.s_dim.value())
        c["n_layers"] = int(self.s_layers.value())
        c["n_heads"] = int(self.s_heads.value())
        c["block_size"] = int(self.s_ctx.value())
        vocab = self.win.state.tokenizer.vocab_size if self.win.state.tokenizer else 256
        hidden = int(c["dim"] * c["ffn_mult"]); hidden += hidden % 2
        per_layer = 3 * c["dim"] ** 2 + c["dim"] ** 2 + 2 * c["dim"] * hidden
        params = vocab * c["dim"] + c["n_layers"] * per_layer + c["n_layers"] * 2 * c["dim"] + c["dim"]
        self.st_params.set(f"{params:,}")
        self.st_ram.set(f"~{human_size(params * 4)}")
        self.st_vocab.set(str(vocab))
        self.st_ctx.set(f"{c['block_size']} симв.")

    def refresh(self) -> None:
        self._update_summary()
        if self.win.state.model:
            self.status.setObjectName("statusOk")
            self.status.setText(f"Активна модель: "
                                f"{self.win.state.model.num_params():,} параметров.")
            self.status.style().unpolish(self.status)
            self.status.style().polish(self.status)


class TrainPage(Page):
    def __init__(self, win):
        super().__init__(
            win, "Обучение",
            "Нажмите «Начать» и смотрите, как лосс ползёт вниз. Чекпоинты "
            "сохраняются автоматически.")
        row = QHBoxLayout()
        row.setSpacing(12)

        left = card()
        left.setFixedWidth(300)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(16, 14, 16, 14)
        ll.addWidget(label("Гиперпараметры", "cardTitle"))
        tc = win.state.train_cfg
        self.s_steps = ParamSlider("шагов обучения", 200, 20000, tc["steps"], 100)
        self.s_lr = ParamSlider("learning rate", 0.0005, 0.01, tc["lr"], 0.0005,
                                fmt=lambda v: f"{v:.4f}")
        self.s_bs = ParamSlider("batch size", 4, 48, tc["batch_size"], 4)
        self.s_steps.changed.connect(lambda v: tc.__setitem__("steps", int(v)))
        self.s_lr.changed.connect(lambda v: tc.__setitem__("lr", float(v)))
        self.s_bs.changed.connect(lambda v: tc.__setitem__("batch_size", int(v)))
        for s in (self.s_steps, self.s_lr, self.s_bs):
            ll.addWidget(s)
        seed_row = QHBoxLayout()
        seed_row.addWidget(label("seed", "statLabel"))
        self.seed_edit = QLineEdit(str(tc["seed"]))
        self.seed_edit.setFixedWidth(90)
        self.seed_edit.editingFinished.connect(self._seed)
        seed_row.addWidget(self.seed_edit)
        seed_row.addStretch(1)
        ll.addLayout(seed_row)

        self.btn_start = QPushButton("▶  Начать обучение")
        self.btn_start.setProperty("variant", "primary")
        self.btn_start.clicked.connect(self.win.start_train)
        self.btn_stop = QPushButton("■  Остановить")
        self.btn_stop.setProperty("variant", "danger")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.win.stop_train)
        ll.addWidget(self.btn_start)
        ll.addWidget(self.btn_stop)
        self.status = label("Готово к запуску", "statusDim")
        self.status.setWordWrap(True)
        ll.addWidget(self.status)
        ll.addStretch(1)
        row.addWidget(left)

        right = QVBoxLayout()
        right.setSpacing(10)
        chart_card = card()
        cl = QVBoxLayout(chart_card)
        cl.setContentsMargins(12, 10, 12, 10)
        self.chart = LossChart()
        cl.addWidget(self.chart, 1)
        right.addWidget(chart_card, 1)

        metrics = QHBoxLayout()
        self.m_loss = MetricTile("лосс")
        self.m_val = MetricTile("валидация")
        self.m_ppl = MetricTile("перплексия")
        self.m_speed = MetricTile("токенов/с")
        for m in (self.m_loss, self.m_val, self.m_ppl, self.m_speed):
            metrics.addWidget(m)
        right.addLayout(metrics)

        rwrap = QWidget()
        rwrap.setLayout(right)
        rwrap.setStyleSheet("background: transparent")
        row.addWidget(rwrap, 1)

        wrap = QWidget()
        wrap.setLayout(row)
        wrap.setStyleSheet("background: transparent")
        self.add(wrap, 1)

    def _seed(self) -> None:
        try:
            self.win.state.train_cfg["seed"] = int(self.seed_edit.text() or 0)
        except ValueError:
            pass

    def set_metrics(self, snap: dict) -> None:
        self.m_loss.set(f"{snap['loss']:.3f}" if snap["loss"] is not None else "—")
        self.m_val.set(f"{snap['val_loss']:.3f}" if snap["val_loss"] is not None else "—")
        self.m_ppl.set(f"{snap['val_ppl']:.1f}" if snap["val_ppl"] else "—")
        self.m_speed.set(f"{snap['tok_per_s']:,.0f}" if snap["tok_per_s"] else "—")

    def refresh(self) -> None:
        if self.win.state.history:
            self.chart.set_data(self.win.state.history)


class PlayPage(Page):
    def __init__(self, win):
        super().__init__(
            win, "Песочница",
            "Поговорите со своей моделью. Генерация идёт локально, на вашем "
            "процессоре.")
        c = card()
        cl = QVBoxLayout(c)
        cl.setContentsMargins(10, 10, 10, 10)
        self.chat = ChatWidget()
        self.chat.send.connect(self.win.on_chat_send)
        cl.addWidget(self.chat, 1)
        self.add(c, 1)

        opts = card()
        ol = QHBoxLayout(opts)
        ol.setContentsMargins(16, 12, 16, 12)
        self.s_temp = ParamSlider("температура", 0.1, 1.5, 0.8, 0.05,
                                  fmt=lambda v: f"{v:.2f}")
        self.s_max = ParamSlider("макс. длина", 16, 256, 96, 16)
        self.s_rep = ParamSlider("штраф повтора", 1.0, 1.6, 1.15, 0.05,
                                 fmt=lambda v: f"{v:.2f}")
        for s in (self.s_temp, self.s_max, self.s_rep):
            ol.addWidget(s, 1)
        self.add(opts, 0)

        self.chat.add("system",
                      "Это песочница. Загрузите или обучите модель — и пишите "
                      "в поле ниже. Модель продолжит текст.")
        self.chat.set_busy(False)


# ================================================================ окно

class MainWindow(QMainWindow):
    generation_done = Signal(str)
    train_tick = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} — конструктор языковых моделей")
        self.resize(1120, 740)
        self.setMinimumSize(960, 660)

        self.state = ProjectState()
        self.trainer = None
        self._gen_busy = False

        self.generation_done.connect(self._chat_reply)
        self.train_tick.connect(self._on_train_tick)

        root = QWidget()
        root.setObjectName("root")
        lay = QHBoxLayout(root)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # ---------------- сайдбар
        side = QWidget()
        side.setObjectName("sidebar")
        side.setFixedWidth(198)
        sl = QVBoxLayout(side)
        sl.setContentsMargins(10, 0, 10, 0)
        sl.setSpacing(2)
        sl.addWidget(label(f"◆ {APP_NAME}", "appTitle"))
        sl.addWidget(label(f"v{VERSION} · локально · офлайн", "appSub"))

        self.nav_group = QButtonGroup(self)
        self.nav_buttons = {}
        for key, text, icon_name in PAGES:
            b = QPushButton("  " + text)
            b.setProperty("class", "nav")
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.setIcon(icons.icon(icon_name, icons.normal_color()))
            b.setIconSize(QSize(20, 20))
            b.clicked.connect(lambda _=False, k=key: self.show_page(k))
            self.nav_group.addButton(b)
            self.nav_buttons[key] = b
            sl.addWidget(b)
        self.nav_buttons["overview"].setChecked(True)

        sl.addStretch(1)
        foot = QWidget()
        foot.setObjectName("sidebarFooter")
        fl = QVBoxLayout(foot)
        fl.setContentsMargins(8, 10, 8, 12)
        self.foot_corpus = StatRow("Корпус")
        self.foot_model = StatRow("Модель")
        self.foot_step = StatRow("Шаг")
        for s in (self.foot_corpus, self.foot_model, self.foot_step):
            fl.addWidget(s)
        sl.addWidget(foot)

        # ---------------- стек страниц
        self.stack = QStackedWidget()
        self.pages = {}
        for key, _text, _icon in PAGES:
            pg = {
                "overview": OverviewPage,
                "data": DataPage,
                "tokenizer": TokenizerPage,
                "model": ModelPage,
                "train": TrainPage,
                "play": PlayPage,
            }[key](self)
            self.pages[key] = pg
            self.stack.addWidget(pg)

        lay.addWidget(side)
        lay.addWidget(self.stack, 1)
        self.setCentralWidget(root)

        # таймер обновления обучения
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll_train)
        self._timer.start(400)

    # ------------------------------------------------------------ навигация
    def show_page(self, key: str) -> None:
        self.nav_group.setExclusive(False)
        for k, b in self.nav_buttons.items():
            b.setChecked(k == key)
            # Перекрашиваем SVG-иконку: активный пункт — акцентный.
            icon_name = dict((p[0], p[2]) for p in PAGES)[k]
            color = icons.active_color() if k == key else icons.normal_color()
            b.setIcon(icons.icon(icon_name, color))
        self.nav_group.setExclusive(True)
        self.stack.setCurrentWidget(self.pages[key])
        self.pages[key].refresh()
        self._refresh_footer()

    def _refresh_footer(self) -> None:
        st = self.state
        self.foot_corpus.set(f"{st.corpus_stats()['chars']:,} симв."
                             if st.text else "не загружен")
        self.foot_model.set(f"{st.model.num_params():,} пар." if st.model
                            else "не собрана")

    # ------------------------------------------------------------ действия
    def load_demo(self) -> None:
        self.state.load_demo_corpus()
        self.pages["data"].refresh()
        self._refresh_footer()
        self.show_page("data")

    def pick_text(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите текстовый файл", "",
            "Текстовые файлы (*.txt *.md *.csv *.log);;Все файлы (*.*)")
        if not path:
            return
        try:
            self.state.load_text(path)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось прочитать файл:\n{e}")
            return
        self.pages["data"].refresh()
        self._refresh_footer()

    def open_model_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Открыть модель", "", "LinguaForge модель (*.lfmodel)")
        if not path:
            return
        try:
            self.state.load_model(path)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось открыть модель:\n{e}")
            return
        self._refresh_footer()
        QMessageBox.information(self, "Модель загружена", os.path.basename(path))
        self.show_page("play")

    def train_tokenizer(self) -> None:
        st = self.state
        if not st.text:
            QMessageBox.warning(self, "Нужен корпус",
                                "Сначала загрузите текст на странице «Данные» "
                                "или возьмите демо-корпус.")
            return
        try:
            tok = st.build_tokenizer()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))
            return
        pg = self.pages["tokenizer"]
        pg.status.setText(f"Готово: {tok.vocab_size} токенов · "
                          f"{len(st.tokenize_corpus()):,} токенов в корпусе")
        pg.status.setObjectName("statusOk")
        pg.status.style().unpolish(pg.status)
        pg.status.style().polish(pg.status)
        pg._show_preview()
        self._refresh_footer()

    def build_model(self) -> None:
        st = self.state
        try:
            if not st.tokenizer:
                st.build_tokenizer()
            c = st.model_cfg
            heads = c["n_heads"]
            if c["dim"] % heads:
                c["dim"] = (c["dim"] // heads) * heads
                self.pages["model"].s_dim.set_value(c["dim"])
            model = st.build_model(force=True)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось собрать модель:\n{e}")
            return
        pg = self.pages["model"]
        pg.status.setObjectName("statusOk")
        pg.status.setText(f"Модель готова: {model.num_params():,} параметров. "
                          f"Переходите к обучению.")
        pg.status.style().unpolish(pg.status)
        pg.status.style().polish(pg.status)
        self._refresh_footer()

    # ------------------------------------------------------------ обучение
    def start_train(self) -> None:
        st = self.state
        try:
            st.ensure_model_for_training()
        except Exception as e:
            QMessageBox.warning(self, "Не готово", str(e))
            return
        tc = dict(st.train_cfg)
        tc["seq_len"] = min(int(tc.get("seq_len", st.model_cfg["block_size"])),
                            st.model_cfg["block_size"])
        train_ids, val_ids = st.train_val_ids()
        if train_ids is None or train_ids.size < tc["seq_len"] + 2:
            QMessageBox.warning(self, "Мало данных",
                                "Корпус слишком короткий для выбранных настроек.")
            return
        self.trainer = Trainer(st.model, st.tokenizer, tc, train_ids, val_ids,
                               st.dir, on_done=self.train_tick.emit)
        st.history = []
        self.pages["train"].chart.clear()
        self.trainer.start()
        self.pages["train"].btn_start.setEnabled(False)
        self.pages["train"].btn_stop.setEnabled(True)
        self._set_train_status("Обучение идёт…", "statusAccent")

    def stop_train(self) -> None:
        if self.trainer:
            self.trainer.stop()

    def _set_train_status(self, text: str, obj: str) -> None:
        s = self.pages["train"].status
        s.setText(text)
        s.setObjectName(obj)
        s.style().unpolish(s)
        s.style().polish(s)

    def _poll_train(self) -> None:
        tr = self.trainer
        if not tr:
            return
        snap = tr.state.snapshot()
        self.state.history = snap["history"]
        pg = self.pages["train"]
        pg.chart.set_data(snap["history"])
        pg.set_metrics(snap)
        self.foot_step.set(str(snap["step"] or "—"))

    def _on_train_tick(self) -> None:
        tr = self.trainer
        if not tr:
            return
        snap = tr.state.snapshot()
        if snap["error"]:
            self._set_train_status(f"Ошибка: {snap['error']}", "statusErr")
            QMessageBox.critical(self, "Обучение прервано", snap["error"])
        else:
            ok = snap["message"] == "Готово"
            self._set_train_status(f"{snap['message']} · шаг {snap['step']}",
                                   "statusOk" if ok else "statusWarn")
        self.pages["train"].btn_start.setEnabled(True)
        self.pages["train"].btn_stop.setEnabled(False)

    # ------------------------------------------------------------ песочница
    def on_chat_send(self, text: str) -> None:
        st = self.state
        if self._gen_busy:
            return
        if not st.model or not st.tokenizer:
            self.pages["play"].chat.add(
                "system", "Сначала соберите модель на странице «Модель» "
                          "(или откройте .lfmodel).")
            return
        self.pages["play"].chat.add("user", text)
        self.pages["play"].chat.set_busy(True)
        self._gen_busy = True
        temp = self.pages["play"].s_temp.value()
        max_new = int(self.pages["play"].s_max.value())
        rep = self.pages["play"].s_rep.value()

        def worker():
            try:
                tok = st.tokenizer
                # Формат вопроса-ответа из корпуса: модель видит «Вопрос: …»
                # и продолжает уже ответом, а не стихами.
                prompt = f"Вопрос: {text.strip()}\nОтвет:"
                ids = tok.encode(prompt)[-st.model.block_size:]
                if not ids:
                    ids = tok.encode("Вопрос: Привет!\nОтвет:")
                ctx = np.array([ids], dtype=np.int32)
                out = st.model.generate(ctx, max_new_tokens=max_new,
                                        temperature=temp, repetition_penalty=rep,
                                        top_p=0.95, stop_at_newline=True)
                reply = tok.decode(out[0][len(ids):])
            except Exception as e:
                reply = f"(ошибка генерации: {e})"
            self.generation_done.emit(reply)

        threading.Thread(target=worker, daemon=True).start()

    def _chat_reply(self, reply: str) -> None:
        self.pages["play"].chat.add("model", reply if reply.strip() else "(пусто)")
        self.pages["play"].chat.set_busy(False)
        self._gen_busy = False

    def closeEvent(self, e) -> None:
        if self.trainer is not None and self.trainer.is_running():
            self.trainer.stop()
        super().closeEvent(e)


def run_app() -> int:
    import sys
    app = QApplication(sys.argv)
    app.setStyleSheet(T.qss())
    app.setFont(QFont("Segoe UI", 10))
    win = MainWindow()
    from ui_qt.icons import icon
    win.setWindowIcon(icon("spark", T.ACCENT))
    win.show()
    return app.exec()
