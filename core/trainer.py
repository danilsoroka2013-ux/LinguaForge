# -*- coding: utf-8 -*-
"""Тренер: цикл обучения, чекпоинты, валидация, журнал, мягкая остановка."""
from __future__ import annotations

import math
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from core.config import Scheduler, clip_grad_norm, save_checkpoint
from core.data import make_batches
from core.model import TinyGPT
from core.optimizer import AdamW


class TrainState:
    """Снимок прогресса для UI (потокобезопасно по замене ссылки)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.step = 0
        self.total_steps = 0
        self.loss: Optional[float] = None
        self.val_loss: Optional[float] = None
        self.val_ppl: Optional[float] = None
        self.lr = 0.0
        self.grad_norm = 0.0
        self.tok_per_s = 0.0
        self.history: List[Dict[str, float]] = []
        self.finished = False
        self.error: Optional[str] = None
        self.message = "Готов"

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "step": self.step,
                "total_steps": self.total_steps,
                "loss": self.loss,
                "val_loss": self.val_loss,
                "val_ppl": self.val_ppl,
                "lr": self.lr,
                "grad_norm": self.grad_norm,
                "tok_per_s": self.tok_per_s,
                "history": list(self.history[-512:]),
                "finished": self.finished,
                "error": self.error,
                "message": self.message,
            }


class Trainer:
    """Обучение в отдельном потоке с мягкой остановкой."""

    def __init__(self, model: TinyGPT, tokenizer, cfg: Dict[str, Any],
                 train_ids: np.ndarray, val_ids: Optional[np.ndarray],
                 project_dir: str, on_done: Optional[Callable[[], None]] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = dict(cfg)
        self.train_ids = train_ids
        self.val_ids = val_ids
        self.project_dir = project_dir
        self.on_done = on_done

        self.state = TrainState()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------- контроль
    def start(self) -> None:
        if self.is_running():
            return
        self._stop.clear()
        self.state.finished = False
        self.state.error = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def is_running(self) -> None | bool:
        return bool(self._thread and self._thread.is_alive())

    # ------------------------------------------------------------- цикл
    def _run(self) -> None:
        cfg = self.cfg
        st = self.state
        try:
            steps = int(cfg["steps"])
            bs = int(cfg["batch_size"])
            seq = int(cfg["seq_len"])
            eval_every = max(0, int(cfg.get("eval_every", 200)))
            ckpt_every = max(0, int(cfg.get("ckpt_every", 500)))
            keep = max(1, int(cfg.get("keep_checkpoints", 3)))

            total_tokens_target = steps * bs * seq
            sched = Scheduler(
                base_lr=float(cfg["lr"]),
                warmup=int(cfg.get("warmup", max(10, steps // 20))),
                total=steps,
                min_lr=float(cfg.get("min_lr", 0.0)),
            )
            opt = AdamW(self.model.params(), lr=float(cfg["lr"]),
                        weight_decay=float(cfg.get("weight_decay", 0.05)))
            get_batch = make_batches(self.train_ids, bs, seq, steps,
                                     seed=int(cfg.get("seed", 0)))

            os.makedirs(self.project_dir, exist_ok=True)
            val_batch = None
            if self.val_ids is not None and self.val_ids.size > seq + 1:
                vb = make_batches(self.val_ids, min(bs, 8), seq, 1,
                                  seed=123)
                val_batch = vb()

            st.total_steps = steps
            st.message = "Обучение"
            t0 = time.time()
            tokens_seen = 0
            # BLAS-потоки: задаём ДО первого matmul, setdefault не перезапишет
            # то, что выставил пользователь.
            threads = max(1, min(8, os.cpu_count() or 1))
            for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                os.environ.setdefault(var, str(threads))

            for step in range(1, steps + 1):
                if self._stop.is_set():
                    st.message = "Остановлено пользователем"
                    break

                x, y = get_batch()
                logits, loss = self.model.forward(x, y)
                opt.zero_grad()
                loss.backward()
                gnorm = clip_grad_norm(
                    {k: v.grad for k, v in self.model.params().items() if v.grad is not None},
                    float(cfg.get("grad_clip", 1.0)))
                lr = sched.step()
                opt.step(lr)
                if step % 25 == 0 or step == 1:
                    st.message = f"Шаг {step}/{steps} · лосс {lv:.3f}"

                lv = float(loss.data)
                tokens_seen += x.size
                st.step = step
                st.loss = lv
                st.lr = lr
                st.grad_norm = gnorm
                st.tok_per_s = tokens_seen / max(1e-6, time.time() - t0)

                # Журнал: на больших прогонах не копим тысячи точек —
                # достаточно ~10 точек на декаду шагов для графика.
                if (step <= 100 or step % max(1, steps // 100) == 0):
                    with st.lock:
                        st.history.append({"step": step, "loss": lv, "lr": lr,
                                           "val": st.val_loss if st.val_loss else float("nan")})

                if eval_every and val_batch is not None and step % eval_every == 0:
                    self._evaluate(val_batch, st)

                if ckpt_every and step % ckpt_every == 0:
                    self._checkpoint(step, keep)

            # финал
            self._evaluate(val_batch, st) if val_batch is not None else None
            self._checkpoint(max(1, st.step), keep, final=True)
            st.finished = True
            if st.message != "Остановлено пользователем":
                st.message = "Готово"
        except Exception as e:  # UI покажет
            st.error = f"{type(e).__name__}: {e}"
            st.finished = True
        finally:
            if self.on_done:
                try:
                    self.on_done()
                except Exception:
                    pass

    def _evaluate(self, val_batch, st: TrainState) -> None:
        x, y = val_batch
        logits, loss = self.model.forward(x, y)
        v = float(loss.data)
        st.val_loss = v
        st.val_ppl = math.exp(min(20.0, v))

    def _checkpoint(self, step: int, keep: int, final: bool = False) -> None:
        tag = "final" if final else f"step{step:06d}"
        path = os.path.join(self.project_dir, f"model_{tag}.lfmodel")
        save_checkpoint(
            path,
            model_params={k: v.data for k, v in self.model.params().items()},
            config={**self.model.config_dict(), "tokenizer": self.tokenizer.to_json()},
            extra={"step": step,
                   "loss": self.state.loss,
                   "val_loss": self.state.val_loss},
        )
        self.state.message = f"Сохранено: {os.path.basename(path)}"
        # ротация старых чекпоинтов
        from core.config import list_checkpoints
        ckpts = [p for p in list_checkpoints(self.project_dir)
                 if "final" not in os.path.basename(p)]
        while len(ckpts) > keep:
            oldest = ckpts.pop()
            try:
                os.remove(oldest)
            except OSError:
                pass
