# -*- coding: utf-8 -*-
"""Обучение демо-модели LinguaForge на диалоговом корпусе (без GUI).

Запуск:
    python scripts/train_demo.py              # пресет по умолчанию
    python scripts/train_demo.py --steps 4000

Результат: ~/.linguaforge/projects/Демо-модель/model_final.lfmodel
"""
import argparse
import gc
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from core.config import (Scheduler, clip_grad_norm, save_checkpoint,  # noqa: E402
                         set_seed, PROJECTS_DIR)
from core.data import load_text_file, make_batches  # noqa: E402
from core.model import TinyGPT                      # noqa: E402
from core.optimizer import AdamW                    # noqa: E402
from core.tokenizers import build_tokenizer         # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.004)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--project", type=str, default="Демо-модель")
    args = ap.parse_args()

    set_seed(args.seed)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    corpus_path = os.path.join(root, "assets", "corpus.txt")
    text = load_text_file(corpus_path)

    tok = build_tokenizer("char", text, 512)
    ids = np.array(tok.encode(text), dtype=np.int32)
    n_val = int(len(ids) * 0.05)
    train_ids, val_ids = ids[:-n_val], ids[-n_val:]
    print(f"корпус: {len(text):,} симв. · словарь: {tok.vocab_size} · "
          f"токенов: {len(ids):,} (val {len(val_ids):,})", flush=True)

    model = TinyGPT(vocab_size=tok.vocab_size, dim=args.dim, n_layers=args.layers,
                    n_heads=args.heads, block_size=args.block)
    model.set_tokenizer_vocab(tok.itos)
    print(f"модель: {model.num_params():,} параметров · dim={args.dim} "
          f"L={args.layers} H={args.heads} ctx={args.block}", flush=True)

    seq = args.block
    get_batch = make_batches(train_ids, args.batch, seq, args.steps, seed=args.seed)
    val_batch = make_batches(val_ids, 8, seq, 1, seed=123)()

    from core.tensor import Tensor

    def eval_val() -> float:
        with Tensor.infer_mode():
            _logits, vloss = model.forward(val_batch[0], val_batch[1])
        return float(vloss.data)

    opt = AdamW(model.params(), lr=args.lr, weight_decay=0.05)
    sched = Scheduler(base_lr=args.lr, warmup=max(10, args.steps // 20),
                      total=args.steps, min_lr=args.lr * 0.1)
    params = model.params()

    t0 = time.time()
    tokens = 0
    loss_val = 0.0
    for step in range(1, args.steps + 1):
        x, y = get_batch()
        _, loss = model.forward(x, y)
        opt.zero_grad()
        loss.backward()
        clip_grad_norm({k: v.grad for k, v in params.items() if v.grad is not None}, 1.0)
        lr = sched.step()
        opt.step(lr)
        tokens += x.size
        loss_val = float(loss.data)

        if step % 50 == 0 or step == 1 or step == args.steps:
            dt = time.time() - t0
            print(f"шаг {step:5d}/{args.steps} · лосс {loss_val:.3f} · "
                  f"lr {lr:.5f} · {tokens / dt:,.0f} ток/с", flush=True)

    vl = eval_val()
    print(f"валидация: лосс {vl:.3f} · перплексия {math.exp(min(20, vl)):.1f}")

    project_dir = os.path.join(PROJECTS_DIR, args.project)
    os.makedirs(project_dir, exist_ok=True)
    path = os.path.join(project_dir, "model_final.lfmodel")
    save_checkpoint(
        path,
        model_params={k: v.data for k, v in params.items()},
        config={**model.config_dict(), "tokenizer": tok.to_json()},
        extra={"step": args.steps, "loss": loss_val, "val_loss": vl},
    )
    print(f"сохранено: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
