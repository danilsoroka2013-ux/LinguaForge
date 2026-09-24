# -*- coding: utf-8 -*-
"""Быстрые проверки ядра: градиенты, токенизатор, smoke-обучение."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.tensor import Tensor, cross_entropy                      # noqa: E402
from core.model import TinyGPT                                     # noqa: E402
from core.tokenizers import CharTokenizer, BPETokenizer            # noqa: E402


def gradcheck() -> None:
    """Численная проверка градиентов узла модели на float64."""
    rng = np.random.default_rng(0)

    class Sub(TinyGPT):
        def __init__(self):  # tiny: 1 блок, крошечные размеры
            from core.model import TransformerBlock, RMSNorm
            self.vocab_size = 7
            self.dim = 8
            self.n_layers = 1
            self.n_heads = 2
            self.block_size = 6
            self.ffn_mult = 2.667
            self.embed = __import__("core.model", fromlist=["Embedding"]).Embedding(7, 8)
            self.blocks = [TransformerBlock(8, 2, 2.667)]
            self.ln_f = RMSNorm(8)
            self.registered_mask = np.triu(
                np.full((6, 6), -1e9, dtype=np.float32), k=1)
            self._rope = {}

    m = Sub()
    # float64 для численной проверки
    for p in m.params().values():
        p.data = p.data.astype(np.float64)

    B, T = 2, 5
    x = rng.integers(0, 7, size=(B, T))
    y = rng.integers(0, 7, size=(B, T))
    logits, loss = m.forward(x, y)
    loss.backward()

    def numerical_loss():
        _, l = m.forward(x, y)
        return float(l.data)

    worst = 0.0
    checked = 0
    for name, p in list(m.params().items()):
        g = p.grad
        if g is None:
            continue
        flat = p.data.ravel()
        gflat = g.ravel()
        # проверяем до 8 случайных элементов на параметр — этого достаточно
        idxs = rng.choice(flat.size, size=min(8, flat.size), replace=False)
        for i in idxs:
            eps = 1e-5
            old = flat[i]
            flat[i] = old + eps
            lp = numerical_loss()
            flat[i] = 2 * old - flat[i]  # = old - eps
            lm = numerical_loss()
            flat[i] = old
            num = (lp - lm) / (2 * eps)
            ana = float(gflat[i])
            denom = max(1e-8, abs(num) + abs(ana))
            err = abs(num - ana) / denom
            worst = max(worst, err)
            checked += 1
    print(f"[gradcheck] проверено {checked}, макс. ошибка {worst:.2e}")
    assert worst < 1e-3, f"градиенты разошлись: {worst}"


def tokenizer_roundtrip() -> None:
    text = "Кот сидит на окне. Лиса живёт в лесу!"
    ct = CharTokenizer()
    ct.train(text, 60)
    ids = ct.encode(text)
    assert ct.decode(ids).replace(" ", "") == text.replace(" ", "")
    bt = BPETokenizer()
    bt.train(text * 3, 40)
    ids2 = bt.encode(text)
    assert bt.decode(ids2).replace(" ", "") == text.replace(" ", "")
    rt = CharTokenizer.from_json(ct.to_json())
    assert rt.encode(text) == ids
    print(f"[tokenizer] ok (char {ct.vocab_size}, bpe {bt.vocab_size})")


def smoke_train() -> None:
    """Крошечная модель на крошечном корпусе должна снизить лосс заметно."""
    from core.data import make_batches
    from core.config import set_seed
    from core.optimizer import AdamW
    from core.config import clip_grad_norm
    from core.trainer import Trainer  # noqa: F401 (импорт не должен падать)

    set_seed(7)
    text = ("кот сидит на окне и смотрит на улицу. пёс бегает по двору. "
            "лиса живёт в лесу и знает все тихие тропинки. ") * 20
    tok = CharTokenizer()
    tok.train(text, 60)
    ids = np.array(tok.encode(text), dtype=np.int32)
    model = TinyGPT(vocab_size=tok.vocab_size, dim=32, n_layers=2, n_heads=2,
                    block_size=24)
    batch = make_batches(ids, 8, 24, 1, seed=0)
    opt = AdamW(model.params(), lr=3e-3, weight_decay=0.0)
    first = None
    for i in range(1, 61):
        x, y = batch()
        _, loss = model.forward(x, y)
        opt.zero_grad()
        loss.backward()
        clip_grad_norm({k: v.grad for k, v in model.params().items()}, 1.0)
        opt.step(3e-3 * min(1.0, i / 10.0))
        lv = float(loss.data)
        if first is None:
            first = lv
        if i in (1, 30, 60):
            print(f"[smoke] step {i:3d} loss {lv:.3f}")
    assert lv < first * 0.7, f"лосс не снизился: {first:.3f} -> {lv:.3f}"
    # генерация не падает
    import random
    ctx = np.array([tok.encode("кот")][:1], dtype=np.int32).reshape(1, -1)
    out = model.generate(ctx, max_new_tokens=12, temperature=0.9, rng=random.Random(3))
    assert len(out[0]) >= 13
    print("[smoke] ok")


if __name__ == "__main__":
    tokenizer_roundtrip()
    smoke_train()
    gradcheck()
    print("ALL CORE TESTS PASSED")
