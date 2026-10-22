# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scikit-learn", "matplotlib", "torch"]
# ///
"""
6차시 보충 — 그래디언트 클리핑과 학습률 스케줄.

  uv run code/06_extras.py          (CPU로 1분 안쪽)

06_nn.py 와 같은 데이터(손글씨 300장, 라벨 20% 틀림)와 같은 MLP를 쓴다.
  [1] 클리핑: 과적합 방지가 아니라 "발산 방지"다. 학습률을 일부러 키워 터뜨려 본다.
  [2] 스케줄: 학습률을 끝으로 갈수록 줄이면 무엇이 달라지나.
"""

import importlib
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from common import INK_SOFT, ORANGE, SEED, TEAL, log_axis, save, setup_style, table

base = importlib.import_module("06_nn")

CLIP_LR = 0.5          # 일부러 큰 학습률
MOMENTUM = 0.9
CLIP_EPOCHS = 60
MAX_NORM = 1.0         # 기울기 전체 길이를 이 값 이하로 자른다
SCHED_EPOCHS = 300
RIGHT_LABEL_SHARE = 1 - base.LABEL_NOISE  # 맞는 라벨 비율 — 그래프의 기준선


def run_sgd(d: base.Split, clip: float | None) -> dict[str, object]:
    """같은 시드·같은 학습률. clip 이 있으면 매 걸음 기울기 길이를 자른다."""
    torch.manual_seed(SEED)
    model = base.mlp(0.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=CLIP_LR, momentum=MOMENTUM)
    losses, norms, nan_epoch = [], [], None
    for epoch in range(1, CLIP_EPOCHS + 1):
        model.train()
        for idx in torch.randperm(len(d.y_tr)).split(base.BATCH):
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(d.x_tr[idx]), d.y_tr[idx])
            loss.backward()
            # clip_grad_norm_ 은 자르기 전의 길이를 돌려준다 — 얼마나 컸는지 기록해 둔다
            limit = clip if clip is not None else math.inf
            norms.append(float(nn.utils.clip_grad_norm_(model.parameters(), limit)))
            optimizer.step()
        train_loss, _ = base.evaluate(model, d.x_tr, d.y_tr)
        losses.append(train_loss)
        if not math.isfinite(train_loss):
            nan_epoch = epoch
            break
    _, test_acc = base.evaluate(model, d.x_te, d.y_te)
    finite = [n for n in norms if math.isfinite(n)]
    return {"losses": losses, "nan_epoch": nan_epoch, "test_acc": test_acc, "memorized": base.memorized(model, d),
            "max_norm": max(finite) if finite else math.nan, "median_norm": float(np.median(finite))}


def clipping(d: base.Split) -> None:
    runs = {"클리핑 없음": run_sgd(d, None), f"클리핑 {MAX_NORM}": run_sgd(d, MAX_NORM)}
    print(f"\n[1] 그래디언트 클리핑 — SGD lr={CLIP_LR}, 모멘텀 {MOMENTUM}, {CLIP_EPOCHS} epoch")
    table(["설정", "발산(NaN) epoch", "마지막 학습 손실", "틀린 라벨 외운 비율", "시험 정확도", "기울기 길이 최대"],
          [[name, r["nan_epoch"] or "-", r["losses"][-1], r["memorized"][1], r["test_acc"], r["max_norm"]]
           for name, r in runs.items()])
    fig, ax = plt.subplots()
    for (name, r), color in zip(runs.items(), [ORANGE, TEAL]):
        shown = [loss if math.isfinite(loss) else np.nan for loss in r["losses"]]
        ax.plot(range(1, len(shown) + 1), shown, "o-", ms=3, color=color, label=name)
    log_axis(ax)
    ax.set_xlabel("epoch")
    ax.set_ylabel("학습 손실")
    ax.legend()
    save(fig, "06_clipping")


def train_sched(d: base.Split, cosine: bool) -> tuple[list[tuple[float, float]], float]:
    torch.manual_seed(SEED)
    model = base.mlp(0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base.LR, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SCHED_EPOCHS) if cosine else None
    history = []
    for _ in range(SCHED_EPOCHS):
        model.train()
        for idx in torch.randperm(len(d.y_tr)).split(base.BATCH):
            optimizer.zero_grad()
            nn.functional.cross_entropy(model(d.x_tr[idx]), d.y_tr[idx]).backward()
            optimizer.step()
        if scheduler:
            scheduler.step()  # epoch마다 학습률을 코사인 곡선을 따라 줄인다
        history.append((base.evaluate(model, d.x_va, d.y_va)[1], base.evaluate(model, d.x_tr, d.y_tr)[1]))
    return history, base.evaluate(model, d.x_te, d.y_te)[1]


def schedule(d: base.Split) -> None:
    runs = {f"고정 lr {base.LR:g}": train_sched(d, False), f"코사인 {base.LR:g} → 0": train_sched(d, True)}
    print(f"\n[2] 학습률 스케줄 — AdamW, 규제 없음, {SCHED_EPOCHS} epoch")
    rows = []
    for name, (h, test_acc) in runs.items():
        val = [v for v, _ in h]
        best = int(np.argmax(val))
        rows.append([name, h[-1][1], val[best], best + 1, test_acc])
    # 최고 검증은 고른 값이라 낙관적이다(2차시 ⑤). 끝까지 돌린 모델의 성능은 시험으로 잰다.
    table(["설정", "마지막 학습 정확도", "검증 최고(엿봄)", "그때 epoch", "끝까지 돌린 시험 정확도"], rows)
    fig, ax = plt.subplots()
    for (name, (h, _)), color in zip(runs.items(), [ORANGE, TEAL]):
        ax.plot(range(1, len(h) + 1), [v for v, _ in h], color=color, label=name)
    ax.set_xlabel("epoch")
    ax.set_ylabel("검증 정확도")
    ax.axhline(RIGHT_LABEL_SHARE, color=INK_SOFT, ls="--", lw=1)
    ax.legend()
    save(fig, "06_schedule")


def main() -> None:
    setup_style()
    d = base.data()
    clipping(d)
    schedule(d)


if __name__ == "__main__":
    main()
