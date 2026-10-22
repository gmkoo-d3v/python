# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scikit-learn", "matplotlib", "torch"]
# ///
"""
6차시 — 신경망 ① 조기 종료 · 가중치 감쇠 · 드롭아웃.

  uv run code/06_nn.py          (CPU로 수십 초)

손글씨 숫자 300장으로 은닉층 512×2의 MLP를 학습시킨다 — 파라미터 30만 개가 300장을 외우기는 쉽다.
현실처럼 학습 라벨의 20%를 틀리게 둔다(3차시의 라벨 노이즈). 신경망은 틀린 정답까지 외운다.
같은 모델·같은 데이터에 막는 법을 하나씩 붙여 검증 정확도가 어떻게 달라지는지 본다.
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

from common import GOLD, INK_SOFT, ORANGE, SEED, TEAL, save, setup_style, table

N_TRAIN = 300
N_VAL = 500          # 멈출 때를 고르고 곡선을 그리는 데 쓴다
HIDDEN = 512
N_CLASSES = 10
N_PIXELS = 64        # 8×8 손글씨
EPOCHS = 300
LR = 1e-3
BATCH = 64
PATIENCE = 20        # 검증 손실 최저 기록을 이만큼 못 깨면 멈춘다
LABEL_NOISE = 0.2    # 학습 라벨 중 틀리게 둘 비율. 검증·시험 라벨은 깨끗하다
WEIGHT_DECAY = 0.1
DROPOUT = 0.5
# history 한 줄 = (학습 손실, 학습 정확도, 검증 손실, 검증 정확도) — 그림에 쓰는 열 번호
VAL_LOSS_COL = 2
VAL_ACC_COL = 3


@dataclass
class Setting:
    name: str
    dropout: float = 0.0
    weight_decay: float = 0.0
    early_stop: bool = False


@dataclass
class Split:
    """학습 300 · 검증 500(멈춤·곡선) · 시험 997(보고). wrong = 학습 라벨이 틀린 자리."""
    x_tr: torch.Tensor
    y_tr: torch.Tensor
    x_va: torch.Tensor
    y_va: torch.Tensor
    x_te: torch.Tensor
    y_te: torch.Tensor
    wrong: torch.Tensor


SETTINGS = [
    Setting("아무것도 안 함"),
    Setting("조기 종료", early_stop=True),
    Setting("가중치 감쇠", weight_decay=WEIGHT_DECAY),
    Setting("드롭아웃", dropout=DROPOUT),
    Setting("셋 다", dropout=DROPOUT, weight_decay=WEIGHT_DECAY, early_stop=True),
]


def data() -> Split:
    """멈출 시점을 고른 데이터로 성능까지 보고하면 그 숫자는 낙관적이다(2차시 ⑤). 그래서 셋으로 나눈다."""
    x, y = load_digits(return_X_y=True)
    x_tr, x_rest, y_tr, y_rest = train_test_split(x, y, train_size=N_TRAIN, stratify=y, random_state=SEED)
    x_va, x_te, y_va, y_te = train_test_split(x_rest, y_rest, train_size=N_VAL, stratify=y_rest, random_state=SEED)
    rng = np.random.default_rng(SEED)
    wrong = rng.choice(N_TRAIN, int(LABEL_NOISE * N_TRAIN), replace=False)
    y_tr = y_tr.copy()
    y_tr[wrong] = (y_tr[wrong] + rng.integers(1, N_CLASSES, len(wrong))) % N_CLASSES
    scaler = StandardScaler().fit(x_tr)  # 평균·분산은 학습 데이터로만 잰다 (2차시의 누수)

    def to_x(a: np.ndarray) -> torch.Tensor:
        return torch.tensor(scaler.transform(a), dtype=torch.float32)

    mask = torch.zeros(N_TRAIN, dtype=torch.bool)
    mask[torch.tensor(wrong)] = True
    return Split(to_x(x_tr), torch.tensor(y_tr), to_x(x_va), torch.tensor(y_va), to_x(x_te), torch.tensor(y_te), mask)


def mlp(dropout: float) -> nn.Module:
    return nn.Sequential(
        nn.Linear(N_PIXELS, HIDDEN), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(HIDDEN, N_CLASSES),
    )


@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    model.eval()  # 드롭아웃을 끈다 — 평가 때도 켜 두면 점수가 매번 달라진다
    logits = model(x)
    return nn.functional.cross_entropy(logits, y).item(), (logits.argmax(1) == y).float().mean().item()


def memorized(model: nn.Module, d: Split) -> tuple[float, float]:
    """학습 라벨이 맞는 자리와 틀린 자리를 따로 잰다. 틀린 자리의 정확도 = 틀린 정답을 외운 비율."""
    _, right_acc = evaluate(model, d.x_tr[~d.wrong], d.y_tr[~d.wrong])
    _, wrong_acc = evaluate(model, d.x_tr[d.wrong], d.y_tr[d.wrong])
    return right_acc, wrong_acc


def train(setting: Setting, d: Split) -> dict[str, object]:
    torch.manual_seed(SEED)
    model = mlp(setting.dropout)
    # AdamW 의 weight_decay 가 가중치 감쇠다. 매 걸음 가중치를 조금씩 0 쪽으로 당긴다.
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=setting.weight_decay)
    history, best_loss, best_epoch, waited = [], float("inf"), 0, 0
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for idx in torch.randperm(len(d.y_tr)).split(BATCH):
            optimizer.zero_grad()
            nn.functional.cross_entropy(model(d.x_tr[idx]), d.y_tr[idx]).backward()
            optimizer.step()
        val_loss, val_acc = evaluate(model, d.x_va, d.y_va)
        history.append((*evaluate(model, d.x_tr, d.y_tr), val_loss, val_acc))
        if val_loss < best_loss:
            # 얕은 복사(dict(...))로는 안 된다 — 텐서가 계속 갱신돼 결국 마지막 가중치가 남는다
            best_loss, best_epoch, waited = val_loss, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif setting.early_stop:
            waited += 1
            if waited >= PATIENCE:
                break
    if setting.early_stop:
        model.load_state_dict(best_state)  # 멈춘 시점이 아니라 가장 좋았던 시점으로 되돌린다
    return {"history": np.array(history), "train_acc": evaluate(model, d.x_tr, d.y_tr)[1],
            "memorized": memorized(model, d), "test_acc": evaluate(model, d.x_te, d.y_te)[1],
            "best_epoch": best_epoch, "epochs": len(history)}


def plot_curves(results: dict[str, dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    styles = {"아무것도 안 함": ORANGE, "드롭아웃": TEAL, "셋 다": GOLD}
    for name, color in styles.items():
        h = results[name]["history"]
        axes[0].plot(range(1, len(h) + 1), h[:, VAL_LOSS_COL], color=color, label=name)
        axes[1].plot(range(1, len(h) + 1), h[:, VAL_ACC_COL], color=color, label=name)
    base = results["아무것도 안 함"]
    axes[0].axvline(base["best_epoch"], color=INK_SOFT, ls="--", lw=1)
    axes[0].text(base["best_epoch"] + 4, axes[0].get_ylim()[1] * 0.9, f"검증 손실 최저\n{base['best_epoch']} epoch",
                 fontsize=9, color=INK_SOFT, va="top")
    axes[0].set_title("검증 손실", fontsize=11)
    axes[1].set_title("검증 정확도", fontsize=11)
    for ax in axes:
        ax.set_xlabel("epoch")
    axes[1].legend()
    save(fig, "06_curves")


def main() -> None:
    setup_style()
    d = data()
    print(f"\n손글씨 숫자 — 학습 {len(d.y_tr)}장(라벨 {LABEL_NOISE:.0%} 틀림) · 검증 {len(d.y_va)}장(멈춤) · "
          f"시험 {len(d.y_te)}장(보고), MLP {N_PIXELS}→{HIDDEN}→{HIDDEN}→{N_CLASSES} "
          f"(파라미터 {sum(p.numel() for p in mlp(0).parameters()):,}개)")
    results = {s.name: train(s, d) for s in SETTINGS}
    n_wrong = int(d.wrong.sum())
    table(["방법", "학습한 epoch", "학습 전체", f"맞는 라벨 {len(d.y_tr) - n_wrong}장",
           f"틀린 라벨 {n_wrong}장(외운 비율)", "시험 정확도"],
          [[name, r["epochs"], r["train_acc"], *r["memorized"], r["test_acc"]] for name, r in results.items()])
    base_loss = results["아무것도 안 함"]["best_epoch"]
    print(f"  → '아무것도 안 함'의 검증 손실 최저 epoch: {base_loss}")
    plot_curves(results)


if __name__ == "__main__":
    main()
