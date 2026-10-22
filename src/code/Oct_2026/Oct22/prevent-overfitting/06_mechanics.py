# /// script
# requires-python = ">=3.10"
# dependencies = ["torch"]
# ///
"""
6·7차시 보충 — 드롭아웃과 배치정규화가 안에서 실제로 하는 일.

  uv run code/06_mechanics.py          (CPU 몇 초)

모델 학습이 아니라, 층 하나에 숫자를 넣고 나오는 숫자를 직접 본다.
"""

import torch
from torch import nn

SEED = 0
P_DROP = 0.5
DROP_PASSES = 3
MEAN_PASSES = 10_000
N_NEURONS = 8
FEATURES = 4
BN_BATCH = 64
# 특징마다 크기가 전혀 다른 입력의 평균과 표준편차
BN_CENTERS = [5.0, -3.0, 100.0, 0.01]
BN_SCALES = [2.0, 1.0, 30.0, 0.001]
RUN_MEAN = 10.0      # 이동평균 실험: 배치를 뽑는 진짜 평균
RUN_STD = 2.0        #               진짜 표준편차
RUN_BATCH = 32
PROBE_SIGMAS = 2     # eval 때 넣어 볼 입력 — 평균에서 표준편차 몇 개 위인가
ERROR_PREVIEW = 60   # 오류 메시지는 앞부분만 찍는다
BATCH_SIZES = [2, 8, 64]
BATCH_REPEATS = 1_000
BN_STEPS = 200


def show(name: str, t: torch.Tensor) -> None:
    print(f"  {name:<22}" + "  ".join(f"{v:6.2f}" for v in t.tolist()))


def dropout_demo() -> None:
    """같은 입력을 train 모드로 여러 번 — 매번 다른 칸이 0이 되고, 남은 칸은 1/(1-p) 배."""
    torch.manual_seed(SEED)
    drop = nn.Dropout(P_DROP)
    x = torch.ones(N_NEURONS)
    print(f"\n[1] 드롭아웃 p={P_DROP} — 입력은 전부 1.0 인 뉴런 {N_NEURONS}개")
    show("입력", x)
    drop.train()
    for i in range(DROP_PASSES):
        show(f"train 모드 {i + 1}번째", drop(x))
    drop.eval()
    show("eval 모드", drop(x))
    # 남은 칸을 2배로 키우는 이유: 평균을 학습·평가 때 같게 맞추려고
    drop.train()
    mean = torch.stack([drop(x) for _ in range(MEAN_PASSES)]).mean().item()
    print(f"  → train 모드 {MEAN_PASSES:,}번 평균: {mean:.3f} (eval 모드 출력 1.000 과 같아야 한다)")


def batchnorm_demo() -> None:
    """한 배치 안에서 특징마다 평균 0·분산 1로 맞춘다. γ·β는 그 뒤에 다시 늘리고 옮긴다."""
    torch.manual_seed(SEED)
    bn = nn.BatchNorm1d(FEATURES)
    centers = torch.tensor(BN_CENTERS)
    scales = torch.tensor(BN_SCALES)
    x = centers + scales * torch.randn(BN_BATCH, FEATURES)
    print(f"\n[2] 배치정규화 — 배치 {BN_BATCH}개 · 특징 {FEATURES}개 (특징마다 크기가 제각각)")
    show("입력 평균", x.mean(0))
    show("입력 표준편차", x.std(0))
    bn.train()
    y = bn(x)
    show("출력 평균", y.mean(0))
    show("출력 표준편차", y.std(0))
    print(f"  학습할 파라미터 γ(늘리기) 초기값 {bn.weight.tolist()} · β(옮기기) {bn.bias.tolist()}")


def running_stats_demo() -> None:
    """학습 중에는 이동평균을 쌓고, eval 때는 배치 대신 그 이동평균으로 정규화한다."""
    torch.manual_seed(SEED)
    bn = nn.BatchNorm1d(1)
    bn.train()
    for _ in range(BN_STEPS):
        bn(RUN_MEAN + RUN_STD * torch.randn(RUN_BATCH, 1))
    print(f"\n[3] 이동평균 — 평균 {RUN_MEAN:g}·표준편차 {RUN_STD:g} 인 배치를 {BN_STEPS}번 통과시킨 뒤")
    print(f"  running_mean = {bn.running_mean.item():.3f}   running_var = {bn.running_var.item():.3f} "
          f"(진짜 분산 {RUN_STD ** 2:g})")
    probe = RUN_MEAN + PROBE_SIGMAS * RUN_STD
    one = torch.tensor([[probe]])
    bn.eval()
    print(f"  eval 모드에서 입력 {probe:.1f} → {bn(one).item():.3f} "
          f"(≈ ({probe:g}-{RUN_MEAN:g})/{RUN_STD:g} = {PROBE_SIGMAS:.1f})")
    bn.train()
    try:
        bn(one)
    except ValueError as error:
        print(f"  train 모드에서 1개짜리 배치 → 오류: {str(error)[:ERROR_PREVIEW]}…")


def batch_noise_demo() -> list[tuple[int, float]]:
    """배치가 작을수록 배치 평균이 흔들린다 — 같은 샘플도 어느 배치에 끼느냐에 따라 다르게 정규화된다."""
    torch.manual_seed(SEED)
    rows = []
    print(f"\n[4] 배치 크기와 흔들림 — 평균 0·표준편차 1 인 데이터에서 배치 평균을 {BATCH_REPEATS:,}번 잼")
    for size in BATCH_SIZES:
        means = torch.randn(BATCH_REPEATS, size).mean(1)
        rows.append((size, means.std().item()))
        print(f"  배치 {size:>3}  배치 평균의 표준편차 {means.std().item():.3f}   (이론값 1/√{size} = {size ** -0.5:.3f})")
    return rows


def main() -> None:
    dropout_demo()
    batchnorm_demo()
    running_stats_demo()
    batch_noise_demo()


if __name__ == "__main__":
    main()
