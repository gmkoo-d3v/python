# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scikit-learn", "matplotlib"]
# ///
"""
1차시 — 과적합을 눈으로 보기.

  uv run code/01_see_overfit.py

sin 곡선에 노이즈를 얹은 점 12개에 다항식을 맞춘다. 차수를 올릴수록 학습 오차는 계속 줄어
11차에서는 점을 전부 지나지만(학습 오차 0), 검증 오차는 어느 지점부터 다시 오른다 — 그 갈림길이 과적합이다.
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial
from sklearn.metrics import mean_squared_error

from common import INK_SOFT, ORANGE, SEED, TEAL, log_axis, save, setup_style, table

N_TRAIN = 12
N_VAL = 200
NOISE = 0.25
SHOWN_DEGREES = [1, 3, 11]
MAX_DEGREE = N_TRAIN - 1  # 점 12개를 전부 지나는 가장 낮은 차수
SWEEP_FLOOR = 1e-3        # 오차 곡선 그림의 바닥. 이보다 작은 값은 "≈ 0"으로 적는다
SWEEP_CEILING = 3e3
BV_REPEATS = 300          # 편향·분산을 재려고 데이터를 새로 뽑는 횟수
BV_DEGREES = [1, 3, 6, 9]
BV_POINTS = 50
BV_RANGE = (0.05, 0.95)   # 편향·분산을 재는 x 구간 — 다항식이 크게 튀는 양 끝은 뺀다
PLOT_POINTS = 400         # 곡선을 매끄럽게 그리려는 격자 점 수


def truth(x: np.ndarray) -> np.ndarray:
    """우리가 맞히고 싶은 진짜 규칙. 모델은 이것을 모른다."""
    return np.sin(2 * np.pi * x)


def sample(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    x = rng.uniform(0, 1, n)
    return x, truth(x) + rng.normal(0, NOISE, n)


def fit(degree: int, x: np.ndarray, y: np.ndarray) -> Polynomial:
    """최소제곱 다항식. Polynomial.fit은 x를 [-1, 1]로 옮겨 풀어서 고차에서도 수치가 덜 흔들린다."""
    return Polynomial.fit(x, y, degree)


def sweep(x_tr: np.ndarray, y_tr: np.ndarray, x_va: np.ndarray, y_va: np.ndarray) -> list[tuple[int, float, float]]:
    """차수마다 학습·검증 오차(MSE)."""
    rows = []
    for degree in range(1, MAX_DEGREE + 1):
        model = fit(degree, x_tr, y_tr)
        rows.append((degree, mean_squared_error(y_tr, model(x_tr)), mean_squared_error(y_va, model(x_va))))
    return rows


def plot_fits(x_tr: np.ndarray, y_tr: np.ndarray) -> None:
    grid = np.linspace(0, 1, PLOT_POINTS)
    fig, axes = plt.subplots(1, len(SHOWN_DEGREES), figsize=(10.5, 3.4), sharey=True)
    for ax, degree in zip(axes, SHOWN_DEGREES):
        model = fit(degree, x_tr, y_tr)
        ax.plot(grid, truth(grid), color=INK_SOFT, ls="--", lw=1.2, label="진짜 규칙")
        ax.plot(grid, model(grid), color=TEAL, lw=2.2, label=f"{degree}차 다항식")
        ax.scatter(x_tr, y_tr, color=ORANGE, s=22, zorder=3, label=f"학습 점 {N_TRAIN}개")
        ax.set_ylim(-1.8, 1.8)
        ax.set_title(f"{degree}차", fontsize=12)
    axes[0].legend(fontsize=8, loc="lower left")
    save(fig, "01_fits")


def plot_sweep(rows: list[tuple[int, float, float]]) -> None:
    degrees, train, val = zip(*rows)
    # 11차의 학습 오차는 1e-24 처럼 사실상 0이다. 그대로 그리면 축이 끌려 내려가 볼 구간이 눌린다.
    shown = [max(t, SWEEP_FLOOR) for t in train]
    fig, ax = plt.subplots()
    ax.plot(degrees, shown, "o-", color=TEAL, label="학습 오차")
    ax.plot(degrees, val, "o-", color=ORANGE, label="검증 오차")
    for degree, value in zip(degrees, train):
        if value < SWEEP_FLOOR:
            ax.annotate("≈ 0", (degree, SWEEP_FLOOR), textcoords="offset points", xytext=(0, 8),
                        ha="center", color=TEAL, fontsize=10)
    log_axis(ax)
    ax.set_ylim(SWEEP_FLOOR / 2, SWEEP_CEILING)
    ax.set_xticks(degrees)
    ax.set_xlabel("다항식 차수 (모델 복잡도)")
    ax.set_ylabel("MSE (로그 눈금)")
    ax.legend(loc="upper left")
    save(fig, "01_sweep")


def bias_variance(rng: np.random.Generator) -> list[tuple[int, float, float, float]]:
    """같은 규칙에서 노이즈만 BV_REPEATS번 새로 뽑아, 고정된 점에서 예측이 얼마나 흔들리나 잰다.
    x 위치까지 매번 새로 뽑으면 점이 한쪽에 몰린 판 하나가 평균을 통째로 끌고 가 숫자를 읽을 수 없다."""
    x = np.linspace(0, 1, N_TRAIN)
    points = np.linspace(*BV_RANGE, BV_POINTS)
    rows = []
    for degree in BV_DEGREES:
        preds = np.array([fit(degree, x, truth(x) + rng.normal(0, NOISE, N_TRAIN))(points) for _ in range(BV_REPEATS)])
        bias2 = float(np.mean((preds.mean(axis=0) - truth(points)) ** 2))
        variance = float(np.mean(preds.var(axis=0)))
        rows.append((degree, bias2, variance, bias2 + variance + NOISE ** 2))
    return rows


def main() -> None:
    setup_style()
    rng = np.random.default_rng(SEED)
    x_tr, y_tr = sample(N_TRAIN, rng)
    x_va, y_va = sample(N_VAL, rng)

    print("\n[1] 차수별 학습·검증 오차 (MSE)")
    rows = sweep(x_tr, y_tr, x_va, y_va)
    table(["차수", "학습", "검증"], [[d, tr, va] for d, tr, va in rows])
    best = min(rows, key=lambda r: r[2])
    print(f"  → 검증 오차가 가장 낮은 차수: {best[0]} (검증 {best[2]:.4f})")
    print(f"  → 노이즈 자체의 MSE(넘을 수 없는 바닥): {NOISE ** 2:.4f}")

    plot_fits(x_tr, y_tr)
    plot_sweep(rows)

    print(f"\n[2] 편향² · 분산 — 같은 x 에 노이즈만 {BV_REPEATS}번 새로 뽑아 잰 값")
    table(["차수", "편향²", "분산", "예상 오차"], [list(r) for r in bias_variance(rng)])


if __name__ == "__main__":
    main()
