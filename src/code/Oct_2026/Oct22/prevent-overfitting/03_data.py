# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scikit-learn", "matplotlib"]
# ///
"""
3차시 — 데이터로 막기.

  uv run code/03_data.py

과적합은 "모델에 비해 데이터가 모자란" 상태다. 그러니 데이터 쪽에서 세 가지를 볼 수 있다 —
양(더 모으면 나아지나), 특징(쓸모없는 열이 섞였나), 라벨(틀린 정답이 섞였나).
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from common import ORANGE, SEED, TEAL, save, setup_style, table

FOLDS = 5
CURVE_SIZES = np.linspace(0.05, 1.0, 10)
NOISE_COLUMNS = [0, 10, 30, 100, 300]
LABEL_FLIPS = [0.0, 0.1, 0.2, 0.3, 0.4]
N_DIGITS = 10
VAL_FRACTION = 0.3     # 라벨 노이즈 실험에서 깨끗한 검증으로 떼는 비율
SHALLOW_DEPTH = 8      # 얕은 트리의 최대 깊이
SHALLOW_LEAF = 5       # 얕은 트리의 잎 최소 샘플 수
LOGISTIC_MAX_ITER = 3000
TABLE_STRIDE = 3       # 표에는 학습 곡선 점을 이만큼 건너뛰며 싣는다


def cv() -> StratifiedKFold:
    return StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)


def curves() -> None:
    """학습 곡선 두 개 — 간격이 벌어진 모델(분산)과 둘 다 낮게 붙은 모델(편향)."""
    x, y = load_digits(return_X_y=True)
    models = [("깊은 결정트리 — 분산이 큰 모델", DecisionTreeClassifier(random_state=SEED)),
              ("나이브 베이즈 — 편향이 큰 모델", GaussianNB())]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=True)
    print(f"\n[1] 학습 곡선 — 손글씨 숫자 {len(y)}개, 학습 데이터를 {CURVE_SIZES[0]:.0%}→{CURVE_SIZES[-1]:.0%}로 늘림 (정확도)")
    for ax, (title, model) in zip(axes, models):
        sizes, train, val = learning_curve(model, x, y, cv=cv(), train_sizes=CURVE_SIZES)
        train_mean, val_mean = train.mean(axis=1), val.mean(axis=1)
        ax.plot(sizes, train_mean, "o-", color=TEAL, label="학습")
        ax.plot(sizes, val_mean, "o-", color=ORANGE, label="검증")
        ax.fill_between(sizes, val_mean - val.std(axis=1), val_mean + val.std(axis=1), color=ORANGE, alpha=0.15)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("학습 데이터 수")
        print(f"  {title}")
        table(["학습 수", "학습", "검증", "간격"],
              [[int(n), float(t), float(v), float(t - v)] for n, t, v in zip(sizes, train_mean, val_mean)][::TABLE_STRIDE])
    axes[0].set_ylabel("정확도")
    axes[0].legend()
    save(fig, "03_learning_curves")


def noise_columns() -> None:
    """쓸모없는 무작위 열을 붙일수록 KNN의 '가까움'이 노이즈로 정해진다."""
    x, y = load_breast_cancer(return_X_y=True)
    rng = np.random.default_rng(SEED)
    knn = make_pipeline(StandardScaler(), KNeighborsClassifier())
    pick = make_pipeline(StandardScaler(), SelectKBest(f_classif, k=x.shape[1]), KNeighborsClassifier())
    rows = []
    for extra in NOISE_COLUMNS:
        noisy = np.hstack([x, rng.normal(size=(len(y), extra))])
        rows.append([extra, float(cross_val_score(knn, noisy, y, cv=cv()).mean()),
                     float(cross_val_score(pick, noisy, y, cv=cv()).mean())])
    print(f"\n[2] 노이즈 열 — 유방암 데이터 진짜 특징 {x.shape[1]}개에 무작위 열을 붙임 (KNN 교차검증 정확도)")
    table(["붙인 열", "그대로", f"상위 {x.shape[1]}개만 골라서"], rows)
    fig, ax = plt.subplots()
    extra, plain, picked = zip(*rows)
    ax.plot(extra, plain, "o-", color=ORANGE, label="그대로")
    ax.plot(extra, picked, "o-", color=TEAL, label="특징 선택 후 (교차검증 안에서)")
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xticks(NOISE_COLUMNS, [str(v) for v in NOISE_COLUMNS])
    ax.set_xlabel("붙인 무작위 열 수")
    ax.set_ylabel("교차검증 정확도")
    ax.legend()
    save(fig, "03_noise_columns")


def flip(y: np.ndarray, rate: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """정답의 일부를 다른 숫자로 바꾼다. 라벨링 실수를 흉내 낸다. 바꾼 자리도 함께 돌려준다."""
    noisy = y.copy()
    idx = rng.choice(len(y), int(rate * len(y)), replace=False)
    noisy[idx] = (noisy[idx] + rng.integers(1, N_DIGITS, len(idx))) % N_DIGITS
    return noisy, idx


def label_noise() -> None:
    """학습 라벨만 틀리게 하고, 검증은 깨끗한 정답으로 잰다."""
    x, y = load_digits(return_X_y=True)
    x_tr, x_va, y_tr, y_va = train_test_split(x, y, test_size=VAL_FRACTION, stratify=y, random_state=SEED)
    rng = np.random.default_rng(SEED)
    models = {"깊은 트리": DecisionTreeClassifier(random_state=SEED),
              f"얕은 트리(깊이 {SHALLOW_DEPTH}·잎 {SHALLOW_LEAF})": DecisionTreeClassifier(
                  max_depth=SHALLOW_DEPTH, min_samples_leaf=SHALLOW_LEAF, random_state=SEED),
              "로지스틱": make_pipeline(StandardScaler(), LogisticRegression(max_iter=LOGISTIC_MAX_ITER))}
    rows, memo_rows = [], []
    for rate in LABEL_FLIPS:
        y_noisy, idx = flip(y_tr, rate, rng)
        row, memo = [f"{rate:.0%}"], [f"{rate:.0%}"]
        for model in models.values():
            model.fit(x_tr, y_noisy)
            row += [float(model.score(x_tr, y_noisy)), float(model.score(x_va, y_va))]
            # 틀린 라벨 자리에서 그 틀린 라벨을 그대로 맞혔다면, 틀린 정답을 외운 것이다
            memo.append(float(model.score(x_tr[idx], y_noisy[idx])) if len(idx) else "-")
        rows.append(row)
        memo_rows.append(memo)
    print("\n[3] 라벨 노이즈 — 학습 정답의 일부를 틀리게, 검증은 깨끗한 정답 (정확도)")
    headers = ["틀린 라벨"] + [f"{name} {kind}" for name in models for kind in ("학습", "검증")]
    table(headers, rows)
    print("\n[3-2] 틀린 라벨을 외운 비율 — 틀린 자리에서 모델이 그 틀린 라벨을 내놓은 비율")
    table(["틀린 라벨"] + list(models), memo_rows)


def main() -> None:
    setup_style()
    curves()
    noise_columns()
    label_noise()


if __name__ == "__main__":
    main()
