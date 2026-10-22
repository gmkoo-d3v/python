# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scikit-learn", "matplotlib"]
# ///
"""
2차시 — 검증을 제대로 하기.

  uv run code/02_validation.py

과적합을 막으려면 먼저 과적합을 **잴 수** 있어야 한다. 그런데 검증 숫자 자체가 거짓말을 하는
경우가 다섯 가지 있다. 하나씩 일부러 만들어 본다.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from common import INK_SOFT, ORANGE, SEED, TEAL, save, setup_style, table

SPLIT_REPEATS = 50       # 홀드아웃을 다르게 나눠 보는 횟수
FOLDS = 5
LEAK_SAMPLES = 100       # 누수 실험: 샘플은 적고
LEAK_FEATURES = 5000     #             특징은 아주 많다 (유전자 데이터처럼)
LEAK_KEEP = 20
TS_POINTS = 300
SELECT_CANDIDATES = 200  # 검증셋으로 고르는 후보 "모델" 수
SELECT_SAMPLES = 100     # 찍기 실험의 검증·시험 문제 수
HOLDOUT_FRACTION = 0.2   # 홀드아웃에서 검증으로 떼는 비율
STRATIFY_SAMPLES = 200
MAJORITY_SHARE = 0.95    # 음성 비율 — 양성은 5%뿐
TS_TREND = 0.02          # 시점마다 오르는 기울기
TS_PERIOD = 8            # sin(t / TS_PERIOD) — 오르내림의 주기를 정한다
TS_NOISE = 0.3
TS_NEIGHBORS = 5
MAX_ITER = 2000          # 로지스틱 회귀가 수렴할 만큼 넉넉한 반복 수
CHANCE_ACCURACY = 0.5    # 동전 던지기 정확도


def logistic() -> Pipeline:
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=MAX_ITER))


def holdout_luck() -> None:
    """같은 데이터·같은 모델인데 나누는 운에 따라 점수가 얼마나 흔들리나."""
    x, y = load_breast_cancer(return_X_y=True)
    scores = []
    for seed in range(SPLIT_REPEATS):
        x_tr, x_va, y_tr, y_va = train_test_split(x, y, test_size=HOLDOUT_FRACTION, random_state=seed)
        scores.append(logistic().fit(x_tr, y_tr).score(x_va, y_va))
    cv = cross_val_score(logistic(), x, y, cv=StratifiedKFold(FOLDS, shuffle=True, random_state=SEED))
    print(f"\n[1] 홀드아웃 운 — 유방암 데이터 {len(y)}개, {HOLDOUT_FRACTION:.0%}를 검증으로 {SPLIT_REPEATS}번 다르게 나눔")
    table(["방법", "평균", "최저", "최고", "표준편차"],
          [["홀드아웃 1회", float(np.mean(scores)), float(np.min(scores)), float(np.max(scores)), float(np.std(scores))],
           [f"{FOLDS}-fold 교차검증", float(cv.mean()), float(cv.min()), float(cv.max()), float(cv.std())]])
    fig, ax = plt.subplots()
    ax.hist(scores, bins=12, color=TEAL, edgecolor="white")
    ax.axvline(cv.mean(), color=ORANGE, lw=2.2, label=f"{FOLDS}-fold 평균 {cv.mean():.3f}")
    ax.set_xlabel("검증 정확도 (홀드아웃 한 번)")
    ax.set_ylabel("나눈 횟수")
    ax.legend()
    save(fig, "02_holdout_luck")


def stratify() -> None:
    """양성이 5%뿐인 데이터를 그냥 자르면 어떤 폴드에는 양성이 거의 없다."""
    x, y = make_classification(n_samples=STRATIFY_SAMPLES, weights=[MAJORITY_SHARE], flip_y=0, random_state=SEED)
    rows = []
    for name, splitter in [("KFold", KFold(FOLDS, shuffle=True, random_state=SEED)),
                           ("StratifiedKFold", StratifiedKFold(FOLDS, shuffle=True, random_state=SEED))]:
        counts = [int(y[va].sum()) for _, va in splitter.split(x, y)]
        rows.append([name, " ".join(map(str, counts)), min(counts), max(counts)])
    print(f"\n[2] 계층 분할 — {STRATIFY_SAMPLES}개 중 양성 {int(y.sum())}개, 폴드마다 검증에 든 양성 수")
    table(["분할", "폴드별 양성", "최소", "최대"], rows)


def leakage() -> tuple[float, float]:
    """라벨이 동전 던지기인 순수 노이즈. 맞힐 수 있는 것이 없으니 정답은 50%다."""
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(LEAK_SAMPLES, LEAK_FEATURES))
    y = rng.integers(0, 2, LEAK_SAMPLES)
    cv = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
    # 틀린 순서: 전체 데이터로 특징을 고른 뒤 교차검증 — 검증 폴드의 라벨이 이미 선택에 쓰였다
    x_selected = SelectKBest(f_classif, k=LEAK_KEEP).fit_transform(x, y)
    leaked = cross_val_score(LogisticRegression(max_iter=MAX_ITER), x_selected, y, cv=cv).mean()
    # 옳은 순서: 선택을 파이프라인에 넣어 폴드마다 학습 데이터로만 고른다
    honest = cross_val_score(make_pipeline(SelectKBest(f_classif, k=LEAK_KEEP), LogisticRegression(max_iter=MAX_ITER)),
                             x, y, cv=cv).mean()
    print(f"\n[3] 누수 — 샘플 {LEAK_SAMPLES}개 · 특징 {LEAK_FEATURES}개 · 라벨은 동전 던지기")
    table(["순서", "교차검증 정확도"], [["특징 선택 → 교차검증 (누수)", float(leaked)],
                                   ["교차검증 안에서 특징 선택", float(honest)]])
    return float(leaked), float(honest)


def plot_leakage(leaked: float, honest: float) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    bars = ax.barh(["교차검증 안에서 선택", "먼저 고르고 교차검증"], [honest, leaked], color=[TEAL, ORANGE])
    ax.axvline(CHANCE_ACCURACY, color=INK_SOFT, ls="--", lw=1.2)
    ax.text(0.505, -0.45, f"동전 던지기 {CHANCE_ACCURACY:.0%}", color=INK_SOFT, fontsize=9)
    ax.bar_label(bars, fmt="%.2f", padding=4)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("교차검증 정확도 (라벨은 순수 노이즈)")
    save(fig, "02_leakage")


def time_series() -> None:
    """미래로 갈수록 오르는 값. 무작위로 섞어 자르면 '어제와 내일 사이'를 맞히는 문제가 된다."""
    rng = np.random.default_rng(SEED)
    t = np.arange(TS_POINTS).reshape(-1, 1)
    y = TS_TREND * t.ravel() + np.sin(t.ravel() / TS_PERIOD) + rng.normal(0, TS_NOISE, TS_POINTS)
    model = KNeighborsRegressor(n_neighbors=TS_NEIGHBORS)
    shuffled = -cross_val_score(model, t, y, cv=KFold(FOLDS, shuffle=True, random_state=SEED),
                                scoring="neg_mean_absolute_error").mean()
    forward = -cross_val_score(model, t, y, cv=TimeSeriesSplit(FOLDS), scoring="neg_mean_absolute_error").mean()
    print(f"\n[4] 시계열 — 시점 {TS_POINTS}개, 앞 시점으로 뒤 시점 값을 맞히기 (MAE, 낮을수록 좋음)")
    table(["분할", "MAE"], [["KFold(섞어서)", float(shuffled)], ["TimeSeriesSplit(과거→미래)", float(forward)]])
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.plot(t, y, color=TEAL, lw=1.2)
    ax.set_xlabel("시점")
    ax.set_ylabel("값")
    save(fig, "02_timeseries")


def selection_overfit() -> None:
    """후보를 많이 시험해 검증 점수가 가장 높은 것을 고르면, 그 점수는 운까지 포함한 값이다."""
    rng = np.random.default_rng(SEED)
    y_val = rng.integers(0, 2, SELECT_SAMPLES)
    y_test = rng.integers(0, 2, SELECT_SAMPLES)
    # 후보 "모델"은 전부 아무것도 모르는 찍기다. 실력 차이는 0이다.
    val_scores = [(rng.integers(0, 2, SELECT_SAMPLES) == y_val).mean() for _ in range(SELECT_CANDIDATES)]
    best = int(np.argmax(val_scores))
    test_score = (rng.integers(0, 2, SELECT_SAMPLES) == y_test).mean()
    print(f"\n[5] 검증셋 고르기 과적합 — 찍기 모델 {SELECT_CANDIDATES}개 중 검증 1등을 고름")
    table(["무엇", "정확도"], [["1등의 검증 점수", float(val_scores[best])], ["같은 실력의 새 시험 점수", float(test_score)],
                            ["후보 전체 검증 평균", float(np.mean(val_scores))]])


def main() -> None:
    setup_style()
    holdout_luck()
    stratify()
    plot_leakage(*leakage())
    time_series()
    selection_overfit()


if __name__ == "__main__":
    main()
