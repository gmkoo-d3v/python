# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scikit-learn", "matplotlib", "threadpoolctl"]
# ///
"""
8차시 — 종합: 넘겨받은 모델을 진단하고 고치기.

  uv run code/08_diagnose.py

상황: 설비 고장 예측 모델을 넘겨받았다. 전임자 보고서에는 "테스트 정확도 96.7%"라고 적혀 있다(predecessor()가 그 방식을 재현한다).
그 숫자를 믿기 전에 순서도대로 하나씩 확인한다.
  ① 숫자가 정직한가 (누수·테스트 재사용)  ② 과적합인가 과소적합인가  ③ 무엇으로 막나  ④ 마지막 시험은 한 번
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from common import SEED, setup_style, table

N_SAMPLES = 1200
N_FEATURES = 20
N_INFORMATIVE = 6          # 정답을 실제로 정하는 특징 수
N_REDUNDANT = 4            # 정보 특징을 섞어 만든 중복 특징 수
LABEL_FLIP = 0.08          # 라벨을 무작위로 바꾼 비율 — 어떤 모델도 넘을 수 없는 노이즈
CLASS_SEP = 0.8            # 두 클래스가 떨어진 정도 (작을수록 어렵다)
LEAK_NOISE = 0.03          # 누수 열이 정답과 다른 비율
TEST_FRACTION = 0.2        # 시험(전임자는 검증 겸용)으로 떼는 비율
FOLDS = 5
MAX_ITER = 150
CURVE_SIZES = np.linspace(0.2, 1.0, 5)
EARLY_VAL_FRACTION = 0.15  # 조기 종료가 멈출 때를 보려고 떼는 비율
EARLY_PATIENCE = 10        # 검증 점수가 이만큼 연속 안 오르면 멈춘다
LOGISTIC_MAX_ITER = 2000
GRID = {"max_depth": [3, None], "min_samples_leaf": [5, 20, 50], "l2_regularization": [0.0, 1.0]}


def make_data() -> tuple[np.ndarray, np.ndarray]:
    """특징 20개 + 마지막 열은 '수리 기록 있음' — 고장이 난 뒤에야 생기는 값이라 예측 시점에는 없다."""
    x, y = make_classification(n_samples=N_SAMPLES, n_features=N_FEATURES, n_informative=N_INFORMATIVE,
                               n_redundant=N_REDUNDANT, flip_y=LABEL_FLIP, class_sep=CLASS_SEP, random_state=SEED)
    rng = np.random.default_rng(SEED)
    leak = np.where(rng.random(N_SAMPLES) < LEAK_NOISE, 1 - y, y)
    return np.column_stack([x, leak]), y


def boosting(**params: float | None) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=MAX_ITER, early_stopping=False, random_state=SEED, **params)


def predecessor(x: np.ndarray, y: np.ndarray) -> float:
    """전임자의 방식 — 전체로 스케일링, 테스트로 하이퍼파라미터를 고르고 같은 테스트로 보고."""
    x_scaled = StandardScaler().fit_transform(x)
    x_tr, x_te, y_tr, y_te = train_test_split(x_scaled, y, test_size=TEST_FRACTION, random_state=SEED)
    scores = [boosting(max_depth=d).fit(x_tr, y_tr).score(x_te, y_te) for d in GRID["max_depth"]]
    return max(scores)


def cv() -> StratifiedKFold:
    """모든 비교가 같은 폴드를 쓴다. 그래야 폴드끼리 짝지어 차이를 볼 수 있다."""
    return StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)


def step1_honest(x_dev: np.ndarray, y_dev: np.ndarray) -> None:
    with_leak = cross_val_score(make_pipeline(StandardScaler(), boosting()), x_dev, y_dev, cv=cv()).mean()
    without = cross_val_score(make_pipeline(StandardScaler(), boosting()), x_dev[:, :-1], y_dev, cv=cv()).mean()
    # 한 열만으로 거의 다 맞힌다면, 그 열이 정답을 들고 있지 않은지 의심한다
    alone = cross_val_score(LogisticRegression(), x_dev[:, -1:], y_dev, cv=cv()).mean()
    print("\n[①] 숫자가 정직한가 — 개발용 데이터로만 교차검증")
    print(f"  '수리 기록 있음' 한 열만으로: {alone:.4f}")
    table(["확인", "교차검증 정확도"], [["누수 열 포함", float(with_leak)], ["누수 열 제거 (예측 시점에 있는 것만)", float(without)]])


def step2_diagnose(x_dev: np.ndarray, y_dev: np.ndarray) -> None:
    sizes, train, val = learning_curve(boosting(), x_dev, y_dev, train_sizes=CURVE_SIZES, cv=cv())
    print("\n[②] 과적합인가 — 학습 곡선 (누수 열 제거 후, 기본 설정)")
    table(["학습 수", "학습", "검증", "간격"],
          [[int(n), float(t), float(v), float(t - v)] for n, t, v in zip(sizes, train.mean(1), val.mean(1))])
    gap = float(train.mean(1)[-1] - val.mean(1)[-1])
    last_gain = float(val.mean(1)[-1] - val.mean(1)[-2])
    spread = float(val.std(1)[-1])
    print(f"  → 마지막 간격 {gap:.3f} · 마지막 구간 검증 증가 {last_gain:+.3f} (폴드 표준편차 {spread:.3f})")
    print("  → 간격이 크면 분산형. 마지막 증가가 흩어짐보다 작으면 '더 모으면 나아진다'고 단정할 수 없다")


def step3_fix(x_dev: np.ndarray, y_dev: np.ndarray) -> GridSearchCV:
    search = GridSearchCV(boosting(), GRID, cv=cv()).fit(x_dev, y_dev)
    early_model = HistGradientBoostingClassifier(max_iter=MAX_ITER, early_stopping=True, validation_fraction=EARLY_VAL_FRACTION,
                                                 n_iter_no_change=EARLY_PATIENCE, random_state=SEED)
    early_scores = cross_val_score(early_model, x_dev, y_dev, cv=cv())
    base = cross_val_score(boosting(), x_dev, y_dev, cv=cv())
    # 단순한 모델이 얼마나 하나도 같이 본다. 크게 낮다면 이 문제엔 복잡한 모델이 필요하다는 뜻이다.
    simple = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=LOGISTIC_MAX_ITER)), x_dev, y_dev, cv=cv())
    chosen = cross_val_score(boosting(**search.best_params_), x_dev, y_dev, cv=cv())
    chosen_std = float(search.cv_results_["std_test_score"][search.best_index_])
    print(f"\n[③] 무엇으로 막나 — 교차검증으로 비교 (후보 {len(search.cv_results_['params'])}개)")
    # 평균만 보면 0.6%p 차이도 개선처럼 보인다. 폴드 사이 흩어짐을 같이 찍어 판단한다(2차시).
    table(["방법", "교차검증 평균", "± 표준편차"],
          [["로지스틱 회귀 (단순 모델)", float(simple.mean()), float(simple.std())],
           ["기본 설정", float(base.mean()), float(base.std())],
           ["조기 종료만", float(early_scores.mean()), float(early_scores.std())],
           [f"규제 탐색 최선 (후보 {len(search.cv_results_['params'])}개 중 1등 — 낙관적)", float(search.best_score_), chosen_std]])
    print(f"  → 고른 규제: {search.best_params_} (min_samples_leaf 기본값은 20 — 5는 기본보다 덜 묶은 것)")
    # 같은 폴드끼리 짝지어 빼면 폴드 난이도 차이가 지워진다. 그래도 0을 넘나들면 개선이라 말할 수 없다.
    diff = chosen - base
    print(f"  → 폴드별 차이(고른 규제 − 기본): {' '.join(f'{v:+.3f}' for v in diff)}  평균 {diff.mean():+.3f} ± {diff.std():.3f}")
    return search


def main() -> None:
    setup_style()
    # 데이터가 작은데 부스팅이 코어 수만큼 스레드를 띄우면 서로 기다리느라 오히려 몇 배 느려진다(실측 6분 → 수십 초).
    threadpool_limits(1)
    x, y = make_data()
    print(f"설비 {N_SAMPLES}대 · 특징 {N_FEATURES}개 + '수리 기록 있음' 1개 · 고장 비율 {y.mean():.0%}")
    print(f"\n[보고서] 전임자의 '테스트 정확도': {predecessor(x, y):.4f}")
    # 시험지는 맨 처음 떼어 서랍에 넣는다. 마지막에 딱 한 번만 연다.
    x_dev, x_test, y_dev, y_test = train_test_split(x, y, test_size=TEST_FRACTION, stratify=y, random_state=SEED + 1)
    step1_honest(x_dev, y_dev)
    x_dev, x_test = x_dev[:, :-1], x_test[:, :-1]
    step2_diagnose(x_dev, y_dev)
    search = step3_fix(x_dev, y_dev)
    print("\n[④] 마지막 시험 — 서랍의 시험지를 처음이자 마지막으로 연다")
    table(["모델", "시험 정확도"], [["기본 설정", float(boosting().fit(x_dev, y_dev).score(x_test, y_test))],
                                 ["고른 규제", float(search.best_estimator_.score(x_test, y_test))]])


if __name__ == "__main__":
    main()
