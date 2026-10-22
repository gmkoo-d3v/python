# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scikit-learn", "matplotlib"]
# ///
"""
4차시 — 규제.

  uv run code/04_regularization.py

1차시의 11차 다항식은 점 12개를 전부 지나려고 계수를 수천 단위로 키웠다. 규제는 "계수가 크면
벌점"을 손실에 더해, 데이터를 덜 맞히더라도 계수를 작게 두게 만든다.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split, validation_curve
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from common import (
    INK_SOFT,
    ORANGE,
    SEED,
    TEAL,
    log_axis,
    log_x,
    save,
    setup_style,
    table,
)

N_TRAIN = 12
N_VAL = 200
NOISE = 0.25
DEGREE = 11
RIDGE_ALPHAS = [0.0, 1e-6, 1e-4, 1e-2, 1.0, 100.0]
SHOWN_ALPHAS = [0.0, 1e-4, 1.0]
LASSO_FEATURES = 50
LASSO_INFORMATIVE = 5
LASSO_SAMPLES = 100
LASSO_NOISE = 10
LASSO_VAL_FRACTION = 0.4   # Lasso 실험에서 검증으로 떼는 비율
LASSO_MAX_ITER = 50_000    # 작은 α에서도 좌표하강이 수렴하도록 넉넉히
LASSO_ALPHAS = [0.001, 0.01, 0.1, 1.0, 3.0, 10.0]
CURVE_ALPHAS = np.logspace(-6, 3, 19)
CURVE_POINTS = 15  # 40개쯤 되면 11차도 과적합이 약해져 규제가 필요 없다는 답이 나온다
FOLDS = 5
PLOT_POINTS = 400  # 곡선을 매끄럽게 그리려는 격자 점 수
TABLE_STRIDE = 3   # 표에는 α 후보를 이만큼 건너뛰며 싣는다


def truth(x: np.ndarray) -> np.ndarray:
    return np.sin(2 * np.pi * x)


class ExactFit(BaseEstimator, RegressorMixin):
    """규제 없는 11차 — 점 12개 = 미지수 12개(절편 + 계수 11)라 연립방정식을 그대로 푼다.
    LinearRegression 은 이 행렬이 거의 특이해서 작은 특이값을 잘라내고, 점을 다 지나지 못한다
    (학습 MSE 0.0255). 그러면 1차시의 11차(학습 0, 검증 868)와 다른 모델이 된다."""

    def fit(self, z: np.ndarray, y: np.ndarray) -> "ExactFit":
        solution = np.linalg.solve(np.column_stack([np.ones(len(z)), z]), y)
        self.intercept_, self.coef_ = solution[0], solution[1:]
        return self

    def predict(self, z: np.ndarray) -> np.ndarray:
        return self.intercept_ + z @ self.coef_


def poly_model(alpha: float) -> Pipeline:
    """같은 11차 다항식. alpha=0이면 규제 없이 점을 전부 지나는 해다.
    특징을 표준화하는 이유: [0,1]에서 x, x², …, x¹¹ 열의 표준편차가 0.34 ~ 0.15로 달라
    같은 α가 열마다 다른 세기로 걸린다. 표준화하면 모든 열에 같은 세기로 걸린다."""
    regressor = ExactFit() if alpha == 0 else Ridge(alpha=alpha)
    return make_pipeline(PolynomialFeatures(DEGREE, include_bias=False), StandardScaler(), regressor)


def ridge_sweep(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    x_tr = rng.uniform(0, 1, N_TRAIN).reshape(-1, 1)
    y_tr = truth(x_tr.ravel()) + rng.normal(0, NOISE, N_TRAIN)
    x_va = rng.uniform(0, 1, N_VAL).reshape(-1, 1)
    y_va = truth(x_va.ravel()) + rng.normal(0, NOISE, N_VAL)
    rows = []
    for alpha in RIDGE_ALPHAS:
        model = poly_model(alpha).fit(x_tr, y_tr)
        coef = model[-1].coef_
        rows.append(["규제 없음" if alpha == 0 else f"{alpha:g}", mean_squared_error(y_tr, model.predict(x_tr)),
                     mean_squared_error(y_va, model.predict(x_va)), float(np.abs(coef).max())])
    print(f"\n[1] Ridge — 점 {N_TRAIN}개, {DEGREE}차 다항식, 벌점 세기 α를 바꿈")
    table(["α", "학습 MSE", "검증 MSE", "가장 큰 계수"], rows)
    return x_tr, y_tr


def plot_ridge(x_tr: np.ndarray, y_tr: np.ndarray) -> None:
    grid = np.linspace(0, 1, PLOT_POINTS).reshape(-1, 1)
    fig, axes = plt.subplots(1, len(SHOWN_ALPHAS), figsize=(10.5, 3.4), sharey=True)
    for ax, alpha in zip(axes, SHOWN_ALPHAS):
        model = poly_model(alpha).fit(x_tr, y_tr)
        ax.plot(grid, truth(grid.ravel()), color=INK_SOFT, ls="--", lw=1.2, label="진짜 규칙")
        ax.plot(grid, model.predict(grid), color=TEAL, lw=2.2, label=f"{DEGREE}차 + Ridge")
        ax.scatter(x_tr, y_tr, color=ORANGE, s=22, zorder=3, label="학습 점")
        ax.set_ylim(-1.8, 1.8)
        ax.set_title("규제 없음" if alpha == 0 else f"α = {alpha:g}", fontsize=12)
    axes[0].legend(fontsize=8, loc="lower left")
    save(fig, "04_ridge_fits")


def lasso_sweep() -> None:
    """특징 50개 중 진짜는 5개. Lasso는 나머지 계수를 정확히 0으로 만든다."""
    x, y, true_coef = make_regression(n_samples=LASSO_SAMPLES, n_features=LASSO_FEATURES,
                                      n_informative=LASSO_INFORMATIVE, noise=LASSO_NOISE, coef=True, random_state=SEED)
    x_tr, x_va, y_tr, y_va = train_test_split(x, y, test_size=LASSO_VAL_FRACTION, random_state=SEED)
    real = set(np.flatnonzero(true_coef))
    rows = [["규제 없음", LASSO_FEATURES, "-", *_scores(LinearRegression().fit(x_tr, y_tr), x_tr, y_tr, x_va, y_va)]]
    for alpha in LASSO_ALPHAS:
        model = Lasso(alpha=alpha, max_iter=LASSO_MAX_ITER).fit(x_tr, y_tr)
        kept = set(np.flatnonzero(model.coef_))
        rows.append([f"{alpha:g}", len(kept), f"{len(kept & real)}/{LASSO_INFORMATIVE}",
                     *_scores(model, x_tr, y_tr, x_va, y_va)])
    print(f"\n[2] Lasso — 샘플 {len(y_tr)}개로 학습, 특징 {LASSO_FEATURES}개 중 진짜 {LASSO_INFORMATIVE}개")
    table(["α", "0 아닌 계수", "진짜를 남김", "학습 MSE", "검증 MSE"], rows)


def _scores(model: RegressorMixin, x_tr: np.ndarray, y_tr: np.ndarray, x_va: np.ndarray,
            y_va: np.ndarray) -> list[float]:
    return [mean_squared_error(y_tr, model.predict(x_tr)), mean_squared_error(y_va, model.predict(x_va))]


def pick_alpha(rng: np.random.Generator) -> None:
    """α도 하이퍼파라미터다. 테스트가 아니라 교차검증으로 고른다."""
    x = rng.uniform(0, 1, CURVE_POINTS).reshape(-1, 1)
    y = truth(x.ravel()) + rng.normal(0, NOISE, CURVE_POINTS)
    train, val = validation_curve(poly_model(1.0), x, y, param_name="ridge__alpha", param_range=CURVE_ALPHAS,
                                  cv=KFold(FOLDS, shuffle=True, random_state=SEED), scoring="neg_mean_squared_error")
    train_mse, val_mse = -train.mean(axis=1), -val.mean(axis=1)
    best = CURVE_ALPHAS[int(np.argmin(val_mse))]
    print(f"\n[3] α 고르기 — 점 {CURVE_POINTS}개, {FOLDS}-fold 교차검증")
    table(["α", "학습 MSE", "검증 MSE"], [[f"{a:.2g}", float(t), float(v)] for a, t, v in
                                     zip(CURVE_ALPHAS, train_mse, val_mse)][::TABLE_STRIDE])
    print(f"  → 검증 MSE가 가장 낮은 α: {best:g}")
    fig, ax = plt.subplots()
    ax.plot(CURVE_ALPHAS, train_mse, "o-", color=TEAL, label="학습")
    ax.plot(CURVE_ALPHAS, val_mse, "o-", color=ORANGE, label="검증 (교차검증 평균)")
    ax.axvline(best, color=INK_SOFT, ls="--", lw=1.2)
    log_x(ax)
    log_axis(ax)
    ax.set_xlabel("α (오른쪽일수록 규제가 셈)")
    ax.set_ylabel("MSE")
    ax.legend()
    save(fig, "04_validation_curve")


def main() -> None:
    setup_style()
    rng = np.random.default_rng(SEED)
    plot_ridge(*ridge_sweep(rng))
    lasso_sweep()
    pick_alpha(rng)


if __name__ == "__main__":
    main()
