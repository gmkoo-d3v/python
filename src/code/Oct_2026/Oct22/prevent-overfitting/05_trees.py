# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scikit-learn", "matplotlib"]
# ///
"""
5차시 — 나무 모델의 과적합.

  uv run code/05_trees.py

결정트리는 멈추라는 말이 없으면 잎 하나에 점 하나가 남을 때까지 쪼갠다 — 학습 정확도 100%.
멈추는 법 세 가지(깊이·잎 크기·가지치기)와, 여러 그루를 평균 내는 법(배깅·랜덤포레스트)을 본다.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

from common import ORANGE, ORANGE_PALE, SEED, TEAL, TEAL_PALE, save, setup_style, table

N_SAMPLES = 1000  # 검증이 200개면 한 개가 0.5%p 라 표가 들쭉날쭉하다
MOON_NOISE = 0.35
DEPTHS = [1, 2, 3, 4, 6, 8, 12, None]
LEAVES = [1, 2, 5, 10, 20, 50]
FOREST_SIZES = [1, 5, 10, 50, 200]
FOLDS = 5
VAL_FRACTION = 0.5     # 반은 학습, 반은 검증
GRID_STEP = 0.02
BOUNDARY_MARGIN = 0.5  # 결정 경계 그림을 데이터 범위보다 이만큼 넓게 그린다
SHOWN_DEPTH = 3
SHOWN_FOREST = FOREST_SIZES[-1]
# None 은 랜덤포레스트, 나머지는 결정트리에 넘길 인자
SHOWN: list[tuple[str, dict[str, int] | None]] = [
    (f"깊이 {SHOWN_DEPTH}", {"max_depth": SHOWN_DEPTH}),
    ("제한 없음", {}),
    (f"랜덤포레스트 {SHOWN_FOREST}그루", None),
]


def data() -> list[np.ndarray]:
    x, y = make_moons(N_SAMPLES, noise=MOON_NOISE, random_state=SEED)
    return train_test_split(x, y, test_size=VAL_FRACTION, stratify=y, random_state=SEED)


def tree(**kwargs: float | None) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(random_state=SEED, **kwargs)


def depth_and_leaf(x_tr: np.ndarray, x_va: np.ndarray, y_tr: np.ndarray, y_va: np.ndarray) -> None:
    rows = []
    for depth in DEPTHS:
        model = tree(max_depth=depth).fit(x_tr, y_tr)
        rows.append([depth or "제한 없음", model.get_n_leaves(), float(model.score(x_tr, y_tr)), float(model.score(x_va, y_va))])
    print(f"\n[1] 깊이 — 반달 데이터 학습 {len(y_tr)}개·검증 {len(y_va)}개 (정확도)")
    table(["최대 깊이", "잎 수", "학습", "검증"], rows)
    rows = []
    for leaf in LEAVES:
        model = tree(min_samples_leaf=leaf).fit(x_tr, y_tr)
        rows.append([leaf, model.get_n_leaves(), float(model.score(x_tr, y_tr)), float(model.score(x_va, y_va))])
    print("\n[2] 잎 크기 — 잎 하나에 최소 몇 개가 남아야 쪼갤 수 있나")
    table(["min_samples_leaf", "잎 수", "학습", "검증"], rows)


def pruning(x_tr: np.ndarray, x_va: np.ndarray, y_tr: np.ndarray, y_va: np.ndarray) -> None:
    """다 키운 뒤 쓸모가 적은 가지부터 잘라낸다. 얼마나 자를지(ccp_alpha)는 교차검증으로 고른다."""
    path = tree().cost_complexity_pruning_path(x_tr, y_tr)
    alphas = path.ccp_alphas[:-1]  # 마지막 값은 뿌리만 남기는 극단이라 뺀다
    cv = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
    scores = [cross_val_score(tree(ccp_alpha=a), x_tr, y_tr, cv=cv).mean() for a in alphas]
    best = float(alphas[int(np.argmax(scores))])
    full, pruned = tree().fit(x_tr, y_tr), tree(ccp_alpha=best).fit(x_tr, y_tr)
    print(f"\n[3] 가지치기 — 후보 ccp_alpha {len(alphas)}개를 {FOLDS}-fold로 비교, 고른 값 {best:.4f}")
    table(["트리", "잎 수", "학습", "검증"],
          [["다 키운 것", full.get_n_leaves(), float(full.score(x_tr, y_tr)), float(full.score(x_va, y_va))],
           ["가지치기", pruned.get_n_leaves(), float(pruned.score(x_tr, y_tr)), float(pruned.score(x_va, y_va))]])


def ensembles(x_tr: np.ndarray, x_va: np.ndarray, y_tr: np.ndarray, y_va: np.ndarray) -> None:
    """깊은 트리 여러 그루를 평균 내면, 각자의 과적합(분산)이 서로 상쇄된다."""
    rows = []
    for n in FOREST_SIZES:
        bag = BaggingClassifier(tree(), n_estimators=n, random_state=SEED).fit(x_tr, y_tr)
        forest = RandomForestClassifier(n_estimators=n, random_state=SEED).fit(x_tr, y_tr)
        rows.append([n, float(bag.score(x_tr, y_tr)), float(bag.score(x_va, y_va)),
                     float(forest.score(x_tr, y_tr)), float(forest.score(x_va, y_va))])
    print("\n[4] 앙상블 — 깊이 제한 없는 트리를 여러 그루")
    table(["그루 수", "배깅 학습", "배깅 검증", "포레스트 학습", "포레스트 검증"], rows)


def plot_boundaries(x_tr: np.ndarray, y_tr: np.ndarray) -> None:
    lo, hi = x_tr.min(axis=0) - BOUNDARY_MARGIN, x_tr.max(axis=0) + BOUNDARY_MARGIN
    gx, gy = np.meshgrid(np.arange(lo[0], hi[0], GRID_STEP), np.arange(lo[1], hi[1], GRID_STEP))
    cmap = ListedColormap([ORANGE_PALE, TEAL_PALE])
    fig, axes = plt.subplots(1, len(SHOWN), figsize=(10.5, 3.5), sharey=True)
    for ax, (title, kwargs) in zip(axes, SHOWN):
        model = (RandomForestClassifier(n_estimators=SHOWN_FOREST, random_state=SEED) if kwargs is None
                 else tree(**kwargs)).fit(x_tr, y_tr)
        ax.contourf(gx, gy, model.predict(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape), cmap=cmap)
        ax.scatter(*x_tr.T, c=[ORANGE if v == 0 else TEAL for v in y_tr], s=10, edgecolors="white", linewidths=0.4)
        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    save(fig, "05_boundaries")


def main() -> None:
    setup_style()
    x_tr, x_va, y_tr, y_va = data()
    depth_and_leaf(x_tr, x_va, y_tr, y_va)
    pruning(x_tr, x_va, y_tr, y_va)
    ensembles(x_tr, x_va, y_tr, y_va)
    plot_boundaries(x_tr, y_tr)


if __name__ == "__main__":
    main()
