# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib", "scikit-learn", "torch", "torchvision"]
# ///
"""
해커톤 사전 수업 1차시 — 빌려 온 몸통을 특징 추출기로 쓰기.

  uv run code/09_features.py          (맥 M 시리즈 MPS 로 약 1분)

ResNet18 의 마지막 층(fc)을 떼면 사진 한 장이 512개 숫자가 된다. 그 숫자로 로지스틱 회귀·kNN 을 돌린다.
신경망 학습은 한 번도 하지 않는다 — 특징을 한 번 뽑아 두고 scikit-learn 으로 분류한다.
A. 사전학습 몸통 vs 무작위 몸통   B. 한 클래스에 몇 장이면 되나
C. 입력 크기                       D. 정규화(평균·표준편차)를 틀리면
E. 로지스틱 회귀의 규제 C — 4차시의 α 와 같은 손잡이
"""

import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt
from common import INK_SOFT, ORANGE, SEED, TEAL, save, setup_style, table

DATA_DIR = Path(os.environ.get("DATA_DIR", Path.home() / ".cache" / "overfit-appendix"))
PER_CLASS = 200          # 클래스당 학습 사진 — 7차시와 같은 2,000장
VAL_SIZE = 2000
N_CLASSES = 10
MAIN_SIZE = 96           # 7차시 전이학습과 같은 크기
SIZES = (32, 64, 96, 160)
SHOTS = (5, 10, 20, 50, 100, 200)
KNN_K = 5
MAX_ITER = 3000          # 로지스틱 회귀가 수렴할 만큼
C_VALUES = (0.001, 0.01, 0.1, 1.0)  # 작을수록 규제가 세다 (C = 1/α)
EVAL_BATCH = 256
PCA_POINTS = 1000        # 산점도에 찍을 검증 사진 수
LEGEND_ROW = (8, 1)      # 산점도 : 범례 줄 높이
LEGEND_COLS = 5          # 범례 10개를 두 줄로
FILTER_COLS = 8          # conv1 필터 64개를 8 × 8 로 늘어놓는다
IMAGENET_STATS = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
CIFAR_STATS = ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
CLASS_NAMES = ["비행기", "자동차", "새", "고양이", "사슴", "개", "개구리", "말", "배", "트럭"]
CLASS_COLORS = ["#2A9D8F", "#B3541E", "#D9A02B", "#47576B", "#5B57C4",
                "#7FD1C4", "#E07A5F", "#3D5A80", "#9C6644", "#81B29A"]


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available(): # apple metal
        return torch.device("mps")
    return torch.device("cpu")


def transform(size: int, stats: tuple[tuple[float, ...], tuple[float, ...]] | None) -> transforms.Compose:
    steps = [transforms.Resize(size)] if size != 32 else []
    norm = [transforms.Normalize(*stats)] if stats else []
    return transforms.Compose([*steps, transforms.ToTensor(), *norm])


def subset_indices(targets: list[int]) -> list[int]:
    """7차시와 같은 규칙: 클래스마다 시드로 섞어 PER_CLASS 장."""
    rng = np.random.default_rng(SEED)
    labels = np.array(targets)
    return sorted(int(i) for c in range(N_CLASSES) for i in rng.permutation(np.flatnonzero(labels == c))[:PER_CLASS])


def body(pretrained: bool) -> nn.Module:
    """fc 를 떼어 낸 ResNet18 — 사진 → 512차원."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    torch.manual_seed(SEED)
    model = models.resnet18(weights=weights)
    model.fc = nn.Identity()
    return model.eval()


@torch.no_grad()
def extract(model: nn.Module, size: int, stats: tuple[tuple[float, ...], tuple[float, ...]] | None
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """학습 2,000장과 검증 2,000장의 특징·라벨. 몸통은 eval — BN 은 ImageNet 통계 그대로."""
    dev = device()
    model.to(dev)
    tf = transform(size, stats)
    train = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=tf)
    val = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=tf)
    out = []
    for ds in (Subset(train, subset_indices(train.targets)), Subset(val, list(range(VAL_SIZE)))):
        feats, labels = [], []
        for x, y in DataLoader(ds, EVAL_BATCH):
            feats.append(model(x.to(dev)).cpu().numpy())
            labels.append(y.numpy())
        out += [np.concatenate(feats), np.concatenate(labels)]
    return out[0], out[1], out[2], out[3]


def logistic(c: float = 1.0) -> object:
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=MAX_ITER))


def knn() -> object:
    return make_pipeline(StandardScaler(), KNeighborsClassifier(KNN_K))


def score(clf: object, xtr: np.ndarray, ytr: np.ndarray, xva: np.ndarray, yva: np.ndarray) -> tuple[float, float]:
    """(학습 정확도, 검증 정확도)"""
    clf.fit(xtr, ytr)
    return clf.score(xtr, ytr), clf.score(xva, yva)


def experiment_body() -> dict[str, tuple[np.ndarray, ...]]:
    feats = {}
    rows = []
    for name, pretrained in (("사전학습", True), ("무작위", False)):
        feats[name] = extract(body(pretrained), MAIN_SIZE, IMAGENET_STATS)
        for clf_name, clf in (("로지스틱", logistic()), (f"kNN(k={KNN_K})", knn())):
            tr, va = score(clf, *feats[name])
            rows.append([f"{name} 몸통 + {clf_name}", tr, va, tr - va])
    print(f"\n[A] 몸통 비교 — {MAIN_SIZE}px · 학습 {PER_CLASS * N_CLASSES}장 · 신경망 학습 없음")
    table(["방법", "학습", "검증", "간격"], rows)
    return feats


def experiment_shots(xtr: np.ndarray, ytr: np.ndarray, xva: np.ndarray, yva: np.ndarray) -> list[float]:
    """같은 특징에서 클래스당 n장만 남겨 로지스틱 회귀."""
    rng = np.random.default_rng(SEED)
    order = {c: rng.permutation(np.flatnonzero(ytr == c)) for c in range(N_CLASSES)}
    rows, curve = [], []
    for n in SHOTS:
        keep = np.sort(np.concatenate([order[c][:n] for c in range(N_CLASSES)]))  # 원래 순서 — 200장이면 [A] 와 같은 입력
        tr, va = score(logistic(), xtr[keep], ytr[keep], xva, yva)
        rows.append([n, n * N_CLASSES, tr, va])
        curve.append(va)
    print("\n[B] 클래스당 몇 장 — 사전학습 몸통 + 로지스틱")
    table(["클래스당", "학습 장수", "학습", "검증"], rows)
    return curve


def experiment_size() -> None:
    model = body(True)
    rows = []
    for size in SIZES:
        start = time.perf_counter()
        feats = extract(model, size, IMAGENET_STATS)
        seconds = time.perf_counter() - start
        rows.append([f"{size}px", score(logistic(), *feats)[1], f"{seconds:.0f}초"])
    print("\n[C] 입력 크기 — 32px 원본을 키워서 넣는다 (사전학습 몸통 + 로지스틱)")
    table(["입력", "검증", "특징 뽑는 시간"], rows)


def experiment_norm() -> None:
    model = body(True)
    rows = []
    for name, stats in (("ImageNet 평균·표준편차", IMAGENET_STATS), ("CIFAR 평균·표준편차", CIFAR_STATS),
                        ("정규화 안 함 (0~1)", None)):
        rows.append([name, score(logistic(), *extract(model, MAIN_SIZE, stats))[1]])
    print(f"\n[D] 정규화 — {MAIN_SIZE}px · 사전학습 몸통 + 로지스틱")
    table(["입력 정규화", "검증"], rows)


def experiment_c(xtr: np.ndarray, ytr: np.ndarray, xva: np.ndarray, yva: np.ndarray) -> None:
    rows = []
    for c in C_VALUES:
        tr, va = score(logistic(c), xtr, ytr, xva, yva)
        rows.append([c, tr, va, tr - va])
    print(f"\n[E] 규제 C — 사전학습 몸통 · {MAIN_SIZE}px · 로지스틱 (C 를 검증으로 고르면 엿봄이다 — 실습에서는 교차검증으로)")
    table(["C", "학습", "검증", "간격"], rows)


def plot_filters() -> None:
    """conv1 의 7×7 필터 64개. ImageNet 에서 배운 첫 층은 줄무늬·경계·색 덩어리를 본다."""
    weights = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).conv1.weight.detach().numpy()
    fig, axes = plt.subplots(len(weights) // FILTER_COLS, FILTER_COLS, figsize=(6.4, 6.4))
    for ax, w in zip(axes.flat, weights):
        w = w.transpose(1, 2, 0)
        ax.imshow((w - w.min()) / (w.max() - w.min()))
        ax.axis("off")
    fig.suptitle("ResNet18 conv1 — ImageNet 사전학습 필터 64개", fontsize=11)
    save(fig, "09_filters")


def plot_pca(feats: dict[str, tuple[np.ndarray, ...]]) -> None:
    """검증 사진 1,000장의 512차원 특징을 2차원으로 눌러 클래스별 색으로 찍는다."""
    fig = plt.figure(figsize=(7.2, 4.2))
    grid = fig.add_gridspec(2, 2, height_ratios=LEGEND_ROW)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    legend_ax = fig.add_subplot(grid[1, :])  # 아래 줄 전체가 범례 자리
    for ax, (name, (_, _, xva, yva)) in zip(axes, feats.items()):
        xy = PCA(2, random_state=SEED).fit_transform(StandardScaler().fit_transform(xva[:PCA_POINTS]))
        ax.scatter(xy[:, 0], xy[:, 1], c=[CLASS_COLORS[c] for c in yva[:PCA_POINTS]], s=6, alpha=0.8)
        ax.set_title(f"{name} 몸통", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    handles = [plt.Line2D([], [], ls="", marker="o", color=color, label=name)
               for name, color in zip(CLASS_NAMES, CLASS_COLORS)]
    legend_ax.axis("off")
    legend_ax.legend(handles=handles, loc="center", ncol=LEGEND_COLS, fontsize=9, frameon=False,
                     handletextpad=0.2, columnspacing=1.2)
    save(fig, "09_pca")


def plot_shots(curve: list[float]) -> None:
    fig, ax = plt.subplots()
    ax.plot(SHOTS, curve, color=TEAL, lw=2, marker="o")
    ax.axhline(1 / N_CLASSES, color=INK_SOFT, ls=":", lw=1)
    ax.text(SHOTS[0], 1 / N_CLASSES + 0.02, "찍기 (10%)", color=INK_SOFT, fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks(SHOTS, [str(n) for n in SHOTS])
    ax.set_xlabel("클래스당 학습 사진 수")
    ax.set_ylabel("검증 정확도")
    ax.set_ylim(0, 1)
    ax.set_title("사전학습 몸통 + 로지스틱 회귀", fontsize=11)
    for n, v in zip(SHOTS, curve):
        ax.annotate(f"{v:.2f}", (n, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9,
                    color=ORANGE)
    save(fig, "09_shots")


def main() -> None:
    setup_style()
    print(f"장치: {device()} · 데이터 {DATA_DIR}")
    plot_filters()
    feats = experiment_body()
    plot_pca(feats)
    plot_shots(experiment_shots(*feats["사전학습"]))
    experiment_size()
    experiment_norm()
    experiment_c(*feats["사전학습"])


if __name__ == "__main__":
    main()
