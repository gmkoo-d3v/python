# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib", "torch", "torchvision"]
# ///
"""
7차시 — 신경망 ② 이미지: 데이터 증강 · 배치정규화 · 전이학습.

  uv run code/07_images.py            (GPU 권장. 맥 M 시리즈 MPS 로 약 10분)
  DATA_DIR=/data uv run code/07_images.py

CIFAR-10 에서 학습 이미지를 2,000장만 쓴다(클래스당 200장). 사진이 적으면 CNN은 사진 자체를 외운다.
A. 작은 CNN을 처음부터 — 증강·배치정규화·라벨 스무딩·Mixup이 얼마나 막나
B. ResNet18 — 처음부터 vs 사전학습 몸통 동결 vs 전부 미세조정
"""

import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt
from common import GOLD, INK_SOFT, ORANGE, SEED, TEAL, save, setup_style, table

DATA_DIR = Path(os.environ.get("DATA_DIR", Path.home() / ".cache" / "overfit-appendix"))
PER_CLASS = 200          # 클래스당 학습 이미지 수
VAL_SIZE = 2000
EPOCHS_SMALL = 40
EPOCHS_RESNET = 12
BATCH = 64
LR = 1e-3
LR_BODY = 1e-4          # 사전학습 몸통은 작게 — 크게 흔들면 ImageNet 에서 배운 것을 잃는다
N_CLASSES = 10
IMAGE_SIZE = 32
CROP_PAD = 4
EVAL_BATCH = 256
CNN_WIDTHS = (32, 64, 128)
CNN_HIDDEN = 256
LABEL_SMOOTHING = 0.1   # 0.1을 10칸 전부에 고르게 나누고 정답 칸에 0.9를 더한다 → 정답 0.91, 나머지 0.01씩
MIXUP_ALPHA = 0.2       # 두 사진을 섞는 비율 λ ~ Beta(α, α)
RESNET_SIZE = 96         # 사전학습 ResNet 은 큰 사진에 익숙하다. 32px 를 키워서 넣는다
CIFAR_STATS = ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
# 사전학습 ResNet 은 ImageNet 사진을 이 평균·표준편차로 정규화해 배웠다. 같은 자로 재야 한다.
IMAGENET_STATS = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def transform(augment: bool, size: int, stats: tuple[tuple[float, ...], tuple[float, ...]]) -> transforms.Compose:
    steps = [transforms.RandomCrop(IMAGE_SIZE, padding=CROP_PAD), transforms.RandomHorizontalFlip()] if augment else []
    if size != IMAGE_SIZE:
        steps.append(transforms.Resize(size))
    return transforms.Compose([*steps, transforms.ToTensor(), transforms.Normalize(*stats)])


def subset_indices(targets: list[int]) -> list[int]:
    """클래스마다 시드로 섞어 PER_CLASS 장을 고른다. 시드가 같으니 어느 실험이든 같은 2,000장이다."""
    rng = np.random.default_rng(SEED)
    labels = np.array(targets)
    return sorted(int(i) for c in range(N_CLASSES) for i in rng.permutation(np.flatnonzero(labels == c))[:PER_CLASS])


def loaders(augment: bool, size: int = IMAGE_SIZE,
            stats: tuple[tuple[float, ...], tuple[float, ...]] = CIFAR_STATS) -> tuple[DataLoader, DataLoader, DataLoader]:
    """학습용 · 학습 정확도 재기용 · 검증용.
    학습 정확도는 증강을 끈 같은 사진으로 잰다 — 증강을 켠 채 재면 낮게 나와 간격 비교가 안 된다."""
    train = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=transform(augment, size, stats))
    plain = datasets.CIFAR10(DATA_DIR, train=True, transform=transform(False, size, stats))
    val = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=transform(False, size, stats))
    indices = subset_indices(train.targets)
    generator = torch.Generator().manual_seed(SEED)
    return (DataLoader(Subset(train, indices), BATCH, shuffle=True, generator=generator),
            DataLoader(Subset(plain, indices), EVAL_BATCH),
            DataLoader(Subset(val, list(range(VAL_SIZE))), EVAL_BATCH))


def small_cnn(batchnorm: bool) -> nn.Module:
    def block(c_in: int, c_out: int) -> list[nn.Module]:
        norm = [nn.BatchNorm2d(c_out)] if batchnorm else []
        return [nn.Conv2d(c_in, c_out, 3, padding=1), *norm, nn.ReLU(), nn.MaxPool2d(2)]
    widths = (3, *CNN_WIDTHS)
    side = IMAGE_SIZE // 2 ** len(CNN_WIDTHS)  # 풀링 세 번 → 32px 가 4px
    layers = [m for c_in, c_out in zip(widths, widths[1:]) for m in block(c_in, c_out)]
    return nn.Sequential(*layers, nn.Flatten(), nn.Linear(CNN_WIDTHS[-1] * side * side, CNN_HIDDEN), nn.ReLU(),
                         nn.Linear(CNN_HIDDEN, N_CLASSES))


@torch.no_grad()
def accuracy(model: nn.Module, loader: DataLoader, dev: torch.device) -> float:
    model.eval()
    hits = total = 0
    for x, y in loader:
        hits += (model(x.to(dev)).argmax(1).cpu() == y).sum().item()
        total += len(y)
    return hits / total


def batch_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor, mode: str) -> torch.Tensor:
    """mode: plain · smooth(라벨 스무딩) · mixup(두 사진과 두 정답을 같은 비율로 섞는다)."""
    if mode == "smooth":
        return nn.functional.cross_entropy(model(x), y, label_smoothing=LABEL_SMOOTHING)
    if mode == "mixup":
        lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
        perm = torch.randperm(len(y), device=x.device)
        logits = model(lam * x + (1 - lam) * x[perm])
        return lam * nn.functional.cross_entropy(logits, y) + (1 - lam) * nn.functional.cross_entropy(logits, y[perm])
    return nn.functional.cross_entropy(model(x), y)


def keep_frozen_bn(model: nn.Module) -> None:
    """requires_grad=False 로 동결해도 model.train() 이면 BN 의 running_mean/var 는 계속 바뀐다.
    파라미터가 모두 동결된 BN 층은 eval 로 돌려 ImageNet 통계를 그대로 둔다."""
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d) and not any(p.requires_grad for p in module.parameters()):
            module.eval()


def fit(model: nn.Module, data: tuple[DataLoader, DataLoader, DataLoader], epochs: int,
        mode: str = "plain", groups: list[dict[str, object]] | None = None) -> list[tuple[float, float]]:
    """epoch마다 (학습 정확도, 검증 정확도). groups 가 있으면 층마다 다른 학습률을 쓴다."""
    train, plain_train, val = data
    dev = device()
    model.to(dev)
    optimizer = torch.optim.Adam(groups or [p for p in model.parameters() if p.requires_grad], lr=LR)
    history = []
    for _ in range(epochs):
        model.train()
        keep_frozen_bn(model)
        for x, y in train:
            optimizer.zero_grad()
            batch_loss(model, x.to(dev), y.to(dev), mode).backward()
            optimizer.step()
        history.append((accuracy(model, plain_train, dev), accuracy(model, val, dev)))
    return history


def experiment_small() -> dict[str, list[tuple[float, float]]]:
    runs = {"기준선": (False, False, "plain"), "증강": (True, False, "plain"), "배치정규화": (False, True, "plain"),
            "증강 + BN": (True, True, "plain"), "증강 + BN + 라벨 스무딩": (True, True, "smooth"),
            "증강 + BN + Mixup": (True, True, "mixup")}
    results = {}
    for name, (augment, batchnorm, mode) in runs.items():
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        results[name] = fit(small_cnn(batchnorm), loaders(augment), EPOCHS_SMALL, mode)
        print(f"  {name} 끝")
    return results


def resnet(pretrained: bool, freeze: bool) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    if freeze:
        for p in model.parameters():
            p.requires_grad = False  # 몸통은 ImageNet 에서 배운 그대로 둔다
    model.fc = nn.Linear(model.fc.in_features, N_CLASSES)  # 새 머리는 항상 학습한다
    return model


def finetune_groups(model: nn.Module) -> list[dict[str, object]]:
    """미세조정: 몸통은 작은 학습률, 새 머리는 큰 학습률."""
    head = list(model.fc.parameters())
    body = [p for name, p in model.named_parameters() if not name.startswith("fc.")]
    return [{"params": body, "lr": LR_BODY}, {"params": head, "lr": LR}]


def experiment_transfer() -> dict[str, list[tuple[float, float]]]:
    runs = {"처음부터": (False, False), "몸통 동결": (True, True), "전부 미세조정": (True, False)}
    results = {}
    for name, (pretrained, freeze) in runs.items():
        torch.manual_seed(SEED)
        model = resnet(pretrained, freeze)
        stats = IMAGENET_STATS if pretrained else CIFAR_STATS
        groups = finetune_groups(model) if pretrained and not freeze else None
        results[name] = fit(model, loaders(augment=True, size=RESNET_SIZE, stats=stats), EPOCHS_RESNET, groups=groups)
        print(f"  {name} 끝")
    return results


def report(title: str, results: dict[str, list[tuple[float, float]]]) -> None:
    print(f"\n{title}")
    rows = []
    for name, history in results.items():
        train_acc, val_acc = history[-1]
        best = max(v for _, v in history)
        rows.append([name, train_acc, val_acc, best, train_acc - val_acc])
    table(["방법", "학습", "검증(마지막)", "검증(최고 — 엿봄)", "간격"], rows)


def plot(results: dict[str, list[tuple[float, float]]], name: str, title: str) -> None:
    colors = [ORANGE, TEAL, GOLD, INK_SOFT, "#5B57C4", "#7FD1C4"]
    fig, ax = plt.subplots()
    for color, (label, history) in zip(colors, results.items()):
        h = np.array(history)
        ax.plot(range(1, len(h) + 1), h[:, 1], color=color, lw=2, label=f"{label} 검증")
        ax.plot(range(1, len(h) + 1), h[:, 0], color=color, lw=1, ls=":", alpha=0.8)
    ax.set_title(f"{title}  (점선 = 학습)", fontsize=11)
    ax.set_xlabel("epoch")
    ax.set_ylabel("정확도")
    ax.legend(fontsize=9)
    save(fig, name)


def main() -> None:
    setup_style()
    print(f"장치: {device()} · 학습 {PER_CLASS * 10}장 · 검증 {VAL_SIZE}장 · 데이터 {DATA_DIR}")
    small = experiment_small()
    report(f"[A] 작은 CNN 처음부터 — {EPOCHS_SMALL} epoch (정확도)", small)
    plot(small, "07_small_cnn", "작은 CNN")
    transfer = experiment_transfer()
    report(f"[B] ResNet18 — {EPOCHS_RESNET} epoch, 셋 다 증강 켬 (정확도)", transfer)
    plot(transfer, "07_transfer", "ResNet18")


if __name__ == "__main__":
    main()
