# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib", "torch", "torchvision"]
# ///
"""
해커톤 사전 수업 2차시 — 미세조정: 어디까지 풀고, 얼마나 세게 움직이나.

  uv run code/10_finetune.py          (GPU 권장. 맥 M 시리즈 MPS 로 약 2분)

7차시와 같은 CIFAR-10 2,000장 · 96px · 증강 켬. ResNet18 사전학습 몸통에서 출발한다.
A. 푸는 깊이 — 머리만 / layer4 부터 / layer3 부터 / 전부
B. 몸통 학습률 — 전부 풀고 1e-5 · 1e-4 · 1e-3 (머리는 1e-3 고정)
C. 데이터 양 — 클래스당 20장이면 머리만 vs 전부, 답이 바뀌나
"""

import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt
from common import GOLD, INK_SOFT, ORANGE, SEED, TEAL, save, setup_style, table

DATA_DIR = Path(os.environ.get("DATA_DIR", Path.home() / ".cache" / "overfit-appendix"))
PER_CLASS = 200
SMALL_PER_CLASS = 20     # [C] 사진이 아주 적을 때
VAL_SIZE = 2000
N_CLASSES = 10
EPOCHS = 10
BATCH = 64
EVAL_BATCH = 256
LR_HEAD = 1e-3
LR_BODY = 1e-4
BODY_LRS = (1e-5, 1e-4, 1e-3)
IMAGE_SIZE = 32
CROP_PAD = 4
RESNET_SIZE = 96
IMAGENET_STATS = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
# ResNet18 의 몸통은 conv1 → layer1 → layer2 → layer3 → layer4 → fc 순서다. 뒤에서부터 푼다.
DEPTHS = {"머리만": ("fc",), "layer4 부터": ("layer4", "fc"), "layer3 부터": ("layer3", "layer4", "fc"),
          "전부": ("",)}


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def transform(augment: bool) -> transforms.Compose:
    steps = [transforms.RandomCrop(IMAGE_SIZE, padding=CROP_PAD), transforms.RandomHorizontalFlip()] if augment else []
    return transforms.Compose([*steps, transforms.Resize(RESNET_SIZE), transforms.ToTensor(),
                               transforms.Normalize(*IMAGENET_STATS)])


def subset_indices(targets: list[int], per_class: int) -> list[int]:
    """7차시와 같은 규칙. per_class 가 작으면 같은 순서의 앞부분만 쓴다."""
    rng = np.random.default_rng(SEED)
    labels = np.array(targets)
    return sorted(int(i) for c in range(N_CLASSES) for i in rng.permutation(np.flatnonzero(labels == c))[:per_class])


def loaders(per_class: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    """학습용(증강) · 학습 정확도 재기용(증강 끔) · 검증용."""
    train = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=transform(True))
    plain = datasets.CIFAR10(DATA_DIR, train=True, transform=transform(False))
    val = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=transform(False))
    indices = subset_indices(train.targets, per_class)
    generator = torch.Generator().manual_seed(SEED)
    return (DataLoader(Subset(train, indices), BATCH, shuffle=True, generator=generator),
            DataLoader(Subset(plain, indices), EVAL_BATCH),
            DataLoader(Subset(val, list(range(VAL_SIZE))), EVAL_BATCH))


def build(trainable: tuple[str, ...]) -> nn.Module:
    """trainable 에 적힌 이름으로 시작하는 층만 학습한다. ("",) 은 전부."""
    torch.manual_seed(SEED)
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, N_CLASSES)
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith(trainable)
    return model


def keep_frozen_bn(model: nn.Module) -> None:
    """얼린 층의 BN 은 eval — running_mean/var 를 ImageNet 값 그대로 둔다 (7차시 함정)."""
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d) and not any(p.requires_grad for p in module.parameters()):
            module.eval()


def groups(model: nn.Module, lr_body: float) -> list[dict[str, object]]:
    head = [p for n, p in model.named_parameters() if n.startswith("fc.")]
    body = [p for n, p in model.named_parameters() if not n.startswith("fc.") and p.requires_grad]
    return [{"params": head, "lr": LR_HEAD}] + ([{"params": body, "lr": lr_body}] if body else [])


@torch.no_grad()
def accuracy(model: nn.Module, loader: DataLoader, dev: torch.device) -> float:
    model.eval()
    hits = total = 0
    for x, y in loader:
        hits += (model(x.to(dev)).argmax(1).cpu() == y).sum().item()
        total += len(y)
    return hits / total


def fit(trainable: tuple[str, ...], lr_body: float = LR_BODY, per_class: int = PER_CLASS) -> dict[str, object]:
    """epoch마다 (학습, 검증) 정확도와 epoch당 초, 학습하는 파라미터 수."""
    dev = device()
    model = build(trainable).to(dev)
    train, plain, val = loaders(per_class)
    optimizer = torch.optim.Adam(groups(model, lr_body))
    history, seconds = [], []
    for _ in range(EPOCHS):
        start = time.perf_counter()
        model.train()
        keep_frozen_bn(model)
        for x, y in train:
            optimizer.zero_grad()
            nn.functional.cross_entropy(model(x.to(dev)), y.to(dev)).backward()
            optimizer.step()
        seconds.append(time.perf_counter() - start)
        history.append((accuracy(model, plain, dev), accuracy(model, val, dev)))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"history": history, "sec": float(np.mean(seconds)), "params": n_params}


def report(title: str, results: dict[str, dict[str, object]], timing: bool = False) -> None:
    print(f"\n{title}")
    rows = []
    for name, r in results.items():
        train_acc, val_acc = r["history"][-1]
        row = [name, train_acc, val_acc, train_acc - val_acc]
        if timing:
            row += [f"{r['params'] / 1e6:.2f}M", f"{r['sec']:.1f}초"]
        rows.append(row)
    table(["설정", "학습", "검증(마지막)", "간격"] + (["학습 파라미터", "epoch당"] if timing else []), rows)


def plot(results: dict[str, dict[str, object]], name: str, title: str) -> None:
    colors = [ORANGE, GOLD, TEAL, INK_SOFT]
    fig, ax = plt.subplots()
    for color, (label, r) in zip(colors, results.items()):
        h = np.array(r["history"])
        ax.plot(range(1, len(h) + 1), h[:, 1], color=color, lw=2, label=f"{label} 검증")
        ax.plot(range(1, len(h) + 1), h[:, 0], color=color, lw=1, ls=":", alpha=0.8)
    ax.set_title(f"{title}  (점선 = 학습)", fontsize=11)
    ax.set_xlabel("epoch")
    ax.set_ylabel("정확도")
    ax.legend(fontsize=9)
    save(fig, name)


def main() -> None:
    setup_style()
    print(f"장치: {device()} · 학습 {PER_CLASS * N_CLASSES}장 · {EPOCHS} epoch · 데이터 {DATA_DIR}")
    depth = {name: fit(trainable) for name, trainable in DEPTHS.items()}
    report(f"[A] 푸는 깊이 — 머리 {LR_HEAD:g} · 몸통 {LR_BODY:g}", depth, timing=True)
    plot(depth, "10_depth", "푸는 깊이")
    lrs = {f"몸통 {lr:g}": (depth["전부"] if lr == LR_BODY else fit(("",), lr_body=lr)) for lr in BODY_LRS}
    report(f"[B] 몸통 학습률 — 전부 풂 · 머리 {LR_HEAD:g}", lrs)
    plot(lrs, "10_body_lr", "몸통 학습률")
    small = {name: fit(DEPTHS[name], per_class=SMALL_PER_CLASS) for name in ("머리만", "전부")}
    report(f"[C] 클래스당 {SMALL_PER_CLASS}장 (학습 {SMALL_PER_CLASS * N_CLASSES}장)", small)


if __name__ == "__main__":
    main()
