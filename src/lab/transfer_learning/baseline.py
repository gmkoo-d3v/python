# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "torchvision", "pillow"]
# ///
"""
전이학습 해커톤 강화 기준선 — Flowers-102 (102종, 학습 1,020장), ResNet18.

  uv run baseline.py

패널 컨센서스 v7:
  [1] 224px + RandomResizedCrop/Flip/ColorJitter(약하게)·B는 TrivialAugmentWide
  [2] 전체 미세조정 + 판별 LR(몸통 2e-4 / 머리 1e-3)
  [3] CE + label_smoothing 0.1
  [4] AdamW + cosine(warmup 2ep)
  [5] 120 epoch(과소적합 해소) · raw_bn vs swa_bn(마지막 40ep 평균) 둘 다 BN 재추정 후 공정 비교
      → 채택 · 가중치 영속화(checkpoints/v7_{A,B}.pt)
  predict: 앙상블(2모델) × 2뷰 × flip softmax 확률 합

룰 준수: resnet18 + IMAGENET1K_V1, 외부 데이터 없음, 입력 224px 이하,
val 합침은 README 30행 허용(선택 신호는 train 내부 holdout 102장).
"""

import copy
import math

import torch
from torch import nn
from torchvision import models, transforms

from common import (HERE, IMAGENET_STATS, N_CLASSES, SEED, TestImages, accuracy, device, labeled,
                    loader, write_submission)
from torch.utils.data import ConcatDataset  # train+val 합침(README 30행 허용)

IMAGE_SIZE = 224
BATCH = 32
EPOCHS = 120            # v5(60)는 3,660 step — 표준의 1/3, 과소적합 (패널2 L1 봉합)
WARMUP = 2
SWA_EPOCHS = 40         # 마지막 40 에폭 = 전체의 1/3 창
LR_BODY = 2e-4
LR_HEAD = 1e-3
WD_BODY = 0.05


# ── [바꿀 곳 1] 변환 ── 학습은 확률적 증강, 평가는 결정론적
def train_transform(scale: tuple[float, float] = (0.65, 1.0), trivial: bool = False) -> transforms.Compose:
    aug: list[transforms.Transform] = [
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=scale, ratio=(0.75, 4 / 3),
                                     interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip()]
    aug.append(transforms.TrivialAugmentWide() if trivial
               else transforms.ColorJitter(0.2, 0.2, 0.2, 0.05))
    return transforms.Compose(aug + [transforms.ToTensor(), transforms.Normalize(*IMAGENET_STATS)])


def eval_transform() -> transforms.Compose:
    return transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.CenterCrop(IMAGE_SIZE),
                               transforms.ToTensor(), transforms.Normalize(*IMAGENET_STATS)])


def squish_transform() -> transforms.Compose:
    """종횡비 유지 없이 224×224 로 늘리는 뷰. TTA 다양성용 (입력 224px, 룰 준수)."""
    return transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                               transforms.ToTensor(), transforms.Normalize(*IMAGENET_STATS)])


# ── [바꿀 곳 2] 모델 ── 전체 학습하되 몸통은 낮은 학습률로 [4]에서 분리
def build_model() -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, N_CLASSES)
    return model


# ── [바꿀 곳 3] 손실 ── CE + label smoothing
def loss_fn(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, y, label_smoothing=0.1)


# ── [바꿀 곳 4] 옵티마이저 · 스케줄 ── AdamW + 판별 LR + cosine/warmup
def build_optimizer(model: nn.Module) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler | None]:
    head, body = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_weight = name.endswith(".weight")
        is_fc = name.startswith("fc.")
        is_bn = "bn" in name  # BN affine 전부
        if is_fc or (not is_weight) or is_bn:
            head.append(p)
        else:
            body.append(p)
    optimizer = torch.optim.AdamW(
        [{"params": body, "lr": LR_BODY, "weight_decay": WD_BODY},
         {"params": head, "lr": LR_HEAD, "weight_decay": 0.0}])

    def lr_scale(epoch: int) -> float:  # epoch: scheduler.step() 횟수, 0-based
        if epoch < WARMUP:
            return (epoch + 1) / WARMUP
        t = (epoch - WARMUP) / max(1, EPOCHS - WARMUP - 1)
        return 0.5 * (1 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    return optimizer, scheduler


# ── SWA ── 마지막 N 에폭 체크포인트의 동일 가중 평균. 추론 파라미터 집합 1개(모델 1개)
def swa_average(snapshots: list[dict]) -> dict:
    keys = snapshots[0].keys()
    return {k: torch.stack([s[k].float() for s in snapshots]).mean(0).to(snapshots[0][k].dtype)
            for k in keys}


@torch.no_grad()
def reestimate_bn(model: nn.Module, train_data, dev: torch.device) -> None:
    """평균 후 BN running stats 를 학습 데이터 forward 로 다시 누적."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None  # 이동평균 대신 누적평균
    model.train()  # BN 통계 갱신 활성 — no_grad 라 가중치는 불변
    for x, _ in train_data:
        model(x.to(dev))
    model.eval()


def freeze_bn_stats(model: nn.Module) -> None:
    """BN running stats 가 1,020장 소배치 노이즈로 흔들리지 않게 통계 고정(가중치는 학습)."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


# ── [바꿀 곳 5] 학습 루프 ── 120 epoch · 후보(raw_bn vs swa_bn, 둘 다 BN 재추정 공정 비교) + 최고점 채택
def train(model: nn.Module, dev: torch.device, train_tf: transforms.Compose, tag: str = "A",
          merged: bool = False) -> nn.Module:
    """merged: 검증 사진을 학습에 합침(README 30행 허용). 선택 신호는 train 클래스당
    첫 1장(102장)으로 보존 — val은 학습 데이터에 풀 투입.
    v7: raw 후보도 BN 재추정 후 비교(기존엔 SWA만 재추정 → 비교 편향, 패널2 DeepSeek·Big Pickle)."""
    from torch.utils.data import Subset
    if merged:
        sel_ds = labeled("train", eval_transform())
        sel_idx, fit_idx, seen = [], [], set()
        for i, t in enumerate(sel_ds.targets):  # 폴더 순서 결정적 — 클래스당 첫 장을 선택용으로
            (sel_idx if t not in seen else fit_idx).append(i)
            seen.add(t)
        fit = ConcatDataset([Subset(labeled("train", train_tf), fit_idx),
                             labeled("val", train_tf)])
        sel_data = loader(Subset(sel_ds, sel_idx), 64)  # 102장 selection 전용(train 도메인)
        bn_data = ConcatDataset([sel_ds, labeled("val", eval_transform())])
    else:
        fit = labeled("train", train_tf)
        sel_data = loader(labeled("val", eval_transform()))
        bn_data = labeled("train", eval_transform())
    train_data = loader(fit, BATCH, shuffle=True)
    optimizer, scheduler = build_optimizer(model)
    snapshots: list[dict] = []
    for epoch in range(EPOCHS):
        model.train()
        freeze_bn_stats(model)
        for x, y in train_data:
            optimizer.zero_grad()
            loss_fn(model(x.to(dev)), y.to(dev)).backward()
            optimizer.step()
        if scheduler:
            scheduler.step()
        raw_acc = accuracy(model, sel_data, dev)
        if epoch >= EPOCHS - SWA_EPOCHS:
            snapshots.append({k: v.clone() for k, v in model.state_dict().items()})
        print(f"epoch {epoch + 1:2d} [{tag}]  {('내부holdout' if merged else '검증')} {raw_acc:.4f}")
    # 후보 1: 최종 에폭 가중치 + BN 재추정(공정 비교 — 종전 best 롤백은 holdout 노이즈 선택이므로 폐기)
    raw_bn = copy.deepcopy(model)
    reestimate_bn(raw_bn, loader(bn_data, 64), dev)
    raw_acc = accuracy(raw_bn, sel_data, dev)
    # 후보 2: SWA(마지막 40 에폭 동일 가중 평균) + BN 재추정
    swa = copy.deepcopy(model)
    swa.load_state_dict(swa_average(snapshots))
    reestimate_bn(swa, loader(bn_data, 64), dev)
    swa_acc = accuracy(swa, sel_data, dev)
    if swa_acc >= raw_acc:
        best_kind, best_state = "swa+bn", swa.state_dict()
        best_acc = swa_acc
    else:
        best_kind, best_state = "raw+bn", raw_bn.state_dict()
        best_acc = raw_acc
    model.load_state_dict(best_state)
    print(f"[모델 {tag}] 채택: {best_kind}  선택점수 {best_acc:.4f}  (raw_bn {raw_acc:.4f} / swa_bn {swa_acc:.4f})")
    # ── 가중치 영속화: v5 가중치 소실(블랙홀) 재발 방지 (패널2 공통 지적)
    ckpt_dir = HERE / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "kind": best_kind,
                "sel_acc": best_acc, "merged": merged}, ckpt_dir / f"v7_{tag}.pt")
    print(f"[모델 {tag}] 가중치 저장: {ckpt_dir / f'v7_{tag}.pt'}")
    return model


@torch.no_grad()
def tta_logits(models_list: list[nn.Module], images: torch.Tensor, dev: torch.device) -> torch.Tensor:
    """(모델 수)×2뷰×(원본+flip) softmax 확률 합 — 앙상블은 모델 2개까지(룰)."""
    total = torch.zeros(len(images), N_CLASSES, device=dev)
    for m in models_list:
        total += torch.softmax(m(images.to(dev)), 1)
        total += torch.softmax(m(torch.flip(images, dims=[3]).to(dev)), 1)
    return total


@torch.no_grad()
def evaluate_tta(models_list: list[nn.Module], dev: torch.device, split: str = "val") -> float:
    """같은 TTA를 주어 split(train/val 이미지셋)에 적용해 평가 — 선택 기준과
    실제 성능의 간극을 없앰(패널 합의). merged 런이면 val은 학습에 이미 쓰였으므로
    그 경우 split='train' 내부 holdout 점수만 신뢰."""
    ys: list[int] = []
    probs: torch.Tensor | None = None
    for k, view in enumerate((eval_transform(), squish_transform())):
        p_all: list[torch.Tensor] = []
        for x, y in loader(labeled(split, view)):
            p_all.append(tta_logits(models_list, x, dev).cpu())
            if k == 0:
                ys.extend(y.tolist())
        p = torch.cat(p_all)
        probs = p if probs is None else probs + p
    if len(probs) != len(ys):  # 뷰 간 샘플 수 불일치 = 데이터 증감 문제 — 신뢰 불가
        raise ValueError(f"TA 뷰 간 샘플 수 불일치: {len(probs)} vs {len(ys)}")
    hits = (probs.argmax(1) == torch.tensor(ys)).sum().item()
    return hits / len(ys)


@torch.no_grad()
def predict(models_list: list[nn.Module], dev: torch.device) -> tuple[list[str], list[int]]:
    """앙상블(모델 2개) × 뷰 2종 ×(원본+flip) 의 softmax 확률 합."""
    for m in models_list:
        m.eval()
    ids: list[str] = []
    scores: torch.Tensor | None = None
    for k, view in enumerate((eval_transform(), squish_transform())):
        batch_scores = []
        for x, names in loader(TestImages(view)):
            batch_scores.append(tta_logits(models_list, x, dev).cpu())
            if k == 0:
                ids.extend(names)
        view_scores = torch.cat(batch_scores)
        scores = view_scores if scores is None else scores + view_scores
    labels = scores.argmax(1).tolist()
    return ids, labels


def main() -> None:
    dev = device()
    print(f"장치: {dev}")
    # v7: 등록 레시피 유지, EPOCHS 120 + SWA 40. val 합침(README 30행 허용) + train 내부 holdout escolha.
    torch.manual_seed(SEED)
    model_a = train(build_model().to(dev), dev,
                    train_transform(scale=(0.65, 1.0)), tag="A", merged=True)
    torch.manual_seed(SEED + 1)
    model_b = train(build_model().to(dev), dev,
                    train_transform(scale=(0.5, 1.0), trivial=True), tag="B", merged=True)
    output = write_submission(*predict([model_a, model_b], dev))
    print(f"제출 파일: {output}")


if __name__ == "__main__":
    main()
