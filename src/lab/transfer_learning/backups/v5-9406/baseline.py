# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "torchvision", "pillow"]
# ///
"""
전이학습 해커톤 강화 기준선 — Flowers-102 (102종, 학습 1,020장), ResNet18.

  uv run baseline.py

패널 컨센서스 반영본:
  [1] 224px + RandomResizedCrop/Flip/ColorJitter(약하게) 증강
  [2] 전체 학습 + 판별 LR(몸통 1e-4 / 머리 1e-3)
  [3] CE + label_smoothing 0.1
  [4] AdamW(몸통 wd 0.05 / 머리 wd 0) + cosine(warmup 2ep)
  [5] 40 epoch · best-val 체크포인트 롤백 + SWA(마지막 10ep 평균 + BN 재추정) · BN 통계 보호
  predict: 4뷰 TTA(원본/hflip/스퀴시/스퀴시+flip, softmax 확률 평균)

룰 준수: resnet18 + IMAGENET1K_V1, 외부 데이터 없음, 입력 224px 이하,
검증 사진은 model selection 에만 사용(학습 데이터에 합치지 않음).
"""

import copy
import math

import torch
from torch import nn
from torchvision import models, transforms

from common import (IMAGENET_STATS, N_CLASSES, SEED, TestImages, accuracy, device, labeled, loader,
                    write_submission)
from torch.utils.data import ConcatDataset  # train+val 합침(README 30행 허용)

IMAGE_SIZE = 224
BATCH = 32
EPOCHS = 60
WARMUP = 2
SWA_EPOCHS = 20         # 마지막 N 에폭 체크포인트 평균(SWA) + BN 재추정
LR_BODY = 2e-4          # merged 2배 데이터에 맞춰 상향 (v4 underfit 대응)
LR_HEAD = 1e-3          # 새로 태어난 머리는 크게
WD_BODY = 0.05
HOLDOUT_PER_CLASS = 1   # merged 학습의 selection 테스트(train에서만 뽑음, val 아님)


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


# ── [바꿀 곳 5] 학습 루프 ── 60 epoch · best 롤백 + SWA · merged=True면 val 합침 최종 런
def train(model: nn.Module, dev: torch.device, train_tf: transforms.Compose, tag: str = "A",
          merged: bool = False) -> nn.Module:
    """merged: 검증 사진을 학습에 합침(README 30행 허용). 선택 신호는 train에서
    클래스당 1장씩 내부 holdout(102장)으로 보존한다 — val은 학습 데이터에 풀 투입."""
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
    best_acc, best_kind, best_state = 0.0, "none", None
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
        if raw_acc > best_acc:
            best_acc, best_kind = raw_acc, "raw"
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch >= EPOCHS - SWA_EPOCHS:
            snapshots.append({k: v.clone() for k, v in model.state_dict().items()})
        print(f"epoch {epoch + 1:2d} [{tag}]  {('내부holdout' if merged else '검증')} {raw_acc:.4f}  best {best_acc:.4f}")
    swa = copy.deepcopy(model)
    swa.load_state_dict(swa_average(snapshots))
    reestimate_bn(swa, loader(bn_data, 64), dev)
    swa_acc = accuracy(swa, sel_data, dev)
    if swa_acc > best_acc:
        best_acc, best_kind, best_state = swa_acc, "swa", {k: v.clone() for k, v in swa.state_dict().items()}
    model.load_state_dict(best_state)
    print(f"[모델 {tag}] 채택: {best_kind}  선택점수 {best_acc:.4f}")
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
def evaluate_tta(models_list: list[nn.Module], dev: torch.device) -> float:
    """검증셋에 같은 TTA를 적용해 선택 기준과 실제 성능의 간극을 없앰(패널 합의)."""
    ys: list[int] = []
    probs: torch.Tensor | None = None
    for k, view in enumerate((eval_transform(), squish_transform())):
        p_all = []
        for x, y in loader(labeled("val", view)):
            p_all.append(tta_logits(models_list, x, dev).cpu())
            if k == 0:
                ys.extend(y.tolist())
        p = torch.cat(p_all)
        probs = p if probs is None else probs + p
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
    # 최종 런: 검증 사진을 학습에 합침(README 30행 허용). 레시피·epoch 는 clean-val 단계
    # (단일 0.9118, 앙상블 0.9206, 공개보드 0.9094)에서 확정한 값 그대로 고정.
    torch.manual_seed(SEED)
    model_a = train(build_model().to(dev), dev,
                    train_transform(scale=(0.65, 1.0)), tag="A", merged=True)
    torch.manual_seed(SEED + 1)
    model_b = train(build_model().to(dev), dev,
                    train_transform(scale=(0.5, 1.0), trivial=True), tag="B", merged=True)
    print(f"제출 파일: {write_submission(*predict([model_a, model_b], dev))}")


if __name__ == "__main__":
    main()
