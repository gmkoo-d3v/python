# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "torchvision", "pillow"]
# ///
"""
Flowers-102 v6 — v5의 학습 설정을 유지하고 선택·저장을 개선한다.

기본 실행: python baseline.py
저장 모델로 재예측: python baseline.py --predict-from runs/<실행 폴더>

train의 클래스당 첫 1장(102장)은 모델 선택 전용이다.
나머지 train+val(1,938장)로 gradient 학습 및 BN 재추정을 한다.
선택은 제출과 동일한 4뷰 TTA를 사용한다. test는 최종 예측에만 사용한다.
ResNet18 IMAGENET1K_V1, 입력 224px, 최종 모델 2개를 유지한다.
실행별 코드·설정·모델·로그·제출 파일을 runs/에 보존한다.
"""

import argparse
import copy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil

import torch
from torch import nn
from torchvision import models, transforms

from common import (IMAGENET_STATS, N_CLASSES, SEED, TestImages, accuracy, device, labeled, loader,
                    write_submission)
from torch.utils.data import ConcatDataset, Subset

IMAGE_SIZE = 224
BATCH = 32
EPOCHS = 60
WARMUP = 2
SWA_EPOCHS = 20         # 마지막 N 에폭 체크포인트 평균(SWA) + BN 재추정
LR_BODY = 2e-4          # merged 2배 데이터에 맞춰 상향 (v4 underfit 대응)
LR_HEAD = 1e-3          # 새로 태어난 머리는 크게
WD_BODY = 0.05
HOLDOUT_PER_CLASS = 1   # train에서만 뽑는 모델 선택용 사진 수(클래스당)


def cpu_state(model: nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def save_checkpoint(path: Path, model: nn.Module, metadata: dict) -> None:
    """CPU 텐서와 기본 자료형만 저장해 weights_only=True로 재로드한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save({'state_dict': cpu_state(model), 'metadata': metadata}, temp)
    temp.replace(path)


def write_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


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
    if not snapshots:
        raise ValueError('평균할 체크포인트가 없습니다')
    return {
        k: (torch.stack([s[k] for s in snapshots]).mean(0)
            if value.is_floating_point() else snapshots[-1][k].clone())
        for k, value in snapshots[0].items()
    }


@torch.no_grad()
def reestimate_bn(model: nn.Module, train_data, dev: torch.device) -> None:
    """fit 데이터로만 BN 갱신. 호출 전 모드와 momentum은 복원한다."""
    modes = {m: m.training for m in model.modules()}
    bn_layers = {m: m.momentum for m in model.modules() if isinstance(m, nn.BatchNorm2d)}
    try:
        model.eval()  # Dropout 등은 끈 채 BN만 통계를 갱신한다.
        for m in bn_layers:
            m.reset_running_stats()
            m.momentum = None
            m.train()
        for x, _ in train_data:
            model(x.to(dev))
    finally:
        for m, momentum in bn_layers.items():
            m.momentum = momentum
        for m, mode in modes.items():
            m.training = mode


def freeze_bn_stats(model: nn.Module) -> None:
    """BN running stats 가 1,020장 소배치 노이즈로 흔들리지 않게 통계 고정(가중치는 학습)."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def prepare_datasets(train_tf: transforms.Compose, merged: bool = False):
    """fit, selection의 두 뷰, fit 전용 BN 데이터를 반환. test는 읽지 않는다."""
    view_transforms = (eval_transform(), squish_transform())
    if merged:
        sel_ds = labeled("train", eval_transform())
        sel_idx, fit_idx, seen = [], [], {}
        for i, target in enumerate(sel_ds.targets):
            count = seen.get(target, 0)
            (sel_idx if count < HOLDOUT_PER_CLASS else fit_idx).append(i)
            seen[target] = count + 1
        if HOLDOUT_PER_CLASS < 1 or any(n <= HOLDOUT_PER_CLASS for n in seen.values()):
            raise ValueError('클래스마다 학습 사진과 selection 사진이 모두 필요합니다')
        fit = ConcatDataset([Subset(labeled("train", train_tf), fit_idx),
                             labeled("val", train_tf)])
        selection = [Subset(labeled('train', tf), sel_idx) for tf in view_transforms]
        bn_data = ConcatDataset([Subset(sel_ds, fit_idx), labeled('val', eval_transform())])
    else:
        fit = labeled("train", train_tf)
        selection = [labeled('val', tf) for tf in view_transforms]
        bn_data = labeled("train", eval_transform())
    return fit, selection, bn_data


def select_candidate(candidates: dict[str, nn.Module], selection_views,
                     dev: torch.device) -> tuple[nn.Module, str, dict[str, float]]:
    """모든 후보를 같은 selection 4뷰로 비교. 동률이면 앞선 후보를 유지."""
    scores = {name: evaluate_tta([m], dev, selection_views) for name, m in candidates.items()}
    name = max(scores, key=scores.get)
    return candidates[name], name, scores


# ── [바꿀 곳 5] 60epoch, selection TTA, raw/평균 각각 BN 재추정 비교
def train(model: nn.Module, dev: torch.device, train_tf: transforms.Compose, tag: str = 'A',
          merged: bool = False, run_dir: Path | None = None) -> nn.Module:
    fit, selection, bn_data = prepare_datasets(train_tf, merged)
    train_data = loader(fit, BATCH, shuffle=True)
    selection_views = [loader(ds, 64) for ds in selection]
    bn_train_data = loader(bn_data, 64)
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
    optimizer, scheduler = build_optimizer(model)
    snapshots: list[dict] = []
    best_acc, best_state, best_epoch = -1.0, None, None
    report = {'tag': tag, 'fit_images': len(fit), 'selection_images': len(selection[0]),
              'bn_images': len(bn_data), 'selection_metric': '4-view TTA', 'epochs': []}
    print(f'[{tag}] fit={len(fit)} selection={len(selection[0])} BN={len(bn_data)} (test 미사용)')
    for epoch in range(EPOCHS):
        model.train()
        freeze_bn_stats(model)
        for x, y in train_data:
            optimizer.zero_grad()
            loss_fn(model(x.to(dev)), y.to(dev)).backward()
            optimizer.step()
        if scheduler:
            scheduler.step()
        raw_acc = accuracy(model, selection_views[0], dev)
        tta_acc = evaluate_tta([model], dev, selection_views)
        if tta_acc > best_acc:
            best_acc, best_epoch, best_state = tta_acc, epoch + 1, cpu_state(model)
            if run_dir is not None:
                save_checkpoint(run_dir / f'{tag}_raw.pt', model,
                                {'kind': 'raw', 'epoch': best_epoch, 'score': best_acc})
        if epoch >= EPOCHS - SWA_EPOCHS:
            snapshots.append(cpu_state(model))
        report['epochs'].append({'epoch': epoch + 1, 'plain': raw_acc, 'tta': tta_acc})
        if run_dir is not None:
            write_json(run_dir / f'{tag}.json', report)
        print(f'epoch {epoch + 1:2d} [{tag}] selection plain {raw_acc:.4f} '
              f'TTA {tta_acc:.4f} best {best_acc:.4f}', flush=True)
    model.load_state_dict(best_state)
    candidates = {'raw': model, 'raw_bn': copy.deepcopy(model)}
    reestimate_bn(candidates['raw_bn'], bn_train_data, dev)
    if snapshots:
        averaged = copy.deepcopy(model)
        averaged.load_state_dict(swa_average(snapshots))
        candidates['swa'] = averaged
        candidates['swa_bn'] = copy.deepcopy(averaged)
        reestimate_bn(candidates['swa_bn'], bn_train_data, dev)
    selected, kind, scores = select_candidate(candidates, selection_views, dev)
    report.update(selected=kind, candidate_scores=scores, raw_best_epoch=best_epoch)
    if run_dir is not None:
        # 후보도 보존하지만 최종 추론은 A.pt와 B.pt 두 개만 사용한다.
        for name, candidate in candidates.items():
            save_checkpoint(run_dir / f'{tag}_{name}.pt', candidate,
                            {'kind': name, 'score': scores[name], 'raw_best_epoch': best_epoch})
        save_checkpoint(run_dir / f'{tag}.pt', selected,
                        {'kind': kind, 'score': scores[kind], 'raw_best_epoch': best_epoch})
        write_json(run_dir / f'{tag}.json', report)
    print(f'[{tag}] 후보 TTA {scores} | 채택 {kind} ({scores[kind]:.4f})', flush=True)
    return selected.eval()


@torch.no_grad()
def tta_logits(models_list: list[nn.Module], images: torch.Tensor, dev: torch.device) -> torch.Tensor:
    """(모델 수)×2뷰×(원본+flip) softmax 확률 합 — 앙상블은 모델 2개까지(룰)."""
    total = torch.zeros(len(images), N_CLASSES, device=dev)
    for m in models_list:
        total += torch.softmax(m(images.to(dev)), 1)
        total += torch.softmax(m(torch.flip(images, dims=[3]).to(dev)), 1)
    return total


@torch.no_grad()
def evaluate_tta(models_list: list[nn.Module], dev: torch.device, view_loaders=None) -> float:
    """명시된 선택셋에 제출과 같은 TTA 적용. merged 실행에서는 val을 평가하지 않는다."""
    for model in models_list:
        model.eval()
    if view_loaders is None:
        view_loaders = [loader(labeled('val', tf)) for tf in (eval_transform(), squish_transform())]
    ys: torch.Tensor | None = None
    probs: torch.Tensor | None = None
    for data in view_loaders:
        p_all, y_all = [], []
        for x, y in data:
            p_all.append(tta_logits(models_list, x, dev).cpu())
            y_all.append(y.cpu())
        if not p_all:
            raise ValueError('선택셋이 비어 있습니다')
        p = torch.cat(p_all)
        labels = torch.cat(y_all)
        if ys is None:
            ys = labels
        elif not torch.equal(ys, labels):
            raise ValueError('TTA 뷰 사이에 샘플 순서/라벨이 다릅니다')
        probs = p if probs is None else probs + p
    if probs is None:
        raise ValueError('평가할 TTA 뷰가 없습니다')
    hits = (probs.argmax(1) == ys).sum().item()
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predict-from', type=Path, help='저장된 A.pt/B.pt로 학습 없이 재예측')
    args = parser.parse_args()
    dev = device()
    print(f"장치: {dev}")
    root = Path(__file__).resolve().parent
    run_dir = root / 'runs' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, run_dir / 'baseline.py')
    if (root / 'submission.csv').exists():
        shutil.copy2(root / 'submission.csv', run_dir / 'previous_submission.csv')
    config = {
        'image_size': IMAGE_SIZE, 'batch': BATCH, 'epochs': EPOCHS, 'warmup': WARMUP,
        'swa_epochs': SWA_EPOCHS, 'lr_body': LR_BODY, 'lr_head': LR_HEAD,
        'weight_decay_body': WD_BODY, 'holdout_per_class': HOLDOUT_PER_CLASS,
        'seeds': [SEED, SEED + 1], 'merged': True, 'model': 'resnet18',
        'weights': 'IMAGENET1K_V1', 'selection_metric': '4-view TTA',
        'tta': ['center', 'center_flip', 'squish', 'squish_flip'],
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'torch': str(torch.__version__), 'device': str(dev),
        'predict_from': str(args.predict_from.resolve()) if args.predict_from else None,
    }
    write_json(run_dir / 'config.json', config)
    print(f'실행 기록: {run_dir}', flush=True)
    if args.predict_from:
        models_list = load_models(args.predict_from, dev)
    else:
        torch.manual_seed(SEED)
        model_a = train(build_model().to(dev), dev,
                        train_transform(scale=(0.65, 1.0)), tag='A', merged=True, run_dir=run_dir)
        torch.manual_seed(SEED + 1)
        model_b = train(build_model().to(dev), dev,
                        train_transform(scale=(0.5, 1.0), trivial=True), tag='B', merged=True, run_dir=run_dir)
        models_list = [model_a, model_b]
    submission = write_submission(*predict(models_list, dev))
    shutil.copy2(submission, run_dir / 'submission.csv')
    print(f'제출 파일: {submission}\n보존 사본: {run_dir / "submission.csv"}')


def load_models(run_dir: Path, dev: torch.device) -> list[nn.Module]:
    """자체 생성한 두 최종 체크포인트만 로드. 사전학습 다운로드를 호출하지 않는다."""
    models_list = []
    for tag in ('A', 'B'):
        saved = torch.load(run_dir / f'{tag}.pt', map_location='cpu', weights_only=True)
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, N_CLASSES)
        model.load_state_dict(saved['state_dict'])
        models_list.append(model.to(dev).eval())
        print(f'[{tag}] 저장 모델 로드: {saved["metadata"]}')
    return models_list


if __name__ == "__main__":
    main()
