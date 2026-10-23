"""
해커톤 공용 도구 — 데이터 읽기, 정확도, 제출 파일. 이 파일은 고치지 않는다.

data/train, data/val 은 <라벨 세 자리>/<id>.jpg 모양이라 ImageFolder 가 그대로 읽는다.
data/test 는 라벨이 없다. 예측을 write_submission 으로 저장해 제출한다.
"""

import csv
import os
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

HERE = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", HERE / "data"))
SUBMISSION = HERE / "submission.csv"
N_CLASSES = 102
SEED = 0
EVAL_BATCH = 64
NUM_WORKERS = 2  # jpg 풀기를 CPU 두 개로 나눈다. 오류가 나면 0 으로
# 사전학습 ResNet 은 ImageNet 사진을 이 평균·표준편차로 정규화해 배웠다. 같은 자로 재야 한다.
IMAGENET_STATS = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def labeled(split: str, transform: transforms.Compose) -> datasets.ImageFolder:
    """split: train · val. 폴더 이름(000~101)이 곧 라벨 번호다."""
    ds = datasets.ImageFolder(DATA_DIR / split, transform=transform)
    if ds.classes != [f"{c:03d}" for c in range(N_CLASSES)]:
        raise RuntimeError(f"{DATA_DIR / split} 의 라벨 폴더가 000~101 이 아니다 — 배포 zip 을 다시 풀 것")
    return ds


class TestImages(Dataset):
    """data/test 의 사진. 라벨 대신 id(파일 이름)를 돌려준다."""

    def __init__(self, transform: transforms.Compose) -> None:
        self.files = sorted((DATA_DIR / "test").glob("*.jpg"))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, str]:
        with Image.open(self.files[i]) as image:
            return self.transform(image.convert("RGB")), self.files[i].stem


def loader(ds: Dataset, batch: int = EVAL_BATCH, shuffle: bool = False) -> DataLoader:
    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(ds, batch, shuffle=shuffle, generator=generator, num_workers=NUM_WORKERS,
                      persistent_workers=NUM_WORKERS > 0)


@torch.no_grad()
def accuracy(model: nn.Module, data: DataLoader, dev: torch.device) -> float:
    model.eval()
    hits = total = 0
    for x, y in data:
        hits += (model(x.to(dev)).argmax(1).cpu() == y).sum().item()
        total += len(y)
    return hits / total


def write_submission(ids: list[str], labels: list[int]) -> Path:
    """id,label 두 열. 시험 사진 전부에 답이 있어야 채점된다."""
    expected = {p.stem for p in (DATA_DIR / "test").glob("*.jpg")}
    if set(ids) != expected or len(ids) != len(expected):
        raise ValueError(f"시험 사진 {len(expected)}장 전부에 한 번씩 답해야 한다 (받은 답 {len(ids)}개)")
    if any(not 0 <= label < N_CLASSES for label in labels):
        raise ValueError(f"라벨은 0~{N_CLASSES - 1} 이어야 한다")
    with SUBMISSION.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])
        writer.writerows(zip(ids, labels))
    return SUBMISSION
