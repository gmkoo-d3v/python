"""2일차 실습 공통 부분: lab5가 만든 데이터와 best.pt를 찾고, YOLO 라벨을 픽셀 박스로 읽습니다."""
import os
import warnings
from pathlib import Path

os.environ.setdefault("YOLO_VERBOSE", "False")   # ultralytics import 전에 정해야 적용된다
warnings.filterwarnings("ignore", message=".*does not have a deterministic implementation.*")

import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
YOLO_ROOT = HERE / "data" / "PennFudan_yolo"
BEST = HERE / "runs" / "person" / "weights" / "best.pt"
IMAGE_SIZE = 416


def pick_device() -> int | str:
    """cuda는 번호로, 그 밖에는 이름으로 넘긴다(ultralytics 규칙)."""
    return 0 if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def require_lab5() -> None:
    """lab5를 먼저 돌려야 데이터와 best.pt가 생긴다."""
    if not BEST.exists() or not (YOLO_ROOT / "data.yaml").exists():
        raise SystemExit(f"lab5_yolo11.py를 먼저 돌리세요. 없음: {BEST}")


def valid_images() -> list[Path]:
    return sorted((YOLO_ROOT / "valid" / "images").glob("*.png"))


def read_truth(image: Path) -> list[list[float]]:
    """YOLO 라벨(중심, 크기, 0~1)을 xyxy 픽셀 박스로 되돌린다."""
    with Image.open(image) as im:
        width, height = im.size
    label = image.parent.parent / "labels" / f"{image.stem}.txt"
    boxes = []
    for line in label.read_text().strip().splitlines():
        cx, cy, w, h = (float(v) for v in line.split()[1:])
        boxes.append([(cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height])
    return boxes
