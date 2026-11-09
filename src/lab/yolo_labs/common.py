"""YOLO 덱 실습판 공통 부분: Penn-Fudan 데이터를 6강 노트북과 같은 방식으로 읽고 나눕니다."""
import os
import random
import shutil
import ssl
import urllib.request
import zipfile
from pathlib import Path

import certifi
import numpy as np
from PIL import Image

PENN_URL = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"
DOWNLOAD_TIMEOUT_SEC = 180
# Windows 기본 인증서 저장소에서 Penn-Fudan 서버 체인을 못 찾는 경우가 있어 certifi를 쓴다
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
SPLIT_SEED = 1729
TRAIN_RATIO = 0.8
# 데이터 위치는 환경변수로 바꿀 수 있다. 없으면 실습 폴더 아래 data/에 내려받는다
DATA_ROOT = Path(os.environ.get("PENN_DATA_ROOT", Path(__file__).parent / "data"))


def penn_root() -> Path:
    """Penn-Fudan 폴더를 돌려줍니다. 없으면 내려받아 풉니다."""
    target = DATA_ROOT / "PennFudanPed"
    if target.exists():
        return target
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_ROOT / "PennFudanPed.zip"
    with urllib.request.urlopen(PENN_URL, timeout=DOWNLOAD_TIMEOUT_SEC, context=SSL_CONTEXT) as response, open(zip_path, "wb") as f:
        shutil.copyfileobj(response, f)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(DATA_ROOT)
    return target


def mask_to_boxes(mask_path: Path) -> list[list[int]]:
    """인스턴스 마스크에서 사람별 xyxy 박스를 구합니다. 노트북 1.4절과 같은 규칙입니다."""
    mask = np.array(Image.open(mask_path))
    boxes = []
    for object_id in np.unique(mask)[1:]:   # 0은 배경
        ys, xs = np.where(mask == object_id)
        boxes.append([int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1])
    return boxes


def load_split() -> tuple[list[dict], list[dict]]:
    """샘플을 읽고 노트북과 같은 난수로 섞어 8:2로 나눕니다."""
    root = penn_root()
    samples = []
    for image_path in sorted((root / "PNGImages").glob("*.png")):
        with Image.open(image_path) as image:
            width, height = image.size
        boxes = mask_to_boxes(root / "PedMasks" / f"{image_path.stem}_mask.png")
        samples.append({"path": image_path, "width": width, "height": height, "boxes": boxes})
    shuffled = samples.copy()
    random.Random(SPLIT_SEED).shuffle(shuffled)
    num_train = int(len(shuffled) * TRAIN_RATIO)
    return shuffled[:num_train], shuffled[num_train:]


