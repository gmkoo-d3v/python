"""YOLO 덱 실습판 공통 부분: Penn-Fudan 데이터를 6강 노트북과 같은 방식으로 읽고 나눕니다."""
import os
import random
import shutil
import ssl
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

PENN_URL = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"
DOWNLOAD_TIMEOUT_SEC = 180
SPLIT_SEED = 1729
TRAIN_RATIO = 0.8
# 데이터 위치는 환경변수로 바꿀 수 있다. 없으면 실습 폴더 아래 data/에 내려받는다
DATA_ROOT = Path(os.environ.get("PENN_DATA_ROOT", Path(__file__).parent / "data"))


def ssl_context() -> ssl.SSLContext:
    """인증서 목록을 고릅니다. 파이썬 기본 목록만으로는 이 서버의 인증서를 확인하지 못하는 컴퓨터가 있습니다.

    1) truststore가 깔려 있으면 운영체제의 인증서 저장소를 그대로 씁니다(학원·회사 보안 프록시 인증서도 포함).
    2) 없으면 certifi(ultralytics와 함께 깔림)의 목록을 씁니다. 서버 인증서의 뿌리(emSign Root CA - G1)가 들어 있습니다.
    3) 둘 다 없으면 파이썬 기본값.
    """
    try:
        import truststore
        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except ImportError:
        pass
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def penn_root() -> Path:
    """Penn-Fudan 폴더를 돌려줍니다. 없으면 내려받아 풉니다."""
    target = DATA_ROOT / "PennFudanPed"
    if target.exists():
        return target
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_ROOT / "PennFudanPed.zip"
    try:
        with urllib.request.urlopen(PENN_URL, timeout=DOWNLOAD_TIMEOUT_SEC, context=ssl_context()) as response, open(zip_path, "wb") as f:
            shutil.copyfileobj(response, f)
    except urllib.error.URLError as error:
        zip_path.unlink(missing_ok=True)   # 반쯤 받은 파일이 남으면 다음 실행에서 깨진 zip을 풀게 된다
        if not isinstance(error.reason, ssl.SSLCertVerificationError):
            raise
        raise SystemExit(
            f"인증서 확인에 실패해 데이터를 받지 못했습니다: {error.reason}\n"
            "학원·회사 네트워크의 보안 장비가 인증서를 바꿔 끼우는 경우가 많습니다. 둘 중 하나를 하세요.\n"
            "  1) 명령에 --with truststore를 더해 다시 실행합니다(운영체제 인증서 저장소를 씁니다).\n"
            f"  2) 브라우저로 {PENN_URL} 을 받아 {DATA_ROOT} 에 풀고 다시 실행합니다.\n"
            "     다른 곳에 풀었다면 PENN_DATA_ROOT에 PennFudanPed 폴더의 상위 폴더를 알려 줍니다."
        ) from error
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


