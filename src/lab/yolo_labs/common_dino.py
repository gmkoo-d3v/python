"""4일차(DINOv3) 공통 부분: 모델 불러오기, 특징 뽑기, EuroSAT 나누기.

Built with DINOv3. 가중치는 Meta의 DINOv3 License를 따릅니다
(https://ai.meta.com/resources/models-and-libraries/dinov3-license/).
timm에 올라온 가중치를 쓰므로 Hugging Face 로그인 없이 받아집니다.
"""

import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import timm
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "runs" / "dino"
MODEL = "vit_small_patch16_dinov3.lvd1689m"      # ViT-S/16, 2,160만 파라미터
SMALL = "convnext_tiny.dinov3_lvd1689m"          # 합성곱 판, 2,780만 파라미터(20차시)
SEED = 7
POOL_PER_CLASS = 50     # 라벨을 고를 수 있는 사진(클래스마다)
TEST_PER_CLASS = 100    # 시험 사진(클래스마다)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load(name: str = MODEL, pretrained: bool = True) -> torch.nn.Module:
    """분류 머리 없이 특징만 내는 모델. pretrained=False면 같은 구조의 무작위 초기값(견줄 거리)."""
    torch.manual_seed(SEED)
    return timm.create_model(name, pretrained=pretrained, num_classes=0).eval().to(pick_device())


def _tensor(model: torch.nn.Module, images: list[Image.Image], size: tuple[int, int]) -> torch.Tensor:
    """사진을 정한 크기(높이, 너비)로 바꾸고 모델이 배운 평균·표준편차로 정규화한다."""
    cfg = model.pretrained_cfg
    mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"]).view(1, 3, 1, 1)
    arrays = [np.asarray(im.convert("RGB").resize((size[1], size[0]), Image.BICUBIC), np.float32) / 255 for im in images]
    x = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2)
    return ((x - mean) / std).to(pick_device())


@torch.no_grad()
def embed(model: torch.nn.Module, images: list[Image.Image], size: int | tuple[int, int] = 224,
          batch: int = 64) -> np.ndarray:
    """사진마다 특징 벡터 하나(길이 1로 맞춤). 코사인 유사도 = 내적. size는 정사각 한 변 또는 (높이, 너비)."""
    hw = (size, size) if isinstance(size, int) else size
    chunks = []
    for i in range(0, len(images), batch):
        feats = model(_tensor(model, images[i:i + batch], hw)).float().cpu().numpy()
        chunks.append(feats / np.linalg.norm(feats, axis=1, keepdims=True))
    return np.concatenate(chunks)


@torch.no_grad()
def patch_tokens(model: torch.nn.Module, image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    """ViT의 패치마다 특징(높이/16 × 너비/16 × 384). 앞쪽의 CLS·레지스터 토큰은 뺀다."""
    tokens = model.forward_features(_tensor(model, [image], size))[0]
    patches = tokens[model.num_prefix_tokens:].float().cpu().numpy()
    return patches.reshape(size[0] // 16, size[1] // 16, -1)


def pixels(images: list[Image.Image], side: int = 32) -> np.ndarray:
    """견줄 거리: 사진을 작게 줄인 픽셀값 그대로(길이 1로 맞춤)."""
    flat = np.stack([np.asarray(im.convert("RGB").resize((side, side)), np.float32).ravel() for im in images])
    flat -= flat.mean(axis=1, keepdims=True)
    return flat / np.linalg.norm(flat, axis=1, keepdims=True)


def eurosat() -> tuple[list[Image.Image], np.ndarray, list[Image.Image], np.ndarray, list[str]]:
    """EuroSAT(위성사진 10종, 64×64, MIT 라이선스)에서 클래스마다 고를 거리 50장, 시험 100장을 뽑는다."""
    from torchvision.datasets import EuroSAT
    ds = EuroSAT(str(DATA), download=not (DATA / "eurosat" / "2750").exists())
    labels = np.array(ds.targets)
    rng = np.random.default_rng(SEED)
    pool_idx, test_idx = [], []
    for c in range(len(ds.classes)):
        idx = rng.permutation(np.nonzero(labels == c)[0])
        pool_idx += idx[:POOL_PER_CLASS].tolist()
        test_idx += idx[POOL_PER_CLASS:POOL_PER_CLASS + TEST_PER_CLASS].tolist()
    load_image = lambda i: Image.open(ds.samples[i][0]).convert("RGB")
    return ([load_image(i) for i in pool_idx], labels[pool_idx],
            [load_image(i) for i in test_idx], labels[test_idx], ds.classes)


def cached(name: str, make) -> np.ndarray:
    """오래 걸리는 특징 뽑기는 runs/dino에 저장해 두고 다시 쓴다."""
    path = OUT / f"{name}.npy"
    if path.exists():
        return np.load(path)
    OUT.mkdir(parents=True, exist_ok=True)
    value = make()
    np.save(path, value)
    return value
