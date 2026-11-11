"""16차시: 라벨 없이 배운 특징 들여다보기(패치 특징 PCA, 그리고 한 번 찍기).

DINOv3는 사람 라벨 없이 사진만 보고 배웠다. 패치마다 나온 384차원 특징을 PCA로 3차원까지 줄여 색으로
칠하면 사람이 배경과 다른 색으로 떠오른다(runs/dino/pca_*.png).
1) PCA 첫째 성분의 부호로 '사람 자리'를 고르면? → 거리 사진에서는 첫째 성분이 하늘·땅을 가른다.
2) 사람 위의 패치 하나를 찍고(클릭 한 번), 특징이 그와 비슷한 패치를 모으면? → 사람들이 잡힌다.
둘 다 Penn-Fudan 마스크와 IoU로 재고, 픽셀값 패치와 견준다.
"""
import numpy as np
from PIL import Image

from common_dino import MODEL, OUT, load, patch_tokens
from common_yolo import YOLO_ROOT

PENN = YOLO_ROOT.parent / "PennFudanPed"
SIZE = (448, 448)          # 28×28 패치
SHOW = 3                   # 그림으로 남길 사진 수
THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5)


def person_grid(stem: str, grid: tuple[int, int]) -> np.ndarray:
    """정답 마스크(사람 모두)를 패치 격자 크기로 줄인다. 패치의 절반 넘게 사람이면 사람 칸."""
    mask = np.array(Image.open(PENN / "PedMasks" / f"{stem}_mask.png")) > 0
    small = Image.fromarray(mask.astype(np.uint8) * 255).resize((grid[1], grid[0]), Image.BOX)
    return np.asarray(small) > 127


def pixel_patches(image: Image.Image) -> np.ndarray:
    """견줄 거리: 16×16 패치의 RGB 값 768개를 그대로 특징으로."""
    a = np.asarray(image.convert("RGB").resize((SIZE[1], SIZE[0])), np.float32) / 255
    gh, gw = SIZE[0] // 16, SIZE[1] // 16
    return a.reshape(gh, 16, gw, 16, 3).transpose(0, 2, 1, 3, 4).reshape(gh, gw, -1)


def foreground(feats: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """모든 사진의 패치를 모아 PCA. 첫째 성분 > 0을 전경으로 보되, 사진 테두리가 배경 쪽이 되게 부호를 정한다."""
    flat = np.concatenate([f.reshape(-1, f.shape[-1]) for f in feats])
    mean = flat.mean(axis=0)
    _, _, vt = np.linalg.svd(flat - mean, full_matrices=False)
    masks, colors = [], []
    for f in feats:
        proj = (f - mean) @ vt[:3].T                          # (높이, 너비, 3)
        border = np.r_[proj[0, :, 0], proj[-1, :, 0], proj[:, 0, 0], proj[:, -1, 0]]
        first = proj[..., 0] * (-1 if border.mean() > 0 else 1)
        masks.append(first > 0)
        lo, hi = proj.min(axis=(0, 1)), proj.max(axis=(0, 1))
        colors.append(((proj - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8))
    return masks, colors


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max((a | b).sum(), 1))


def click(stem: str, grid: tuple[int, int]) -> tuple[int, int]:
    """첫째 사람의 몸 가운데 패치(사람이 화면을 한 번 찍는 셈). 정답은 찍을 자리에만 쓴다."""
    mask = np.array(Image.open(PENN / "PedMasks" / f"{stem}_mask.png")) == 1
    ys, xs = np.nonzero(mask)
    return int(np.median(ys) * grid[0] / mask.shape[0]), int(np.median(xs) * grid[1] / mask.shape[1])


def similar_to_click(feats: np.ndarray, at: tuple[int, int], thr: float) -> np.ndarray:
    """사진 안에서 평균을 빼고(위치에 따라 천천히 변하는 성분을 덜어 냄) 찍은 패치와 코사인 유사도 > thr."""
    g = feats.reshape(-1, feats.shape[-1]).astype(np.float64)
    g -= g.mean(axis=0)
    g /= np.linalg.norm(g, axis=1, keepdims=True) + 1e-9
    sim = g @ g[at[0] * feats.shape[1] + at[1]]
    return sim.reshape(feats.shape[:2]) > thr


def main() -> None:
    images = sorted((YOLO_ROOT / "valid" / "images").glob("*.png"))
    photos = [Image.open(PENN / "PNGImages" / p.name).convert("RGB") for p in images]
    grid = (SIZE[0] // 16, SIZE[1] // 16)
    truth = [person_grid(p.stem, grid) for p in images]
    model = load()
    clicks = [click(p.stem, grid) for p in images]
    features = {"DINOv3": [patch_tokens(model, im, SIZE) for im in photos],
                "픽셀값": [pixel_patches(im) for im in photos]}
    print("1) PCA 첫째 성분 > 0을 사람 자리로 볼 때, 정답 마스크와의 IoU 평균")
    for name, feats in features.items():
        masks, colors = foreground(feats)
        print(f"   {name:<8} {np.mean([iou(m, t) for m, t in zip(masks, truth)]):.3f}")
        if name.startswith("DINOv3"):
            OUT.mkdir(parents=True, exist_ok=True)
            for k in range(SHOW):
                big = Image.fromarray(colors[k]).resize((SIZE[1], SIZE[0]), Image.NEAREST)
                pair = Image.new("RGB", (SIZE[1] * 2, SIZE[0]))
                pair.paste(photos[k].resize((SIZE[1], SIZE[0])), (0, 0))
                pair.paste(big, (SIZE[1], 0))
                pair.save(OUT / f"pca_{k}.png")
    print("\n2) 사람 위 패치 하나를 찍고, 그와 비슷한 패치(코사인 유사도 > 기준)를 모을 때의 IoU 평균")
    print("   기준        " + "  ".join(f"{t:>5}" for t in THRESHOLDS))
    for name, feats in features.items():
        row = [np.mean([iou(similar_to_click(f, c, t), m) for f, c, m in zip(feats, clicks, truth)]) for t in THRESHOLDS]
        print(f"   {name:<8}  " + "  ".join(f"{v:5.3f}" for v in row))
    print(f"\n모델 {MODEL}, 사진 {SIZE[1]}×{SIZE[0]} → 패치 {grid[1]}×{grid[0]}, 사진 {len(images)}장. "
          f"그림: {OUT}/pca_0~{SHOW - 1}.png")
    print("특징은 라벨을 한 번도 보지 않았다. 그래도 한 사람을 찍으면 같은 '사람다움'을 가진 패치가 함께 잡힌다.")


if __name__ == "__main__":
    main()
