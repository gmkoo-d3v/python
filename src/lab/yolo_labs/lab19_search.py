"""18차시: 비슷한 것 찾기(사진 검색과 부위 짝짓기).

1) 사진 검색: EuroSAT 시험 사진 1,000장을 하나씩 질문으로 던져, 고를 거리 500장 가운데 특징이 가장 가까운
   5장을 꺼낸다. 5장 중 질문과 같은 종류의 비율(precision@5)을 잰다. 라벨은 채점에만 쓴다.
2) 부위 짝짓기: Penn-Fudan의 서로 다른 두 사람을 세로로 세워 놓고, A의 머리 패치와 발 패치마다
   B에서 특징이 가장 가까운 패치를 찾는다. 그 짝이 B의 같은 부위(위 1/4, 아래 1/4)에 떨어지는 비율을 잰다.
"""
import numpy as np
from PIL import Image

from common_dino import cached, embed, eurosat, load, patch_tokens, pixels
from common_yolo import YOLO_ROOT

PENN = YOLO_ROOT.parent / "PennFudanPed"
TOP_K = 5
PERSON_SIZE = (256, 128)      # 사람 조각(높이, 너비) → 패치 16×8
PAIRS = 60
SEED = 3


def retrieval(pool: np.ndarray, pool_y: np.ndarray, test: np.ndarray, test_y: np.ndarray) -> float:
    nearest = np.argsort(-(test @ pool.T), axis=1)[:, :TOP_K]
    return float((pool_y[nearest] == test_y[:, None]).mean())


def person_crops() -> list[Image.Image]:
    """검증 사진의 사람을 마스크 상자로 잘라 배경을 회색으로 지운다(높이 150픽셀 넘는 사람만)."""
    crops = []
    for path in sorted((YOLO_ROOT / "valid" / "images").glob("*.png")):
        rgb = np.array(Image.open(PENN / "PNGImages" / path.name).convert("RGB"))
        mask = np.array(Image.open(PENN / "PedMasks" / f"{path.stem}_mask.png"))
        for k in range(1, int(mask.max()) + 1):
            ys, xs = np.nonzero(mask == k)
            if len(ys) == 0 or ys.max() - ys.min() < 150:
                continue
            box = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
            crop = np.where((mask[box] == k)[..., None], rgb[box], 128).astype(np.uint8)
            crops.append(Image.fromarray(crop))
    return crops


def pixel_grid(image: Image.Image) -> np.ndarray:
    a = np.asarray(image.resize((PERSON_SIZE[1], PERSON_SIZE[0])), np.float32) / 255
    gh, gw = PERSON_SIZE[0] // 16, PERSON_SIZE[1] // 16
    return a.reshape(gh, 16, gw, 16, 3).transpose(0, 2, 1, 3, 4).reshape(gh, gw, -1)


def part_agreement(grids: list[np.ndarray], pairs: list[tuple[int, int]]) -> tuple[float, float]:
    """A의 머리(위 1/4)·발(아래 1/4) 패치 → B에서 가장 가까운 패치가 같은 띠에 떨어지는 비율."""
    rows = grids[0].shape[0]
    band = rows // 4
    head_ok, feet_ok = [], []
    for a, b in pairs:
        fa = grids[a].reshape(-1, grids[a].shape[-1])
        fb = grids[b].reshape(-1, grids[b].shape[-1])
        fa = fa / (np.linalg.norm(fa, axis=1, keepdims=True) + 1e-9)
        fb = fb / (np.linalg.norm(fb, axis=1, keepdims=True) + 1e-9)
        hit_row = np.argmax(fa @ fb.T, axis=1) // grids[b].shape[1]
        src_row = np.arange(len(fa)) // grids[a].shape[1]
        head_ok += (hit_row[src_row < band] < band).tolist()
        feet_ok += (hit_row[src_row >= rows - band] >= rows - band).tolist()
    return float(np.mean(head_ok)), float(np.mean(feet_ok))


def main() -> None:
    pool_images, pool_y, test_images, test_y, classes = eurosat()
    dino = (cached("eurosat_pool_dino", lambda: embed(load(), pool_images)),
            cached("eurosat_test_dino", lambda: embed(load(), test_images)))
    pix = (pixels(pool_images), pixels(test_images))
    print(f"1) 사진 검색: 질문 {len(test_images)}장, 찾을 곳 {len(pool_images)}장, 가까운 {TOP_K}장 중 같은 종류 비율")
    print(f"   DINOv3 {retrieval(dino[0], pool_y, dino[1], test_y):.3f}   픽셀값 {retrieval(pix[0], pool_y, pix[1], test_y):.3f}"
          f"   (아무렇게나 꺼내면 0.100)")
    nearest = np.argsort(-(dino[1] @ dino[0].T), axis=1)[:, :TOP_K]
    for q in (0, 300, 800):
        print(f"   예: {classes[test_y[q]]:<20} → " + ", ".join(classes[pool_y[i]] for i in nearest[q]))

    crops = person_crops()
    rng = np.random.default_rng(SEED)
    pairs = [tuple(rng.choice(len(crops), 2, replace=False)) for _ in range(PAIRS)]
    model = load()
    grids = {"DINOv3": [patch_tokens(model, c, PERSON_SIZE) for c in crops],
             "픽셀값": [pixel_grid(c) for c in crops]}
    print(f"\n2) 부위 짝짓기: 사람 조각 {len(crops)}개, 서로 다른 두 사람 {PAIRS}쌍, 조각 {PERSON_SIZE[1]}×{PERSON_SIZE[0]}")
    print("   A의 패치 짝이 B의 같은 띠에 떨어지는 비율(띠 높이 1/4이라 아무렇게나 고르면 0.25)")
    for name, g in grids.items():
        head, feet = part_agreement(g, pairs)
        print(f"   {name:<8} 머리 {head:.3f}   발 {feet:.3f}")


if __name__ == "__main__":
    main()
