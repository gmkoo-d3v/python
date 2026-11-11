"""17차시: 라벨 몇 장으로 분류하기(k-NN과 선형 분류).

EuroSAT 위성사진 10종(밭, 숲, 고속도로, 주거지, 강, 바다…)에서 클래스마다 라벨 1·5·20·50장만 주고
나머지 1,000장(클래스마다 100장)을 맞힌다. DINOv3는 위성사진을 따로 배우지 않았다(lvd1689m = 일반 사진).
같은 일을 세 가지 특징으로 해서 견준다.
  - DINOv3 특징(라벨 없이 배운 것)
  - 같은 구조의 무작위 초기값 ViT 특징(배운 것이 없을 때)
  - 32×32로 줄인 픽셀값
라벨을 무작위로 고르는 것을 5번 되풀이해 평균과 흔들림을 본다.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

from common_dino import MODEL, cached, embed, eurosat, load, pixels

SHOTS = (1, 5, 20, 50)
REPEATS = 5


def few_shot(pool: np.ndarray, pool_y: np.ndarray, test: np.ndarray, test_y: np.ndarray, shots: int,
             seed: int) -> tuple[float, float]:
    """클래스마다 shots장을 골라 1-NN(코사인)과 로지스틱 회귀로 시험 사진을 맞힌다."""
    rng = np.random.default_rng(seed)
    pick = np.concatenate([rng.choice(np.nonzero(pool_y == c)[0], shots, replace=False) for c in np.unique(pool_y)])
    x, y = pool[pick], pool_y[pick]
    knn = float((y[np.argmax(test @ x.T, axis=1)] == test_y).mean())
    linear = LogisticRegression(max_iter=3000, C=10.0).fit(x, y) if shots > 1 else None
    lin = float((linear.predict(test) == test_y).mean()) if linear else float("nan")
    return knn, lin


def main() -> None:
    pool_images, pool_y, test_images, test_y, classes = eurosat()
    print(f"EuroSAT {len(classes)}종: {', '.join(classes)}")
    print(f"라벨을 고를 사진 {len(pool_images)}장, 시험 사진 {len(test_images)}장(클래스마다 100장)\n")
    feats = {
        "DINOv3": (cached("eurosat_pool_dino", lambda: embed(load(), pool_images)),
                   cached("eurosat_test_dino", lambda: embed(load(), test_images))),
        "무작위 ViT": (cached("eurosat_pool_rand", lambda: embed(load(pretrained=False), pool_images)),
                     cached("eurosat_test_rand", lambda: embed(load(pretrained=False), test_images))),
        "픽셀값": (pixels(pool_images), pixels(test_images)),
    }
    print("정확도(5번 평균 ± 표준편차). 아무렇게나 찍으면 0.10")
    print(f"{'특징':<10}{'방법':<8}" + "".join(f"{f'라벨 {s}장':>15}" for s in SHOTS))
    for name, (pool, test) in feats.items():
        runs = {s: [few_shot(pool, pool_y, test, test_y, s, seed) for seed in range(REPEATS)] for s in SHOTS}
        for k, method in enumerate(("1-NN", "선형")):
            cells = []
            for s in SHOTS:
                vals = np.array([r[k] for r in runs[s]])
                cells.append("-" if np.isnan(vals).all() else f"{vals.mean():.3f} ± {vals.std():.3f}")
            print(f"{name if k == 0 else '':<10}{method:<8}" + "".join(f"{c:>15}" for c in cells))
    print(f"\n모델 {MODEL}. 선형 분류는 라벨 1장일 때 클래스마다 점 하나라 1-NN과 같아서 뺐다.")
    print("라벨 50장은 고를 거리 전부를 쓰는 것이라 되풀이해도 같은 결과(흔들림 0)다.")


if __name__ == "__main__":
    main()
