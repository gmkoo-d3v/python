"""20차시: 작은 모델로 옮기기(ViT-S와 ConvNeXt-T), 그리고 금요일 예고.

DINOv3는 큰 ViT(7B)에서 배운 것을 작은 모델에 옮겨(증류) 함께 내놓았다. 합성곱 판 ConvNeXt-T도 있다.
같은 일(EuroSAT 라벨 5장·20장 선형 분류)을 두 모델로 하고, CPU와 가속기에서 1초에 몇 장을 처리하는지 잰다.
Jetson으로 옮길 때는 2일차 lab10처럼 ONNX로 내보낸 뒤 TensorRT로 바꾼다(로봇 과정 22장).
"""

import time

import numpy as np
import torch

from common_dino import MODEL, SMALL, cached, embed, eurosat, load, pick_device
from lab18_fewshot import REPEATS, few_shot

BATCH = 32
WARMUP = 2
TIMED = 5


@torch.no_grad()
def speed(model: torch.nn.Module, device: torch.device) -> float:
    """224×224 사진 BATCH장을 TIMED번 넣어 1초에 몇 장인지."""
    model = model.to(device)
    x = torch.randn(BATCH, 3, 224, 224, device=device)
    for _ in range(WARMUP):
        model(x)
    sync = (
        torch.mps.synchronize
        if device.type == "mps"
        else (torch.cuda.synchronize if device.type == "cuda" else lambda: None)
    )
    sync()
    start = time.perf_counter()
    for _ in range(TIMED):
        model(x)
    sync()
    return BATCH * TIMED / (time.perf_counter() - start)


def main() -> None:
    pool_images, pool_y, test_images, test_y, _ = eurosat()
    accel = pick_device()
    print(
        f"{'모델':<34}{'파라미터':>10}{'CPU 장/초':>12}{f'{accel.type} 장/초':>12}{'라벨 5장':>10}{'라벨 20장':>10}"
    )
    for name, key in [(MODEL, "dino"), (SMALL, "convnext")]:
        model = load(name)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        pool = cached(f"eurosat_pool_{key}", lambda: embed(model, pool_images))
        test = cached(f"eurosat_test_{key}", lambda: embed(model, test_images))
        acc = [
            np.mean(
                [
                    few_shot(pool, pool_y, test, test_y, s, seed)[1]
                    for seed in range(REPEATS)
                ]
            )
            for s in (5, 20)
        ]
        cpu = speed(load(name), torch.device("cpu"))
        fast = speed(model, accel) if accel.type != "cpu" else cpu
        print(
            f"{name:<34}{params:>9.1f}M{cpu:>12.0f}{fast:>12.0f}{acc[0]:>10.3f}{acc[1]:>10.3f}"
        )
    print(
        "\n장/초는 이 컴퓨터에서 잰 값이라 기계마다 다르다. 정확도는 선형 분류, 5번 평균."
    )


if __name__ == "__main__":
    main()
