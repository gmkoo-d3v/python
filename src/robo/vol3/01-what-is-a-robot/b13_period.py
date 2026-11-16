import argparse

import matplotlib.pyplot as plt
import numpy as np


# plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["font.family"] = "D2Coding"
plt.rcParams["axes.unicode_minus"] = False

목표 = 1.0          # 직선 위 1m 앞까지 간다(1차원 점 로봇)
parser = argparse.ArgumentParser(description="지연·속도 제한 없는 1차원 비례 제어 모델")
parser.add_argument("--gain", type=float, default=4.0, help="비례 이득 (1/s)")
args = parser.parse_args()
if not np.isfinite(args.gain) or args.gain <= 0:
    parser.error("gain은 양의 유한한 수여야 합니다")
이득 = args.gain
끝_시각 = 4.0

그림, 칸 = plt.subplots(figsize=(7, 3.4))
for 주기 in [0.02, 0.2, 0.4, 0.6]:
    x, 기록 = 0.0, [0.0]
    for _ in range(round(끝_시각 / 주기)):
        속도 = 이득 * (목표 - x)            # 판단은 주기마다 한 번
        x = x + 속도 * 주기                 # 다음 판단 때까지 같은 속도로 간다
        기록.append(x)
    기록 = np.array(기록)
    print(f"주기 {주기:.2f}초(한 번에 고치는 비율 {이득 * 주기:.2f}): "
          f"최대 지나침 {max(기록.max() - 목표, 0):.3f}m, 끝 오차 {abs(목표 - 기록[-1]):.3f}m")
    칸.plot(np.arange(len(기록)) * 주기, 기록, "o-", markersize=3, label=f"주기 {주기}초")
칸.axhline(목표, color="gray", linestyle="--", linewidth=1)
칸.set_xlabel("시간(초)")
칸.set_ylabel("위치(m)")
칸.set_ylim(-1.5, 3.5)
칸.legend(fontsize=8, ncol=2)
그림.tight_layout()
그림.savefig("control_period.png", dpi=200)
print("저장: control_period.png")
