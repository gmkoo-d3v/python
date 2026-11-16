import matplotlib.pyplot as plt
import numpy as np

# plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["font.family"] = "D2Coding"
plt.rcParams["axes.unicode_minus"] = False

# b20_friction.py 와 같은 조건: 처음 속도 1 m/s, 중력 9.81
속도, 중력 = 1.0, 9.81
마찰_목록 = [1.0, 0.5, 0.2]
# 슬라이드 59쪽에 출력된 MuJoCo 결과를 그대로 옮긴 값 (여기서는 MuJoCo를 돌리지 않는다)
시뮬레이터 = {1.0: 0.048, 0.5: 0.101, 0.2: 0.254}


def 공식(마찰):
    return 속도 ** 2 / (2 * 마찰 * 중력)             # 미끄러져 멈추는 거리 = 속도² / (2 × 마찰 × 중력)


for 마찰 in 마찰_목록:
    print(f"마찰 {마찰}: 시뮬레이터 {시뮬레이터[마찰]:.3f}m, 공식 {공식(마찰):.4f}m, 차이 {시뮬레이터[마찰] - 공식(마찰):+.4f}m")
print(f"거리 비 (마찰 0.2 / 1.0): 시뮬레이터 {시뮬레이터[0.2] / 시뮬레이터[1.0]:.2f}배, 공식 {공식(0.2) / 공식(1.0):.2f}배")

그림 = plt.figure(figsize=(10, 6))
칸 = 그림.add_axes([0.09, 0.17, 0.88, 0.70])
μ = np.linspace(0.15, 1.1, 300)
칸.plot(μ, 공식(μ), color="gray", linewidth=2)
칸.text(0.30, 0.40, "회색 곡선: 공식  d = v² / (2·μ·g)", color="dimgray", fontsize=12)
칸.scatter(마찰_목록, [시뮬레이터[m] for m in 마찰_목록], s=90, color="tab:orange", zorder=3)
칸.text(0.30, 0.35, "주황 점: 시뮬레이터가 낸 거리", color="tab:orange", fontsize=12)

# 점마다 옆 빈 자리에 시뮬레이터 값과 공식 값
글자_자리 = {1.0: (0.74, 0.145), 0.5: (0.50, 0.20), 0.2: (0.27, 0.27)}
for 마찰 in 마찰_목록:
    x, y = 글자_자리[마찰]
    칸.text(x, y, f"μ = {마찰}\n시뮬레이터 {시뮬레이터[마찰]:.3f} m\n공식 {공식(마찰):.3f} m",
          color="tab:orange", fontsize=11, ha="left", va="center")
    if 마찰 != 0.2:
        칸.plot([x + 0.03, 마찰], [y - 0.042, 시뮬레이터[마찰] + 0.006], color="tab:orange", linewidth=0.8)

# 마찰 0.2 와 1.0 의 비교: 세로 화살표
칸.annotate("", xy=(1.08, 시뮬레이터[0.2]), xytext=(1.08, 시뮬레이터[1.0]),
          arrowprops=dict(arrowstyle="<->", color="tab:purple", linewidth=2, shrinkA=0, shrinkB=0))
칸.text(1.045, 0.31, f"마찰 5배 차이\n→ 거리 {시뮬레이터[0.2] / 시뮬레이터[1.0]:.1f}배\n(공식 {공식(0.2) / 공식(1.0):.0f}배)",
      color="tab:purple", fontsize=11, ha="right", va="center")

칸.set_xlim(0, 1.15)
칸.set_ylim(0, 0.45)
칸.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
칸.set_xlabel("마찰 계수 μ (넣은 숫자)")
칸.set_ylabel("미끄러진 거리 (m)")
칸.grid(alpha=0.3)
그림.text(0.5, 0.93, "마찰 계수 하나만 바꾸면: 1 m/s로 민 상자가 미끄러지는 거리", ha="center", fontsize=13)
그림.text(0.5, 0.045, "두 번 돌려도 끝자리까지 같다 (x = 0.436 m, b19)  ·  시뮬레이터는 넣은 숫자만큼만 맞다",
        ha="center", fontsize=11, color="dimgray")

그림.savefig("s59_friction.png", dpi=200)
print("저장: s59_friction.png")
