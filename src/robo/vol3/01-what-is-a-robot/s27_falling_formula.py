import matplotlib.pyplot as plt
import numpy as np

# plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["font.family"] = "D2Coding"
plt.rcParams["axes.unicode_minus"] = False

중력 = 9.81                                   # b16_falling_box.py와 같은 값
처음_높이 = 1.0                               # box_scene.py: 상자 pos="0 0 1"
바닥_높이 = 0.1                               # 상자 한 변 0.2m → 중심은 바닥 위 0.1m에서 멈춘다


def 공식(t):
    return np.maximum(처음_높이 - 0.5 * 중력 * t ** 2, 바닥_높이)   # 슬라이드의 공식 줄과 같다


# 슬라이드 27쪽 출력에서 그대로 옮긴 시뮬레이터 값(MuJoCo는 여기서 돌리지 못한다)
시각 = np.array([0.0, 0.2, 0.4, 0.6, 1.0])
시뮬 = np.array([1.000, 0.802, 0.211, 0.100, 0.100])
슬라이드_공식 = np.array([1.000, 0.804, 0.215, 0.100, 0.100])   # 슬라이드에 찍힌 공식 값(대조용)
차이 = (공식(시각) - 시뮬) * 1000                                 # 공식 - 시뮬레이터, mm
착지 = np.sqrt(2 * (처음_높이 - 바닥_높이) / 중력)                  # 바닥에 닿는 시각

for t, s, f, p in zip(시각, 시뮬, 공식(시각), 슬라이드_공식):
    print(f"{t:.1f}초: 시뮬 {s:.3f}m  공식 {f:.4f}m (슬라이드 {p:.3f}m)  차이 {(f - s) * 1000:+.1f}mm")
print(f"착지 시각 {착지:.3f}초")

그림 = plt.figure(figsize=(10, 5.6))
그림.text(0.5, 0.95, "시뮬레이터는 같은 물리를 0.002초씩 끊어 계산한다 → 공식과 거의 같다", ha="center", fontsize=13)

# 왼쪽: 높이 대 시간
왼 = 그림.add_axes([0.08, 0.14, 0.52, 0.70])
t = np.linspace(0, 1, 400)
왼.plot(t, 공식(t), color="tab:blue", lw=2.5)
왼.scatter(시각, 시뮬, s=70, color="tab:orange", edgecolor="black", zorder=5)
왼.axhline(바닥_높이, color="gray", ls="--", lw=1)
왼.axvline(착지, color="gray", ls=":", lw=1)
왼.set_xlim(-0.03, 1.05)
왼.set_ylim(0, 1.1)
왼.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
왼.set_yticks([0.1, 0.5, 1.0])
왼.grid(alpha=0.3)
왼.set_xlabel("시간 (초)", fontsize=11)
왼.set_ylabel("상자 중심 높이 (m)", fontsize=11)
왼.text(0.55, 0.93, "공식 (선)\n1 − ½·g·t²", color="tab:blue", fontsize=11, va="center")
왼.text(0.55, 0.74, "시뮬레이터 (점)\n슬라이드 출력값", color="tab:orange", fontsize=11, va="center")
왼.text(1.03, 0.14, "바닥 위 0.1m에서 멈춤\n(접촉 계산)", color="dimgray", fontsize=10, ha="right", va="bottom")
왼.text(착지 + 0.02, 0.40, f"착지\n약 {착지:.2f}초", color="dimgray", fontsize=10, va="center")

# 오른쪽: 차이(mm)
오 = 그림.add_axes([0.69, 0.14, 0.29, 0.70])
오.bar(range(len(시각)), 차이, color="tab:purple", width=0.55)
오.axhline(0, color="black", lw=1)
오.set_xticks(range(len(시각)))
오.set_xticklabels([f"{x:.1f}" for x in 시각])
오.set_xlabel("시간 (초)", fontsize=11)
오.set_ylabel("공식 − 시뮬레이터 (mm)", fontsize=11)
오.set_ylim(0, 5.5)
오.set_yticks([0, 2, 4])
오.grid(axis="y", alpha=0.3)
for 번호, 값 in enumerate(차이):
    오.text(번호, 값 + 0.15, f"{값:.1f}", ha="center", fontsize=10)
오.text(2.5, 5.0, "낙하 중 몇 mm", ha="center", fontsize=10, color="tab:purple")
오.text(3.5, 1.0, "착지 후\n차이 0", ha="center", fontsize=10, color="dimgray")
그림.text(0.37, 0.89, "높이 (슬라이드 숫자)", ha="center", fontsize=11)
그림.text(0.835, 0.89, "차이는 mm", ha="center", fontsize=11)

그림.text(0.5, 0.015, "차이의 이유: 시간을 한 걸음(0.002초)씩 끊어 계산하는 오차 — 슬라이드 출력은 소수 셋째 자리까지만 보여 준다",
          ha="center", fontsize=9, color="gray")
그림.savefig("s27_falling_formula.png", dpi=200)
print("저장: s27_falling_formula.png")
