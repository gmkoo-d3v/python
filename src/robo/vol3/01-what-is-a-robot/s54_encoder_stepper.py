import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch

# plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["font.family"] = "D2Coding"
plt.rcParams["axes.unicode_minus"] = False

# b08_parts_numbers.py 의 숫자를 그대로 옮긴다
바퀴_반지름 = 0.035                  # m
한바퀴_눈금 = 1024
눈금당_거리 = 2 * np.pi * 바퀴_반지름 / 한바퀴_눈금
걸음_각도_도 = 1.8
조각_수 = 16
예_각도 = 90
눈금_mm = 눈금당_거리 * 1000
한바퀴_걸음 = 360 / 걸음_각도_도
예_걸음 = 예_각도 / 걸음_각도_도
조각_각도 = 걸음_각도_도 / 조각_수
센_눈금 = int(0.5 / 눈금당_거리)
print(f"눈금 하나 = {눈금_mm:.3f}mm")
print(f"0.5m 굴렀을 때 센 눈금 {센_눈금}개 → 계산한 거리 {센_눈금 * 눈금당_거리:.5f}m")
print(f"한 바퀴 = {한바퀴_걸음:.0f}걸음, 90도 = {예_걸음:.0f}걸음")
print(f"한 걸음을 {조각_수}조각으로 나누면(마이크로스텝) 한 바퀴 = {한바퀴_걸음 * 조각_수:.0f}조각, 조각 하나 = {조각_각도:.4f}도")

그림 = plt.figure(figsize=(10, 5.2))
왼 = 그림.add_axes([0.07, 0.14, 0.42, 0.70])
원 = 그림.add_axes([0.53, 0.30, 0.44, 0.55])
막대 = 그림.add_axes([0.75, 0.22, 0.22, 0.12])

# --- 엔코더: 계단 ---
개수 = 5                               # 보여 줄 눈금 수
x = np.linspace(0, 개수, 1000) * 눈금당_거리
센_거리 = np.floor(x / 눈금당_거리 + 1e-9) * 눈금당_거리
왼.plot(x / 눈금당_거리, x / 눈금당_거리, color="gray", linestyle="--", lw=1.5)
왼.plot(x / 눈금당_거리, 센_거리 / 눈금당_거리, color="tab:blue", lw=2.5)
왼.text(0.15, 3.3, "실제 거리\n(매끄러운 대각선)", color="dimgray", fontsize=11, ha="left")
왼.text(3.1, 0.9, "엔코더가 보고하는 거리\n(계단)", color="tab:blue", fontsize=11, ha="left", va="center")
왼.annotate("", xy=(3.0, 3.0), xytext=(3.0, 2.0), arrowprops=dict(arrowstyle="<->", color="tab:green", lw=1.8))
왼.text(3.12, 2.45, f"눈금 하나\n= {눈금_mm:.3f} mm", color="tab:green", fontsize=11, ha="left", va="center")
왼.annotate("", xy=(3.97, 4.0), xytext=(3.97, 3.0), arrowprops=dict(arrowstyle="<->", color="tab:red", lw=1.8))
왼.text(4.12, 3.45, "가장 큰 오차\n= 눈금 하나", color="tab:red", fontsize=11, ha="left", va="center")
왼.set_xlim(0, 6.2)
왼.set_ylim(0, 5)
왼.set_xticks(range(개수 + 1))
왼.set_yticks(range(개수 + 1))
왼.set_xlabel("실제로 굴러간 거리 (눈금 개수만큼)")
왼.set_ylabel("엔코더가 센 눈금 (개)")
왼.grid(alpha=0.3)
왼.set_aspect("equal")

# --- 스테핑 모터: 원 + 눈금 ---
for d in np.arange(한바퀴_걸음) * 걸음_각도_도:
    a = np.deg2rad(d)
    강조 = d < 예_각도 - 1e-9
    원.plot([np.cos(a), 1.1 * np.cos(a)], [np.sin(a), 1.1 * np.sin(a)],
            color="tab:orange" if 강조 else "gray", lw=1.2 if 강조 else 0.7)
t = np.linspace(0, 2 * np.pi, 300)
원.plot(np.cos(t), np.sin(t), color="gray", lw=1)
호 = np.deg2rad(np.linspace(0, 예_각도, 60))
원.plot(0.6 * np.cos(호), 0.6 * np.sin(호), color="tab:orange", lw=3)
원.plot([0, 1.0], [0, 0], color="dimgray", lw=1)
원.plot([0, 0], [0, 1.0], color="dimgray", lw=1)
원.text(1.4, 0.75, f"{예_각도}° = {예_걸음:.0f}걸음", color="tab:orange", fontsize=12, ha="left", va="center")
원.annotate("", xy=(0.8, 0.62), xytext=(1.38, 0.75), arrowprops=dict(arrowstyle="-", color="tab:orange"))
원.text(0, -0.4, f"한 바퀴\n= {한바퀴_걸음:.0f}걸음", ha="center", va="center", fontsize=11, color="dimgray")
원.plot([1.0, 1.1], [0, 0], color="tab:purple", lw=3)
원.text(1.4, 0.3, f"한 걸음 = {걸음_각도_도}°", color="tab:purple", fontsize=11, ha="left", va="center")
원.annotate("", xy=(1.1, -0.01), xytext=(1.38, 0.27), arrowprops=dict(arrowstyle="-", color="tab:purple"))
원.set_xlim(-1.3, 2.6)
원.set_ylim(-1.2, 1.2)
원.set_aspect("equal")
원.set_axis_off()

# --- 확대: 한 걸음 = 16조각 ---
막대.set_xlim(-0.05, 걸음_각도_도 + 0.05)
막대.set_ylim(0, 1)
for k in range(조각_수 + 1):
    끝 = k in (0, 조각_수)
    막대.plot([k * 조각_각도] * 2, [0.8, 0.5 if 끝 else 0.62], color="tab:purple", lw=1.8 if 끝 else 0.9)
막대.plot([0, 걸음_각도_도], [0.8, 0.8], color="tab:purple", lw=1.8)
막대.text(0, 0.42, "0°", color="tab:purple", fontsize=11, ha="center", va="top")
막대.text(걸음_각도_도, 0.42, f"{걸음_각도_도}°", color="tab:purple", fontsize=11, ha="center", va="top")
막대.text(걸음_각도_도 / 2, -0.1, f"확대: 한 걸음을\n{조각_수}조각으로\n조각 하나 = {조각_각도:.4f}°", color="tab:purple",
        fontsize=11, ha="center", va="top")
막대.set_axis_off()
for 끝_각 in (0, 걸음_각도_도):
    그림.add_artist(ConnectionPatch((1.1, 0.0), (끝_각, 0.8), "data", "data", axesA=원, axesB=막대,
                                   color="tab:purple", lw=0.8, linestyle=":"))

그림.text(0.28, 0.93, "엔코더: 눈금 단위로만 안다", ha="center", fontsize=12)
그림.text(0.75, 0.93, "스테핑 모터: 걸음 단위로만 돈다", ha="center", fontsize=12)
그림.savefig("s54_encoder_stepper.png", dpi=200)
print("저장: s54_encoder_stepper.png")
