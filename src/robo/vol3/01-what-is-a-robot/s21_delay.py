import matplotlib.pyplot as plt
import numpy as np

# plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["font.family"] = "D2Coding"
plt.rcParams["axes.unicode_minus"] = False

# b14_delay.py 의 반복문을 그대로 옮긴다 (늦음 0.2초만)
목표, 이득, 주기 = 1.0, 4.0, 0.02
늦음 = 0.2
늦은_박자 = round(늦음 / 주기)
x, 기록, 명령 = 0.0, [0.0], []
for _ in range(round(6.0 / 주기)):
    잰_x = 기록[max(len(기록) - 1 - 늦은_박자, 0)]
    명령.append(이득 * (목표 - 잰_x))          # 로봇이 내리는 속도 명령
    x = x + 이득 * (목표 - 잰_x) * 주기
    기록.append(x)
기록 = np.array(기록)
명령 = np.array(명령)
시각 = np.arange(len(기록)) * 주기
잰값 = np.concatenate([np.zeros(늦은_박자), 기록[:-늦은_박자]])   # 로봇이 받는 측정 =늦음만큼 옛날 위치

번 = int(np.argmax(기록 >= 목표))               # 처음 목표에 닿은 박자
도착 = 시각[번]
믿음 = 잰값[번]
최대 = 기록.max() - 목표
꼭대기 = 시각[np.argmax(기록)]
print(f"늦음 {늦음}초: 최대 지나침 {최대:.3f}m (슬라이드 표 0.341m)")
print(f"실제 위치가 처음 1m에 닿은 시각 t = {도착:.2f}s")
print(f"그 순간 로봇이 받은 측정 위치 {믿음:.3f}m, 속도 명령 {명령[번]:.3f}m/s")

그림 = plt.figure(figsize=(10, 6.6))
위 = 그림.add_axes([0.09, 0.40, 0.88, 0.50])
아래 = 그림.add_axes([0.09, 0.09, 0.88, 0.20], sharex=위)
그림.text(0.09, 0.945, f"측정이 {늦음}초 늦게 도착하면: 로봇은 옛날 위치를 본다", fontsize=13, weight="bold")
그림.text(0.09, 0.325, "그때의 속도 명령 = 이득 × (목표 − 받은 측정)", fontsize=11, weight="bold")

위.plot(시각, 기록, color="tab:blue", lw=2.6)
위.plot(시각, 잰값, color="tab:orange", lw=2.6, ls="--")
위.axhline(목표, color="gray", ls="--", lw=1.4)
위.text(5.9, 목표 - 0.04, "목표 x = 1 m", color="gray", fontsize=11, ha="right", va="top")
위.text(0.02, 1.28, "실제 위치", color="tab:blue", fontsize=11, ha="left", va="center")
위.text(2.0, 0.6, "로봇이 받는 측정\n(0.2초 전 위치)", color="tab:orange", fontsize=11, ha="left", va="center")

# 늦음 표시: 같은 높이 0.5 에서 두 곡선의 가로 간격
높이 = 0.5
t실 = 시각[np.argmax(기록 >= 높이)]
t측 = 시각[np.argmax(잰값 >= 높이)]
위.annotate("", xy=(t측, 높이), xytext=(t실, 높이), arrowprops=dict(arrowstyle="<->", color="black", lw=1.4, shrinkA=0, shrinkB=0))
위.text(t측 + 0.08, 높이 + 0.03, f"{늦음}초 늦음", fontsize=11, va="bottom")

# 도착 순간: 실제는 1 m, 받은 측정은 0.24 m
위.plot([도착], [목표], "o", color="tab:blue", ms=9, zorder=5)
위.plot([도착], [믿음], "o", color="tab:orange", ms=9, zorder=5)
위.annotate("", xy=(도착, 믿음), xytext=(도착, 목표), arrowprops=dict(arrowstyle="<->", color="tab:red", lw=1.8, shrinkA=4, shrinkB=4))
위.text(0.62, 0.2,f"실제로는 이미 도착 (t = {도착:.2f}s)\n그런데 받은 측정은 {믿음:.2f} m\n→ \"아직 멀다\"며 계속 달린다", color="tab:red", fontsize=11, va="center")
위.plot([도착 + 0.02, 0.6], [(목표 + 믿음) / 2 - 0.25, 0.22], color="tab:red", lw=0.8)

위.plot([꼭대기], [기록.max()], "v", color="tab:blue", ms=8)
위.text(꼭대기 + 0.1, 기록.max() + 0.005, f"최대 지나침 {최대:.3f} m", color="tab:blue", fontsize=11, va="bottom")
위.set_ylim(0, 1.5)
위.set_xlim(0, 3.5)
위.set_ylabel("위치 x (m)")
위.set_yticks([0, 0.5, 1.0, 1.5])
위.grid(alpha=0.3)
위.tick_params(labelbottom=False)

아래.step(시각[:-1], 명령, where="post", color="tab:red", lw=2.4)
아래.axhline(0, color="gray", lw=1.2)
아래.axvline(도착, color="tab:red", ls=":", lw=1.4)
아래.plot([도착], [명령[번]], "o", color="tab:red", ms=8, zorder=5)
아래.text(도착 + 0.1, 명령[번] + 0.4, f"도착한 순간에도 명령 +{명령[번]:.2f} m/s\n(계속 앞으로 민다)", color="tab:red", fontsize=11, va="bottom")
아래.text(2.4, -0.9, "음수: 뒤로 되돌아온다", color="tab:red", fontsize=11, va="center")
아래.set_ylim(-1.5, 6.0)
아래.set_yticks([0, 2, 4])
아래.set_ylabel("속도 명령 (m/s)")
아래.set_xlabel("시간 (s)")
아래.grid(alpha=0.3)

그림.savefig("s21_delay.png", dpi=200)
print("저장: s21_delay.png")
