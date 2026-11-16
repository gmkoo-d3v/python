import matplotlib.pyplot as plt
import numpy as np

# plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["font.family"] = "D2Coding"
plt.rcParams["axes.unicode_minus"] = False

# b21_watchdog.py 의 반복문을 그대로 옮긴다 (point_robot.py 의 움직이기는 x 방향만 필요해서 줄여 씀)
목표 = np.array([1.0, 0.0])
주기, 이득 = 0.1, 1.5
감시_한계 = 0.3
센서_끊김 = (1.0, 2.5)
최대_속도 = 0.5


def 움직이기(위치, 속도_명령, 주기):
    크기 = np.linalg.norm(속도_명령)
    if 크기 > 최대_속도:
        속도_명령 = 속도_명령 * (최대_속도 / 크기)
    return 위치 + 속도_명령 * 주기, 속도_명령[0]


기록 = {}
for 감시견_켜기 in [False, True]:
    위치 = np.array([0.0, 0.0])
    마지막_측정, 마지막_시각 = 위치.copy(), 0.0
    멈춘_횟수, 가장_멀리 = 0, 0.0
    시각, 위치_x, 속도_x, 멈춘_시각 = [], [], [], []
    for 걸음 in range(50):
        t = 걸음 * 주기
        if not (센서_끊김[0] <= t < 센서_끊김[1]):
            마지막_측정, 마지막_시각 = 위치.copy(), t
        if 감시견_켜기 and t - 마지막_시각 > 감시_한계:
            속도_명령 = np.zeros(2)
            멈춘_횟수 += 1
            멈춘_시각.append(t)
        else:
            속도_명령 = 이득 * (목표 - 마지막_측정)
        시각.append(t)
        위치_x.append(위치[0])
        위치, 속도 = 움직이기(위치, 속도_명령, 주기)
        속도_x.append(속도)
        가장_멀리 = max(가장_멀리, 위치[0])
    시각.append(50 * 주기)
    위치_x.append(위치[0])
    속도_x.append(속도_x[-1])
    기록[감시견_켜기] = (np.array(시각), np.array(위치_x), np.array(속도_x), 가장_멀리, 멈춘_시각)
    print(f"감시견 {'켬' if 감시견_켜기 else '끔'}: 가장 멀리 x {가장_멀리:.3f}m, "
          f"멈춘 주기 {멈춘_횟수}번, 마지막 위치 x {위치[0]:.3f}m")
멈춤_시작 = 기록[True][4][0]
print(f"감시견이 처음 멈춘 시각 t = {멈춤_시작:.1f}s (마지막 측정 0.9s + 한계 {감시_한계}s 초과)")

그림 = plt.figure(figsize=(10, 6.2))
위, 아래 = 그림.add_subplot(2, 1, 1), 그림.add_subplot(2, 1, 2, sharex=None)
for 칸 in (위, 아래):
    칸.axvspan(*센서_끊김, color="gray", alpha=0.18, lw=0)
    칸.set_xlim(0, 5)
    칸.grid(alpha=0.3)

색 = {False: "tab:red", True: "tab:blue"}
이름 = {False: "감시견 끔", True: "감시견 켬"}
for 켬, (시각, 위치_x, 속도_x, 최대, _) in 기록.items():
    위.step(시각, 위치_x, where="post", color=색[켬], linewidth=2.2)
    아래.step(시각, 속도_x, where="post", color=색[켬], linewidth=2.2 if 켬 else 3.5, alpha=1 if 켬 else 0.8)
위.axhline(목표[0], color="gray", linestyle="--")
위.text(0.05, 목표[0] + 0.03, "목표 x = 1 m", color="gray", fontsize=11)
위.text(3.0, 1.19, f"감시견 끔: 가장 멀리 x {기록[False][3]:.3f} m\n(목표를 지나쳐 계속 간다)",
        color="tab:red", fontsize=11, va="bottom")
위.text(3.0, 0.7, f"감시견 켬: 가장 멀리 x {기록[True][3]:.3f} m\n(목표 앞에서 멈춘다)",
        color="tab:blue", fontsize=11, va="top")
위.annotate("감시견이 멈춤", xy=(멈춤_시작, 기록[True][1][int(round(멈춤_시작 / 주기))]), xytext=(1.45, 0.3),
          color="tab:blue", fontsize=11, arrowprops=dict(arrowstyle="-|>", color="tab:blue"))
위.text(1.75, 0.04, "센서 끊김 (1.0 ~ 2.5 s)", ha="center", fontsize=11, color="dimgray")
위.set_ylim(0, 1.45)
위.set_ylabel("위치 x (m)")
위.tick_params(labelbottom=False)

아래.axvline(멈춤_시작, color="tab:blue", linestyle=":")
아래.text(1.85, 0.3, "측정값이 안 와서\n빨강 명령은\n0.5로 굳는다", ha="center", va="center", fontsize=11, color="tab:red")
아래.text(1.85, 0.08, "파랑: 속도 0", ha="center", color="tab:blue", fontsize=11, va="center")
아래.text(3.6, -0.25, "빨강: 목표를 지나쳐\n뒤로 돌아온다", color="tab:red", fontsize=11, va="center")
아래.set_ylim(-0.45, 0.6)
아래.set_ylabel("속도 명령 (m/s)")
아래.set_xlabel("시간 (s)")

그림.subplots_adjust(left=0.09, right=0.98, bottom=0.1, top=0.9, hspace=0.12)
for 칸, 제목 in [(위, "시간에 따른 위치: 센서가 끊기면"), (아래, "그때의 속도 명령")]:
    자리 = 칸.get_position()
    그림.text((자리.x0 + 자리.x1) / 2, 자리.y1 + 0.015, 제목, ha="center", fontsize=12)
그림.savefig("s29_watchdog.png", dpi=200)
print("저장: s29_watchdog.png")
