import matplotlib.pyplot as plt
import numpy as np

# plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["font.family"] = "D2Coding"
plt.rcParams["axes.unicode_minus"] = False


# point_robot.py에서 가져온 두 함수(속도 제한 포함, 이름과 매개변수 그대로)
def 속도_제한(속도, 최대=0.5):
    크기 = np.linalg.norm(속도)
    return 속도 * (최대 / 크기) if 크기 > 최대 else 속도


def 움직이기(위치, 속도_명령, 주기, 생성기=None, 미끄러짐=0.0):
    이동 = 속도_제한(속도_명령) * 주기
    if 생성기 is not None and 미끄러짐 > 0:
        덜_감 = 생성기.uniform(1 - 미끄러짐, 1.0)
        옆_비율 = 생성기.normal(미끄러짐 / 2, 미끄러짐 / 2)
        왼쪽 = np.array([-이동[1], 이동[0]])
        이동 = 덜_감 * 이동 + 옆_비율 * 왼쪽
    return 위치 + 이동


def 위치_재기(위치, 생성기=None, 잡음=0.0):
    if 생성기 is None or 잡음 == 0:
        return 위치.copy()
    return 위치 + 생성기.normal(0, 잡음, size=2)


# 1) b10_sensor_average.py: 평균 횟수별 흔들림 (같은 시드, 같은 난수 순서)
생성기 = np.random.default_rng(0)
참_거리, 잡음 = 1.0, 0.02
횟수들, 평균_모음, 흔들림 = [1, 4, 16, 64], [], []
for 횟수 in 횟수들:
    평균들 = (참_거리 + 생성기.normal(0, 잡음, size=(1000, 횟수))).mean(axis=1)
    평균_모음.append(평균들)
    흔들림.append(평균들.std())
    print(f"{횟수:2d}번 평균: 흔들림(표준편차) {평균들.std():.4f}m")

# 2) b11_sensor_bias.py: 닫힌 고리 80번, 치우침 0 / 0.05 (생성기를 이어서 쓴다)
목표, 주기, 이득 = np.array([2.0, 1.0]), 0.1, 1.5
생성기 = np.random.default_rng(0)
길 = {}
for 치우침_x in [0.0, 0.05]:
    위치 = np.array([0.0, 0.0])
    길[치우침_x] = [위치]
    for _ in range(80):
        잰_위치 = 위치_재기(위치, 생성기, 0.02) + np.array([치우침_x, 0.0])
        위치 = 움직이기(위치, 이득 * (목표 - 잰_위치), 주기, 생성기, 0.2)
        길[치우침_x].append(위치)
    print(f"센서 x 치우침 {치우침_x:.2f}m → 멈춘 곳 {np.round(위치, 3)}, 목표와 x 차이 {위치[0] - 목표[0]:+.3f}m")

그림 = plt.figure(figsize=(10, 4.2))
격자 = 그림.add_gridspec(1, 3, width_ratios=[1.15, 1, 1])

# 칸 1: 평균 내면 잡음이 줄어든다 (점 띠: 평균 1000개 중 150개만, 가로로 흩뿌려 겹침을 줄인다)
칸1 = 그림.add_subplot(격자[0])
점찍기 = np.random.default_rng(1)          # 그림용 가로 흩뿌림 전용
for 순번, (횟수, 평균들, 값) in enumerate(zip(횟수들, 평균_모음, 흔들림)):
    칸1.scatter((평균들[:150] - 참_거리) * 100, 순번 + 점찍기.uniform(-0.3, 0.3, 150),
                s=6, color="tab:blue", alpha=0.5, linewidths=0)
    칸1.text(4.6, 순번, f"{횟수}번 평균\n흔들림 {값 * 100:.2f}cm", va="center", fontsize=10)
칸1.axvline(0, color="gray", linestyle="--")
칸1.text(0.2, -0.65, "참값", color="gray", ha="left", va="center", fontsize=10)
칸1.set_xlim(-5, 10.5)
칸1.set_ylim(-0.8, 4.0)
칸1.set_xticks([-4, -2, 0, 2, 4])
칸1.set_yticks([])
칸1.set_xlabel("참값과의 차이 (cm)")
칸1.invert_yaxis()
칸1.text(0.5, 3.75, "횟수 ×4 → 흔들림 ×1/2  (1/√N)", transform=칸1.transData, fontsize=10, color="tab:blue", va="center")
칸1.spines[["top", "right"]].set_visible(False)

# 칸 2: 과녁. 잡음만 vs 치우침 (읽기 300번씩, 위치 센서 잡음 0.02m)
칸2 = 그림.add_subplot(격자[1])
읽기생성기 = np.random.default_rng(2)
편차 = np.array([0.0, 0.0])
잡음만 = np.array([위치_재기(편차, 읽기생성기, 0.02) for _ in range(150)]) * 100
치우침만 = np.array([위치_재기(편차, 읽기생성기, 0.02) + [0.05, 0] for _ in range(150)]) * 100
for 반지름 in [2, 4, 6]:
    칸2.add_patch(plt.Circle((0, 0), 반지름, fill=False, color="lightgray"))
    칸2.text(0, -반지름, f"{반지름}cm", ha="center", va="center", fontsize=9, color="dimgray", bbox=dict(fc="white", ec="none", pad=1), zorder=7)
칸2.scatter(*잡음만.T, s=8, color="tab:orange", alpha=0.6, label="잡음만")
칸2.scatter(*치우침만.T, s=8, color="tab:purple", alpha=0.6, label="치우침")
칸2.scatter(0, 0, marker="+", s=150, color="black", zorder=5)
칸2.text(-0.5, 0.6, "참값", ha="right", va="bottom", fontsize=10, bbox=dict(fc="white", ec="none", pad=1), zorder=7)
칸2.scatter(*잡음만.mean(0), marker="x", s=140, linewidths=2.5, color="darkorange", zorder=6)
칸2.scatter(*치우침만.mean(0), marker="x", s=140, linewidths=2.5, color="indigo", zorder=6)
칸2.annotate("", xy=(치우침만.mean(0)[0] - 0.4, -3.2), xytext=(잡음만.mean(0)[0], -3.2), arrowprops=dict(arrowstyle="-|>", color="tab:red", linewidth=2))
칸2.text(2.5, -4.2, "+5cm", color="tab:red", ha="center", va="top", fontsize=10)
칸2.text(-7.5, 9.2, "잡음만: 참값 둘레로 흩어짐", color="tab:orange", fontsize=10)
칸2.text(-7.5, 7.8, "치우침: 통째로 +5cm(x)", color="tab:purple", fontsize=10)
칸2.set_xlim(-8, 12)
칸2.set_ylim(-7.5, 10.5)
칸2.set_aspect("equal")
칸2.set_xlabel("x 읽기 오차 (cm)")
칸2.set_ylabel("y 읽기 오차 (cm)")
print("읽기 평균(cm): 잡음만", 잡음만.mean(0).round(2), "치우침", 치우침만.mean(0).round(2))

# 칸 3: 닫힌 고리가 멈춘 곳 (목표에서 얼마나 벗어났나, cm). 접근 경로는 마지막 20걸음만 흐리게
칸3 = 그림.add_subplot(격자[2])
끝0, 끝5 = 길[0.0][-1], 길[0.05][-1]
for 치우침_x, 색 in [(0.0, "tab:orange"), (0.05, "tab:purple")]:
    칸3.plot(*((np.array(길[치우침_x][-20:]) - 목표) * 100).T, color=색, linewidth=0.8, alpha=0.35)
차0, 차5 = (끝0 - 목표) * 100, (끝5 - 목표) * 100
칸3.scatter(0, 0, marker="*", s=260, color="gold", edgecolor="black", zorder=5)
칸3.text(0, -0.9, "목표", fontsize=11, ha="center", va="top", zorder=7)
칸3.scatter(*차0, s=50, color="tab:orange", zorder=6)
칸3.scatter(*차5, s=50, color="tab:purple", zorder=6)
칸3.annotate("", xy=차5, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="tab:red", linewidth=2.5, shrinkA=9, shrinkB=4), zorder=4)
칸3.text(차5[0] / 2, 1.5, f"x로 {(끝5[0] - 목표[0]) :+.3f} m\n(약 5cm 못 미침)".replace("-", "−"), color="tab:red", ha="center", va="bottom", fontsize=10)
칸3.text(차5[0], 차5[1] - 0.7, "치우침 5cm:\n멈춘 곳", color="tab:purple", ha="center", va="top", fontsize=10)
칸3.text(4.4, -2.0, "치우침 없음:\n멈춘 곳", color="tab:orange", ha="right", va="top", fontsize=10)
칸3.text(0.5, 0.97, "치우친 센서는 '도착했다'고 믿는다", transform=칸3.transAxes, ha="center", va="top", fontsize=9, color="tab:purple")
칸3.set_xlim(-8.5, 4.5)
칸3.set_ylim(-4, 5)
칸3.set_xticks([-5, 0])
칸3.set_yticks([0])
칸3.set_aspect("equal")
칸3.set_xlabel("목표에서 x 방향 차이 (cm)")
칸3.set_ylabel("목표에서 y 방향 차이 (cm)")
칸3.grid(alpha=0.3)

그림.subplots_adjust(left=0.05, right=0.98, bottom=0.15, top=0.86, wspace=0.32)
for 칸, 제목 in [(칸1, "잡음: 평균 내면 줄어든다"), (칸2, "치우침: 평균 내도 그대로"), (칸3, "치우친 센서로 닫힌 고리")]:
    자리 = 칸.get_position()
    그림.text((자리.x0 + 자리.x1) / 2, 0.93, 제목, ha="center", fontsize=12)
그림.savefig("s19_sensor_errors.png", dpi=200)
print("저장: s19_sensor_errors.png")
