import matplotlib.pyplot as plt
import numpy as np

# plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["font.family"] = "D2Coding"
plt.rcParams["axes.unicode_minus"] = False

# avoidance_tools.py / point_robot.py 의 값과 규칙을 그대로 옮긴다
시작, 목표 = np.array([0.0, 0.0]), np.array([2.0, 1.0])
장애물_반지름, 감지_범위, 주기, 이득, 최대_속도 = 0.25, 0.4, 0.1, 1.5, 0.5


def 속도_제한(속도):
    크기 = np.linalg.norm(속도)
    return 속도 * (최대_속도 / 크기) if 크기 > 최대_속도 else 속도


def 규칙들(위치, 중심):
    """(규칙 1, 규칙 2, 규칙 3) 속도 벡터. 감지 범위 밖이면 규칙 2, 3은 0."""
    r1 = 속도_제한(이득 * (목표 - 위치))
    차이 = 위치 - 중심
    표면_거리 = np.linalg.norm(차이) - 장애물_반지름
    if 표면_거리 > 감지_범위:
        return r1, np.zeros(2), np.zeros(2)
    바깥 = 차이 / np.linalg.norm(차이)
    세기 = 최대_속도 * 2 * (1 - 표면_거리 / 감지_범위)
    둘레 = np.array([-바깥[1], 바깥[0]])
    if np.dot(둘레, 목표 - 위치) < 0:
        둘레 = -둘레
    return r1, 세기 * 바깥, 세기 * 둘레


def 달리기(중심, 걸음_수=150):  # 시도 2: 규칙 1 + 규칙 2 만
    위치, 경로 = 시작.copy(), [시작.copy()]
    for _ in range(걸음_수):
        r1, r2, _ = 규칙들(위치, 중심)
        위치 = 위치 + 속도_제한(r1 + r2) * 주기
        경로.append(위치)
    return np.array(경로)


중심_a, 중심_b = np.array([1.0, 0.5]), np.array([1.0, 0.6])
경로_a, 경로_b = 달리기(중심_a), 달리기(중심_b)
멈춤 = 경로_a[-1]                                            # 시도 2가 멈춘 자리
가까이 = 경로_b[np.argmin(np.linalg.norm(경로_b - 중심_b, axis=1))]  # y=0.6에서 가장 세게 밀리는 자리
print("멈춘 자리:", 멈춤.round(3), "/ 목표까지", round(float(np.linalg.norm(목표 - 멈춤)), 3), "m (슬라이드 1.568m)")

패널 = [("시도 2가 멈추는 자리", 중심_a, 경로_a, 멈춤, False),
      ("규칙 3을 더하면", 중심_a, 경로_a, 멈춤, True),
      ("장애물을 y=0.6m로 옮기면", 중심_b, 경로_b, 가까이, False)]
제목들 = ["① 두 명령이 맞서서 합이 0", "② 옆으로 비켜 가는 규칙을 더하면", "③ 장애물을 y=0.6m로 옮기면"]

그림, 칸들 = plt.subplots(1, 3, figsize=(10, 3.9))
for 번호, (칸, (제목, 중심, 경로, 위치, 옆으로)) in enumerate(zip(칸들, 패널)):
    r1, r2, r3 = 규칙들(위치, 중심)
    합 = r1 + r2 + (r3 if 옆으로 else 0)
    print(f"{제목}: 위치 {위치.round(3)} 규칙1 {r1.round(3)} 규칙2 {r2.round(3)}"
          f"{' 규칙3 ' + str(r3.round(3)) if 옆으로 else ''} 합 {합.round(3)}")
    칸.add_patch(plt.Circle(중심, 장애물_반지름, color="dimgray"))
    칸.add_patch(plt.Circle(중심, 장애물_반지름 + 감지_범위, fill=False, linestyle="--", color="gray"))
    칸.plot([0, 2], [0, 1], color="lightgray", linewidth=1, zorder=0)               # 곧은 길
    칸.plot(*경로[:np.argmin(np.linalg.norm(경로 - 위치, axis=1)) + 1].T, color="tab:orange",
            linewidth=1.2, alpha=0.6)                                                # 시도 2의 지나온 길
    칸.scatter(*시작, s=30, color="black", zorder=5)
    칸.scatter(*목표, marker="*", s=200, color="gold", edgecolors="black", zorder=5)
    칸.scatter(*위치, s=40, color="tab:blue", zorder=6)

    def 화살표(벡터, 색, 굵기=2, 시작점=위치, 머리=14):
        칸.annotate("", xy=시작점 + 벡터, xytext=시작점, zorder=7,
                    arrowprops=dict(arrowstyle="-|>", color=색, linewidth=굵기, shrinkA=0, shrinkB=0,
                                    mutation_scale=머리))

    def 글(내용, 색, 자리, 끝점, 정렬="left"):  # 글자는 빈 곳에 두고, 가는 선으로 자기 화살표와 잇는다
        칸.annotate(내용, xy=끝점, xytext=자리, color=색, fontsize=10, ha=정렬, va="center",
                    fontweight="bold" if 색 == "tab:purple" else "normal",
                    arrowprops=dict(arrowstyle="-", color=색, linewidth=0.8, shrinkA=2, shrinkB=2), zorder=10)

    화살표(r1, "tab:green")
    화살표(r2, "tab:red")
    글("규칙 1\n목표 쪽으로", "tab:green", (0.3, 0.88), 위치 + r1 / 2, "right")
    if 번호 < 2:
        글("규칙 2\n멀어지게", "tab:red", (-0.15, 0.45), 위치 + r2 / 2)
    else:
        글("규칙 2\n멀어지게", "tab:red", (-0.15, -0.4), 위치 + r2)
    if 번호 == 0:
        칸.scatter(*위치, s=300, facecolors="none", edgecolors="tab:purple", linewidths=2.2, zorder=8)
        글("합 ≈ 0\n(줄다리기 멈춤)", "tab:purple", (0.95, -0.4), 위치 + np.array([0.04, -0.06]))
    elif 번호 == 1:
        옮김 = 0.06 * np.array([r3[1], -r3[0]]) / np.linalg.norm(r3)         # 합 화살표를 옆으로 살짝 비켜 그린다
        화살표(합, "tab:purple", 3.5, 위치 + 옮김, 18)
        화살표(r3, "tab:orange")
        글("규칙 3\n둘레 따라 옆으로", "tab:orange", (-0.15, -0.4), 위치 + r3)
        글("합 = 규칙 3\n(1 + 2 = 0)", "tab:purple", (0.95, -0.4), 위치 + 합 + 옮김)
    else:
        화살표(합, "tab:purple", 3.5, 위치, 16)
        글("합 ≠ 0\n(대칭이 깨져 비껴 감)", "tab:purple", (0.8, -0.4), 위치 + 합)
    칸.text(중심[0], 중심[1] + 장애물_반지름 + 0.05, "장애물", fontsize=10, ha="center", va="bottom")
    칸.text(0.0, -0.08, "시작", fontsize=10, ha="center", va="top")
    칸.set_xlim(-0.2, 2.2)
    칸.set_ylim(-0.7, 1.3)
    칸.set_aspect("equal")
    칸.set_xlabel("x (m)")
    칸.set_yticks([-0.5, 0, 0.5, 1.0])
    칸.grid(alpha=0.3)
칸들[0].set_ylabel("y (m)")

그림.subplots_adjust(left=0.07, right=0.99, bottom=0.14, top=0.86, wspace=0.18)
for 칸, 제목 in zip(칸들, 제목들):
    자리 = 칸.get_position()
    그림.text((자리.x0 + 자리.x1) / 2, 0.93, 제목, ha="center", fontsize=11)
그림.savefig("s30_avoid_forces.png", dpi=200)
print("저장: s30_avoid_forces.png")
