import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["font.family"] = "D2Coding"
plt.rcParams["axes.unicode_minus"] = False


def Rz(각도_도: float) -> np.ndarray:
    """z축 기준 반시계 회전 (rotation_tools.Rz와 같은 식)."""
    c, s = np.cos(np.deg2rad(각도_도)), np.sin(np.deg2rad(각도_도))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def 변환(R: np.ndarray, p) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def 깔끔(a):
    return np.round(a, 3) + 0.0          # 6e-17 같은 찌꺼기와 -0.0 없애기


def 글(v):
    return ", ".join(f"{x:g}" for x in v)


# 1) 슬라이드 45, 50쪽의 값 (b18_homogeneous.py, b22_mistakes.py)
T_방_로봇 = 변환(Rz(90), [1.5, 0.0, 0.0])               # 로봇: 방의 (1.5, 0)에서 왼쪽으로 90도
T_로봇_손 = 변환(np.eye(3), [0.3, 0.0, 0.4])            # 손은 로봇 앞 0.3m, 높이 0.4m
손_맞음 = 깔끔((T_방_로봇 @ T_로봇_손)[:3, 3])
손_틀림 = 깔끔((T_로봇_손 @ T_방_로봇)[:3, 3])
print("T_방_로봇 =\n", 깔끔(T_방_로봇))
print("T_방_로봇 @ T_로봇_손:", 손_맞음)
print("T_로봇_손 @ T_방_로봇:", 손_틀림)
T = 깔끔(T_방_로봇)
R, p = T[:3, :3], T[:3, 3]

빨강, 초록, 파랑, 보라, 주황 = "tab:red", "tab:green", "tab:blue", "tab:purple", "tab:orange"
축색 = [빨강, 초록, 파랑, 보라]
흰 = dict(facecolor="white", edgecolor="none", pad=1)
그림 = plt.figure(figsize=(10, 7.6))
격자 = 그림.add_gridspec(2, 2, width_ratios=[1, 1.15])

# 2) 왼쪽 위: 4×4 행렬을 칸마다 그리고 색으로 묶기
칸 = 그림.add_subplot(격자[0, 0])
칸.axis("off")
칸.set(xlim=(-0.9, 4.3), ylim=(-1.4, 5.2))
for 행 in range(4):
    for 열 in range(4):
        색 = "lightgray" if 행 == 3 else 축색[열]
        칸.add_patch(Rectangle((열, 3 - 행), 1, 1, facecolor=색, alpha=0.18 if 행 < 3 else 0.5,
                              edgecolor="white", linewidth=2))
        칸.text(열 + 0.5, 3.5 - 행, f"{T[행, 열]:g}", ha="center", va="center", fontsize=14,
               color="dimgray" if 행 == 3 else "black")
for 변, 방향 in [(-0.08, 1), (4.08, -1)]:               # 대괄호
    칸.plot([변 + 0.15 * 방향, 변, 변, 변 + 0.15 * 방향], [4.05, 4.05, -0.05, -0.05], color="black")
for 열, 이름 in enumerate(["x축", "y축", "z축", "위치"]):
    칸.text(열 + 0.5, 4.25, 이름, color=축색[열], ha="center", fontsize=11)
칸.plot([0.05, 2.95], [4.75, 4.75], color="gray", linewidth=1)
칸.text(1.5, 4.85, "R: 로봇 축의 방향", ha="center", fontsize=11)
칸.text(3.5, 4.85, "p: 로봇의 위치", ha="center", color=보라, fontsize=11)
칸.text(-0.3, 2.0, "T_방_로봇 =", ha="right", va="center", fontsize=11)
칸.text(2.0, -0.45, "맨 아랫줄은 항상 0 0 0 1", ha="center", color="dimgray", fontsize=11)
칸.text(2.0, -1.05, f"z축 열 [{글(R[:, 2])}]: 위쪽은 그대로", ha="center", color=파랑, fontsize=10)

# 3) 방을 위에서 본 그림: p 화살표 + R의 1, 2열을 로봇 위치에 (z는 그리지 않음)
범위 = dict(xlim=(-0.4, 2.4), ylim=(-0.6, 1.15))


def 방_그리기(칸, 축_굵기=2.4, 글자=True):
    칸.plot(0, 0, "ko", markersize=5)
    칸.text(0, 0.07, "방 원점", ha="center", va="bottom", fontsize=10)
    칸.annotate("", xy=p[:2], xytext=(0, 0),           # 아래로 휘게 그려 로봇 y축과 겹치지 않게
                arrowprops=dict(arrowstyle="-|>", color=보라, linewidth=2, connectionstyle="arc3,rad=0.35"))
    for 열 in range(2):                              # R의 1열 = 로봇 x축, 2열 = 로봇 y축 (단위 길이)
        칸.annotate("", xy=p[:2] + R[:2, 열], xytext=p[:2],
                    arrowprops=dict(arrowstyle="-|>", color=축색[열], linewidth=축_굵기))
    칸.plot(*p[:2], "s", color="black", markersize=9)
    if 글자:
        칸.text(0.75, -0.42, f"p = [{글(p)}]", color=보라, ha="center", fontsize=11, bbox=흰)
        칸.text(1.58, 0.75, f"로봇 x축\n= R 1열\n[{글(R[:, 0])}]", color=빨강, fontsize=10, va="center")
        칸.text(1.0, 0.07, f"로봇 y축 = R 2열\n[{글(R[:, 1])}]", color=초록, fontsize=10,
               ha="center", va="bottom")
    칸.set(**범위, xlabel="방 x (m)", ylabel="방 y (m)")
    칸.set_aspect("equal")
    칸.grid(alpha=0.3)


방 = 그림.add_subplot(격자[0, 1])
방_그리기(방)

# 4) 왼쪽 아래: 두 곱셈 순서
글_칸 = 그림.add_subplot(격자[1, 0])
글_칸.axis("off")
글_칸.set(xlim=(0, 1), ylim=(0, 1))
for 높이, 제목, 식, 손, 색 in [(0.88, "맞는 순서: 안쪽 글자(로봇)가 맞물린다", "T_방_로봇 @ T_로봇_손", 손_맞음, 주황),
                         (0.40, "거꾸로 곱한 순서: 안쪽 글자가 안 맞물린다", "T_로봇_손 @ T_방_로봇", 손_틀림, "dimgray")]:
    글_칸.text(0.5, 높이, 제목, ha="center", fontsize=11, color=색)
    글_칸.text(0.5, 높이 - 0.13, 식, ha="center", fontsize=13,
              bbox=dict(boxstyle="round", facecolor="whitesmoke", edgecolor=색))
    글_칸.text(0.5, 높이 - 0.27, f"손 위치 = ({글(손)})", ha="center", fontsize=12, color=색)

# 5) 오른쪽 아래: 두 손 위치를 같은 방 그림에
손칸 = 그림.add_subplot(격자[1, 1])
방_그리기(손칸, 축_굵기=1.4, 글자=False)
손칸.annotate("", xy=손_틀림[:2], xytext=p[:2],        # 거꾸로: 방의 x 방향으로 0.3
             arrowprops=dict(arrowstyle="-|>", color="dimgray", linewidth=1.5, linestyle="--"))
손칸.plot(*손_맞음[:2], "o", color=주황, markersize=11)
손칸.plot(*손_틀림[:2], "X", color="dimgray", markersize=11)
손칸.text(손_맞음[0] - 0.1, 손_맞음[1], f"맞는 순서\n({글(손_맞음[:2])})\n로봇 앞(빨강)으로 0.3",
         color=주황, ha="right", va="bottom", fontsize=10, bbox=흰)
손칸.text(손_틀림[0] + 0.05, 손_틀림[1] - 0.12, f"거꾸로 곱한 순서\n({글(손_틀림[:2])})\n방의 x로 0.3",
         color="dimgray", ha="center", va="top", fontsize=10, bbox=흰)

그림.subplots_adjust(left=0.1, right=0.98, bottom=0.07, top=0.92, wspace=0.15, hspace=0.3)
for 패널, 제목 in [(칸, "4×4 변환 = 방향 + 위치"), (방, "같은 숫자를 방에서 보면 (위에서 본 그림)"),
                 (글_칸, "곱하는 순서"), (손칸, "순서가 틀리면 손이 다른 곳에 (높이 0.4는 생략)")]:
    자리 = 패널.get_position()
    윗줄 = max(p_.get_position().y1 for p_ in ([칸, 방] if 패널 in (칸, 방) else [글_칸, 손칸]))
    그림.text((자리.x0 + 자리.x1) / 2, 윗줄 + 0.02, 제목, ha="center", fontsize=12)
그림.savefig("s45_homogeneous.png", dpi=200)
print("저장: s45_homogeneous.png")
