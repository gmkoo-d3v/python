import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon

# plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["font.family"] = "D2Coding"
plt.rcParams["axes.unicode_minus"] = False


def 회전z(각도_도: float) -> np.ndarray:
    """방 위에서 본 2×2 회전 행렬 (반시계 방향이 +)."""
    c, s = np.cos(np.deg2rad(각도_도)), np.sin(np.deg2rad(각도_도))
    return np.array([[c, -s], [s, c]])


def 말로(앞, 왼쪽) -> str:
    """로봇 좌표를 로봇의 말로: 음수 앞 = 뒤, 음수 왼쪽 = 오른쪽."""
    return f"{'앞' if 앞 >= 0 else '뒤'} {abs(앞):g} m, {'왼쪽' if 왼쪽 >= 0 else '오른쪽'} {abs(왼쪽):g} m"


# b01_two_frames.py / b18_homogeneous.py 와 같은 값 (방 좌표, m)
컵_방 = np.array([2.0, 1.0])
로봇_방 = np.array([1.5, 0.0])
빨강, 초록 = "tab:red", "tab:green"
흰 = dict(facecolor="white", edgecolor="none", pad=1)

그림 = plt.figure(figsize=(10, 4.4))
for 번호, 요 in enumerate([0, 90, 180]):
    R = 회전z(요)
    컵_로봇 = R.T @ (컵_방 - 로봇_방)            # 방 기준 → 로봇 기준
    assert np.allclose(로봇_방 + R @ 컵_로봇, 컵_방)  # 다시 방 기준으로 돌려놓기
    print(f"yaw {요:3d}°: 로봇 기준 컵 = {np.round(컵_로봇, 3)} → {말로(*컵_로봇)}")
    앞, 왼 = R[:, 0], R[:, 1]                  # 로봇의 x(앞), y(왼쪽) 축이 방에서 향하는 방향

    칸 = 그림.add_subplot(1, 3, 번호 + 1)
    칸.set(xlim=(-0.6, 3.4), ylim=(-1.0, 2.0), xticks=range(0, 4), yticks=range(-1, 3))
    칸.set_aspect("equal")
    칸.grid(alpha=0.3)
    칸.tick_params(labelsize=9)
    칸.plot(0, 0, "ks", markersize=6)
    칸.text(0.08, 0.08, "방 원점", fontsize=10, color="dimgray")
    # 점선: 로봇의 앞 방향으로 얼마, 거기서 왼쪽 방향으로 얼마
    꺾임 = 로봇_방 + 컵_로봇[0] * 앞
    점선 = dict(linestyle="--", linewidth=1.6)
    칸.plot(*zip(로봇_방, 꺾임), color=빨강, **점선)
    칸.plot(*zip(꺾임, 컵_방), color=초록, **점선)
    가로 = abs(앞[0]) > 0.5                       # 앞으로 가는 구간이 방에서 가로인가
    앞쪽 = -왼 * np.sign(컵_로봇[1])               # 앞 글씨는 왼쪽 구간 반대편에
    if 가로:
        왼쪽 = np.array([0.0, 1.0])               # 왼쪽 구간(세로) 글씨는 점선 바깥쪽
        왼자리 = 꺾임 + 0.5 * (컵_방 - 꺾임) + 0.1 * np.array([np.sign(꺾임[0] - 로봇_방[0]), 0])
        왼정렬 = ("left" if 앞[0] * np.sign(컵_로봇[0]) > 0 else "right", "center")
    else:
        왼자리 = np.array([컵_방[0] - 0.15 * np.sign(컵_방[0] - 꺾임[0]), 컵_방[1] + 0.07])
        왼정렬 = ("right" if 컵_방[0] > 꺾임[0] else "left", "bottom")
    앞자리 = 로봇_방 + (1.0 if 가로 else 0.8) * (꺾임 - 로봇_방) + 0.15 * 앞쪽
    앞정렬 = ("center", "top" if 앞쪽[1] < -0.5 else "bottom") if 가로 else ("right" if 앞쪽[0] < 0 else "left", "center")
    for 자리, 정렬, 글, 색 in [(앞자리, 앞정렬, f"{'앞' if 컵_로봇[0] >= 0 else '뒤'} {abs(컵_로봇[0]):g}", 빨강),
                             (왼자리, 왼정렬, f"{'왼쪽' if 컵_로봇[1] >= 0 else '오른쪽'} {abs(컵_로봇[1]):g}", 초록)]:
        칸.text(*자리, 글, color=색, fontsize=10, bbox=흰, ha=정렬[0], va=정렬[1])
    # 로봇: 앞을 가리키는 삼각형 몸체와 자기 축
    몸 = np.array([[0.22, 0], [-0.15, 0.15], [-0.15, -0.15]])
    칸.add_patch(Polygon(로봇_방 + 몸 @ R.T, closed=True, facecolor="lightsteelblue", edgecolor="tab:blue", zorder=3))
    for 방향, 색, 이름 in [(앞, 빨강, "x"), (왼, 초록, "y")]:
        칸.annotate("", xy=로봇_방 + 0.4 * 방향, xytext=로봇_방, zorder=4,
                   arrowprops=dict(arrowstyle="-|>", color=색, linewidth=2.2, shrinkA=0, shrinkB=0))
        칸.text(*(로봇_방 + 0.55 * 방향), 이름, color=색, ha="center", va="center", fontsize=11, fontweight="bold")
    칸.annotate("로봇", xy=로봇_방 + [-0.05, -0.05], xytext=(0.85, -0.6), color="tab:blue", fontsize=10, ha="center",
               arrowprops=dict(arrowstyle="-", color="tab:blue", linewidth=0.8))
    칸.add_patch(Circle(컵_방, 0.1, facecolor="tab:orange", edgecolor="black", zorder=5))
    칸.text(컵_방[0] + 0.15, 컵_방[1] + 0.05, "컵", fontsize=11, va="bottom")
    칸.text(0.5, 1.22, f"로봇이 {요}° 돌아 있을 때", transform=칸.transAxes, ha="center", fontsize=11)
    칸.text(0.5, 1.1, 말로(*컵_로봇), transform=칸.transAxes, ha="center", fontsize=11, fontweight="bold")
    칸.text(0.5, -0.2, f"로봇 기준 컵 = [{컵_로봇[0]:g}, {컵_로봇[1]:g}]", transform=칸.transAxes, ha="center", fontsize=10)

그림.subplots_adjust(left=0.04, right=0.99, bottom=0.2, top=0.8, wspace=0.12)
그림.text(0.5, 0.96, "같은 컵, 같은 자리: 컵은 방 기준 (2, 1)  ·  로봇은 방 기준 (1.5, 0)", ha="center", fontsize=12)
그림.text(0.5, 0.03, "컵이 어디냐는 '누가 보느냐'에 따라 답이 달라진다", ha="center", fontsize=12)
그림.savefig("s36_cup_three_yaws.png", dpi=200)
print("저장: s36_cup_three_yaws.png")
