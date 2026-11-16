import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, FancyBboxPatch, Polygon, Rectangle

# plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["font.family"] = "D2Coding"
plt.rcParams["axes.unicode_minus"] = False


def Ry(각도_도: float) -> np.ndarray:
    c, s = np.cos(np.deg2rad(각도_도)), np.sin(np.deg2rad(각도_도))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(각도_도: float) -> np.ndarray:
    c, s = np.cos(np.deg2rad(각도_도)), np.sin(np.deg2rad(각도_도))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def 변환(R: np.ndarray, p) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def 글(v):
    return ", ".join(f"{x:g}" for x in np.round(v, 3) + 0.0)


# 1) b19_chain.py와 같은 값
R_몸_카메라 = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
T_방_로봇 = 변환(Rz(90), [1.5, 0.0, 0.0])                    # 로봇: 방의 (1.5, 0)에서 왼쪽으로 90도
T_로봇_카메라 = 변환(Ry(30) @ R_몸_카메라, [0.1, 0.0, 0.5])  # 카메라: 앞 0.1m, 높이 0.5m, 30도 숙임
컵_카메라 = np.array([0.5, -0.017, 1.029, 1.0])
T_방_카메라 = T_방_로봇 @ T_로봇_카메라
컵_로봇 = T_로봇_카메라 @ 컵_카메라
컵_방 = T_방_카메라 @ 컵_카메라
print("컵_카메라:", 글(컵_카메라[:3]))
print("컵_로봇  :", 글(컵_로봇[:3]))
print("컵_방    :", 글(컵_방[:3]))
캠 = T_로봇_카메라[[0, 2], 3]                     # 옆에서 본 그림 = 로봇의 x(앞), z(위)
캠_z, 캠_y = T_로봇_카메라[[0, 2], 2], T_로봇_카메라[[0, 2], 1]
숙임 = np.degrees(np.arctan2(-캠_z[1], 캠_z[0]))
print("카메라 위치(로봇 x, z):", 글(캠), " 숙인 각도:", round(숙임, 3))
print("렌즈 정면이 바닥에 닿는 곳: 로봇 앞", round(캠[0] - 캠[1] * 캠_z[0] / 캠_z[1], 3), "m")
print("카메라 x축(로봇 기준):", 글(T_로봇_카메라[:3, 0]), "-> 그림 밖(로봇 오른쪽)으로")

빨강, 초록, 파랑, 주황, 보라 = "tab:red", "tab:green", "tab:blue", "tab:orange", "tab:purple"
그림 = plt.figure(figsize=(10, 7.4))

# 2) 위: 로봇을 오른쪽 옆에서 본 그림
옆 = 그림.add_axes([0.08, 0.42, 0.9, 0.5])
옆.set(xlim=(-0.6, 2.0), ylim=(-0.2, 0.85), xlabel="로봇 앞쪽 x (m)", ylabel="높이 z (m)")
옆.set_xticks([-0.5, 0, 0.5, 1, 1.5, 2])
옆.set_yticks([0, 0.5])
옆.set_aspect("equal")
옆.grid(alpha=0.25)
옆.axhline(0, color="gray", linewidth=1.5)
옆.text(1.95, -0.04, "바닥", color="gray", ha="right", va="top", fontsize=10)
옆.add_patch(Rectangle((-0.2, 0.06), 0.4, 0.24, facecolor=주황, alpha=0.25, edgecolor=주황))
for 바퀴_x in (-0.12, 0.12):
    옆.add_patch(Circle((바퀴_x, 0.06), 0.06, facecolor="white", edgecolor="dimgray"))
옆.plot([캠[0], 캠[0]], [0.3, 캠[1]], color="dimgray", linewidth=2)       # 카메라 기둥
옆.plot(0, 0, "o", color=주황, markersize=7)
옆.text(-0.05, -0.05, "로봇 원점", color=주황, ha="right", va="top", fontsize=10)
# 카메라 몸통: 카메라 z(렌즈 앞), y(화면 아래) 방향으로 돌린 작은 상자
모서리 = [캠 + a * 0.07 * 캠_z + b * 0.04 * 캠_y for a, b in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
옆.add_patch(Polygon(모서리, facecolor=보라, alpha=0.6, edgecolor=보라))
# 놓인 자리: 앞 0.1, 높이 0.5
옆.plot([-0.32, -0.32], [0, 캠[1]], color="gray", linewidth=1, marker="_", markersize=8)
옆.text(-0.35, 캠[1] / 2, "높이\n0.5m", color="dimgray", ha="right", va="center", fontsize=10)
옆.plot([0, 캠[0]], [0.66, 0.66], color="gray", linewidth=1, marker="|", markersize=8)
옆.plot([0, 0], [0.36, 0.66], color="gray", linewidth=0.6, linestyle=":")
옆.text(0.05, 0.69, "앞 0.1m", color="dimgray", ha="center", va="bottom", fontsize=10)
# 숙인 각도
옆.plot([캠[0], 캠[0] + 0.45], [캠[1]] * 2, color="gray", linewidth=0.8, linestyle=":")
옆.add_patch(Arc(캠, 0.66, 0.66, theta1=-숙임, theta2=0, color="black", linewidth=1))
옆.text(캠[0] + 0.36, 캠[1] - 0.08, f"{숙임:.0f}° 숙임", fontsize=10, va="center")
# 카메라 축: z, y는 이 평면에, x는 그림 밖으로
옆.annotate("", xy=캠 + 0.45 * 캠_z, xytext=캠, arrowprops=dict(arrowstyle="-|>", color=파랑, linewidth=2))
옆.annotate("", xy=캠 + 0.2 * 캠_y, xytext=캠, arrowprops=dict(arrowstyle="-|>", color=초록, linewidth=2))
옆.text(*(캠 + 0.2 * 캠_y + [-0.03, 0]), "카메라 y\n(화면 아래)", color=초록, ha="right", va="center", fontsize=10)
옆.text(*(캠 + 0.45 * 캠_z + [0.04, 0.03]), "카메라 z (렌즈 앞)", color=파랑, ha="left", va="bottom", fontsize=10)
옆.text(-0.55, 0.78, "⊙ 카메라 x: 로봇 오른쪽\n  (그림 밖으로)", color=빨강, ha="left", va="center", fontsize=10)
# 시선과 컵
옆.plot([캠[0], 컵_로봇[0]], [캠[1], 컵_로봇[2]], "k--", linewidth=1)
시선각 = np.degrees(np.arctan2(컵_로봇[2] - 캠[1], 컵_로봇[0] - 캠[0]))
print("옆에서 본 시선 각도:", round(-시선각, 2), "° 아래")
옆.text(0.62, 캠[1] + (0.62 - 캠[0]) * np.tan(np.radians(시선각)) - 0.03, "시선 (점선)", rotation=시선각,
        rotation_mode="anchor", ha="left", va="top", fontsize=10)
옆.plot(컵_로봇[0], 컵_로봇[2], "o", color="black", markersize=8)
옆.text(1.05, 0.12, f"컵 (로봇 앞 {컵_로봇[0]:.0f}m, 오른쪽 {-컵_로봇[1]:.1f}m)\n"
                   f"카메라 기준 [{글(컵_카메라[:3])}]\n방 기준     [{글(컵_방[:3])}]",
        fontsize=10, va="bottom", linespacing=1.4)
옆.plot([1.0, 1.04], [0.02, 0.12], color="black", linewidth=0.6)

# 3) 아래: 변환의 사슬
사슬 = 그림.add_axes([0.02, 0.0, 0.96, 0.3])
사슬.axis("off")
사슬.set(xlim=(0, 10), ylim=(0, 3))
상자들 = [("방", "black", 1.3, 컵_방), ("로봇", 주황, 5.0, 컵_로봇), ("카메라", 보라, 8.7, 컵_카메라)]
for 이름, 색, x, 컵 in 상자들:
    사슬.add_patch(FancyBboxPatch((x - 1.1, 1.75), 2.2, 0.75, boxstyle="round,pad=0.05",
                                facecolor="whitesmoke", edgecolor=색, linewidth=2))
    사슬.text(x, 2.28, 이름, color=색, ha="center", va="center", fontsize=13)
    사슬.text(x, 1.93, f"컵 [{글(컵[:3])}]", ha="center", va="center", fontsize=10)
for (x1, 이름, 뜻) in [(2.55, "T_방_로봇", "(1.5, 0)에 서서\n왼쪽으로 90° 돈다"),
                      (6.25, "T_로봇_카메라", "앞 0.1, 높이 0.5\n30° 숙임")]:
    사슬.annotate("", xy=(x1, 2.12), xytext=(x1 + 1.2, 2.12),
                arrowprops=dict(arrowstyle="-|>", color="dimgray", linewidth=2))
    사슬.text(x1 + 0.6, 2.62, 이름, ha="center", va="bottom", fontsize=11)
    사슬.text(x1 + 0.6, 1.95, 뜻, ha="center", va="top", fontsize=10, color="dimgray")
# 식: 이름마다 색을 칠해 이어 붙인다 (앞 조각의 오른쪽 끝에 다음 조각)
조각들 = [("T_", "black"), ("방", "black"), ("_", "black"), ("카메라", 보라), (" = T_", "black"), ("방", "black"),
        ("_", "black"), ("로봇", 주황), (" @ T_", "black"), ("로봇", 주황), ("_", "black"), ("카메라", 보라)]
앞 = 사슬.text(2.1, 0.75, "", fontsize=14, va="baseline")
글자들 = []
for 내용, 색 in 조각들:
    앞 = 사슬.annotate(내용, xy=(1, 0), xycoords=앞, va="bottom", fontsize=14, color=색)
    글자들.append(앞)
사슬.text(7.85, 0.75, "← 안쪽 이름(로봇)이\n   맞물려 지워진다", color=주황, fontsize=11, va="center")
사슬.text(5.0, 0.15, "컵은 오른쪽에서 왼쪽으로 옮겨 간다:  컵_방 = T_방_카메라 @ 컵_카메라",
         ha="center", fontsize=11)

그림.canvas.draw()
가 = 글자들[7].get_window_extent().transformed(사슬.transData.inverted())
나 = 글자들[9].get_window_extent().transformed(사슬.transData.inverted())
사슬.plot([가.x0, 가.x0, 나.x1, 나.x1], [가.y0 - 0.05, 가.y0 - 0.15, 가.y0 - 0.15, 가.y0 - 0.05], color=주황)

for 패널, 제목, 높이 in [(옆, "옆에서 본 로봇 (오른쪽에서 봄): 30° 숙인 카메라가 앞 1m 바닥의 컵을 본다", 0.95),
                       (사슬, "변환의 사슬: 카메라 → 로봇 → 방", 0.32)]:
    그림.text(0.5, 높이, 제목, ha="center", fontsize=12)
그림.savefig("s46_chain_side_view.png", dpi=200)
print("저장: s46_chain_side_view.png")
