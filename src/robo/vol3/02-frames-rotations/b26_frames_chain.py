import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

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


def 역변환(T: np.ndarray) -> np.ndarray:
    R, p = T[:3, :3], T[:3, 3]
    return 변환(R.T, -R.T @ p)


def 깔끔(배열, 자리=3):
    return (np.round(배열, 자리) + 0.0).tolist()


# 1) 수업의 값 그대로: 로봇은 방의 (1.5, 0)에서 왼쪽으로 90도, 카메라는 앞 0.1m·높이 0.5m에서 30도 숙임
R_몸_카메라 = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
T_방_로봇 = 변환(Rz(90), [1.5, 0.0, 0.0])
T_로봇_카메라 = 변환(Ry(30) @ R_몸_카메라, [0.1, 0.0, 0.5])
T_방_카메라 = T_방_로봇 @ T_로봇_카메라
컵_방 = np.array([2.0, 1.0, 0.0, 1.0])
컵_로봇 = 역변환(T_방_로봇) @ 컵_방
컵_카메라 = 역변환(T_방_카메라) @ 컵_방
print("T_방_로봇 =\n", np.round(T_방_로봇, 3) + 0.0)
print("T_로봇_카메라 =\n", np.round(T_로봇_카메라, 3) + 0.0)
print("방 기준 컵    :", 깔끔(컵_방[:3]))
print("로봇 기준 컵  :", 깔끔(컵_로봇[:3]), "(슬라이드 36·45: [1, -0.5, 0])")
print("카메라 기준 컵:", 깔끔(컵_카메라[:3]), "(b19: [0.5, -0.017, 1.029])")
print("사슬로 되돌린 방 기준 컵:", 깔끔((T_방_카메라 @ np.array([0.5, -0.017, 1.029, 1.0]))[:3]), "(슬라이드 46: ≈ [2, 1, 0])")

앞뒤 = f"{'앞' if 컵_로봇[0] >= 0 else '뒤'} {abs(컵_로봇[0]):g} m"
좌우 = f"{'왼쪽' if 컵_로봇[1] >= 0 else '오른쪽'} {abs(컵_로봇[1]):g} m"
빨강, 초록, 파랑, 주황 = "tab:red", "tab:green", "tab:blue", "tab:orange"
그림 = plt.figure(figsize=(12, 4.8))

# 2) 왼쪽: 위에서 본 방 (스케치 왼쪽 그대로)
칸 = 그림.add_axes([0.0525, 0.2, 0.3675, 0.62])
화살 = dict(arrowstyle="-|>", linewidth=1.6, mutation_scale=14)
for 끝, 글, 자리 in [((2.55, 0), "방 x", (2.55, -0.17)), ((0, 1.4), "방 y", (0.08, 1.33))]:
    칸.annotate("", xy=끝, xytext=(0, 0), arrowprops=dict(**화살, color="gray"))
    칸.text(*자리, 글, color="gray", fontsize=10, ha="right" if 글 == "방 x" else "left", va="center")
칸.text(-0.06, -0.08, "방 원점", color="gray", fontsize=10, ha="left", va="top")
로봇 = T_방_로봇[:2, 3]
for 열, 색, 글, 밀기 in [(0, 빨강, "x (앞)", (0.05, 0.02)), (1, 초록, "y (왼쪽)", (-0.02, 0.06))]:
    끝 = 로봇 + 0.5 * T_방_로봇[:2, 열]
    칸.annotate("", xy=끝, xytext=로봇, arrowprops={**화살, "color": 색, "linewidth": 2.4})
    칸.text(*(끝 + 밀기), 글, color=색, fontsize=10, ha="center" if 열 == 1 else "left", va="bottom")
칸.add_patch(FancyArrowPatch(로봇 + [0.2, 0.02], 로봇 + [0.02, 0.2], connectionstyle="arc3,rad=0.4",
                             arrowstyle="-|>", mutation_scale=10, color="black", linewidth=1))
칸.text(*(로봇 + [-0.06, 0.3]), "왼쪽으로\n90° 돌았다", fontsize=10, ha="right", va="center")
칸.plot(*로봇, marker="o", color=파랑, markersize=7, zorder=5)
칸.text(*(로봇 + [0, -0.3]), "로봇 (1.5, 0)", color=파랑, fontsize=10, ha="center", va="top")
모퉁이 = np.array([컵_방[0], 로봇[1]])
점선화살 = dict(arrowstyle="-|>", linestyle="--", linewidth=1.6, mutation_scale=12, shrinkA=0, shrinkB=0)
칸.annotate("", xy=모퉁이, xytext=로봇, arrowprops=dict(**점선화살, color=초록))
칸.annotate("", xy=컵_방[:2], xytext=모퉁이, arrowprops=dict(**점선화살, color=빨강))
칸.text(*((로봇 + 모퉁이) / 2 + [0, -0.1]), 좌우, color=초록,
        fontsize=10, ha="center", va="top")
칸.text(*((모퉁이 + 컵_방[:2]) / 2 + [0.07, 0]), 앞뒤, color=빨강, fontsize=10, ha="left", va="center")
칸.plot(*컵_방[:2], marker="o", color=주황, markersize=11, markeredgecolor="black", zorder=5)
칸.text(*(컵_방[:2] + [0.1, 0.06]), "컵", fontsize=11, ha="left", va="bottom")
칸.set(xlim=(-0.35, 2.65), ylim=(-0.5, 1.5), xticks=[0, 0.5, 1, 1.5, 2, 2.5], yticks=[0, 0.5, 1])
칸.set_aspect("equal")
칸.grid(alpha=0.3)
칸.text(0.5, -0.13, f"방 기준 컵 = ({컵_방[0]:g}, {컵_방[1]:g})     "
                    f"로봇 기준 컵 = [{컵_로봇[0]:g}, {컵_로봇[1]:g}]", transform=칸.transAxes,
        fontsize=10.5, ha="center", va="top")

# 3) 오른쪽: 세 좌표계를 따로 그린 기호 (스케치처럼 떨어뜨려 놓고 선으로 잇는다)
#    축 방향은 모두 방 축으로 적은 실제 회전, 자리는 보기 좋게 벌려 놓은 것
p = T_방_로봇[:3, 3]
돌기 = np.rad2deg(np.arctan2(T_방_로봇[1, 0], T_방_로봇[0, 0]))          # 로봇 x축이 방 x에서 돈 각도
q = T_로봇_카메라[:3, 3]
숙임 = np.rad2deg(np.arcsin(-T_로봇_카메라[2, 2]))                       # 카메라 z(렌즈 앞)가 아래로 기운 각도
말_로봇 = f"({p[0]:g}, {p[1]:g})으로\n옮기고\n{'왼쪽' if 돌기 > 0 else '오른쪽'}으로 {abs(돌기):g}°"
말_카메라 = f"{'앞' if q[0] >= 0 else '뒤'} {abs(q[0]):g} m,\n위 {q[2]:g} m,\n{숙임:g}° 숙임"
print("말:", 말_로봇.replace("\n", " "), "/", 말_카메라.replace("\n", " "))

기호들 = []
for 번호, (이름, T, 컵) in enumerate([("방", np.eye(4), 컵_방), ("로봇", T_방_로봇, 컵_로봇),
                                    ("카메라", T_방_카메라, 컵_카메라)]):
    기호 = 그림.add_axes([0.46 + 0.195 * 번호, 0.3, 0.115, 0.42], projection="3d")
    if 이름 != "방":                         # 방의 축을 흐린 회색으로 뒤에
        for 열 in range(3):
            기호.quiver(0, 0, 0, *np.eye(3)[:, 열], color="lightgray", linewidth=1.5, arrow_length_ratio=0.12)
    for 열, 축색, 축글 in [(0, 빨강, "x"), (1, 초록, "y"), (2, 파랑, "z")]:
        축 = T[:3, 열]
        기호.quiver(0, 0, 0, *축, color=축색, linewidth=2.6, arrow_length_ratio=0.18)
        기호.text(*(1.3 * 축), 축글, color=축색, fontsize=11, ha="center", va="center")
    기호.set(xlim=(-0.7, 0.7), ylim=(-0.7, 0.7), zlim=(-0.7, 0.7))     # 좁게 잡아 기호를 크게
    기호.set_box_aspect((1, 1, 1))
    기호.view_init(elev=35, azim=-30)
    기호.set_axis_off()
    자리 = 기호.get_position()
    가로, 세로 = (자리.x0 + 자리.x1) / 2, (자리.y0 + 자리.y1) / 2
    그림.text(가로, 세로 + 0.2, 이름, ha="center", va="bottom", fontsize=12)
    그림.text(가로, 세로 - 0.19, f"컵 [{', '.join(f'{v:g}' for v in 깔끔(컵[:3]))}]",
              ha="center", va="top", fontsize=10)
    기호들.append(기호)

for (앞칸, 뒤칸), 이름, 말 in [(기호들[:2], "T_방_로봇", 말_로봇), (기호들[1:], "T_로봇_카메라", 말_카메라)]:
    a, b = 앞칸.get_position(), 뒤칸.get_position()
    시작, 끝 = (a.x1 + 0.003, (a.y0 + a.y1) / 2), (b.x0 + 0.012, (b.y0 + b.y1) / 2)
    그림.patches.append(FancyArrowPatch(시작, 끝, transform=그림.transFigure, arrowstyle="-|>",
                                        mutation_scale=14, color="dimgray", linewidth=1.6))
    가운데 = ((시작[0] + 끝[0]) / 2, (시작[1] + 끝[1]) / 2)
    그림.text(가운데[0], 가운데[1] + 0.03, 이름, ha="center", va="bottom", fontsize=10)
    그림.text(가운데[0], 가운데[1] - 0.03, 말, ha="center", va="top", fontsize=10, color="dimgray",
              linespacing=1.3)
그림.text(0.73, 0.15, "축 방향은 실제(회색 = 방의 축), 위치는 보기 좋게 벌려 그림", ha="center", va="top",
          fontsize=10, color="gray")

for 제목, 가운데 in [("위에서 본 방: 컵은 로봇의 " + 앞뒤 + ", " + 좌우, 0.0525 + 0.3675 / 2),
                    ("변환의 사슬: 방 → 로봇 → 카메라", 0.73)]:
    그림.text(가운데, 0.92, 제목, ha="center", fontsize=12)
그림.text(0.5, 0.03, "같은 컵도 어느 좌표계에서 재느냐에 따라 숫자가 다르다 · 좌표계는 사슬처럼 이어진다",
          ha="center", fontsize=11)
그림.savefig("frames_chain.png", dpi=200)
print("저장: frames_chain.png")
