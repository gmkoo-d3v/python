import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle

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
컵_로봇 = T_로봇_카메라 @ 컵_카메라
컵_방 = T_방_로봇 @ 컵_로봇
print("카메라 기준 컵:", 글(컵_카메라[:3]), " 렌즈까지 거리:", round(np.linalg.norm(컵_카메라[:3]), 3))
print("로봇 기준 컵  :", 글(컵_로봇[:3]))
print("방 기준 컵    :", 글(컵_방[:3]))
# ① 그림의 '같은 말': 카메라 바로 위에서 본 컵 = 앞 0.9, 오른쪽 0.5 (로봇 기준 컵 - 카메라 자리, 수평 성분)
앞_오른 = np.array([컵_로봇[0] - T_로봇_카메라[0, 3], -컵_로봇[1]])
print("위에서 본 카메라→컵 (앞, 오른쪽):", 글(앞_오른))

회색, 주황, 보라 = "dimgray", "tab:orange", "tab:purple"
그림 = plt.figure(figsize=(12, 7.6))
판 = 그림.add_axes([0, 0, 1, 1])              # 판서 칠판: 단위 = 인치
판.set(xlim=(0, 12), ylim=(0, 7.6))
판.axis("off")


def 단계(y0, h, 번호, 제목):
    판.add_patch(FancyBboxPatch((0.15, y0), 11.7, h, boxstyle="round,pad=0,rounding_size=0.12",
                                facecolor="none", edgecolor="lightgray", linewidth=1.2))
    판.add_patch(Circle((0.47, y0 + h - 0.3), 0.17, color="black"))
    판.text(0.47, y0 + h - 0.3, 번호, color="white", ha="center", va="center", fontsize=12)
    판.text(0.75, y0 + h - 0.3, 제목, va="center", fontsize=14)


def 말풍선(x, y, w, h, 내용, 색, 꼬리):
    """둥근 상자 + 말하는 쪽(꼬리 끝)으로 뾰족한 꼬리."""
    판.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.15",
                                facecolor="white", edgecolor=색, linewidth=1.6, zorder=2))
    밑 = x + 0.35
    판.add_patch(Polygon([(밑, y + 0.01), (밑 + 0.25, y + 0.01), 꼬리], facecolor="white",
                         edgecolor=색, linewidth=1.6, zorder=1))
    판.plot([밑 + 0.02, 밑 + 0.23], [y, y], color="white", linewidth=2.5, zorder=3)
    판.text(x + w / 2, y + h / 2, 내용, ha="center", va="center", fontsize=12, linespacing=1.4, zorder=4)


def 카메라(x, y):
    판.add_patch(FancyBboxPatch((x - 0.3, y - 0.18), 0.5, 0.36, boxstyle="round,pad=0,rounding_size=0.05",
                                facecolor=보라, alpha=0.7, edgecolor=보라))
    판.add_patch(Polygon([(x + 0.2, y - 0.08), (x + 0.36, y - 0.15), (x + 0.36, y + 0.15), (x + 0.2, y + 0.08)],
                         facecolor=보라, edgecolor=보라))


# ① 컵을 본 건 카메라뿐이다
단계(4.05, 3.35, "1", "컵을 본 건 카메라뿐이다")
카메라(0.95, 5.15)
판.text(0.95, 4.85, "카메라", color=보라, ha="center", va="top", fontsize=11)
말풍선(0.55, 5.65, 3.1, 0.9, "컵은 내 렌즈 앞 1.03 m,\n오른쪽 0.5 m에 있어", 보라, (1.05, 5.37))
판.text(1.6, 5.15, f"카메라 기준 컵 [{글(컵_카메라[:3])}]", color=보라, va="center", fontsize=11)
판.text(0.55, 4.45, "하지만 카메라는 자기가 방의\n어디에서 어느 쪽을 보는지 모른다", color=회색, va="center",
        fontsize=11, linespacing=1.3)

방그림 = 그림.add_axes([4.75 / 12, 4.2 / 7.6, 3.0 / 12, 2.25 / 7.6])
방그림.set(xlim=(-0.1, 3.1), ylim=(-0.1, 2.3))
방그림.set_aspect("equal")
방그림.axis("off")
방그림.add_patch(Rectangle((0, 0), 3, 2.2, facecolor="whitesmoke", edgecolor=회색, linewidth=1.5))
방그림.text(2.95, 0.05, "방 (위에서)", color=회색, ha="right", va="bottom", fontsize=10)
for 자리, 향 in [((0.35, 0.35), 50), ((1.5, 0.1), 90), ((2.7, 0.6), 150)]:   # 가운데 = 수업의 실제 자세
    앞 = np.array([np.cos(np.radians(향)), np.sin(np.radians(향))])
    오른 = np.array([앞[1], -앞[0]])
    컵 = np.array(자리) + 앞_오른[0] * 앞 + 앞_오른[1] * 오른
    시야 = [자리, 자리 + 0.45 * (앞 + 0.45 * 오른), 자리 + 0.45 * (앞 - 0.45 * 오른)]
    방그림.add_patch(Polygon(시야, facecolor=보라, alpha=0.15, edgecolor="none"))
    방그림.plot(*자리, "s", color=보라, markersize=7)
    방그림.plot([자리[0], 컵[0]], [자리[1], 컵[1]], color=보라, linewidth=1, linestyle=":")
    방그림.plot(*컵, "o", color=주황, markeredgecolor="black", markersize=9)
    방그림.text(컵[0], 컵[1] + 0.12, "?", ha="center", va="bottom", fontsize=12)
판.text(6.25, 6.75, "같은 말인데, 카메라(■)가 어디 있느냐에\n따라 방에서 컵(●) 자리가 다르다",
        ha="center", va="center", fontsize=11, linespacing=1.3)

# 일상의 예: 사람
판.add_patch(Circle((8.75, 5.35), 0.17, facecolor="white", edgecolor="black", linewidth=1.5))
판.plot([8.75, 8.75], [5.18, 4.75], color="black", linewidth=1.5)
판.plot([8.5, 8.75, 9.0], [4.95, 5.08, 4.95], color="black", linewidth=1.5)
판.plot([8.55, 8.75, 8.95], [4.45, 4.75, 4.45], color="black", linewidth=1.5)
말풍선(8.35, 5.75, 3.3, 0.7, "컵은 내 오른손 앞 30 cm", "black", (8.8, 5.55))
판.text(9.25, 5.1, "…그래서 당신은 어디에 서서,\n   어느 쪽을 보고 있는데?", va="center", fontsize=11,
        linespacing=1.3, color=회색)
판.text(9.25, 4.45, "그걸 알기 전엔 못 찾는다", va="center", fontsize=11, color=회색)

# ② 빠진 정보 두 가지는 다른 데서 온다
단계(0.25, 3.55, "2", "빠진 정보 두 가지는 다른 데서 온다")
마디 = [("컵", 1.2, 주황), ("카메라", 4.1, 보라), ("로봇", 7.3, 주황), ("방", 10.5, 회색)]
y_마디 = 2.45
for 이름, x, 색 in 마디:
    if 이름 == "컵":
        판.plot(x, y_마디, "o", color=주황, markeredgecolor="black", markersize=16)
        판.text(x - 0.25, y_마디, "컵", ha="right", va="center", fontsize=13)
        continue
    판.add_patch(FancyBboxPatch((x - 0.65, y_마디 - 0.25), 1.3, 0.5, boxstyle="round,pad=0.03",
                                facecolor="whitesmoke", edgecolor=색, linewidth=2))
    판.text(x, y_마디, 이름, color=색, ha="center", va="center", fontsize=13)
for x0, x1 in [(1.4, 3.42), (4.78, 6.62), (7.98, 9.82)]:
    판.plot([x0, x1], [y_마디, y_마디], color="black", linewidth=1.5)
연결 = [(2.65, "카메라 기준 컵 = 잰 것", 보라, "카메라가 잰다 (지금 이 순간)", "렌즈 앞 1.03 m, 오른쪽 0.5 m"),
        (5.7, "로봇 기준 카메라 = 빠진 것 1", 주황, "만들 때 정해진다 (나사로 고정)",
         "몸 앞 0.1 m, 높이 0.5 m, 30° 아래"),
        (8.9, "방 기준 로봇 = 빠진 것 2", 회색, "로봇이 지금 서 있는 자리", "(1.5, 0)에 서서 왼쪽으로 90°")]
for x, 무엇, 색, 누가, 값 in 연결:
    판.text(x, 2.05, 무엇, color=색, ha="center", va="center", fontsize=12)
    판.text(x, 1.75, 누가, ha="center", va="center", fontsize=11)
    판.text(x, 1.45, 값, color=회색, ha="center", va="center", fontsize=11)
# 아무도 재지 않은 것: 방 기준 컵
판.annotate("", xy=(1.25, 2.72), xytext=(10.5, 2.75),
            arrowprops=dict(arrowstyle="-|>", color="black", linestyle="--", linewidth=1.3,
                            connectionstyle="arc3,rad=0.12"))
판.text(7.7, 3.48, "방 기준 컵: 아무도 재지 않았다 → 이걸 구한다", ha="center", va="center", fontsize=12)
# 이야기 순서: 본 것에서 출발
판.annotate("", xy=(10.9, 1.0), xytext=(1.0, 1.0),
            arrowprops=dict(arrowstyle="simple,head_width=1.1,head_length=0.7,tail_width=0.6",
                            mutation_scale=30, facecolor="gainsboro", edgecolor="none"))
for x, 말 in [(2.65, "카메라가 본 것에서 출발"), (5.7, "로봇 기준으로 넓히고"), (8.9, "방 기준으로 넓힌다")]:
    판.text(x, 1.0, 말, ha="center", va="center", fontsize=11)
판.text(6.0, 0.5, "식에는 이름이 방 · 로봇 · 카메라 순으로 적히지만, 이야기는 카메라에서 시작한다",
        ha="center", va="center", fontsize=11, color=회색)

그림.savefig("s46_chain_why.png", dpi=200)
print("저장: s46_chain_why.png")
