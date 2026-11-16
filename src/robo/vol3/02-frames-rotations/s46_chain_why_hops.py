import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Wedge

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


def 깔끔(a):
    return np.round(a, 3) + 0.0


def 글(v):
    return ", ".join(f"{x:g}" for x in 깔끔(v))


# 1) b19_chain.py와 같은 값
R_몸_카메라 = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
T_방_로봇 = 변환(Rz(90), [1.5, 0.0, 0.0])                    # 로봇: 방의 (1.5, 0)에서 왼쪽으로 90도
T_로봇_카메라 = 변환(Ry(30) @ R_몸_카메라, [0.1, 0.0, 0.5])  # 카메라: 앞 0.1m, 높이 0.5m, 30도 숙임
T_방_카메라 = T_방_로봇 @ T_로봇_카메라
컵_카메라 = np.array([0.5, -0.017, 1.029, 1.0])
컵_로봇 = T_로봇_카메라 @ 컵_카메라
컵_방 = T_방_카메라 @ 컵_카메라

# 2) 세 토막을 모두 방의 말(방 x, y, z)로: 원점→로봇, 로봇→카메라, 카메라→컵
로봇_방, 카메라_방 = T_방_로봇[:3, 3], T_방_카메라[:3, 3]
토막 = [로봇_방, 카메라_방 - 로봇_방, T_방_카메라[:3, :3] @ 컵_카메라[:3]]
for 이름, v in zip(["원점→로봇", "로봇→카메라", "카메라→컵"], 토막):
    print(f"토막 {이름:6s}: {글(v)}")
print("세 토막의 합:", 글(sum(토막)), " = 방 기준 컵", 글(컵_방[:3]), np.allclose(sum(토막), 컵_방[:3], atol=1e-3))
# 3) 말 바꾸기 두 번: 고쳐 말하고(회전) 더한다(이동)
고친1 = T_로봇_카메라[:3, :3] @ 컵_카메라[:3]
고친2 = T_방_로봇[:3, :3] @ 컵_로봇[:3]
print("카메라 말 → 로봇 방향으로 고치면:", 글(고친1), "+ 카메라 자리", 글(T_로봇_카메라[:3, 3]), "=", 글(컵_로봇[:3]))
print("로봇 말 → 방 방향으로 고치면  :", 글(고친2), "+ 로봇 자리", 글(로봇_방), "=", 글(컵_방[:3]))

회색, 주황, 보라 = "dimgray", "tab:orange", "tab:purple"
그림 = plt.figure(figsize=(12, 10.2))
판 = 그림.add_axes([0, 0, 1, 1])              # 판서 칠판: 단위 = 인치
판.set(xlim=(0, 12), ylim=(0, 10.2))
판.axis("off")


def 단계(y0, h, 번호, 제목):
    판.add_patch(FancyBboxPatch((0.15, y0), 11.7, h, boxstyle="round,pad=0,rounding_size=0.12",
                                facecolor="none", edgecolor="lightgray", linewidth=1.2))
    판.add_patch(Circle((0.47, y0 + h - 0.3), 0.17, color="black"))
    판.text(0.47, y0 + h - 0.3, 번호, color="white", ha="center", va="center", fontsize=12)
    판.text(0.75, y0 + h - 0.3, 제목, va="center", fontsize=14)


def 축칸(x0, y0, 범위x, 범위y, 배율):
    """인치 단위 자리와 m당 인치로 축을 만든다 (가로세로 같은 축척)."""
    칸 = 그림.add_axes([x0 / 12, y0 / 10.2, (범위x[1] - 범위x[0]) * 배율 / 12, (범위y[1] - 범위y[0]) * 배율 / 10.2])
    칸.set(xlim=범위x, ylim=범위y)
    칸.set_aspect("equal")
    return 칸


def 로봇(칸, 크기=0.09):
    칸.add_patch(Circle(로봇_방[:2], 크기, facecolor=주황, alpha=0.35, edgecolor=주황, linewidth=1.5))
    칸.add_patch(Wedge(로봇_방[:2], 크기, 60, 120, facecolor=주황, edgecolor="none"))   # 앞 = 방 +y


화살 = dict(arrowstyle="-|>", linewidth=2.2, shrinkA=0, shrinkB=0)

# ③ 세 토막을 이어 붙이면 컵에 닿는다 (위에서 본 방, 실제 축척)
단계(5.35, 4.65, "3", "세 토막을 이어 붙이면 컵에 닿는다")
지도 = 축칸(0.95, 6.1, (-0.3, 2.6), (-0.4, 1.35), 1.65)
지도.set_xticks([0, 1, 2])
지도.set_yticks([0, 1])
지도.grid(alpha=0.25)
지도.tick_params(labelsize=10)
지도.set_xlabel("방 x (m)", fontsize=10)
지도.set_ylabel("방 y (m)", fontsize=10)
지도.annotate("", xy=로봇_방[:2], xytext=(0, 0), arrowprops=dict(화살, color=회색))
지도.annotate("", xy=컵_방[:2], xytext=카메라_방[:2], arrowprops=dict(화살, color=보라))
지도.annotate("", xy=컵_방[:2], xytext=(0, 0), arrowprops=dict(화살, color="black", linestyle="--", linewidth=1.3))
로봇(지도)
지도.plot(*카메라_방[:2], "s", color=보라, markersize=6)
지도.plot(*컵_방[:2], "o", color=주황, markeredgecolor="black", markersize=12)
지도.plot(0, 0, "o", color="black", markersize=5)
지도.text(0.03, -0.07, "방 원점", ha="left", va="top", fontsize=10)
지도.text(0.75, -0.05, "토막 1", color=회색, ha="center", va="top", fontsize=11)
지도.text(1.5, -0.13, "로봇", color=주황, ha="center", va="top", fontsize=10)
지도.text(1.66, 0.06, "토막 2\n카메라(■)까지 0.1 m", color=주황, ha="left", va="center", fontsize=11,
          linespacing=1.3)
지도.text(1.86, 0.5, "토막 3", color=보라, ha="left", va="center", fontsize=11)
지도.text(2.0, 1.1, "컵", ha="center", va="bottom", fontsize=11)
지도.text(0.62, 0.36, "곧바로 가면 (아무도 모름)", rotation=np.degrees(np.arctan2(1, 2)), rotation_mode="anchor",
          ha="center", va="bottom", fontsize=10)

x표, 열 = 6.25,[9.35, 10.35, 11.35]
for x, 머리 in zip(열, ["방 x", "방 y", "높이"]):
    판.text(x, 9.15, 머리, ha="right", va="center", fontsize=11, color=회색)
줄들 = [("토막 1  방 원점 → 로봇", 회색, "방 기준 로봇", 토막[0]),
        ("토막 2  로봇 → 카메라", 주황, "로봇 기준 카메라", 토막[1]),
        ("토막 3  카메라 → 컵", 보라, "카메라가 잰 것", 토막[2])]
for i, (이름, 색, 출처, v) in enumerate(줄들):
    y = 8.7 - i * 0.6
    판.text(x표, y, 이름, color=색, va="center", fontsize=12)
    판.text(x표 + 0.25, y - 0.24, f"({출처})", color=회색, va="center", fontsize=10)
    for x, 값 in zip(열, 깔끔(v)):
        판.text(x, y, f"{값:g}", ha="right", va="center", fontsize=12, color=색)
판.plot([x표, 11.5], [7.15, 7.15], color="black", linewidth=1)
판.text(x표, 6.85, "합 = 방 기준 컵", va="center", fontsize=12)
for x, 값 in zip(열, 깔끔(sum(토막))):
    판.text(x, 6.85, f"{값:g}", ha="right", va="center", fontsize=12)
판.text(x표, 6.3, "컵은 그대로다. 원점에서 컵까지의 길을, 아는 사람별로\n세 토막으로 잘랐을 뿐 → 다시 이으면 컵에 닿는다.",
        va="center", fontsize=11, linespacing=1.4)
판.text(x표, 5.65, "위에서 본 그림이라 높이는 안 보인다: 카메라까지 0.5 올라가고,\n"
                  "카메라에서 컵까지 0.5 내려와서 합은 0 (바닥).", va="center", fontsize=10, color=회색,
        linespacing=1.3)

# ④ 토막마다 '앞'의 뜻이 다르다
단계(0.75, 4.4, "4", "토막마다 '앞'의 뜻이 다르다: 같은 말로 고쳐야 더할 수 있다")
판.text(0.5, 4.45, "카메라의 '앞'은 30° 숙인 렌즈 쪽, 로봇의 '앞'은 방의 +y 쪽(왼쪽으로 90° 돌았으니).",
        va="center", fontsize=11, color=회색)
판.text(0.5, 4.15, "그래서 '카메라 앞 1.03'과 '방 x로 1.5'는 그냥 더할 수 없다.", va="center", fontsize=11, color=회색)
작은 = 축칸(0.5, 0.95, (1.1, 2.5), (-0.4, 1.25), 1.85)
작은.axis("off")
x0, y0 = 로봇_방[:2]
x1, y1 = 컵_방[:2]
점선 = dict(linewidth=1.6, linestyle=":")
작은.plot([x0, x0, x1], [y0, y1, y1], color=주황, **점선)       # 로봇의 말: 앞 1, 오른쪽 0.5
작은.plot([x0, x1, x1], [y0, y0, y1], color=회색, **점선)       # 방의 말: x로 0.5, y로 1
작은.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(화살, color=주황))
로봇(작은)
작은.plot(x1, y1, "o", color=주황, markeredgecolor="black", markersize=12)
작은.text(x0 - 0.05, (y0 + y1) / 2, "로봇의 말: 앞 1 m", color=주황, rotation=90, ha="right", va="center",
          fontsize=11)
작은.text((x0 + x1) / 2, y1 + 0.05, "오른쪽 0.5 m", color=주황, ha="center", va="bottom", fontsize=11)
작은.text(x0 + 0.13, y0 - 0.06, "방의 말: x로 0.5 m", color=회색, ha="left", va="top", fontsize=11)
작은.text(x1 + 0.05, (y0 + y1) / 2, "y로 1 m", color=회색, rotation=90, ha="left", va="center", fontsize=11)
작은.text(x0 - 0.08, y0 - 0.08, "로봇", color=주황, ha="right", va="top", fontsize=10)
작은.text(x1 + 0.08, y1 + 0.05, "컵", ha="left", va="bottom", fontsize=11)
작은.text(1.55, 0.85, "같은\n화살표", color=주황, ha="left", va="center", fontsize=10, linespacing=1.2)

# 오른쪽: 같은 컵을 세 가지 말로
def 말상자(y, 누구, 말, 값, 색):
    판.add_patch(FancyBboxPatch((4.4, y - 0.27), 7.2, 0.54, boxstyle="round,pad=0,rounding_size=0.12",
                                facecolor="white", edgecolor=색, linewidth=1.8))
    판.text(4.6, y, 누구, color=색, va="center", fontsize=12)
    판.text(6.7, y, 말, va="center", fontsize=12)
    판.text(11.45, y, 값, color=색, ha="right", va="center", fontsize=12)


def 바꿈(y위, y아래, 첫줄, 둘째줄):
    판.annotate("", xy=(5.0, y아래), xytext=(5.0, y위), arrowprops=dict(arrowstyle="-|>", color="black", linewidth=1.5))
    판.text(5.25, (y위 + y아래) / 2 + 0.16, 첫줄, va="center", fontsize=11)
    판.text(5.25, (y위 + y아래) / 2 - 0.16, 둘째줄, va="center", fontsize=11)


말상자(3.6, "카메라의 말", "렌즈 앞 1.03 m, 오른쪽 0.5 m", f"[{글(컵_카메라[:3])}]", 보라)
바꿈(3.33, 2.67, f"고쳐 말하기: 로봇 방향으로 → 앞 {고친1[0]:.1f}, 오른쪽 {-고친1[1]:.1f}, 아래 {-고친1[2]:.1f}",
     f"더하기: 카메라 자리 (앞 {T_로봇_카메라[0, 3]:g}, 위 {T_로봇_카메라[2, 3]:g})")
말상자(2.4, "로봇의 말", f"앞 {컵_로봇[0]:.0f} m, 오른쪽 {-컵_로봇[1]:.1f} m", f"[{글(컵_로봇[:3])}]", 주황)
바꿈(2.13, 1.47, f"고쳐 말하기: 방 방향으로 → x로 {고친2[0]:.1f}, y로 {고친2[1]:.0f} (왼쪽 그림)",
     f"더하기: 로봇 자리 ({로봇_방[0]:g}, {로봇_방[1]:g})")
말상자(1.2, "방의 말", f"x로 {컵_방[0]:.0f} m, y로 {컵_방[1]:.0f} m", f"({글(컵_방[:3])})", 회색)
판.text(6.0, 0.35, "4×4 행렬은 이 '고쳐 말하고(R) 더하기(p)'를 한 번에 적은 것, 두 행렬을 곱하면 그걸 두 번 하는 것이다.",
        ha="center", va="center", fontsize=11)

그림.savefig("s46_chain_why_hops.png", dpi=200)
print("저장: s46_chain_why_hops.png")
