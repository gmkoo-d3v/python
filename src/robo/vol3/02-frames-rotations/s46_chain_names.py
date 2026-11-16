import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

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


def 깔끔(a, 자리=3):
    return np.round(a, 자리) + 0.0


def 글(v, 자리=2):
    return "[" + ", ".join(f"{x:g}" for x in 깔끔(v, 자리)) + "]"


# 슬라이드 46의 값 (b19_chain.py, b18_homogeneous.py)
R_몸_카메라 = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
T_방_로봇 = 변환(Rz(90), [1.5, 0.0, 0.0])
T_로봇_카메라 = 변환(Ry(30) @ R_몸_카메라, [0.1, 0.0, 0.5])
T_방_카메라 = T_방_로봇 @ T_로봇_카메라            # 안쪽 이름(로봇)이 맞물린다
컵_카메라 = np.array([0.5, -0.017, 1.029, 1.0])
컵_로봇 = T_로봇_카메라 @ 컵_카메라
컵_방 = T_방_카메라 @ 컵_카메라
print("T_방_로봇 =\n", 깔끔(T_방_로봇))
print("T_로봇_카메라 =\n", 깔끔(T_로봇_카메라))
print("T_방_카메라 = T_방_로봇 @ T_로봇_카메라 =\n", 깔끔(T_방_카메라))
print("컵_카메라:", 컵_카메라[:3], " 컵_로봇:", 깔끔(컵_로봇[:3]), " 컵_방:", 깔끔(컵_방[:3]))

주황, 보라, 파랑, 갈색 = "tab:orange", "tab:purple", "tab:blue", "tab:brown"
색 = {"방": "dimgray", "로봇": 주황, "카메라": 보라, "a": "dimgray", "b": 파랑, "c": 갈색, "d": "dimgray"}
그림 = plt.figure(figsize=(12, 9.25))
판 = 그림.add_axes([0, 0, 1, 1])              # 판서 칠판: 단위 = 인치 (④는 s46_chain_names_back.py로)
판.set(xlim=(0, 12), ylim=(2.95, 12.2))
판.axis("off")
H = 0.55                                      # 칸 높이


def 단계(y0, h, 번호, 제목):
    판.add_patch(FancyBboxPatch((0.15, y0), 11.7, h, boxstyle="round,pad=0,rounding_size=0.12",
                                facecolor="none", edgecolor="lightgray", linewidth=1.2))
    판.add_patch(Circle((0.47, y0 + h - 0.3), 0.17, color="black"))
    판.text(0.47, y0 + h - 0.3, 번호, color="white", ha="center", va="center", fontsize=12)
    판.text(0.75, y0 + h - 0.3, 제목, va="center", fontsize=14)


def 칸(x, y, 이름, w, 지움=False):
    """이름 하나 = 색칸 하나. 지움이면 가로줄로 지운다."""
    c = 색.get(이름, "black")
    판.add_patch(Rectangle((x, y), w, H, facecolor=to_rgba(c if 이름 in 색 else "lightgray", 0.15),
                           edgecolor=c if 이름 in 색 else "gray", linewidth=1.5))
    판.text(x + w / 2, y + H / 2, 이름, color=c, ha="center", va="center", fontsize=15)
    if 지움:
        판.plot([x + 0.12, x + w - 0.12], [y + H / 2, y + H / 2], color=c, lw=1.8)
    return x + w


def 변환칸(x, y, 왼, 오, w=0.9, 지움=(False, False), 코드=None):
    """'받는 쪽 T 주는 쪽' 타일. 끝 x를 돌려준다."""
    판.add_patch(FancyBboxPatch((x - 0.06, y - 0.06), 2 * w + 0.52, H + 0.12,
                                boxstyle="round,pad=0,rounding_size=0.08", facecolor="none",
                                edgecolor="gray", linewidth=1))
    x1 = 칸(x, y, 왼, w, 지움[0])
    판.text(x1 + 0.2, y + H / 2, "T", ha="center", va="center", fontsize=15)
    끝 = 칸(x1 + 0.4, y, 오, w, 지움[1])
    if 코드:
        판.text((x + 끝) / 2, y - 0.12, 코드, ha="center", va="top", fontsize=11, color="dimgray")
    return 끝


def 기호(x, y, s):
    판.text(x, y + H / 2, s, ha="center", va="center", fontsize=16)


def 화살(시작, 끝, c, 굽음=0.0):
    판.add_patch(FancyArrowPatch(시작, 끝, connectionstyle=f"arc3,rad={굽음}", arrowstyle="-|>",
                                 mutation_scale=16, color=c, linewidth=1.8))


def 맞물림(x0, x1, y, c, 말):
    """맞붙은 같은 이름 둘을 아래에서 묶는다."""
    판.plot([x0, x0, x1, x1], [y, y - 0.15, y - 0.15, y], color=c, lw=1.8)
    판.text((x0 + x1) / 2, y - 0.2, 말, ha="center", va="top", fontsize=11, color=c)


# ① 이름 읽는 법
단계(9.85, 2.2, "1", "이름 읽는 법: (받는 쪽) T (주는 쪽)")
y = 10.3
끝 = 변환칸(0.8, y, "방", "카메라", w=1.1)
판.text(0.8 + 0.55, y - 0.12, "받는 쪽", ha="center", va="top", fontsize=11, color="dimgray")
판.text(끝 - 0.55, y - 0.12, "주는 쪽", ha="center", va="top", fontsize=11, color=보라)
화살((끝 - 0.55, y + H + 0.1), (0.8 + 0.55, y + H + 0.1), "black", 굽음=0.2)
판.text((0.8 + 끝) / 2, y + H + 0.38, "카메라에서 방으로", ha="center", va="bottom", fontsize=11)
판.text(5.0, 11.2, "코드로 쓰면  T_방_카메라", fontsize=13, va="center")
판.text(5.0, 10.75, "카메라 기준 점을 넣으면 → 방 기준 점이 나온다", fontsize=12, va="center")
판.text(5.0, 10.3, "점은 오른쪽(주는 쪽)으로 들어가 왼쪽(받는 쪽)으로 나온다", fontsize=11,
       va="center", color="dimgray")

# ② 점에 곱할 때
단계(7.45, 2.2, "2", "점에 곱할 때: 맞닿는 이름이 같아야 한다")
y = 8.25
x = 칸(0.8, y, "컵", 0.6)
결과끝 = 칸(x, y, "방", 0.9)
기호(결과끝 + 0.3, y, "=")
T시작 = 결과끝 + 0.6
T끝 = 변환칸(T시작, y, "방", "카메라", 지움=(False, True))
기호(T끝 + 0.3, y, "@")
x = 칸(T끝 + 0.6, y, "컵", 0.6)
벡터끝 = 칸(x, y, "카메라", 0.9, 지움=True)
맞물림(T끝 - 0.45, 벡터끝 - 0.45, y - 0.1, 보라, "이름 끝(카메라) = 벡터 끝(카메라): 맞물려 지워진다")
화살((T시작 + 0.45, y - 0.1), (결과끝 - 0.45, y - 0.1), "dimgray", 굽음=-0.4)
판.text((T시작 + 결과끝) / 2, y - 0.55, "남는 이름(방)", ha="center", va="top", fontsize=11,
       color="dimgray")
판.text(7.3, 8.9, "컵_방 = T_방_카메라 @ 컵_카메라", fontsize=13, va="center")
판.text(7.3, 8.45, "벡터 이름 끝 = 그 점을 잰 기준", fontsize=11, va="center", color="dimgray")
판.text(7.3, 8.1, "T의 주는 쪽과 같아야 넣을 수 있다", fontsize=11, va="center", color="dimgray")

# ③ 사슬
단계(3.1, 4.15, "3", "사슬: 안쪽 이름이 맞물려 지워진다")
y = 6.0
x = 0.8
끝들 = []
for 왼, 오, 지움 in [("a", "b", (False, True)), ("b", "c", (True, True)), ("c", "d", (True, False))]:
    끝들.append((x, 변환칸(x, y, 왼, 오, w=0.6, 지움=지움)))
    x = 끝들[-1][1] + 0.6
    기호(x - 0.3, y, "@" if 오 != "d" else "=")
변환칸(x, y, "a", "d", w=0.6)
for (_, 끝), (다음, _), c, 이름 in zip(끝들, 끝들[1:], [파랑, 갈색], "bc"):
    맞물림(끝 - 0.3, 다음 + 0.3, y - 0.1, c, f"{이름} | {이름}")
판.text(x + 2.0, y + 0.45, "바깥 이름 a, d만 남는다", fontsize=12, va="center")
판.text(x + 2.0, y + 0.05, "사이 이름이 몇 개여도 같다", fontsize=11, va="center", color="dimgray")

y = 4.5
판.text(0.5, 5.3, "수업의 사슬: 카메라가 잰 컵 → 로봇 기준 → 방 기준", fontsize=12,
       va="center", color="dimgray")
x1 = 변환칸(0.8, y, "방", "로봇", 지움=(False, True), 코드="T_방_로봇")
x2 = 변환칸(x1 + 0.6, y, "로봇", "카메라", 지움=(True, False), 코드="T_로봇_카메라")
기호(x1 + 0.3, y, "@")
기호(x2 + 0.3, y, "=")
변환칸(x2 + 0.6, y, "방", "카메라", 코드="T_방_카메라")
맞물림(x1 - 0.55, x1 + 0.9, y - 0.1, 주황, "로봇 | 로봇")
판.text(x2 + 3.4, y + H / 2, "안쪽 이름(로봇)이 맞물린다", fontsize=12, va="center", color=주황)

y = 3.35
조각 = [("카메라 기준 컵", 보라, 글(컵_카메라[:3], 3)), ("로봇 기준 컵", 주황, "≈ " + 글(컵_로봇[:3])),
        ("방 기준 컵", "black", "≈ " + 글(컵_방[:3]))]
for i, (이름, c, 값) in enumerate(조각):
    x = 0.8 + i * 3.9
    판.text(x, y + 0.42, 이름, fontsize=11, color=c, va="center")
    판.text(x, y + 0.08, 값, fontsize=13, color=c, va="center")
    if i < 2:
        화살((x + 2.6, y + 0.08), (x + 3.7, y + 0.08), "gray")
        판.text(x + 3.15, y + 0.22, ["T_로봇_카메라 @", "T_방_로봇 @"][i], fontsize=10, ha="center",
               va="bottom", color="dimgray")

그림.savefig("s46_chain_names.png", dpi=200)
print("저장: s46_chain_names.png")
