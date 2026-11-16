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


def 역변환(T: np.ndarray) -> np.ndarray:
    R, p = T[:3, :3], T[:3, 3]
    return 변환(R.T, -R.T @ p)


def 깔끔(a, 자리=3):
    return np.round(a, 자리) + 0.0


def 글(v, 자리=3):
    return "[" + ", ".join(f"{x:g}" for x in 깔끔(v, 자리)) + "]"


# 슬라이드 46의 값 (b19_chain.py, b18_homogeneous.py)
R_몸_카메라 = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
T_방_로봇 = 변환(Rz(90), [1.5, 0.0, 0.0])
T_로봇_카메라 = 변환(Ry(30) @ R_몸_카메라, [0.1, 0.0, 0.5])
T_방_카메라 = T_방_로봇 @ T_로봇_카메라
T_카메라_방 = 역변환(T_로봇_카메라) @ 역변환(T_방_로봇)    # 거꾸로: 타일도 뒤집고 순서도 뒤집는다
컵_카메라 = np.array([0.5, -0.017, 1.029, 1.0])
컵_방 = np.array([2.0, 1.0, 0.0, 1.0])                       # 슬라이드의 방 기준 컵
같다 = np.allclose(T_카메라_방, 역변환(T_방_카메라))
print("T_카메라_방 = 역변환(T_로봇_카메라) @ 역변환(T_방_로봇) =\n", 깔끔(T_카메라_방))
print("역변환(T_방_카메라)와 같은가?", 같다)
print("T_방_카메라 @ 컵_카메라 =", 깔끔((T_방_카메라 @ 컵_카메라)[:3]))
print("T_카메라_방 @ 컵_방    =", 깔끔((T_카메라_방 @ 컵_방)[:3]), " (카메라가 잰 컵", 컵_카메라[:3], ")")

# 일반 규칙 검산: 아무 변환 넷 (T1 = e T d, T2 = d T c, T3 = c T b, T4 = b T a)
T1 = 변환(Rz(30), [1.0, 0.0, 0.0])
T2 = 변환(Ry(45), [0.0, 0.5, 0.2])
T3 = 변환(Rz(-60) @ Ry(20), [0.3, -0.2, 0.0])
T4 = 변환(Ry(-15), [0.0, 0.0, 0.7])
일반 = np.allclose(역변환(T1 @ T2 @ T3 @ T4), 역변환(T4) @ 역변환(T3) @ 역변환(T2) @ 역변환(T1))
print("역변환(T1@T2@T3@T4) == 역변환(T4)@역변환(T3)@역변환(T2)@역변환(T1) ?", 일반)
print("순서를 안 뒤집으면?", np.allclose(역변환(T1 @ T2 @ T3 @ T4),
                                     역변환(T1) @ 역변환(T2) @ 역변환(T3) @ 역변환(T4)))

주황, 보라, 빨강 = "tab:orange", "tab:purple", "tab:red"
색 = {"방": "dimgray", "로봇": 주황, "카메라": 보라, "a": "dimgray", "e": "dimgray",
     "b": "tab:olive", "c": "tab:brown", "d": "tab:blue"}
그림 = plt.figure(figsize=(12, 11.7))
판 = 그림.add_axes([0, 0, 1, 1])              # 판서 칠판: 단위 = 인치
판.set(xlim=(0, 12), ylim=(0, 11.7))
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


def 화살(시작, 끝, c, 굽음=0.0, 양쪽=False):
    판.add_patch(FancyArrowPatch(시작, 끝, connectionstyle=f"arc3,rad={굽음}",
                                 arrowstyle="<|-|>" if 양쪽 else "-|>", mutation_scale=16, color=c,
                                 linewidth=1.8))


def 맞물림(x0, x1, y, c, 말):
    """맞붙은 같은 이름 둘을 아래에서 묶는다."""
    판.plot([x0, x0, x1, x1], [y, y - 0.15, y - 0.15, y], color=c, lw=1.8)
    판.text((x0 + x1) / 2, y - 0.2, 말, ha="center", va="top", fontsize=11, color=c)


# ④ 안 맞물리면 틀린 식 (그대로 옮김)
단계(9.55, 1.95, "4", "안 맞물리면 틀린 식")
y = 10.1
w = 0.75
x1 = 변환칸(0.8, y, "로봇", "카메라", w=w, 코드="T_로봇_카메라")
기호(x1 + 0.3, y, "@")
x2 = 변환칸(x1 + 0.6, y, "방", "로봇", w=w, 코드="T_방_로봇")
판.add_patch(Rectangle((x1 - w - 0.04, y - 0.04), 2 * w + 0.68, H + 0.08, facecolor="none",
                       edgecolor=빨강, linewidth=2, linestyle="--"))
판.text(x2 + 0.35, y + H / 2, "✗", fontsize=22, color=빨강, va="center", ha="center")
판.text(6.6, y + 0.45, "카메라 | 방: 안쪽 글자가 맞물리지 않는다", fontsize=12, color=빨강, va="center")
판.text(6.6, y + 0.05, "numpy는 그래도 숫자를 낸다 → 이름으로 검사", fontsize=11, color="dimgray",
       va="center")

# ⑤ 거꾸로 가는 사슬
단계(0.25, 9.1, "5", "거꾸로 가는 사슬: 방에서 카메라로")


def 사슬줄(y, 결과, 고리들, w, 글자=None):
    """결과 = 고리1 @ 고리2 @ ... 를 한 줄로. 안쪽 이름은 지운다. 고리마다 가운데 x를 돌려준다."""
    끝 = 변환칸(0.8, y, *결과, w=w, 코드=None if 글자 is None else f"T_{결과[0]}_{결과[1]}")
    기호(끝 + 0.3, y, "=")
    x, 가운데 = 끝 + 0.6, []
    for k, (왼, 오) in enumerate(고리들):
        지움 = (k > 0, k < len(고리들) - 1)
        끝 = 변환칸(x, y, 왼, 오, w=w, 지움=지움, 코드=None if 글자 is None else f"T_{왼}_{오}")
        가운데.append((x + 끝) / 2)
        if k < len(고리들) - 1:
            기호(끝 + 0.3, y, "@")
        x = 끝 + 0.6
    return 가운데, x


# (가) 일반 규칙: 고리 넷, 결과는 왼쪽
앞 = [("e", "d"), ("d", "c"), ("c", "b"), ("b", "a")]
뒤 = [(오, 왼) for 왼, 오 in reversed(앞)]            # 하나하나 뒤집고, 순서도 거꾸로
y1, y2 = 7.8, 6.9
가운데, x끝 = 사슬줄(y1, ("e", "a"), 앞, 0.5)
for x, 표 in zip(가운데, "가나다라"):
    판.text(x, y1 + H + 0.25, 표, ha="center", va="center", fontsize=11, color="dimgray")
가운데, _ = 사슬줄(y2, ("a", "e"), 뒤, 0.5)
for x, 표 in zip(가운데, "라다나가"):
    판.text(x, y2 - 0.27, f"{표} 뒤집음", ha="center", va="center", fontsize=10, color="dimgray")
판.text(x끝 - 0.15, y1 + H / 2, "a에서 e로", fontsize=11, va="center")
판.text(x끝 - 0.15, y2 + H / 2, "e에서 a로", fontsize=11, va="center")
판.text(0.6, 6.2, "역행렬: 타일 하나하나를 뒤집고, 순서도 거꾸로 → 안쪽 d | d, c | c, b | b가 다시 맞물린다",
       fontsize=11, va="center", color="dimgray")

# (나) 수업의 사슬로
판.text(0.6, 5.7, "수업의 사슬로: e, a 자리에 방, 카메라", fontsize=12, va="center", color="dimgray")
사슬줄(4.85, ("방", "카메라"), [("방", "로봇"), ("로봇", "카메라")], 0.9, 글자=True)
사슬줄(3.8, ("카메라", "방"), [("카메라", "로봇"), ("로봇", "방")], 0.9, 글자=True)
판.text(9.0, 4.85 + H / 2, "카메라에서 방으로", fontsize=12, va="center")
판.text(9.0, 3.8 + H / 2, "방에서 카메라로", fontsize=12, va="center")

# (다) 점 식 두 줄: 서로 거꾸로
줄들 = [(2.6, "방", "카메라", "카메라에서 방으로", 컵_카메라, T_방_카메라 @ 컵_카메라, 보라),
        (1.45, "카메라", "방", "방에서 카메라로", 컵_방, T_카메라_방 @ 컵_방, "dimgray")]
for y, 받는, 주는, 말, 넣은, 나온, c in 줄들:
    x = 칸(0.8, y, "x", 0.5)
    x = 칸(x, y, 받는, 0.9)
    기호(x + 0.3, y, "=")
    T끝 = 변환칸(x + 0.6, y, 받는, 주는, 지움=(False, True))
    기호(T끝 + 0.3, y, "@")
    x = 칸(T끝 + 0.6, y, "x", 0.5)
    벡터끝 = 칸(x, y, 주는, 0.9, 지움=True)
    맞물림(T끝 - 0.45, 벡터끝 - 0.45, y - 0.1, c, f"{주는} | {주는}")
    판.text(7.8, y + 0.48, f"{말}   T_{받는}_{주는}", fontsize=12, va="center")
    판.text(7.8, y + 0.08, f"컵: {글(넣은[:3])} → {글(나온[:3])}", fontsize=12, va="center",
           color="dimgray")
화살((0.5, 2.6 + H / 2), (0.5, 1.45 + H / 2), "gray", 양쪽=True)
판.text(7.8, 2.3, "↕ 두 식은 서로 거꾸로", fontsize=11, va="center", color="gray")
판.text(0.6, 0.75, f"검산: 역변환(T_로봇_카메라) @ 역변환(T_방_로봇) = 역변환(T_방_카메라)  → {같다}",
       fontsize=11, va="center", color="dimgray")
판.text(0.6, 0.45, f"아무 변환 넷: 역변환(T1@T2@T3@T4) = 역변환(T4)@역변환(T3)@역변환(T2)@역변환(T1)  → {일반}",
       fontsize=11, va="center", color="dimgray")

그림.savefig("s46_chain_names_back.png", dpi=200)
print("저장: s46_chain_names_back.png")
