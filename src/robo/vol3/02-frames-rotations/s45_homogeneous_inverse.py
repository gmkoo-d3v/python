import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

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


def 역변환(T: np.ndarray) -> np.ndarray:
    R, p = T[:3, :3], T[:3, 3]                # 회전은 전치, 이동은 -Rᵀp
    return 변환(R.T, -R.T @ p)


def 깔끔(a):
    return np.round(a, 3) + 0.0              # 6e-17 같은 찌꺼기와 -0.0 없애기


def 글(v):
    return ", ".join(f"{x:g}" for x in v)


# 1) 슬라이드 45의 값 (b18_homogeneous.py)
T = 깔끔(변환(Rz(90), [1.5, 0.0, 0.0]))
R, p = T[:3, :3], T[:3, 3]
T_역 = 깔끔(역변환(T))
컵_방 = np.array([2.0, 1.0, 0.0, 1.0])        # 끝에 1
컵_로봇 = 깔끔(T_역 @ 컵_방)
뺀_컵 = 깔끔(컵_방[:3] - p)                   # 옮긴 것부터 뺀다 (아직 방의 축 방향)
print("역변환(T_방_로봇) =\n", T_역)
print("역변환 @ 컵_방 =", 컵_로봇)
print("np.linalg.inv와 같은가?", np.allclose(T_역, np.linalg.inv(T)))
print("컵_방 - p =", 뺀_컵, " Rᵀ @ (컵_방 - p) =", 깔끔(R.T @ 뺀_컵), " -Rᵀp =", 깔끔(-R.T @ p))
print("그냥 -p를 쓰면:", 깔끔(R.T @ 컵_방[:3] - p))

보라, 주황 = "tab:purple", "tab:orange"
그림 = plt.figure(figsize=(12, 4.6))
판 = 그림.add_axes([0, 0, 1, 1])              # 판서 칠판: 단위 = 인치
판.set(xlim=(0, 12), ylim=(0, 4.6))
판.axis("off")
판.add_patch(FancyBboxPatch((0.15, 0.15), 11.7, 4.3, boxstyle="round,pad=0,rounding_size=0.12",
                            facecolor="none", edgecolor="lightgray", linewidth=1.2))
판.add_patch(Circle((0.47, 4.15), 0.17, color="black"))
판.text(0.47, 4.15, "4", color="white", ha="center", va="center", fontsize=12)
판.text(0.75, 4.15, "거꾸로 가려면: 회전은 전치, 이동은 -Rᵀp", va="center", fontsize=14)


def 세로줄(x, 위, 값들, 바탕, 진하기, s, w=0.55):
    for 행, v in enumerate(값들):
        판.add_patch(Rectangle((x, 위 - (행 + 1) * s), w, s, facecolor=바탕[행], alpha=진하기[행],
                               edgecolor="white", linewidth=2))
        판.text(x + w / 2, 위 - (행 + 0.5) * s, f"{v:g}", ha="center", va="center", fontsize=13,
               color="dimgray" if 행 == 3 else "black")


def 대괄호(x0, x1, 위, 아래):
    for 변, 방향 in [(x0 - 0.04, 1), (x1 + 0.04, -1)]:
        판.plot([변 + 0.08 * 방향, 변, 변, 변 + 0.08 * 방향],
               [위 + 0.03, 위 + 0.03, 아래 - 0.03, 아래 - 0.03], color="black", lw=1.3)


# 2) 왼쪽: 역변환(T) @ 방 기준 컵 = 로봇 기준 컵
s, 위, w = 0.44, 3.15, 0.55
아래 = 위 - 4 * s
회색 = ["gray"] * 3 + ["lightgray"]
for 열 in range(4):
    바탕 = 회색 if 열 < 3 else [보라] * 3 + ["lightgray"]
    세로줄(0.5 + 열 * w, 위, T_역[:, 열], 바탕, [0.18] * 3 + [0.5], s)
판.text(0.5 + 1.5 * w, 위 + 0.12, "Rᵀ", ha="center", va="bottom", fontsize=12)
판.text(0.5 + 3.5 * w, 위 + 0.12, "-Rᵀp", ha="center", va="bottom", fontsize=12, color=보라)
판.text(0.5 + 2 * w, 아래 - 0.12, "역변환(T_방_로봇)", ha="center", va="top", fontsize=11)
판.text(2.95, 위 - 2 * s, "@", ha="center", va="center", fontsize=15)
세로줄(3.2, 위, 컵_방, ["whitesmoke"] * 3 + [보라], [1, 1, 1, 0.25], s)
판.text(3.2 + w / 2, 아래 - 0.12, "방 기준 컵", ha="center", va="top", fontsize=11)
판.text(4.0, 위 - 2 * s, "=", ha="center", va="center", fontsize=15)
세로줄(4.25, 위, 컵_로봇, ["whitesmoke"] * 3 + ["lightgray"], [1, 1, 1, 0.5], s)
판.text(4.25 + w / 2, 아래 - 0.12, "로봇 기준 컵", ha="center", va="top", fontsize=11)
for x0, x1 in [(0.5, 0.5 + 4 * w), (3.2, 3.2 + w), (4.25, 4.25 + w)]:
    대괄호(x0, x1, 위, 아래)
판.text(0.5, 0.45, "np.linalg.inv(T)와 같다. 맨 아랫줄은 그대로 0 0 0 1",
       fontsize=11, color="dimgray", va="center")

# 3) 오른쪽: 왜 그냥 -p가 아닌가 — 앞으로 간 길을 거꾸로 밟는다
판.text(5.6, 3.5, "앞으로: 먼저 R로 돌리고, 다음에 p만큼 옮겼다", fontsize=12, va="center")
판.text(5.6, 3.1, "거꾸로: 옮긴 것부터 빼고, 돈 것을 되돌린다", fontsize=12, va="center")
판.text(5.6, 2.55, f"① 컵_방 - p = [{글(뺀_컵)}]", fontsize=12, va="center", color=보라)
판.text(9.15, 2.55, "아직 방의 축 방향", fontsize=11, va="center", color="dimgray")
판.text(5.6, 2.1, f"② Rᵀ @ [{글(뺀_컵)}] = [{글(깔끔(R.T @ 뺀_컵))}]", fontsize=12, va="center")
판.text(9.15, 2.1, "로봇 기준 컵", fontsize=11, va="center", color="dimgray")
판.text(5.6, 1.5, "묶으면 Rᵀ @ 컵_방 + (-Rᵀp): 이동 칸에 들어가는 것은", fontsize=12, va="center")
판.text(5.6, 1.1, f"-Rᵀp = [{글(깔끔(-R.T @ p))}]", fontsize=12, va="center", color=보라)
판.text(8.2, 1.1, f"그냥 -p = [{글(-p + 0.0)}]가 아니다", fontsize=12, va="center",
       color="dimgray")
판.text(5.6, 0.55, "p는 방의 축으로 잰 숫자라서, 로봇 축으로 돌려 놓아야 한다",
       fontsize=11, va="center", color="dimgray")
그림.savefig("s45_homogeneous_inverse.png", dpi=200)
print("저장: s45_homogeneous_inverse.png")
