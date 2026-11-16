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


def 깔끔(a):
    return np.round(a, 3) + 0.0              # 6e-17 같은 찌꺼기와 -0.0 없애기


def 글(v):
    return ", ".join(f"{x:g}" for x in v)


def 괄호(x):
    return f"({x:g})" if x < 0 else f"{x:g}"


# 1) 슬라이드 45의 값 (b18_homogeneous.py): 로봇은 방의 (1.5, 0)에서 왼쪽으로 90도
T = 깔끔(변환(Rz(90), [1.5, 0.0, 0.0]))
R, p = T[:3, :3], T[:3, 3]
컵_로봇 = np.array([1.0, -0.5, 0.0, 1.0])     # 동차 좌표: 끝에 1을 붙인다
돌린_컵 = 깔끔(R @ 컵_로봇[:3])
컵_방 = 깔끔(T @ 컵_로봇)
항 = 깔끔(T * 컵_로봇)                         # 항[행, 열] = T[행, 열]·점[열], 줄마다 더하면 컵_방
print("T_방_로봇 =\n", T)
print("R @ 컵_로봇 =", 돌린_컵, " + p =", 깔끔(돌린_컵 + p))
print("T_방_로봇 @ 컵_로봇 =", 컵_방)
for 행 in range(4):
    print(f"{행 + 1}행: {' + '.join(f'{x:g}' for x in 항[행])} = {항[행].sum():g}")

빨강, 초록, 파랑, 보라, 주황 = "tab:red", "tab:green", "tab:blue", "tab:purple", "tab:orange"
축색 = [빨강, 초록, 파랑, 보라]
그림 = plt.figure(figsize=(12, 10.5))
판 = 그림.add_axes([0, 0, 1, 1])              # 판서 칠판: 단위 = 인치
판.set(xlim=(0, 12), ylim=(0, 10.5))
판.axis("off")


def 단계(y0, h, 번호, 제목):
    판.add_patch(FancyBboxPatch((0.15, y0), 11.7, h, boxstyle="round,pad=0,rounding_size=0.12",
                                facecolor="none", edgecolor="lightgray", linewidth=1.2))
    판.add_patch(Circle((0.47, y0 + h - 0.3), 0.17, color="black"))
    판.text(0.47, y0 + h - 0.3, 번호, color="white", ha="center", va="center", fontsize=12)
    판.text(0.75, y0 + h - 0.3, 제목, va="center", fontsize=14)


def 칸(x, y, w, h, 내용, 바탕, 진하기, 글색="black", 글자=13):
    판.add_patch(Rectangle((x, y), w, h, facecolor=바탕, alpha=진하기, edgecolor="white", linewidth=2))
    판.text(x + w / 2, y + h / 2, 내용, ha="center", va="center", fontsize=글자, color=글색)


def 대괄호(x0, x1, 위, 아래):
    for 변, 방향 in [(x0 - 0.04, 1), (x1 + 0.04, -1)]:
        판.plot([변 + 0.08 * 방향, 변, 변, 변 + 0.08 * 방향],
               [위 + 0.03, 위 + 0.03, 아래 - 0.03, 아래 - 0.03], color="black", lw=1.3)


def 이어쓰기(x, y, 조각들, **kw):
    """색이 다른 글 조각을 한 줄에 이어 쓴다."""
    그리기 = 그림.canvas.get_renderer()
    for 내용, 색 in 조각들:
        t = 판.text(x, y, 내용, color=색, **kw)
        x = 판.transData.inverted().transform((t.get_window_extent(그리기).x1, 0))[0]


def 행렬식(위, s, 결과, 행렬, 점, 이름들, 전개, 왼쪽):
    """결과 = 행렬 @ 점 을 칸으로 그리고, 오른쪽에 같은 높이로 줄마다 전개한다."""
    w = 0.55
    for 행 in range(4):
        y = 위 - (행 + 1) * s
        칸(0.5, y, w, s, 결과[행], "lightgray" if 행 == 3 else "whitesmoke", 0.5 if 행 == 3 else 1)
        for 열 in range(4):
            if 행 == 3:
                칸(1.55 + 열 * w, y, w, s, 행렬[행][열], "lightgray", 0.5, "dimgray")
            else:
                칸(1.55 + 열 * w, y, w, s, 행렬[행][열], 축색[열], 0.18)
        칸(4.25, y, w, s, 점[행], 보라 if 행 == 3 else "whitesmoke", 0.25 if 행 == 3 else 1,
          보라 if 행 == 3 else "black")
        판.text(5.65, y + s / 2, f"{왼쪽[행]} =", ha="right", va="center", fontsize=13)
        for k in range(4):                   # 앞 셋 = R @ 점(주황), 넷째 = p·1(보라), 맨 아랫줄은 회색
            바탕 = "lightgray" if 행 == 3 else (주황 if k < 3 else 보라)
            칸(5.8 + k * 1.5, y, 1.2, s, 전개[행][k], 바탕, 0.5 if 행 == 3 else 0.15,
              "dimgray" if 행 == 3 else "black")
            if k < 3:
                판.text(5.8 + k * 1.5 + 1.35, y + s / 2, "+", ha="center", va="center", fontsize=13)
    아래 = 위 - 4 * s
    for x0, x1 in [(0.5, 0.5 + w), (1.55, 1.55 + 4 * w), (4.25, 4.25 + w)]:
        대괄호(x0, x1, 위, 아래)
    판.text(1.3, 위 - 2 * s, "=", ha="center", va="center", fontsize=15)
    판.text(4.0, 위 - 2 * s, "@", ha="center", va="center", fontsize=15)
    for x, 이름 in zip([0.5 + w / 2, 1.55 + 2 * w, 4.25 + w / 2], 이름들):
        판.text(x, 위 + 0.12, 이름, ha="center", va="bottom", fontsize=11)
    return 아래


# ① 돌리고, 옮긴다
단계(7.75, 2.55, "1", "할 일은 두 가지: 돌리고, 옮긴다")
판.text(0.5, 9.4, "컵_방 = R @ 컵_로봇 + p", fontsize=14, va="center")
for y, 이름, 값, 색 in [(8.85, "컵_로봇", 컵_로봇[:3], "dimgray"), (8.45, "R로 돌리면", 돌린_컵, 주황),
                     (8.05, "p만큼 옮기면", 깔끔(돌린_컵 + p), 보라)]:
    판.text(0.5, y, 이름, fontsize=12, color=색, va="center")
    판.text(1.85, y, f"[{글(값)}]", fontsize=12, color=색, va="center")
판.text(3.3, 8.05, "= 방 기준 컵", fontsize=12, va="center")
위 = 그림.add_axes([4.75 / 12, 8.15 / 10.5, 2.75 / 12, 1.95 / 10.5])
위.set(xlim=(-0.3, 2.4), ylim=(-0.75, 1.35), xticks=[0, 1, 2], yticks=[0, 1])
위.set_aspect("equal")
위.grid(alpha=0.3)
위.tick_params(labelsize=10)
화살 = dict(arrowstyle="-|>", linewidth=2)
위.annotate("", xy=컵_로봇[:2], xytext=(0, 0), arrowprops=dict(화살, color="gray", linestyle="--"))
위.annotate("", xy=돌린_컵[:2], xytext=(0, 0), arrowprops=dict(화살, color=주황))
위.annotate("", xy=컵_방[:2], xytext=돌린_컵[:2], arrowprops=dict(화살, color=보라))
위.plot(*p[:2], "s", color="black", markersize=8)
위.plot(*컵_방[:2], "o", color=주황, markeredgecolor="black", markersize=11)
위.text(1.08, -0.5, "돌리기 전", color="dimgray", fontsize=10, va="center")
위.text(0.45, 0.42, "R로 돌리기", color=주황, fontsize=10, va="center")
위.text(1.25, 1.1, "+p 옮기기", color=보라, fontsize=10, ha="center", va="bottom")
위.text(1.5, -0.1, "로봇 (=p)", fontsize=10, ha="center", va="top")
위.text(2.0, 0.84, "컵", fontsize=10, ha="center", va="top")
판.text(8.0, 9.25, "곱셈 한 번 + 덧셈 한 번.", fontsize=12, va="center")
판.text(8.0, 8.8, "좌표계를 여러 개 이으면", fontsize=12, va="center", color="dimgray")
판.text(8.0, 8.45, "곱셈과 덧셈이 뒤섞여 식이 길어진다.", fontsize=12, va="center", color="dimgray")
판.text(8.0, 8.0, "→ 둘을 곱셈 하나로 묶고 싶다", fontsize=12, va="center", color=보라)

# ② 기호로: 끝에 1을 붙이면 곱셈 한 번, 줄마다 펼쳐 보면
단계(4.15, 3.4, "2", "끝에 1을 붙이면 곱셈 한 번: 줄마다 펼쳐 보면")
판.text(0.5, 6.9, "T의 한 줄과 세로줄 [x y z 1]을 짝지어 곱하고 더한다", fontsize=11,
       color="dimgray", va="center")
기호 = [[f"r{행 + 1}{열 + 1}" for 열 in range(3)] + [f"p{'xyz'[행]}"] for 행 in range(3)]
기호 += [["0", "0", "0", "1"]]
점기호 = ["x", "y", "z", "1"]
전개기호 = [[f"{기호[행][열]}·{점기호[열]}" for 열 in range(4)] for 행 in range(4)]
아래 = 행렬식(6.3, 0.42, ["x′", "y′", "z′", "1"], 기호, 점기호, ["방 기준", "T_방_로봇", "로봇 기준"],
           전개기호, ["x′", "y′", "z′", "1"])
판.text(7.9, 6.42, "R @ 점: 돌리기", ha="center", va="bottom", fontsize=12, color=주황)
판.text(10.9, 6.42, "+ p: 옮기기\n(붙인 1 덕분)", ha="center", va="bottom", fontsize=12, color=보라,
       linespacing=1.2)
판.text(4.25 + 0.275, 아래 - 0.1, "붙인 1", ha="center", va="top", fontsize=11, color=보라)
판.text(5.8, 아래 - 0.1, "맨 아랫줄 0 0 0 1 → 결과 끝도 1: 다음 변환에 그대로 넣을 수 있다",
       va="top", fontsize=11, color="dimgray")

# ③ 숫자로: 같은 칸에 슬라이드의 숫자를 넣는다
단계(0.65, 3.3, "3", f"숫자로 확인: 로봇 기준 컵 [{글(컵_로봇[:3])}]")
값 = [[f"{T[행, 열]:g}" for 열 in range(4)] for 행 in range(4)]
전개값 = [[f"{괄호(T[행, 열])}·{괄호(컵_로봇[열])}" for 열 in range(4)] for 행 in range(4)]
아래 = 행렬식(2.8, 0.42, [f"{x:g}" for x in 컵_방], 값, [f"{x:g}" for x in 컵_로봇],
           ["방 기준 컵", "T_방_로봇", "로봇 기준 컵"], 전개값, [f"{x:g}" for x in 컵_방])
판.text(0.5, 3.3, f"②와 같은 칸에 숫자를 넣었다:  r12 → {T[0, 1]:g},  y → {컵_로봇[1]:g},  "
                  f"px → {T[0, 3]:g}", fontsize=11, color="dimgray", va="center")
조각 = [("첫 줄: ", "dimgray"), (f"돌리기 {돌린_컵[0]:g}", 주황), (" + ", "black"),
        (f"옮기기 {p[0]:g}", 보라), (f" = {컵_방[0]:g}", "black"),
        (f"   → 방 기준 컵 [{글(컵_방[:3])}]", "black")]
이어쓰기(5.8, 아래 - 0.25, 조각, fontsize=12, va="center")

판.text(6.0, 0.3, "모두 행렬 하나라서, 좌표계를 잇는 것도 곱셈 한 번: "
                  "T_방_카메라 = T_방_로봇 @ T_로봇_카메라", ha="center", va="center", fontsize=11,
       color="dimgray")
그림.savefig("s45_homogeneous_steps.png", dpi=200)
print("저장: s45_homogeneous_steps.png")
