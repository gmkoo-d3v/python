import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle

# plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["font.family"] = "D2Coding"
plt.rcParams["axes.unicode_minus"] = False


def 회전2d(각도_도: float) -> np.ndarray:
    """원점 기준 반시계 방향 2×2 회전 행렬."""
    c, s = np.cos(np.deg2rad(각도_도)), np.sin(np.deg2rad(각도_도))
    return np.array([[c, -s], [s, c]])


# 1) R을 만들고 x, y축을 돌린다: R @ x, R @ y
각도 = 30
R = 회전2d(각도)
x축, y축 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
돌린_x, 돌린_y = R @ x축, R @ y축
print(f"R({각도}°) =\n{np.round(R, 3)}")
print("R @ x =", np.round(돌린_x, 3), " R @ y =", np.round(돌린_y, 3))
print("R의 열 == 돌린 축:", np.allclose(R[:, 0], 돌린_x) and np.allclose(R[:, 1], 돌린_y))

그림 = plt.figure(figsize=(10, 4))
격자 = 그림.add_gridspec(1, 2, width_ratios=[1, 1])
빨강, 초록, 흰 = "tab:red", "tab:green", dict(facecolor="white", edgecolor="none", pad=1)

# 2) 왼쪽: 단위원, 원래 축(회색), 돌린 축(색), 각도 호, 점선 투영
칸 = 그림.add_subplot(격자[0])
칸.add_patch(Circle((0, 0), 1, fill=False, color="lightgray"))
for 끝, 색, 글, 자리, 정렬 in [(x축, "gray", "x = [1, 0]", (1.28, -0.24), "right"),
                              (y축, "gray", "y = [0, 1]", (0.06, 1.15), "left"),
                              (돌린_x, 빨강, "", None, None), (돌린_y, 초록, "", None, None)]:
    칸.annotate("", xy=끝, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=색, linewidth=2.2))
    if 글:
        칸.text(*자리, 글, color="dimgray", ha=정렬, fontsize=10, bbox=흰)
for 시작, 끝 in [(0, 각도), (90, 90 + 각도)]:
    칸.add_patch(Arc((0, 0), 1.1, 1.1, theta1=시작, theta2=끝, color="black"))
    칸.text(*(0.68 * np.array([np.cos(np.deg2rad(시작 + 15)), np.sin(np.deg2rad(시작 + 15))])),
           r"$\theta$", ha="center", va="center", fontsize=12)
점선 = dict(color="gray", linestyle="--", linewidth=0.9)
칸.plot([돌린_x[0]] * 2, [0, 돌린_x[1]], **점선)      # 돌린 x의 가로/세로 성분
칸.plot([0, 돌린_x[0]], [돌린_x[1]] * 2, **점선)
칸.plot([돌린_y[0]] * 2, [0, 돌린_y[1]], **점선)      # 돌린 y의 가로/세로 성분
칸.plot([0, 돌린_y[0]], [돌린_y[1]] * 2, **점선)
for 자리, 글, 색, 정렬 in [((0.33, -0.1), r"$\cos\theta$", 빨강, "center"),
                          ((0.05, 돌린_x[1] - 0.07), r"$\sin\theta$", 빨강, "left"),
                          ((돌린_y[0], -0.1), r"$-\sin\theta$", 초록, "center"),
                          ((0.05, 돌린_y[1] - 0.07), r"$\cos\theta$", 초록, "left")]:
    칸.text(*자리, 글, color=색, ha=정렬, va="center", fontsize=12, bbox=흰)
칸.set(xlim=(-1.3, 1.3), ylim=(-1.3, 1.3), xlabel="x", ylabel="y")
칸.set_aspect("equal")
칸.grid(alpha=0.3)

# 3) 오른쪽: 글자만 (x 관련은 빨강, y 관련은 초록)
글_칸 = 그림.add_subplot(격자[1])
글_칸.axis("off")
글_칸.set(xlim=(0, 1), ylim=(0, 1))
글_칸.text(0.5, 0.88, r"$x$ = [1, 0]   →   [$\cos\theta$, $\sin\theta$]", color=빨강, ha="center", fontsize=13)
글_칸.text(0.5, 0.76, r"$y$ = [0, 1]   →   [$-\sin\theta$, $\cos\theta$]", color=초록, ha="center", fontsize=13)
for 중심, 간격, 여백, 행들 in [(0.15, 0.06, 0.09, ["1", "0", "0", "1"]),
                      (0.70, 0.14, 0.12, [r"$\cos\theta$", r"$-\sin\theta$", r"$\sin\theta$", r"$\cos\theta$"])]:
    for 번호, 글 in enumerate(행들):          # 1열은 빨강(x축), 2열은 초록(y축)
        글_칸.text(중심 + (번호 % 2 - 0.5) * 2 * 간격, 0.54 - 0.12 * (번호 // 2), 글,
                  color=[빨강, 초록][번호 % 2], ha="center", va="center", fontsize=14)
    for 변, 방향 in [(중심 - 간격 - 여백, 1), (중심 + 간격 + 여백, -1)]:      # 대괄호
        글_칸.plot([변 + 0.02 * 방향, 변, 변, 변 + 0.02 * 방향], [0.62, 0.62, 0.36, 0.36], color="black")
글_칸.text(0.39, 0.49, "→", ha="center", va="center", fontsize=18)
글_칸.text(0.15, 0.28, "단위 행렬", ha="center", fontsize=11)
글_칸.text(0.70, 0.28, r"회전 행렬 $R(\theta)$", ha="center", fontsize=11)
글_칸.text(0.5, 0.17, "R의 열 = 돌아간 x축(빨강), 돌아간 y축(초록)", ha="center", fontsize=11)
글_칸.text(0.5, 0.05, "돌린_점 = R(θ) @ 점", ha="center", fontsize=14,
         bbox=dict(boxstyle="round", facecolor="whitesmoke", edgecolor="gray"))

그림.subplots_adjust(left=0.07, right=0.98, bottom=0.13, top=0.87, wspace=0.05)
for 패널, 제목 in [(칸, "축을 θ만큼 돌리기"), (글_칸, "회전 행렬")]:
    자리 = 패널.get_position()
    그림.text((자리.x0 + 자리.x1) / 2, 0.93, 제목, ha="center", fontsize=12)
그림.savefig("rot2d_axes.png", dpi=200)
print("저장: rot2d_axes.png")
