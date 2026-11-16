import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["font.family"] = "D2Coding"
plt.rcParams["axes.unicode_minus"] = False

# b09와 같은 행렬: 각 열은 카메라의 x, y, z축이 몸 좌표에서 어느 방향인지를 적은 것
R_몸_카메라 = np.array([[ 0,  0, 1],
                     [-1,  0, 0],
                     [ 0, -1, 0]])
R = R_몸_카메라
내적표 = R.T @ R                                   # 칸 (i, j) = 열i · 열j
print("행렬식:", round(float(np.linalg.det(R)), 6))
print("직교 행렬인가?", np.allclose(R.T @ R, np.eye(3)))
print("R의 전치 @ R =\n", 내적표)
print("R @ R의 전치 =\n", R @ R.T)
print("열1 × 열2 == 열3 (오른손)?", np.allclose(np.cross(R[:, 0], R[:, 1]), R[:, 2]))
print("R_카메라_몸 = 역행렬 = 전치?", np.allclose(np.linalg.inv(R), R.T))

# 반례: 카메라 x축(열 1)만 뒤집은 행렬
거울 = R.copy()
거울[:, 0] *= -1
거울_det = round(float(np.linalg.det(거울)), 6)
print("거울 =\n", 거울)
print("거울: 전치 @ 거울 == I?", np.allclose(거울.T @ 거울, np.eye(3)), "/ 행렬식:", 거울_det)
print("거울: 열1 × 열2 =", np.cross(거울[:, 0], 거울[:, 1]).tolist(), "/ 열3 =", 거울[:, 2].tolist())

색들 = ["tab:red", "tab:green", "tab:blue"]           # 카메라 x 빨강, y 초록, z 파랑


def 세_화살표(칸, M, 글들, 정렬들):
    for 열, (색, 글, (가로, 세로)) in enumerate(zip(색들, 글들, 정렬들)):
        칸.quiver(0, 0, 0, *M[:, 열], color=색, linewidth=3, arrow_length_ratio=0.12, zorder=3)
        칸.text(*(M[:, 열] * 1.15), 글, color=색, ha=가로, va=세로, fontsize=11, zorder=5)
    칸.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], [0] * 5, color="lightgray", linewidth=0.8)   # 몸 바닥
    칸.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1.2, 0.4))   # 바닥 위로는 그릴 것이 없어 z 위쪽을 줄인다
    칸.set_box_aspect((2, 2, 1.6))                          # 세 방향 눈금 길이는 같게
    칸.view_init(elev=20, azim=-120)                 # s37과 같은 자리(로봇 뒤 오른쪽 위)에서 본다
    칸.set_axis_off()


def 상자(칸, M, 색):
    # 세 열이 만드는 평행육면체(여기서는 정육면체)의 여섯 면
    a, b, c = M[:, 0], M[:, 1], M[:, 2]
    면들 = []
    for p, q, r in [(a, b, c), (b, c, a), (c, a, b)]:
        for 밑 in [np.zeros(3), r]:
            면들.append([밑, 밑 + p, 밑 + p + q, 밑 + q])
    칸.add_collection3d(Poly3DCollection(면들, facecolor=색, edgecolor="gray", alpha=0.12, linewidths=0.6))


그림 = plt.figure(figsize=(10, 6.8))
위 = 그림.add_gridspec(1, 2, width_ratios=[1, 1.45], left=0.0, right=0.99, top=0.89, bottom=0.5, wspace=0.0)
아래 = 그림.add_gridspec(1, 4, width_ratios=[1.15, 0.85, 1.15, 0.85], left=0.0, right=0.99, top=0.42, bottom=-0.02, wspace=0.0)
정렬 = [("left", "top"), ("center", "top"), ("left", "bottom")]
상자_글 = dict(fontsize=11, linespacing=1.6, va="center", ha="left")

# 1) 세 열 = 세 화살표
화살_칸 = 그림.add_subplot(위[0], projection="3d", computed_zorder=False)
세_화살표(화살_칸, R, ["열1 카메라 x", "열2 카메라 y", "열3 카메라 z"], 정렬)

# 2) R의 전치 @ R 표: 칸 (i, j) = 열i · 열j
표_칸 = 그림.add_subplot(위[1])
표_칸.axis("off")
표_칸.set(xlim=(0, 1), ylim=(0, 1))
x0, y0, 칸크기 = 0.09, 0.86, 0.12
for i in range(3):
    표_칸.text(x0 - 0.045, y0 - 칸크기 * (i + 0.5), f"열{i + 1}", color=색들[i], ha="center", va="center", fontsize=11)
    표_칸.text(x0 + 칸크기 * (i + 0.5), y0 + 0.05, f"열{i + 1}", color=색들[i], ha="center", va="center", fontsize=11)
    for j in range(3):
        표_칸.add_patch(Rectangle((x0 + 칸크기 * j, y0 - 칸크기 * (i + 1)), 칸크기, 칸크기,
                                 facecolor="moccasin" if i == j else "white", edgecolor="gray"))
        표_칸.text(x0 + 칸크기 * (j + 0.5), y0 - 칸크기 * (i + 0.5), f"{내적표[i, j]}", ha="center", va="center", fontsize=15)
오른쪽 = x0 + 칸크기 * 3 + 0.05
표_칸.text(오른쪽, 0.86, "칸 (i, j) = 열i · 열j", fontsize=11, va="center")
표_칸.add_patch(Rectangle((오른쪽, 0.66), 0.035, 0.06, facecolor="moccasin", edgecolor="gray"))
표_칸.text(오른쪽 + 0.05, 0.69, "대각선 1: 세 열 모두 길이 1", fontsize=11, va="center")
표_칸.add_patch(Rectangle((오른쪽, 0.52), 0.035, 0.06, facecolor="white", edgecolor="gray"))
표_칸.text(오른쪽 + 0.05, 0.55, "나머지 0: 서로 직각", fontsize=11, va="center")
표_칸.text(오른쪽, 0.36, "RRᵀ = I 도 같은 뜻 (정사각 행렬)\n덤: 역행렬 = 전치\n    R_카메라_몸 = R_몸_카메라ᵀ",
          fontsize=10, color="dimgray", va="center", linespacing=1.5)
표_칸.text(0.5, 0.06, "RᵀR = I  →  직교 행렬인가? True\n= 길이 1인 화살표 셋, 서로 직각",
          ha="center", va="center", fontsize=11, linespacing=1.5,
          bbox=dict(boxstyle="round", facecolor="whitesmoke", edgecolor="gray"))

# 3) det = +1: 세 열이 만드는 정육면체, 오른손
진짜_칸 = 그림.add_subplot(아래[0], projection="3d", computed_zorder=False)
상자(진짜_칸, R, "tab:blue")
세_화살표(진짜_칸, R, ["x", "y", "z"], 정렬)
진짜_글 = 그림.add_subplot(아래[1])
진짜_글.axis("off")
진짜_글.text(0, 0.5, f"행렬식: {round(float(np.linalg.det(R)), 6)}\n\n부피 1: 늘지도\n  줄지도 않음\n"
                    "x → y → z: 오른손\n\n뒤집지 않는\n진짜 회전", **상자_글)

# 4) 반례: 열1만 뒤집으면 RᵀR = I는 그대로, 행렬식은 -1
거울_칸 = 그림.add_subplot(아래[2], projection="3d", computed_zorder=False)
상자(거울_칸, 거울, "tab:gray")
세_화살표(거울_칸, 거울, ["-x", "y", "z"], [("right", "bottom")] + 정렬[1:])
거울_글 = 그림.add_subplot(아래[3])
거울_글.axis("off")
거울_글.text(0, 0.5, f"열1만 뒤집음\nRᵀR = I: True\n행렬식: {거울_det}\n\n"
                    "x → y → z: 왼손\n\n거울에 비친 모양,\n회전이 아니다", **상자_글)

for 왼, 오, 제목, 높이 in [(화살_칸, 화살_칸, "R_몸_카메라의 세 열", 0.95), (표_칸, 표_칸, "RᵀR = I : 열끼리의 내적", 0.95),
                         (진짜_칸, 진짜_글, "det(R) = +1", 0.435), (거울_칸, 거울_글, "반례: 열 하나 뒤집기, det = -1", 0.435)]:
    그림.text((왼.get_position().x0 + 오.get_position().x1) / 2, 높이, 제목, ha="center", fontsize=12)
그림.add_artist(plt.Line2D([0.02, 0.98], [0.47, 0.47], color="lightgray", linewidth=0.8))
그림.savefig("s40_rotation_checks.png", dpi=200)
print("저장: s40_rotation_checks.png")
