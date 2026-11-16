import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "AppleGothic"      # macOS 한글 글꼴
plt.rcParams["axes.unicode_minus"] = False


def 회전2d(각도_도: float) -> np.ndarray:
    라디안 = np.deg2rad(각도_도)
    c, s = np.cos(라디안), np.sin(라디안)
    return np.array([[c, -s], [s, c]])


# 앞(+x)을 가리키는 화살표 모양 로봇. 점 4개를 열로 세워 한 번에 돌린다
로봇_모양 = np.array([[0.6, -0.3, -0.15, -0.3],
                    [0.0,  0.25, 0.0,  -0.25]])

그림, 칸들 = plt.subplots(1, 4, figsize=(9, 2.6))
for 칸, 각도 in zip(칸들, [0, 45, 90, 150]):
    돌린_모양 = 회전2d(각도) @ 로봇_모양          # (2×2) @ (2×4) → (2×4)
    칸.fill(돌린_모양[0], 돌린_모양[1], color="tab:orange", alpha=0.8)
    칸.arrow(0, 0, 0.8, 0, head_width=0.05, color="gray")    # 바닥에 그린 x축
    칸.arrow(0, 0, 0, 0.8, head_width=0.05, color="gray")    # 바닥에 그린 y축
    칸.set_title(f"{각도}도")
    칸.set_xlim(-0.9, 0.9)
    칸.set_ylim(-0.9, 0.9)
    칸.set_aspect("equal")
    칸.set_xticks([])
    칸.set_yticks([])
그림.tight_layout()
그림.savefig("rot2d.png", dpi=200)
print("저장: rot2d.png")
