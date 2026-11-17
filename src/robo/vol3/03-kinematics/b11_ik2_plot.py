import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

위팔, 아래팔 = 0.4, 0.3
목표 = np.array([0.5, 0.3])

# 앞에서 역기구학2가 낸 두 답(도)
그림, 칸 = plt.subplots(figsize=(5, 4.2))
for (어깨_도, 팔꿈치_도), 색 in [((59.5, -68.0), "tab:blue"), ((2.5, 68.0), "tab:orange")]:
    a, b = np.deg2rad(어깨_도), np.deg2rad(팔꿈치_도)
    팔꿈치_위치 = 위팔 * np.array([np.cos(a), np.sin(a)])
    손끝_위치 = 팔꿈치_위치 + 아래팔 * np.array([np.cos(a + b), np.sin(a + b)])
    점 = np.array([[0, 0], 팔꿈치_위치, 손끝_위치])
    칸.plot(점[:, 0], 점[:, 1], "o-", color=색, linewidth=3, label=f"어깨 {어깨_도}도, 팔꿈치 {팔꿈치_도}도")
칸.plot(*목표, "k*", markersize=14, label="목표 (0.5, 0.3)")
칸.plot(0.8, 0.2, "rx", markersize=10, label="닿지 않는 목표 (0.8, 0.2)")
for 반지름 in [위팔 - 아래팔, 위팔 + 아래팔]:
    칸.add_patch(plt.Circle((0, 0), 반지름, fill=False, color="gray", linestyle="--"))
칸.plot(0, 0, "ks")
칸.set_xlim(-0.2, 0.9)
칸.set_ylim(-0.3, 0.75)
칸.set_aspect("equal")
칸.set_xlabel("x (m)")
칸.set_ylabel("y (m)")
칸.legend(fontsize=8, loc="lower right")
그림.tight_layout()
그림.savefig("ik2_two.png", dpi=200)
print("저장: ik2_two.png")
