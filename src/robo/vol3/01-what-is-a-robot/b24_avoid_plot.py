import matplotlib.pyplot as plt

from avoidance_tools import 감지_범위, 달리기, 목표, 시작, 장애물_반지름, 장애물_중심

# plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["font.family"] = "D2Coding"
plt.rcParams["axes.unicode_minus"] = False

그림, 칸 = plt.subplots(figsize=(7, 4.2))
칸.add_patch(plt.Circle(장애물_중심, 장애물_반지름, color="dimgray", label="장애물"))
칸.add_patch(plt.Circle(장애물_중심, 장애물_반지름 + 감지_범위, fill=False,
                        linestyle=":", color="gray", label="감지 범위"))
for 이름, 밀어내기, 옆으로, 색 in [("1. 목표만", False, False, "tab:red"),
                             ("2. 밀려나기", True, False, "tab:orange"),
                             ("3. 밀려나며 옆으로", True, True, "tab:blue")]:
    경로 = 달리기(밀어내기, 옆으로)
    칸.plot(경로[:, 0], 경로[:, 1], "o-", markersize=2, color=색, label=이름)
칸.scatter(*시작, s=40, color="black", zorder=5)
칸.scatter(*목표, marker="*", s=250, color="gold", edgecolors="black", zorder=5, label="목표")
칸.set_xlabel("x (m)")
칸.set_ylabel("y (m)")
칸.set_xlim(-0.2, 2.3)
칸.set_ylim(-0.3, 1.4)
칸.set_aspect("equal")
칸.legend(loc="upper left", fontsize=8)
그림.tight_layout()
그림.savefig("avoid.png", dpi=200)
print("저장: avoid.png")
