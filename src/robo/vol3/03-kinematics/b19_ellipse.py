import matplotlib.pyplot as plt
import numpy as np

from 두링크 import 야코비안2, 팔_점들

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

원 = np.stack([np.cos(np.linspace(0, 2 * np.pi, 100)), np.sin(np.linspace(0, 2 * np.pi, 100))])
그림, 칸 = plt.subplots(figsize=(6.4, 5.2))
for 어깨_도, 팔꿈치_도 in [(60, 100), (25, 60), (-15, 25), (-55, 3)]:
    q = np.deg2rad([어깨_도, 팔꿈치_도])
    점 = 팔_점들(q)
    칸.plot(점[:, 0], 점[:, 1], "o-", color="dimgray", linewidth=3)
    타원 = 점[2][:, None] + 0.25 * 야코비안2(q) @ 원      # 길이 1인 관절 속도들이 만드는 손끝 속도
    선 = 칸.plot(타원[0], 타원[1], linewidth=1.5)[0]
    칸.fill(타원[0], 타원[1], color=선.get_color(), alpha=0.3)
    칸.text(*(점[2] + [0.04, 0.0]), f"팔꿈치 {팔꿈치_도}도", fontsize=9)
칸.plot(0, 0, "ks")
칸.set_xlim(-0.4, 1.0)
칸.set_ylim(-0.8, 0.8)
칸.set_aspect("equal")
칸.set_xlabel("x (m)")
칸.set_ylabel("y (m)")
칸.set_title("관절 속도 크기 1이 만드는 손끝 속도(속도 타원)")
그림.tight_layout()
그림.savefig("ellipse.png", dpi=200)
print("저장: ellipse.png")
