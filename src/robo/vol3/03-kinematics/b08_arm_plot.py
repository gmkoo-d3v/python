import matplotlib.pyplot as plt
import numpy as np

from 팔도구 import 순기구학

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

그림 = plt.figure(figsize=(9, 4.4))
for 번호, 관절각_도 in enumerate([[0, 0, 0, 0, 0, 0], [30, 45, 60, 0, 45, 0]]):
    칸 = 그림.add_subplot(1, 2, 번호 + 1, projection="3d")
    좌표계들 = 순기구학(np.deg2rad(관절각_도))
    점들 = np.array([[0, 0, 0]] + [T[:3, 3] for T in 좌표계들])     # 바닥 → 관절들 → 손끝
    칸.plot(*점들.T, "o-", color="dimgray", linewidth=3, markersize=4)
    for T in [np.eye(4), 좌표계들[-1]]:                          # 바닥과 손끝 좌표계
        for 열, 색 in zip(range(3), ["tab:red", "tab:green", "tab:blue"]):
            칸.plot(*zip(T[:3, 3], T[:3, 3] + 0.12 * T[:3, 열]), color=색, linewidth=2)
    칸.set_title(f"관절 각도(도): {관절각_도}", fontsize=10)
    칸.set_xlim(-0.3, 0.6)
    칸.set_ylim(-0.3, 0.6)
    칸.set_zlim(0, 0.9)
    칸.set_box_aspect((1, 1, 1))
    칸.set_xticks([-0.2, 0.2, 0.6])
    칸.set_yticks([-0.2, 0.2, 0.6])
    칸.set_zticks([0, 0.3, 0.6, 0.9])
    칸.set_xlabel("x (m)")
    칸.set_ylabel("y (m)")
    칸.set_zlabel("z (m)")
    칸.view_init(elev=20, azim=-60)
그림.tight_layout()
그림.savefig("arm6.png", dpi=200)
print("저장: arm6.png")
