import matplotlib.pyplot as plt
import numpy as np

from rotation_tools import Ry, Rz

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


def 변환(R: np.ndarray, p) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


R_몸_카메라 = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
T_방_로봇 = 변환(Rz(90), [1.5, 0.0, 0.0])
T_방_카메라 = T_방_로봇 @ 변환(Ry(30) @ R_몸_카메라, [0.1, 0.0, 0.5])
컵_방 = T_방_카메라 @ np.array([0.5, -0.017, 1.029, 1.0])

그림 = plt.figure(figsize=(6, 5))
칸 = 그림.add_subplot(projection="3d")
for 이름, T, 길이 in [("방", np.eye(4), 0.5), ("로봇", T_방_로봇, 0.35), ("카메라", T_방_카메라, 0.25)]:
    원점 = T[:3, 3]
    for 열, 색 in zip(range(3), ["tab:red", "tab:green", "tab:blue"]):   # x 빨강, y 초록, z 파랑
        끝 = 원점 + 길이 * T[:3, 열]
        칸.plot(*zip(원점, 끝), color=색, linewidth=2)
    칸.text(*(원점 + [0.06, 0.06, 0.04]), 이름)
칸.scatter(*컵_방[:3], color="black", s=40)
칸.text(*(컵_방[:3] + [0.05, 0, 0.05]), "컵")
칸.plot(*zip(T_방_카메라[:3, 3], 컵_방[:3]), "k--", linewidth=1)
칸.set_xlabel("x (m)")
칸.set_ylabel("y (m)")
칸.set_zlabel("z (m)")
칸.set_xlim(0, 2.2)
칸.set_ylim(-0.4, 1.4)
칸.set_zlim(0, 1.0)
칸.set_box_aspect((2.2, 1.8, 1.0))
칸.view_init(elev=25, azim=-60)
그림.tight_layout()
그림.savefig("frames.png", dpi=200)
print("저장: frames.png")
