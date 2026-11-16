import matplotlib.pyplot as plt
import numpy as np

# plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["font.family"] = "D2Coding"
plt.rcParams["axes.unicode_minus"] = False


# rotation_tools.py의 Rx, Ry, Rz와 같은 식
def Rx(각도_도):
    c, s = np.cos(np.deg2rad(각도_도)), np.sin(np.deg2rad(각도_도))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(각도_도):
    c, s = np.cos(np.deg2rad(각도_도)), np.sin(np.deg2rad(각도_도))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(각도_도):
    c, s = np.cos(np.deg2rad(각도_도)), np.sin(np.deg2rad(각도_도))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def 자세(요, 피치, 롤):                       # b11_gimbal.py와 같은 순서 (ROS rpy)
    return Rz(요) @ Ry(피치) @ Rx(롤)


def 사이각(A, B):                            # 두 자세 사이의 회전 각(도)
    return np.rad2deg(np.arccos(np.clip((np.trace(A.T @ B) - 1) / 2, -1, 1)))


# b11_gimbal.py의 세 자세 (요, 피치, 롤)
셋 = [(0, 90, 30), (-30, 90, 0), (20, 90, 50)]
가, 나, 다 = [자세(*t) for t in 셋]
print("가 =\n", np.round(가, 3) + 0.0)
print("요 0, 롤 30  == 요 -30, 롤 0 ?", np.allclose(가, 나), f"(최대 차이 {np.abs(가 - 나).max():.1e})")
print("요 0, 롤 30  == 요 20, 롤 50 ?", np.allclose(가, 다), f"(최대 차이 {np.abs(가 - 다).max():.1e})")
print("롤 - 요:", [r - y for y, _, r in 셋])
for 피치 in (89, 45):
    A, B = 자세(0, 피치, 30), 자세(-30, 피치, 0)
    print(f"피치 {피치}도에서도 같은가? {np.allclose(A, B)}  (최대 차이 {np.abs(A - B).max():.3f}, 사이각 {사이각(A, B):.1f}°)")
차이_89, 차이_45 = 사이각(자세(0, 89, 30), 자세(-30, 89, 0)), 사이각(자세(0, 45, 30), 자세(-30, 45, 0))

축색 = [("x", "tab:red"), ("y", "tab:green"), ("z", "tab:blue")]


def 바탕(칸):                                 # 세계 축(가는 회색), 바닥 테두리, 시점
    for 벡터 in np.eye(3):
        칸.plot(*zip((0, 0, 0), 1.3 * 벡터), color="gray", linewidth=0.8, alpha=0.6)
    칸.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], [-1.1] * 5, color="lightgray", linestyle="--", linewidth=0.8)
    칸.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), zlim=(-1.1, 1.1))
    칸.set_box_aspect((1, 1, 1))
    칸.view_init(elev=22, azim=25)          # 몸체 y·z가 둘 다 비스듬히 보이는 쪽에서
    칸.set_axis_off()


def 몸체축(칸, R, 점선=False):
    for 열, (이름, 색) in enumerate(축색):
        끝 = R[:, 열]
        if 점선:
            칸.plot(*zip((0, 0, 0), 끝), color=색, linewidth=1.6, linestyle=(0, (3, 2)), alpha=0.7)
        else:
            칸.quiver(0, 0, 0, *끝, color=색, linewidth=2.5, arrow_length_ratio=0.15)
            칸.text(*(1.22 * 끝), 이름, color=색, fontsize=12, fontweight="bold", ha="center", va="center")


그림 = plt.figure(figsize=(10.5, 4.6))
칸들 = []
for 번호, (요, 피치, 롤) in enumerate(셋):
    칸 = 그림.add_subplot(1, 4, 번호 + 1, projection="3d")
    바탕(칸)
    몸체축(칸, 자세(요, 피치, 롤))
    칸들.append((칸, f"요 {요}°, 피치 {피치}°, 롤 {롤}°", f"롤 − 요 = {롤 - 요}°", "black"))
칸 = 그림.add_subplot(1, 4, 4, projection="3d")
바탕(칸)
몸체축(칸, 자세(-30, 45, 0), 점선=True)
몸체축(칸, 자세(0, 45, 30))
칸들.append((칸, "피치 45°일 때", "실선 요 0·롤 30 / 점선 요 −30·롤 0", "tab:purple"))

그림.subplots_adjust(left=0.0, right=1.0, bottom=0.17, top=0.78, wspace=0.0)
for 칸, 제목, 아래글, 색 in 칸들:
    자리 = 칸.get_position()
    가운데 = (자리.x0 + 자리.x1) / 2
    그림.text(가운데, 0.8, 제목, ha="center", fontsize=11, color=색)
    그림.text(가운데, 0.745, 아래글, ha="center", fontsize=10, color="dimgray")
# 1~3칸을 묶는 괄호와 4칸 결론
왼, 오 = 칸들[0][0].get_position().x0 + 0.03, 칸들[2][0].get_position().x1 - 0.03
그림.add_artist(plt.Line2D([왼, 왼, 오, 오], [0.855, 0.875, 0.875, 0.855], color="black", linewidth=1))
그림.text((왼 + 오) / 2, 0.89, "피치 90°: 숫자는 다른데 자세는 똑같다", ha="center", fontsize=12, fontweight="bold")
가운데4 = sum(칸들[3][0].get_position().intervalx) / 2
그림.text(가운데4, 0.89, f"다른 자세 (사이 {차이_45:.0f}°)", ha="center", fontsize=12, fontweight="bold", color="tab:purple")
그림.text(0.5, 0.1, "피치 90°에서는 요와 롤이 한 몸이 되어 '롤 − 요'만 자세를 정한다 → 자유도 하나가 사라진다",
         ha="center", fontsize=11)
그림.text(0.5, 0.04, f"굵은 화살표: 몸체의 x·y·z 축   가는 회색 선: 세계 축   "
                    f"(슬라이드의 피치 89°도 다른 자세, 다만 {차이_89:.1f}° 차이)",
         ha="center", fontsize=10, color="dimgray")
그림.savefig("s62_gimbal_lock.png", dpi=200)
print("저장: s62_gimbal_lock.png")
