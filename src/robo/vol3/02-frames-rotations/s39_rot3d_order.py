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


def 글(v):                                   # [0. 1. 0.] → "[0, 1, 0]"
    return "[" + ", ".join(f"{x:g}" for x in (np.round(v, 3) + 0.0).tolist()) + "]"


# 1) 위: 앞 [1, 0, 0]을 각 축으로 90도 (b07_rot3d_axes.py)
앞 = np.array([1.0, 0.0, 0.0])
축들 = {"z": (Rz, "tab:blue", "왼쪽으로 고개 돌리기"), "y": (Ry, "tab:green", "고개 숙이기(앞이 아래로)"),
       "x": (Rx, "tab:red", "고개 갸웃, 시선은 그대로")}
for 이름, (R, _, _) in 축들.items():
    print(f"{이름}축으로 90도:", 글(R(90) @ 앞))

# 2) 아래: 같은 두 회전, 순서만 바꾸기 (b08_order.py) — @는 오른쪽부터 적용된다
가_중간, 가 = Rx(90) @ 앞, Rz(90) @ Rx(90) @ 앞
나_중간, 나 = Rz(90) @ 앞, Rx(90) @ Rz(90) @ 앞
print("x 먼저, z 나중:", 글(가_중간), "→", 글(가))
print("z 먼저, x 나중:", 글(나_중간), "→", 글(나))
print("두 행렬이 같은가?", np.allclose(Rz(90) @ Rx(90), Rx(90) @ Rz(90)))


def 바탕(칸):                                 # 좌표축(가는 선), 바닥 테두리
    for 벡터, 색, 이름 in [((1, 0, 0), "tab:red", "x 앞"), ((0, 1, 0), "tab:green", "y 왼쪽"),
                         ((0, 0, 1), "tab:blue", "z 위")]:
        끝 = (1.2 if 이름[0] == "z" else 1.5) * np.array(벡터)
        칸.plot(*zip((0, 0, 0), 끝), color=색, linewidth=1, alpha=0.7)
        칸.text(*(끝 * 1.15), 이름, color=색, fontsize=10, ha="center", va="center")
    칸.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], [0] * 5, color="lightgray", linestyle="--", linewidth=0.8)
    칸.set(xlim=(-1, 1.3), ylim=(-1, 1.3), zlim=(-1, 1.3))
    칸.set_box_aspect((1, 1, 1), zoom=1.3)
    칸.view_init(elev=25, azim=-125)        # 로봇 뒤 오른쪽 위에서 본다: x는 안쪽, y는 왼쪽, z는 위
    칸.set_axis_off()


def 화살표(칸, 끝, 색, 이름="", 밀기=(0, 0, -0.15), 굵기=2.5, 정렬=None):
    칸.quiver(0, 0, 0, *끝, color=색, linewidth=굵기, arrow_length_ratio=0.15)
    if 이름:                                # 글자는 화살 끝 아래(축 이름과 겹치지 않게)
        칸.text(*(np.array(끝) + 밀기), 이름, color=색, fontsize=10, va="top",
               ha=정렬 or ("right" if 끝[1] > 0.5 else "left" if 끝[0] > 0.5 else "center"),
               bbox=dict(facecolor="white", edgecolor="none", pad=0.5, alpha=0.8))


def 호(칸, 축_이름, 벡터, 색):              # 회전 방향을 보여 주는 90도 호 하나
    R = 축들[축_이름][0]
    if np.allclose(R(90) @ 벡터, 벡터):     # 축과 나란한 벡터: 끝점이 안 움직이므로 축 둘레에 작은 고리
        t = np.deg2rad(np.linspace(-45, 45, 30))
        점들 = np.array([0.6 + 0 * t, 0.3 * np.cos(t), 0.3 * np.sin(t)]).T
    else:                                   # 끝점이 지나가는 길(반지름 0.6)
        점들 = np.array([0.6 * R(a) @ 벡터 for a in np.linspace(0, 90, 30)])
    칸.plot(*점들.T, color=색, linewidth=1.8)
    칸.quiver(*점들[-4], *(점들[-1] - 점들[-4]), color=색, linewidth=1.8, arrow_length_ratio=1.2)


그림 = plt.figure(figsize=(10.5, 7.2))
격자 = 그림.add_gridspec(2, 12)
위_칸들 = []
for 번호, (이름, (R, 색, 설명)) in enumerate(축들.items()):
    칸 = 그림.add_subplot(격자[0, 4 * 번호:4 * 번호 + 4], projection="3d")
    바탕(칸)
    뒤 = R(90) @ 앞
    호(칸, 이름, 앞, 색)
    if np.allclose(뒤, 앞):                 # x축: 앞이 축 위에 있어 그대로 — 한 화살표로 정직하게
        화살표(칸, 뒤, "tab:orange", f"앞 = 결과 {글(뒤)}", 밀기=(0, 0, -0.3), 정렬="center")
    else:
        화살표(칸, 앞, "gray", "앞", 굵기=2)
        화살표(칸, 뒤, "tab:orange", 글(뒤))
    위_칸들.append((칸, f"{이름}축으로 90도", 설명))

아래_칸들 = []
for 번호, (제목, 식, 순서, 중간, 끝) in enumerate([
        ("가: x 먼저, z 나중", "Rz(90) @ Rx(90) @ 앞", "xz", 가_중간, 가),
        ("나: z 먼저, x 나중", "Rx(90) @ Rz(90) @ 앞", "zx", 나_중간, 나)]):
    칸 = 그림.add_subplot(격자[1, 2 + 4 * 번호:6 + 4 * 번호], projection="3d")
    바탕(칸)
    호(칸, 순서[0], 앞, 축들[순서[0]][1])
    호(칸, 순서[1], 중간, 축들[순서[1]][1])
    if np.allclose(중간, 앞):
        화살표(칸, 앞, "gray", f"앞 = ① {글(중간)}", 굵기=2)
    else:
        화살표(칸, 앞, "gray", "앞", 굵기=2)
        화살표(칸, 중간, "tab:purple", f"① {글(중간)}")
    화살표(칸, 끝, "tab:orange", f"② {글(끝)}", **(dict(밀기=(0, -0.12, 0), 정렬="left") if 끝[2] > 0.5 else {}))
    아래_칸들.append((칸, 제목, 식))

그림.subplots_adjust(left=0.0, right=1.0, bottom=0.06, top=0.9, hspace=0.3, wspace=0.0)
for 칸, 제목, 아래글 in 위_칸들 + 아래_칸들:
    자리 = 칸.get_position()
    가운데 = (자리.x0 + 자리.x1) / 2
    그림.text(가운데, 자리.y1 + 0.035, 제목, ha="center", fontsize=12)
    그림.text(가운데, 자리.y1 + 0.008, 아래글, ha="center", fontsize=10, color="dimgray")
그림.text(0.5, 0.025, "@는 오른쪽부터 적용된다. 같은 두 회전, 순서만 바꿨는데 앞이 왼쪽과 위로 갈렸다",
         ha="center", fontsize=11)
그림.savefig("s39_rot3d_order.png", dpi=200)
print("저장: s39_rot3d_order.png")
