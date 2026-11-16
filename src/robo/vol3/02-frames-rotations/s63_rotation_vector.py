import matplotlib.pyplot as plt
import numpy as np

# plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["font.family"] = "D2Coding"
plt.rcParams["axes.unicode_minus"] = False


# b12_rodrigues.py의 축각_행렬과 같은 식
def 축각_행렬(축, 각도_도):
    k = 축 / np.linalg.norm(축)
    라디안 = np.deg2rad(각도_도)
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])        # K @ v == np.cross(k, v)
    return np.eye(3) + np.sin(라디안) * K + (1 - np.cos(라디안)) * (K @ K)


def 글(v):                                   # [0. 1. 0.] → "[0, 1, 0]"
    return "[" + ", ".join(f"{x:g}" for x in (np.round(v, 3) + 0.0).tolist()) + "]"


축 = np.array([1.0, 1.0, 1.0])               # 정육면체의 대각선 방향
k = 축 / np.linalg.norm(축)
R = 축각_행렬(축, 120)
회전벡터 = k * np.deg2rad(120)                # 방향 = 축, 길이 = 각도(라디안)
단위들 = [("x", np.eye(3)[0], "tab:red"), ("y", np.eye(3)[1], "tab:green"), ("z", np.eye(3)[2], "tab:blue")]

# 확인: x → y → z → x, 세 끝점이 축에서 같은 거리
for (이름, v, _), (다음, w, _) in zip(단위들, 단위들[1:] + 단위들[:1]):
    print(f"R @ {이름} = {글(R @ v)} → {다음}와 같은가? {np.allclose(R @ v, w)}")
거리들 = [np.linalg.norm(v - (v @ k) * k) for _, v, _ in 단위들]
print("축에서 끝점까지 거리:", np.round(거리들, 3).tolist())
print("(1,1,1)축으로 120도:\n", np.round(R, 3) + 0.0)
print("회전 벡터:", 글(회전벡터), "/ 길이:", round(float(np.linalg.norm(회전벡터)), 3), "rad =",
      round(float(np.rad2deg(np.linalg.norm(회전벡터))), 3), "도")

그림 = plt.figure(figsize=(10, 6))
칸 = 그림.add_axes([-0.06, 0.08, 0.6, 0.8], projection="3d")

# 정육면체(흐리게, 한 변 1): 좌표 하나만 다른 꼭짓점 쌍이 모서리
꼭짓점 = np.array([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)], dtype=float)
for i in range(8):
    for j in range(i + 1, 8):
        if np.sum(np.abs(꼭짓점[i] - 꼭짓점[j])) == 1:
            칸.plot(*zip(꼭짓점[i], 꼭짓점[j]), color="lightgray", linewidth=0.9, linestyle="--")
칸.text(1.0, 1.0, -0.04, " 정육면체 (한 변 1)", color="gray", fontsize=10, ha="left", va="top")

# 세 끝점이 도는 원: x 끝점을 축 둘레로 0~360도
원 = np.array([축각_행렬(축, a) @ np.eye(3)[0] for a in np.linspace(0, 360, 181)])
칸.plot(*원.T, color="gray", linewidth=1, alpha=0.6)

# 각 끝점이 다음 끝점으로 120도: 호(끝점 가까이는 비움) + 일정한 크기의 화살촉
for 이름, v, 색 in 단위들:
    점들 = np.array([축각_행렬(축, a) @ v for a in np.linspace(10, 110, 40)])
    칸.plot(*점들.T, color="tab:purple", linewidth=2)
    방향 = (점들[-1] - 점들[-2]) / np.linalg.norm(점들[-1] - 점들[-2]) * 0.1
    칸.quiver(*(점들[-1] - 방향), *방향, color="tab:purple", linewidth=2, arrow_length_ratio=1.0)
가운데 = 축각_행렬(축, 60) @ np.eye(3)[0]     # x→y 호의 가운데
칸.text(*(가운데 + 0.12 * (가운데 - k / np.sqrt(3))), "끝점이 120°씩 이동", color="tab:purple",
        fontsize=11, ha="center", va="top")

# x, y, z 단위 벡터
for 이름, v, 색 in 단위들:
    칸.quiver(0, 0, 0, *v, color=색, linewidth=2.5, arrow_length_ratio=0.12)
    칸.text(*(v * 1.18), 이름, color=색, fontsize=14, ha="center", va="center")

# 회전 벡터 = 축 방향, 길이 2.094(rad)
칸.quiver(0, 0, 0, *회전벡터, color="tab:orange", linewidth=3, arrow_length_ratio=0.07)
칸.text(*(회전벡터 * 1.04), "  회전 벡터\n  방향 = 축 (1,1,1)\n  길이 = 2.094 = 120°",
        color="tab:orange", fontsize=11, ha="left", va="center")
칸.set(xlim=(-0.2, 1.3), ylim=(-0.2, 1.3), zlim=(-0.1, 1.4))
칸.set_box_aspect((1, 1, 1))
칸.view_init(elev=25, azim=5)              # 대각선에서 36도 비껴 본다: 원점이 원 안쪽, 세 끝점이 갈라진다
칸.set_axis_off()

# 오른쪽: 슬라이드 숫자
줄들 = [("축 (1,1,1)을 길이 1로", "black", 12), (f"k = {글(k)}", "dimgray", 11), None,
       ("각도", "black", 12), ("θ = 120° = 2.094 rad", "dimgray", 11), None,
       ("회전 벡터 = θ × k", "tab:orange", 12), (f"= {글(회전벡터)}", "tab:orange", 11), None,
       ("(1,1,1)축으로 120도:", "black", 12)]
줄들 += [(" [" + "  ".join(f"{x:g}" for x in (np.round(행, 3) + 0.0).tolist()) + "]", "black", 12) for 행 in R]
줄들 += [None, ("x → y → z → x 자리 바꾸기", "tab:purple", 12)]
y = 0.82
for 줄 in 줄들:
    if 줄 is None:
        y -= 0.03
        continue
    그림.text(0.62, y, 줄[0], color=줄[1], fontsize=줄[2], ha="left", va="top")
    y -= 0.05
그림.text(0.25, 0.93, "어떤 회전이든 축 하나로 한 번", ha="center", fontsize=13)
그림.text(0.62, 0.93, "숫자로 보면", ha="left", fontsize=13)
그림.text(0.5, 0.035, "세 숫자 하나로 회전을 적는다: 방향 = 축, 길이 = 각도(rad)", ha="center", fontsize=12)
그림.savefig("s63_rotation_vector.png", dpi=200)
print("저장: s63_rotation_vector.png")
