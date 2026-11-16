import matplotlib.pyplot as plt
import numpy as np

# plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["font.family"] = "D2Coding"
plt.rcParams["axes.unicode_minus"] = False

# 1) 위치 벡터: 원점 O에서 본 로봇(v1)과 컵(v2)
원점, 로봇, 컵 = np.zeros(2), np.array([1.5, 2.0]), np.array([-1.0, 1.5])
컵에서_로봇 = 로봇 - 컵                      # v1 - v2: 컵에서 출발해 로봇에 닿는 화살표
print("v1 - v2 =", 컵에서_로봇, "/ 길이", round(float(np.linalg.norm(컵에서_로봇)), 3), "m")

# 2) 좌표축: 길이 1인 세 방향(오른손 좌표계)과 외적 규칙
축 = {"x": np.array([1, 0, 0]), "y": np.array([0, 1, 0]), "z": np.array([0, 0, 1])}


def 축_이름(벡터):
    for 이름, 값 in 축.items():
        if np.array_equal(벡터, 값):
            return 이름
        if np.array_equal(벡터, -값):
            return "-" + 이름


규칙들 = []
for 앞, 뒤 in ["xy", "yz", "zx"]:           # 순서대로 곱하면 다음 축, 거꾸로 곱하면 부호가 뒤집힌다
    규칙들.append(f"{앞} × {뒤} = {축_이름(np.cross(축[앞], 축[뒤])):<2}    "
                  f"{뒤} × {앞} = {축_이름(np.cross(축[뒤], 축[앞]))}")
print("\n".join(규칙들))
print("주의: x * y =", 축["x"] * 축["y"], "→ 성분끼리 곱한 값이라 외적이 아니다(np.cross를 쓴다)")

그림 = plt.figure(figsize=(10, 3.8))
격자 = 그림.add_gridspec(1, 3, width_ratios=[1.15, 1, 0.85])

벡터_칸 = 그림.add_subplot(격자[0])
for 시작, 끝, 색, 글, 밀기 in [(원점, 로봇, "tab:blue", r"$\vec{v}_1$", (0.12, -0.2)),
                           (원점, 컵, "tab:orange", r"$\vec{v}_2$", (-0.4, -0.2)),
                           (컵, 로봇, "tab:red", r"$\vec{v}_1 - \vec{v}_2$", (-0.35, 0.15))]:
    벡터_칸.annotate("", xy=끝, xytext=시작, arrowprops=dict(arrowstyle="-|>", color=색, linewidth=2))
    벡터_칸.text(*((시작 + 끝) / 2 + 밀기), 글, color=색, fontsize=13)
for 점, 글, 밀기 in [(원점, "원점 O", (0.1, -0.3)), (로봇, "로봇", (0.1, 0.05)), (컵, "컵", (-0.4, 0.05))]:
    벡터_칸.scatter(*점, s=30, color="black", zorder=5)
    벡터_칸.text(*(점 + 밀기), 글, fontsize=11)
벡터_칸.set_xlim(-1.7, 2.3)
벡터_칸.set_ylim(-0.6, 2.8)
벡터_칸.set_xlabel("x (m)")
벡터_칸.set_ylabel("y (m)")
벡터_칸.set_aspect("equal")
벡터_칸.grid(alpha=0.3)

축_칸 = 그림.add_subplot(격자[1], projection="3d")
for (이름, 벡터), (색, 가로, 세로) in zip(축.items(), [("tab:red", "center", "bottom"),
                                                 ("tab:green", "right", "center"),
                                                 ("tab:blue", "center", "bottom")]):
    축_칸.quiver(0, 0, 0, *벡터, color=색, linewidth=2.5, arrow_length_ratio=0.12)
    축_칸.text(*(벡터 * 1.08), f"{이름} {tuple(벡터.tolist())}", color=색, ha=가로, va=세로, fontsize=11)
축_칸.plot([1, 1, 0], [0, 1, 1], [0, 0, 0], color="lightgray", linestyle="--")   # 바닥(xy 평면)의 테두리
축_칸.set(xlim=(0, 1), ylim=(0, 1), zlim=(0, 1))
축_칸.set_box_aspect((1, 1, 1))
축_칸.view_init(elev=22, azim=155)           # z는 위, y는 왼쪽, x는 화면 안쪽을 향하게 본다
축_칸.set_axis_off()

규칙_칸 = 그림.add_subplot(격자[2])
규칙_칸.axis("off")
규칙_칸.text(0.5, 0.5, "\n\n".join(규칙들), fontsize=12, ha="center", va="center")

그림.subplots_adjust(left=0.07, right=0.98, bottom=0.15, top=0.87, wspace=0.05)
for 칸, 제목 in [(벡터_칸, "위치 벡터의 뺄셈"), (축_칸, "오른손 좌표계"), (규칙_칸, "외적 규칙")]:
    자리 = 칸.get_position()
    그림.text((자리.x0 + 자리.x1) / 2, 0.93, 제목, ha="center", fontsize=12)
그림.savefig("vectors_axes.png", dpi=200)
print("저장: vectors_axes.png")
