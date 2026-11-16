import matplotlib.pyplot as plt
import numpy as np

# plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["font.family"] = "D2Coding"
plt.rcParams["axes.unicode_minus"] = False

# 카메라 좌표(x 오른쪽, y 아래, z 앞)로 잰 값을 로봇 몸 좌표(x 앞, y 왼쪽, z 위)로 옮기는 행렬 (b09와 같다)
R_몸_카메라 = np.array([[ 0,  0, 1],
                     [-1,  0, 0],
                     [ 0, -1, 0]])
print("R_몸_카메라 =\n", R_몸_카메라)
print("행렬식:", round(float(np.linalg.det(R_몸_카메라)), 6))
print("직교 행렬인가?", np.allclose(R_몸_카메라.T @ R_몸_카메라, np.eye(3)))
for 열, 이름 in enumerate("xyz"):
    print(f"열 {열 + 1} = 카메라 {이름}축을 몸 좌표로:", R_몸_카메라[:, 열].tolist())

# 카메라가 본 컵: 화면 가운데에서 오른쪽 0.1m, 위쪽 0.05m, 앞으로 0.8m
컵_카메라 = np.array([0.1, -0.05, 0.8])
컵_몸 = np.round(R_몸_카메라 @ 컵_카메라, 3) + 0.0
print("몸 기준 컵 위치:", 컵_몸)

색들 = ["tab:red", "tab:green", "tab:blue"]           # x 빨강, y 초록, z 파랑
몸_뜻 = ["x 앞", "y 왼쪽", "z 위"]
카메라_뜻 = ["x 화면 오른쪽", "y 화면 아래", "z 렌즈 앞"]


def 바닥(칸):
    # 몸의 바닥(xy 평면) 테두리: 깊이를 읽기 위한 단서
    칸.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], [0] * 5, color="lightgray", linewidth=0.8)
    칸.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))
    칸.set_box_aspect((1, 1, 1))
    칸.view_init(elev=20, azim=-120)     # 로봇 뒤 오른쪽 위에서 본다
    칸.set_axis_off()


그림 = plt.figure(figsize=(10, 4.4))
격자 = 그림.add_gridspec(1, 3, width_ratios=[1, 1.15, 0.95])

# 1) 몸 좌표계
몸_칸 = 그림.add_subplot(격자[0], projection="3d")
for 열, (색, 뜻) in enumerate(zip(색들, 몸_뜻)):
    축 = np.eye(3)[:, 열]
    몸_칸.quiver(0, 0, 0, *축, color=색, linewidth=2.5, arrow_length_ratio=0.12)
    몸_칸.text(*(축 * 1.12), 뜻, color=색, ha=["left", "right", "center"][열], va=["center", "center", "bottom"][열], fontsize=11)
바닥(몸_칸)

# 2) 같은 원점의 카메라 광학 좌표계 (몸 축은 흐린 점선)
카메라_칸 = 그림.add_subplot(격자[1], projection="3d", computed_zorder=False)   # 그린 순서(zorder)대로 겹친다
for 열, 색 in enumerate(색들):
    카메라_칸.plot(*zip([0, 0, 0], np.eye(3)[:, 열]), color=색, alpha=0.35, linewidth=1.2, linestyle="--")
    if 열 > 0:                                      # 몸 x는 카메라 z와 겹쳐서 이름을 생략
        카메라_칸.text(*(np.eye(3)[:, 열] * 1.1), "몸 " + "xyz"[열], color=색, alpha=0.6, fontsize=10, ha="center")
for 열, (색, 뜻) in enumerate(zip(색들, 카메라_뜻)):
    축 = R_몸_카메라[:, 열]                          # 열 = 카메라 축이 몸 좌표에서 가리키는 방향
    몸_축 = int(np.argmax(np.abs(축)))               # 이 열이 몸의 어느 축(과 부호)인지 읽는다
    풀이 = f"(= 몸 {'xyz'[몸_축]} 방향)" if 축[몸_축] > 0 else f"(= 몸 {'xyz'[몸_축]}의 반대)"
    print(f"카메라 {'xyz'[열]}축 {풀이}")
    카메라_칸.quiver(0, 0, 0, *축, color=색, linewidth=3.5, arrow_length_ratio=0.12, zorder=3)
    자리, 가로, 세로 = 축 * 1.15, ["left", "center", "left"][열], ["top", "top", "bottom"][열]
    # 같은 크기·같은 기준점의 두 글상자를 빈 줄로 맞춰, 셋째 줄만 흐리게 쓴다
    카메라_칸.text(*자리, f"{뜻}\n열{열 + 1} = {축.tolist()}\n ", color=색, ha=가로, va=세로, fontsize=10)
    카메라_칸.text(*자리, f" \n \n{풀이}", color=색, alpha=0.6, ha=가로, va=세로, fontsize=10)
카메라_칸.scatter(*컵_몸, color="black", s=45, edgecolors="white", linewidths=1.5, depthshade=False, zorder=10)
카메라_칸.text(*(컵_몸 + [0, 0, -0.14]), "컵", fontsize=10, ha="center", va="top", zorder=10)
바닥(카메라_칸)

# 3) 행렬: 열마다 색을 칠한다
행렬_칸 = 그림.add_subplot(격자[2])
행렬_칸.axis("off")
행렬_칸.set(xlim=(0, 1), ylim=(0, 1))
for 행 in range(3):
    for 열 in range(3):
        행렬_칸.text(0.3 + 0.2 * 열, 0.86 - 0.11 * 행, f"{R_몸_카메라[행, 열]:>2d}", color=색들[열],
                   ha="center", va="center", fontsize=14)
for 끝, 방향 in [(0.13, 1), (0.87, -1)]:          # 대괄호
    행렬_칸.plot([끝 + 0.04 * 방향, 끝, 끝, 끝 + 0.04 * 방향], [0.95, 0.95, 0.55, 0.55], color="black")
for 열, 이름 in enumerate("xyz"):
    행렬_칸.text(0.3 + 0.2 * 열, 0.48, f"카메라\n{이름}축", color=색들[열], ha="center", va="center", fontsize=10)
행렬_칸.text(0.5, 0.3, "열 = 카메라 축을 몸 좌표로 적은 것", ha="center", fontsize=10)
행렬_칸.text(0.5, 0.08, f"컵_카메라 = {컵_카메라.tolist()}\n"
                     f"컵_몸 = R @ 컵_카메라\n      = {컵_몸.tolist()}",
           ha="center", va="bottom", fontsize=10, linespacing=1.4,
           bbox=dict(boxstyle="round", facecolor="whitesmoke", edgecolor="gray"))

그림.subplots_adjust(left=0.0, right=0.99, bottom=0.02, top=0.88, wspace=0.1)
for 칸, 제목 in [(몸_칸, "몸 좌표계"), (카메라_칸, "카메라 광학 좌표계 (흐린 점선 = 몸 축)"),
               (행렬_칸, "R_몸_카메라")]:
    자리 = 칸.get_position()
    그림.text((자리.x0 + 자리.x1) / 2, 0.93, 제목, ha="center", fontsize=12)
그림.savefig("s37_body_camera_frames.png", dpi=200)
print("저장: s37_body_camera_frames.png")
