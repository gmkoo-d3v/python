import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle, Wedge
from matplotlib.path import Path

# plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["font.family"] = "D2Coding"
plt.rcParams["axes.unicode_minus"] = False

# b07_dof.py와 같은 표: 관절 하나가 허락하는 독립 방향의 수
관절_자유도 = {"회전": 1, "직선": 1, "공": 3}
로봇_팔들 = {"평면 2관절 팔": ["회전", "회전"], "스카라 팔(조립용)": ["회전", "회전", "직선", "회전"],
          "6축 산업용 팔": ["회전"] * 6, "사람 팔(어깨~손목)": ["공", "회전", "회전", "회전", "회전"]}
for 이름, 관절들 in 로봇_팔들.items():
    print(f"{이름}: 관절 {len(관절들)}개, 자유도 {sum(관절_자유도[관절] for 관절 in 관절들)}")
print("공간 속 물체:", 3 + 3, "/ 바닥 위 물체(앞뒤·좌우·제자리 돌기):", 2 + 1)
# MuJoCo는 여기서 돌릴 수 없으므로 b18_joint_dof.py의 출력(슬라이드 58쪽)을 그대로 옮겨 적는다
무조코 = {"회전": "hinge(경첩): nq=1, nv=1", "직선": "slide(미닫이): nq=1, nv=1",
       "공": "ball(공): nq=4, nv=3", "자유": "free(자유): nq=7, nv=6"}

축색 = {"x": "tab:red", "y": "tab:green", "z": "tab:blue"}
축벡터 = {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]), "z": np.array([0, 0, 1.0])}
화살표_수 = {}


def 투영(점들):                   # 3D → 그림 평면: y는 오른쪽, z는 위, x는 왼쪽 아래(보는 사람 쪽)
    점들 = np.atleast_2d(점들)
    return np.column_stack([점들[:, 1] - 0.45 * 점들[:, 0], 점들[:, 2] - 0.3 * 점들[:, 0]])


def 원호(중심, 축, 반지름, 시작도, 끝도):  # 축 둘레의 원호(3D 점들)
    u, v = {"x": ("y", "z"), "y": ("z", "x"), "z": ("x", "y")}[축]
    t = np.radians(np.linspace(시작도, 끝도, 60))[:, None]
    return np.asarray(중심) + 반지름 * (np.cos(t) * 축벡터[u] + np.sin(t) * 축벡터[v])


def 화살표(칸, 점들, 색):          # 움직일 수 있는 방향 하나 = 양쪽 화살표 하나
    칸.add_patch(FancyArrowPatch(path=Path(점들), arrowstyle="<|-|>", mutation_scale=13,
                                 color=색, linewidth=2.2, zorder=6))
    화살표_수[칸] = 화살표_수.get(칸, 0) + 1


def 상자(칸, 중심, 반):            # 정육면체를 투영해 보이는 세 면만 칠한다
    c = np.asarray(중심)
    for 고정, a, b in [("x", "y", "z"), ("y", "x", "z"), ("z", "x", "y")]:
        면 = [c + 반 * (축벡터[고정] + s * 축벡터[a] + t * 축벡터[b]) for s, t in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
        칸.add_patch(Polygon(투영(np.array(면)), closed=True, facecolor="navajowhite", edgecolor="dimgray", alpha=0.8))


그림 = plt.figure(figsize=(10, 6.6))
위 = 그림.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.75], left=0.01, right=0.99, top=0.9, bottom=0.55, wspace=0.08)
아래 = 그림.add_gridspec(1, 2, left=0.06, right=0.94, top=0.4, bottom=0.04, wspace=0.15)
칸들 = [그림.add_subplot(위[i]) for i in range(4)] + [그림.add_subplot(아래[i]) for i in range(2)]
회전_칸, 직선_칸, 공_칸, 자유_칸, 팔_칸, 바닥_칸 = 칸들

# 1) 회전(경첩): 고정된 막대에 핀으로 이어진 막대가 한 방향으로만 돈다
회전_칸.plot([-1, 0], [0, 0], color="gray", linewidth=7, solid_capstyle="round")
끝 = 1.0 * np.array([np.cos(np.radians(40)), np.sin(np.radians(40))])
회전_칸.plot([0, 끝[0]], [0, 끝[1]], color="dimgray", linewidth=7, solid_capstyle="round")
회전_칸.add_patch(Circle((0, 0), 0.09, color="white", ec="black", zorder=5))
t = np.radians(np.linspace(8, 75, 40))
화살표(회전_칸, np.column_stack([0.65 * np.cos(t), 0.65 * np.sin(t)]), "tab:orange")

# 2) 직선(미닫이): 레일 위의 블록이 한 줄로만 미끄러진다
직선_칸.add_patch(Rectangle((-1.1, -0.08), 2.2, 0.08, color="gray"))
직선_칸.add_patch(Rectangle((-0.3, 0), 0.6, 0.35, facecolor="navajowhite", edgecolor="dimgray"))
화살표(직선_칸, np.array([[-0.9, 0.55], [0.9, 0.55]]), "tab:orange")

# 3) 공(어깨): 소켓 안의 공이 세 축 둘레로 돈다
공_칸.add_patch(Wedge((0, 0), 0.42, 180, 360, width=0.1, color="gray"))
공_칸.plot(*투영(np.array([[0, 0, 0], [0, 0, 1.0]])).T, color="dimgray", linewidth=6, solid_capstyle="round")
공_칸.add_patch(Circle((0, 0), 0.3, facecolor="navajowhite", edgecolor="dimgray", zorder=4))
화살표(공_칸, 투영(원호([0, 0, 0], "x", 1.15, 65, 115)), 축색["x"])     # 좌우로 기울이기
화살표(공_칸, 투영(원호([0, 0, 0], "y", 1.15, -40, 40)), 축색["y"])     # 앞뒤로 기울이기(x-z 평면)
화살표(공_칸, 투영(원호([0, 0, 0.62], "z", 0.3, -150, 120)), 축색["z"])  # 막대를 비틀기

# 4) 자유(free): 위치 3(직선 화살표) + 자세 3(회전 화살표)
반 = 0.32
for 중심, 종류 in [((0, -1.0, 0), "위치"), ((0, 1.0, 0), "자세")]:
    상자(자유_칸, 중심, 반)
    for 축, 길이 in zip("xyz", [1.25, 0.8, 0.8]):      # x는 비스듬히 짧아 보이므로 길게
        a, b = np.array(중심) - 길이 * 축벡터[축], np.array(중심) + 길이 * 축벡터[축]
        if 종류 == "위치":
            화살표(자유_칸, 투영(np.array([a, b])), 축색[축])
        else:
            자유_칸.plot(*투영(np.array([중심, b])).T, color="gray", linewidth=1, linestyle="--")
            화살표(자유_칸, 투영(원호(b, 축, 0.17, -120, 150)), 축색[축])
    x, y = 투영(np.array(중심) - np.array([0, 0, 0.95]))[0]
    자유_칸.text(x, y, f"{종류} 3", ha="center", va="top", fontsize=11)

# 5) 평면 2관절 팔: 순운동학으로 관절 위치를 계산하고 θ1, θ2를 호로 표시
L1, L2, θ1, θ2 = 1.0, 0.8, np.radians(35), np.radians(55)
팔꿈치 = L1 * np.array([np.cos(θ1), np.sin(θ1)])
손끝 = 팔꿈치 + L2 * np.array([np.cos(θ1 + θ2), np.sin(θ1 + θ2)])
print("팔꿈치", np.round(팔꿈치, 3), "/ 손끝", np.round(손끝, 3))
팔_칸.add_patch(Rectangle((-0.2, -0.15), 0.4, 0.15, color="gray"))
팔_칸.plot([0, 0.8], [0, 0], color="gray", linestyle="--", linewidth=1)
팔_칸.plot(*np.array([팔꿈치, 팔꿈치 + 0.5 * 팔꿈치 / L1]).T, color="gray", linestyle="--", linewidth=1)
팔_칸.plot(*np.array([[0, 0], 팔꿈치, 손끝]).T, color="dimgray", linewidth=6, solid_capstyle="round")
for 점 in [(0, 0), 팔꿈치]:
    팔_칸.add_patch(Circle(점, 0.06, color="white", ec="black", zorder=5))
for 중심, 시작, 크기, 반지름, 색, 글 in [((0, 0), 0, θ1, 0.6, "tab:orange", "θ1"),
                                (팔꿈치, θ1, θ2, 0.42, "tab:purple", "θ2")]:
    t = np.linspace(시작, 시작 + 크기, 40)
    화살표(팔_칸, np.asarray(중심) + 반지름 * np.column_stack([np.cos(t), np.sin(t)]), 색)
    가운데 = 시작 + 크기 / 2
    팔_칸.text(*(np.asarray(중심) + (반지름 + 0.15) * np.array([np.cos(가운데), np.sin(가운데)])), 글, color=색,
              fontsize=13, ha="center", va="center")

# 6) 바닥 위 물체(위에서 본 모습): 앞뒤(x), 좌우(y), 제자리 돌기(z축 둘레)
바닥_칸.add_patch(Rectangle((-0.45, -0.3), 0.9, 0.6, facecolor="navajowhite", edgecolor="dimgray"))
바닥_칸.add_patch(Polygon([[0.45, -0.12], [0.45, 0.12], [0.6, 0]], color="dimgray"))   # 앞쪽 표시
화살표(바닥_칸, np.array([[-1.05, 0], [1.05, 0]]), 축색["x"])
화살표(바닥_칸, np.array([[0, -0.75], [0, 0.75]]), 축색["y"])
t = np.radians(np.linspace(15, 75, 40))
화살표(바닥_칸, 0.9 * np.column_stack([np.cos(t), np.sin(t)]) + [0, 0], 축색["z"])
for 위치, 글, 색 in [((1.1, -0.12), "앞뒤", "x"), ((0.08, -0.82), "좌우", "y"), ((0.75, 0.82), "제자리 돌기", "z")]:
    바닥_칸.text(*위치, 글, color=축색[색], fontsize=11, ha="left", va="top" if 색 != "z" else "bottom")

# 칸 범위, 제목, 자유도 글씨. 그린 화살표 수가 자유도와 같은지 확인한다
자유도들 = [관절_자유도["회전"], 관절_자유도["직선"], 관절_자유도["공"], 3 + 3,
         sum(관절_자유도[관절] for 관절 in 로봇_팔들["평면 2관절 팔"]), 2 + 1]
범위들 = [(-1.1, 1.1, -0.5, 1.35), (-1.1, 1.1, -0.5, 1.35), (-1.1, 1.1, -0.5, 1.35),
        (-2.0, 2.0, -1.25, 1.35), (-0.4, 1.4, -0.25, 1.75), (-1.3, 2.0, -1.0, 1.15)]
제목들 = ["회전 관절", "직선 관절", "공 관절", "자유 물체 (관절 없음)", "평면 2관절 팔", "바닥 위 물체 (위에서 본 모습)"]
아랫글 = [f"자유도 {자유도들[0]}\n{무조코['회전']}", f"자유도 {자유도들[1]}\n{무조코['직선']}",
       f"자유도 {자유도들[2]}\n{무조코['공']}", f"위치 3 + 자세 3 = 자유도 {자유도들[3]}\n{무조코['자유']}",
       f"관절 2개 → 자유도 {자유도들[4]}", f"앞뒤 + 좌우 + 제자리 돌기 → 자유도 {자유도들[5]}"]
for 칸, (x0, x1, y0, y1) in zip(칸들, 범위들):
    칸.set_xlim(x0, x1)
    칸.set_ylim(y0, y1)
    칸.set_aspect("equal")
    칸.axis("off")
    칸.apply_aspect()                       # 비율을 맞춘 뒤의 실제 칸 위치에 글씨를 붙인다
for 줄칸 in [칸들[:4], 칸들[4:]]:             # 한 줄의 제목과 아랫글은 같은 높이에 맞춘다
    위끝 = max(칸.get_position().y1 for 칸 in 줄칸)
    아래끝 = min(칸.get_position().y0 for 칸 in 줄칸)
    for 칸 in 줄칸:
        i, 자리 = 칸들.index(칸), 칸.get_position()
        가운데 = (자리.x0 + 자리.x1) / 2
        그림.text(가운데, 위끝 + 0.025, 제목들[i], ha="center", fontsize=12)
        줄 = 아랫글[i].split("\n")
        그림.text(가운데, 아래끝 - 0.01, 줄[0], ha="center", va="top", fontsize=12)
        if len(줄) > 1:
            그림.text(가운데, 아래끝 - 0.05, 줄[1], ha="center", va="top", fontsize=10, color="gray")
for 칸, 제목, 자유도 in zip(칸들, 제목들, 자유도들):
    print(f"{제목}: 화살표 {화살표_수[칸]}개, 자유도 {자유도}", "일치" if 화살표_수[칸] == 자유도 else "불일치!")
그림.savefig("s17_dof_joints.png", dpi=200)
print("저장: s17_dof_joints.png")
