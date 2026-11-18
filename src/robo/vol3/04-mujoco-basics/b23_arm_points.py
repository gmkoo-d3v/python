import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np

from 팔장면 import XML, 손끝_계산

plt.rcParams["font.family"] = "D2Coding"
plt.rcParams["axes.unicode_minus"] = False

모델 = mujoco.MjModel.from_xml_string(XML)
데이터 = mujoco.MjData(모델)
받침 = 데이터.body("base").xpos[:2].copy()          # 팔 받침의 (x, y): 순기구학 식의 원점

목표들_도 = [(-45, 45), (-60, 90), (-20, 30), (0, 0)]  # (어깨, 팔꿈치) 목표 각도
머무는_시간 = 1.0
시각, 손끝, 관절_도 = [], [], []
with mujoco.viewer.launch_passive(모델, 데이터) as 뷰어:
    while True:  # 반복 실행 (Ctrl+C로 종료)
        for 어깨_도, 팔꿈치_도 in 목표들_도:
            데이터.ctrl[:] = np.deg2rad([어깨_도, 팔꿈치_도])    # 위치 액추에이터에 목표 각도를 준다
            끝 = 데이터.time + 머무는_시간
            while 데이터.time < 끝 - 1e-9:
                mujoco.mj_step(모델, 데이터)
                mujoco.mj_forward(모델, 데이터)                # 사이트 위치를 지금 상태로 맞춘다
                시각.append(데이터.time)
                손끝.append(데이터.site("tip").xpos[:2].copy())
                관절_도.append(np.rad2deg(데이터.qpos[:2]))
                뷰어.sync()
        목표_손끝 = 받침 + 손끝_계산(*np.deg2rad([어깨_도, 팔꿈치_도]))
        오차_mm = np.linalg.norm(손끝[-1] - 목표_손끝) * 1000
        print(f"목표 ({어깨_도:3d}도, {팔꿈치_도:2d}도) → 손끝 목표 {np.round(목표_손끝, 3) + 0.0}, "
              f"도착 {np.round(손끝[-1], 3) + 0.0}, 차이 {오차_mm:.1f}mm")
손끝, 관절_도 = np.array(손끝), np.array(관절_도)
print("상자 위치:", np.round(데이터.body("block").xpos, 3), "(처음 0.4 0.15 0.43)")

그림, (왼쪽, 오른쪽) = plt.subplots(1, 2, figsize=(9, 3.6), gridspec_kw={"width_ratios": [1, 1.3]})
왼쪽.add_patch(plt.Rectangle((-0.1, -0.35), 0.8, 0.7, color="tan", alpha=0.3, label="탁자"))
왼쪽.add_patch(plt.Rectangle((0.37, 0.12), 0.06, 0.06, color="gold", label="상자"))
왼쪽.plot(손끝[:, 0], 손끝[:, 1], color="tab:red", label="손끝 궤적")
for i, (어깨_도, 팔꿈치_도) in enumerate(목표들_도, start=1):
    x, y = 받침 + 손끝_계산(*np.deg2rad([어깨_도, 팔꿈치_도]))
    왼쪽.plot(x, y, "kx", markersize=8)
    왼쪽.annotate(str(i), (x, y), textcoords="offset points", xytext=(6, -12))
왼쪽.plot(*받침, "ko", markersize=5)                    # 팔 받침(어깨)
왼쪽.set_xlim(-0.15, 0.75)
왼쪽.set_ylim(-0.4, 0.4)
왼쪽.set_aspect("equal")
왼쪽.set_xlabel("x (m)")
왼쪽.set_ylabel("y (m)")
왼쪽.legend(fontsize=8, loc="upper left")
오른쪽.plot(시각, 관절_도[:, 0], label="어깨")
오른쪽.plot(시각, 관절_도[:, 1], label="팔꿈치")
오른쪽.set_xlabel("시간(초)")
오른쪽.set_ylabel("관절 각도(도)")
오른쪽.legend(fontsize=8)
그림.tight_layout()
그림.savefig("arm_points.png", dpi=200)
print("저장: arm_points.png")
