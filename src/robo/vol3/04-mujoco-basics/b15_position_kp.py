import matplotlib.pyplot as plt
import mujoco
import numpy as np

plt.rcParams["font.family"] = "D2Coding"
plt.rcParams["axes.unicode_minus"] = False

# 매달린 진자(y축 경첩): 0.5m 막대 끝에 1kg 추
xml = """
<mujoco>
  <worldbody>
    <body pos="0 0 1">
      <joint name="pivot" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.01" mass="0.01"/>
      <geom type="sphere" pos="0 0 -0.5" size="0.04" mass="1"/>
    </body>
  </worldbody>
  <actuator><position name="servo" joint="pivot" {설정}/></actuator>
</mujoco>
"""


def 응답(설정: str, 목표_도: float, 시간: float = 3.0) -> tuple[list[float], np.ndarray]:
    """목표 각도를 주고 시간 동안 관절 각도(도)를 기록한다."""
    모델 = mujoco.MjModel.from_xml_string(xml.replace("{설정}", 설정))
    데이터 = mujoco.MjData(모델)
    데이터.ctrl[0] = np.deg2rad(목표_도)
    시각, 각도 = [], []
    while 데이터.time < 시간:
        mujoco.mj_step(모델, 데이터)
        시각.append(데이터.time)
        각도.append(데이터.qpos[0])
    return 시각, np.rad2deg(각도)


목표_도 = 60
그림, 칸들 = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
for kp in [5, 20, 100]:
    시각, 그냥 = 응답(f'kp="{kp}"', 목표_도)
    _, 감쇠 = 응답(f'kp="{kp}" dampratio="1"', 목표_도)
    print(f"kp={kp:3d}: 감쇠 없음 → {round(그냥.min(), 1) + 0.0:5.1f}~{그냥.max():5.1f}도 사이를 오감 / "
          f"dampratio=1 → 3초 뒤 {감쇠[-1]:5.1f}도 (목표와 {목표_도 - 감쇠[-1]:4.1f}도 차이)")
    칸들[0].plot(시각, 그냥, label=f"kp={kp}")
    칸들[1].plot(시각, 감쇠, label=f"kp={kp}")
for 칸, 제목 in zip(칸들, ["kp만 (감쇠 없음)", "kp + dampratio=1"]):
    칸.axhline(목표_도, color="gray", linestyle="--", linewidth=1)
    칸.set_title(제목, fontsize=10)
    칸.set_xlabel("시간(초)")
칸들[0].set_ylabel("관절 각도(도)")
칸들[1].legend(fontsize=8, loc="lower center", ncol=3)   # 색은 두 칸이 같다
그림.tight_layout()
그림.savefig("position_kp.png", dpi=200)
print("저장: position_kp.png")
