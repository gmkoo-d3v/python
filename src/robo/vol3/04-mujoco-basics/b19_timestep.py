import matplotlib.pyplot as plt
import mujoco
import numpy as np

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# 용수철에 매달린 1kg 공(무중력). 강성 10000N/m라 1초에 약 16번 떤다
xml = """
<mujoco>
  <option timestep="{dt}" gravity="0 0 0"/>
  <worldbody>
    <body>
      <joint type="slide" axis="1 0 0" stiffness="10000"/>
      <geom type="sphere" size="0.05" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""
고유_각진동수 = np.sqrt(10000 / 1)                # √(k/m) = 100 rad/s
print(f"한 번 떠는 시간: {2 * np.pi / 고유_각진동수:.4f}초, 2/ω = {2 / 고유_각진동수:.3f}초")

그림, 칸들 = plt.subplots(1, 3, figsize=(9, 2.8))
for 칸, dt in zip(칸들, [0.002, 0.019, 0.021]):
    모델 = mujoco.MjModel.from_xml_string(xml.replace("{dt}", str(dt)))
    데이터 = mujoco.MjData(모델)
    데이터.qpos[0] = 0.1                           # 10cm 당겼다 놓는다
    시각, 위치 = [0.0], [0.1]
    while 데이터.time < 0.5:
        mujoco.mj_step(모델, 데이터)
        시각.append(데이터.time)
        위치.append(데이터.qpos[0])
    print(f"timestep {dt}: 0.5초 동안 가장 멀리 {np.max(np.abs(위치)):.3g}m")
    칸.plot(시각, 위치, marker=".", markersize=3)
    칸.set_title(f"timestep = {dt}초", fontsize=10)
    칸.set_xlabel("시간(초)")
칸들[0].set_ylabel("위치(m)")
그림.tight_layout()
그림.savefig("timestep.png", dpi=200)
print("저장: timestep.png")
