import mujoco
import numpy as np

# 천장에 매달린 0.5m 진자를 옆으로 90도 들었다가 놓는다
xml = """
<mujoco>
  <worldbody>
    <body pos="0 0 1">
      <joint type="hinge" axis="0 1 0"/>
      <geom type="sphere" pos="0 0 -0.5" size="0.05"/>
    </body>
  </worldbody>
</mujoco>
"""
모델 = mujoco.MjModel.from_xml_string(xml)
데이터 = mujoco.MjData(모델)
데이터.qpos[0] = np.deg2rad(90)                   # 처음 각도

for _ in range(6):
    print(f"{데이터.time:.1f}초: 각도 {np.rad2deg(데이터.qpos[0]):6.1f}도")
    for _ in range(100):                         # 한 걸음 0.002초 × 100 = 0.2초
        mujoco.mj_step(모델, 데이터)
