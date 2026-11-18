"""4장 실습: 탁자 위의 평면 2관절 팔과 작은 상자."""
import numpy as np

L1, L2 = 0.30, 0.25                 # 위팔·아래팔 길이(m)
팔_높이 = 0.45                      # 팔이 움직이는 평면의 높이(탁자 윗면 0.4m보다 5cm 위)

XML = f"""
<mujoco model="평면 2관절 팔">
  <option timestep="0.002" integrator="implicitfast"/>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.95 0.95 1" rgb2="0.6 0.7 0.8" width="256" height="256"/>
    <texture name="격자" type="2d" builtin="checker" rgb1="0.25 0.3 0.35" rgb2="0.75 0.75 0.75"
             width="256" height="256"/>
    <material name="바닥재" texture="격자" texrepeat="8 8"/>
  </asset>
  <default>
    <joint type="hinge" axis="0 0 1" damping="0.05"/>
    <geom friction="0.6 0.005 0.0001"/>
    <position kp="50" dampratio="1" ctrlrange="-3.14 3.14"/>
  </default>
  <worldbody>
    <light pos="0.3 -0.5 2.5" dir="0 0.2 -1"/>
    <camera name="overview" pos="1.0 1.1 1.05" xyaxes="-0.785 0.62 0 -0.31 -0.393 0.866"/>
    <geom name="floor" type="plane" size="2 2 0.1" material="바닥재"/>
    <geom name="table" type="box" pos="0.3 0 0.2" size="0.4 0.35 0.2" rgba="0.75 0.6 0.45 1"/>
    <body name="base" pos="0 0 {팔_높이}">
      <geom type="cylinder" pos="0 0 -0.025" size="0.04 0.025" rgba="0.3 0.3 0.3 1" contype="0" conaffinity="0"/>
      <body name="upper">
        <joint name="shoulder"/>
        <geom type="capsule" fromto="0 0 0 {L1} 0 0" size="0.02" mass="0.3" rgba="0.2 0.5 0.9 1"/>
        <body name="lower" pos="{L1} 0 0">
          <joint name="elbow"/>
          <geom type="capsule" fromto="0 0 0 {L2} 0 0" size="0.018" mass="0.2" rgba="0.2 0.7 0.5 1"/>
          <site name="tip" pos="{L2} 0 0" size="0.012" rgba="1 0.2 0.2 1"/>
        </body>
      </body>
    </body>
    <body name="block" pos="0.4 0.15 0.43">
      <freejoint name="block"/>
      <geom name="block" type="box" size="0.03 0.03 0.03" mass="0.1" rgba="0.95 0.75 0.1 1"/>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder" joint="shoulder"/>
    <position name="elbow" joint="elbow"/>
  </actuator>
</mujoco>
"""


def 손끝_계산(어깨: float, 팔꿈치: float) -> np.ndarray:
    """3장의 순기구학: 관절 각도(rad)로 손끝의 (x, y)를 계산한다(팔 받침 기준)."""
    return np.array([L1 * np.cos(어깨) + L2 * np.cos(어깨 + 팔꿈치),
                     L1 * np.sin(어깨) + L2 * np.sin(어깨 + 팔꿈치)])
