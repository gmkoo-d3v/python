import mujoco

g = 9.81
xml = """
<mujoco>
  <option timestep="{간격}"/>
  <worldbody>
    <body pos="0 0 10"><freejoint/><geom type="sphere" size="0.1"/></body>
  </worldbody>
</mujoco>
"""
for 간격 in [0.002, 0.001, 0.0005, 0.004]:
    모델 = mujoco.MjModel.from_xml_string(xml.replace("{간격}", str(간격)))
    데이터 = mujoco.MjData(모델)
    while 데이터.time < 1.0 - 간격 / 2:         # 1초까지
        mujoco.mj_step(모델, 데이터)
    t = 데이터.time
    높이_차이_mm = (데이터.qpos[2] - (10 - 0.5 * g * t**2)) * 1000
    속도_차이 = 데이터.qvel[2] - (-g * t)
    print(f"간격 {간격}초: 높이 차이 {높이_차이_mm:6.3f}mm (예측 {-0.5 * g * 간격 * t * 1000:6.3f}mm), "
          f"속도 차이 {abs(속도_차이):.1e}m/s")
