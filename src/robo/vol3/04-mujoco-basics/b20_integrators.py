import mujoco

# 1kg 공을 위치 액추에이터로 0.1m에 세운다. kv(속도 감쇠)가 크고 timestep이 0.01초로 크다
xml = """
<mujoco>
  <option timestep="0.01" integrator="{적분기}" gravity="0 0 0"/>
  <worldbody>
    <body>
      <joint name="slide" type="slide" axis="1 0 0"/>
      <geom type="sphere" size="0.05" mass="1"/>
    </body>
  </worldbody>
  <actuator><position joint="slide" kp="100" kv="{kv}"/></actuator>
</mujoco>
"""
for kv in [20, 300]:
    for 적분기 in ["Euler", "implicitfast", "RK4"]:
        모델 = mujoco.MjModel.from_xml_string(xml.replace("{적분기}", 적분기).replace("{kv}", str(kv)))
        데이터 = mujoco.MjData(모델)
        데이터.ctrl[0] = 0.1
        while 데이터.time < 2:
            mujoco.mj_step(모델, 데이터)
        경고 = 데이터.warning[mujoco.mjtWarning.mjWARN_BADQACC].number    # 불안정 경고 횟수
        print(f"kv={kv:3d}, {적분기:12s}: 2초 뒤 위치 {데이터.qpos[0]:.4f}m, 불안정 경고 {경고}번")
