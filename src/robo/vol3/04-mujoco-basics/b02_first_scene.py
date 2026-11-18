import mujoco

xml = """
<mujoco model="첫 장면">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"/>
    <body name="ball" pos="0 0 1">
      <freejoint/>
      <geom name="ball_geom" type="sphere" size="0.1" rgba="0.9 0.3 0.2 1"/>
    </body>
  </worldbody>
</mujoco>
"""
모델 = mujoco.MjModel.from_xml_string(xml)
print("시간 간격  :", 모델.opt.timestep, "초")
print("중력       :", 모델.opt.gravity)
print("몸체 수    :", 모델.nbody, "(world 포함)")
print("모양 수    :", 모델.ngeom)
print("관절 수    :", 모델.njnt)
print("qpos 길이  :", 모델.nq, "/ qvel 길이:", 모델.nv)
print("공의 질량  :", round(float(모델.body("ball").mass[0]), 3), "kg")
