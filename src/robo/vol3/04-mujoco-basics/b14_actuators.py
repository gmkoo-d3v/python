import mujoco
import numpy as np

# 바닥에 누운 막대 세 개(z축 경첩이라 중력이 돌리지 않는다)에 액추에이터를 하나씩 단다
막대 = """
    <body name="{0}" pos="{1} 0 0.1">
      <joint name="{0}" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="0.5"/>
    </body>"""
xml = f"""
<mujoco>
  <worldbody>{막대.format("m", 0)}{막대.format("p", 1)}{막대.format("v", 2)}
  </worldbody>
  <actuator>
    <motor    name="motor"    joint="m" gear="1"/>
    <position name="position" joint="p" kp="2"/>
    <velocity name="velocity" joint="v" kv="0.5"/>
  </actuator>
</mujoco>
"""
모델 = mujoco.MjModel.from_xml_string(xml)
데이터 = mujoco.MjData(모델)
for 이름 in ["motor", "position", "velocity"]:
    a = 모델.actuator(이름)
    print(f"{이름:8s}: gain {a.gainprm[0]:4.1f}, bias {a.biasprm[:3]}")

데이터.ctrl[:] = [0.05, 1.0, 1.0]               # 토크 0.05N·m / 목표 각도 1rad / 목표 속도 1rad/s
for 목표 in [0.5, 1.0, 2.0]:
    while 데이터.time < 목표 - 1e-9:
        mujoco.mj_step(모델, 데이터)
    각도 = 데이터.qpos
    속도 = 데이터.qvel
    print(f"{목표:.1f}초 | motor 각도 {각도[0]:5.2f} 속도 {속도[0]:5.2f} | "
          f"position 각도 {각도[1]:5.2f} | velocity 속도 {속도[2]:5.2f} | 힘 {np.round(데이터.actuator_force, 3)}")
