import mujoco
import numpy as np

xml = """
<mujoco>
  <worldbody>
    <geom type="plane" size="2 2 0.1"/>
    <body name="box" pos="0 0 0.4">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.05" mass="1"/>
      <site name="imu" pos="0 0 0"/>
      <site name="bottom" pos="0 0 -0.05" type="box" size="0.1 0.1 0.01"/>
    </body>
    <body name="arm" pos="1 0 1">
      <joint name="shoulder" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0.4 0 0" size="0.02" mass="0.5"/>
    </body>
  </worldbody>
  <sensor>
    <accelerometer name="가속도" site="imu"/>
    <gyro          name="자이로" site="imu"/>
    <touch         name="접촉"   site="bottom"/>
    <framepos      name="상자위치" objtype="body" objname="box"/>
    <jointpos      name="어깨각도" joint="shoulder"/>
    <jointvel      name="어깨속도" joint="shoulder"/>
  </sensor>
</mujoco>
"""
모델 = mujoco.MjModel.from_xml_string(xml)
데이터 = mujoco.MjData(모델)
print("센서 수:", 모델.nsensor, "/ sensordata 길이:", 모델.nsensordata)
print("센서별 칸 수:", 모델.sensor_dim, "/ 시작 위치:", 모델.sensor_adr)


def 읽기(이름: str) -> np.ndarray:
    return np.round(데이터.sensor(이름).data, 2) + 0.0


for 목표 in [0.1, 1.0]:
    while 데이터.time < 목표 - 1e-9:
        mujoco.mj_step(모델, 데이터)
    print(f"--- {목표}초 ---")
    for 이름 in ["가속도", "자이로", "접촉", "상자위치", "어깨각도", "어깨속도"]:
        print(f"{이름}: {읽기(이름)}")
