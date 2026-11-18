import mujoco
import numpy as np

from 자이로데이터 import 증분_쿼터니언       # 2장 실습의 함수를 그대로 쓴다
from 쿼터니언도구 import 곱, 사이각

# 무중력에서 비스듬히 던진 길쭉한 상자: 각속도가 계속 바뀌며 뒤뚱거린다
xml = """
<mujoco>
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="box">
      <freejoint/>
      <geom type="box" size="0.2 0.1 0.05" mass="1"/>
      <site name="imu"/>
    </body>
  </worldbody>
  <sensor><gyro name="gyro" site="imu"/></sensor>
</mujoco>
"""
모델 = mujoco.MjModel.from_xml_string(xml)
간격 = 모델.opt.timestep
for 언제 in ["걸음 전", "걸음 뒤"]:
    데이터 = mujoco.MjData(모델)
    데이터.qvel[3:6] = [0.3, 4.0, 0.5]           # 처음 각속도(rad/s)
    mujoco.mj_forward(모델, 데이터)
    q = 데이터.body("box").xquat.copy()          # 처음 자세에서 출발
    for _ in range(5000):                        # 0.002초 × 5000 = 10초
        전 = 데이터.sensor("gyro").data.copy()
        mujoco.mj_step(모델, 데이터)
        if 언제 == "걸음 전":
            각속도 = 전
        else:
            mujoco.mj_forward(모델, 데이터)       # 걸음을 마친 상태로 센서를 다시 계산
            각속도 = 데이터.sensor("gyro").data
        q = 곱(q, 증분_쿼터니언(각속도, 간격))      # 2장 (다) 방법으로 쌓기
    print(f"{언제} 자이로로 쌓은 자세 vs MuJoCo xquat: {사이각(q, 데이터.body('box').xquat):.5f}도")
print("10초 뒤 자이로 값:", np.round(데이터.sensor("gyro").data, 3))
