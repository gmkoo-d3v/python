import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from 회전도구 import Rz, 깔끔

xml = """
<mujoco>
  <worldbody>
    <body name="door" pos="1 0 0">
      <joint name="hinge" type="hinge" axis="0 0 1"/>
      <geom type="box" pos="0.4 0 1" size="0.4 0.02 1"/>
      <site name="knob" pos="0.7 -0.05 1"/>
    </body>
  </worldbody>
</mujoco>
"""
모델 = mujoco.MjModel.from_xml_string(xml)
데이터 = mujoco.MjData(모델)

# 이름으로 번호 찾기: 두 가지 방법
번호 = mujoco.mj_name2id(모델, mujoco.mjtObj.mjOBJ_BODY, "door")
print("mj_name2id:", 번호, "/ model.body('door').id:", 모델.body("door").id)
print("없는 이름이면:", mujoco.mj_name2id(모델, mujoco.mjtObj.mjOBJ_BODY, "window"))

데이터.joint("hinge").qpos[0] = np.deg2rad(30)    # 문을 30도 연다
mujoco.mj_forward(모델, 데이터)

문 = 데이터.body("door")
print("문 xpos :", 깔끔(문.xpos))
print("문 xquat:", 깔끔(문.xquat, 4), "(w x y z)")
print("xmat == Rz(30)?", np.allclose(문.xmat.reshape(3, 3), Rz(30)))
print("xquat == scipy?", np.allclose(문.xquat, Rotation.from_matrix(Rz(30)).as_quat(scalar_first=True)))

손잡이_방 = 데이터.site("knob").xpos
손잡이_문 = 모델.site("knob").pos                  # 문 기준으로 적어 둔 위치(설계도)
직접 = 문.xpos + 문.xmat.reshape(3, 3) @ 손잡이_문     # 2장의 T_방_문 @ 손잡이_문
print("손잡이 site xpos:", 깔끔(손잡이_방), "/ 직접 계산:", 깔끔(직접))
