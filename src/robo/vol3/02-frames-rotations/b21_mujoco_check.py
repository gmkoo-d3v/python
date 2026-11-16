import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from rotation_tools import Ry, Rz, 깔끔

R_몸_카메라 = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
q_카메라 = Rotation.from_matrix(Ry(30) @ R_몸_카메라).as_quat(scalar_first=True)   # MuJoCo는 w x y z

# 2.8절의 장면을 MuJoCo 모델로: 몸체(body) 안에 몸체를 넣으면 부모 기준으로 놓인다
xml = f"""
<mujoco>
  <worldbody>
    <body name="robot" pos="1.5 0 0" euler="0 0 90">
      <body name="camera" pos="0.1 0 0.5" quat="{' '.join(map(str, q_카메라))}">
        <site name="cup" pos="0.5 -0.017 1.029"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
모델 = mujoco.MjModel.from_xml_string(xml)
데이터 = mujoco.MjData(모델)
mujoco.mj_forward(모델, 데이터)                 # 시간은 흐르지 않고, 위치만 계산한다

print("MuJoCo가 계산한 컵 위치:", 깔끔(데이터.site("cup").xpos))
print("로봇 몸체 자세 xquat   :", 깔끔(데이터.body("robot").xquat, 4))
print("같은 자세, scipy 순서  :", 깔끔(Rotation.from_matrix(Rz(90)).as_quat(), 4))

# MuJoCo의 euler="..." 기본 순서는 움직이는 축 기준(scipy의 대문자)이다
xml2 = '<mujoco><worldbody><body name="b" euler="10 20 30"/></worldbody></mujoco>'
모델2 = mujoco.MjModel.from_xml_string(xml2)
데이터2 = mujoco.MjData(모델2)
mujoco.mj_forward(모델2, 데이터2)
행렬 = 데이터2.body("b").xmat.reshape(3, 3)
for 순서 in ["xyz", "XYZ"]:
    같음 = np.allclose(행렬, Rotation.from_euler(순서, [10, 20, 30], degrees=True).as_matrix())
    print(f"MuJoCo euler='10 20 30' == scipy '{순서}' ? {같음}")
