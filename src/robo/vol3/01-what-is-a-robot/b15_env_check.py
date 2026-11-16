import platform
import sys

import matplotlib
import mujoco
import numpy as np
import scipy

print("운영 체제   :", platform.system())
print("파이썬      :", sys.version.split()[0])
for 이름, 모듈 in [("NumPy", np), ("SciPy", scipy), ("matplotlib", matplotlib), ("MuJoCo", mujoco)]:
    print(f"{이름:11s}:", 모듈.__version__)

# 화면 창 없이 그림을 그릴 수 있는지도 확인한다(빈 장면 하나를 64×64로 렌더)
모델 = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
with mujoco.Renderer(모델, 64, 64) as 렌더러:
    렌더러.update_scene(mujoco.MjData(모델))
    print("렌더 시험   :", 렌더러.render().shape)
