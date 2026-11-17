import mujoco
import numpy as np
from PIL import Image

from 팔모델 import MJCF

모델 = mujoco.MjModel.from_xml_string(MJCF())
데이터 = mujoco.MjData(모델)
카메라 = mujoco.MjvCamera()
카메라.lookat[:] = [0.15, 0.1, 0.35]        # 카메라가 바라볼 점
카메라.distance, 카메라.azimuth, 카메라.elevation = 1.6, -120, -15

사진들 = []
with mujoco.Renderer(모델, 400, 400) as 그리개:
    for 관절각_도 in [[0, 0, 0, 0, 0, 0], [30, 45, 60, 0, 45, 0]]:
        데이터.qpos[:] = np.deg2rad(관절각_도)
        mujoco.mj_forward(모델, 데이터)
        그리개.update_scene(데이터, camera=카메라)
        사진들.append(그리개.render())
        print(f"{관절각_도} → 손끝(site) 위치 {np.round(데이터.site('tip').xpos, 3) + 0.0}")
Image.fromarray(np.hstack(사진들)).save("arm6_mujoco.png")
print("저장: arm6_mujoco.png")
