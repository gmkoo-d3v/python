import matplotlib.pyplot as plt
import mujoco
import numpy as np

from box_scene import 상자_xml

모델 = mujoco.MjModel.from_xml_string(상자_xml)
데이터 = mujoco.MjData(모델)
사진들 = []
with mujoco.Renderer(모델, 240, 320) as 렌더러:     # 높이 240, 너비 320 픽셀. 화면 창 없이 그린다
    for 목표_시각 in [0.0, 0.3, 0.6]:
        while 데이터.time < 목표_시각 - 1e-9:
            mujoco.mj_step(모델, 데이터)
        mujoco.mj_forward(모델, 데이터)             # 그리기 전에 지금 상태의 위치를 다시 계산
        렌더러.update_scene(데이터, camera="side")
        사진들.append(렌더러.render())              # (높이, 너비, 3) 모양의 RGB 배열

print("사진 한 장의 모양:", 사진들[0].shape, 사진들[0].dtype)
plt.imsave("falling_box.png", np.hstack(사진들))   # 세 장을 옆으로 이어 붙여 저장
print("저장: falling_box.png")
