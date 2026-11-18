import mujoco
from PIL import Image

from 그림도구 import 프레임격자

xml = """
<mujoco>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.95 0.95 1" rgb2="0.6 0.7 0.8" width="256" height="256"/>
    <texture name="격자" type="2d" builtin="checker" rgb1="0.25 0.3 0.35" rgb2="0.75 0.75 0.75"
             width="256" height="256"/>
    <material name="바닥재" texture="격자" texrepeat="6 6"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="1.5 1.5 0.1" material="바닥재"/>
    <camera name="front" pos="0 -2.2 0.6" xyaxes="1 0 0 0 0.15 1"/>
    <body pos="-0.6 0 1">
      <freejoint/>
      <geom type="sphere" size="0.1" rgba="0.9 0.3 0.2 1" solref="-10000 -10"/>
    </body>
  </worldbody>
</mujoco>
"""
모델 = mujoco.MjModel.from_xml_string(xml)
데이터 = mujoco.MjData(모델)
데이터.qvel[0] = 0.8                               # 오른쪽으로 0.8m/s 굴려 보낸다

초당_프레임 = 30
프레임들, 시각들 = [], []
with mujoco.Renderer(모델, 240, 320) as 렌더러:
    while 데이터.time < 1.5:
        mujoco.mj_step(모델, 데이터)
        if len(프레임들) < 데이터.time * 초당_프레임:    # 1/30초마다 한 장씩 찍는다
            mujoco.mj_forward(모델, 데이터)              # 지금 상태의 위치로 맞춘 뒤 찍는다
            렌더러.update_scene(데이터, camera="front")
            프레임들.append(렌더러.render())
            시각들.append(데이터.time)
print("찍은 프레임 수:", len(프레임들))

고른_번호 = [0, 9, 13, 18, 25, 44]
프레임격자([프레임들[i] for i in 고른_번호], [f"{시각들[i]:.2f}초" for i in 고른_번호], "bounce_frames.png")
print("저장: bounce_frames.png")

그림들 = [Image.fromarray(f) for f in 프레임들]
그림들[0].save("bounce.gif", save_all=True, append_images=그림들[1:], duration=1000 // 초당_프레임, loop=0)
print("저장: bounce.gif")
