import mujoco

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
    <camera name="front" pos="0 -2.2 0.9" xyaxes="1 0 0 0 0.35 1"/>
    <body pos="-0.3 0 0.1"><geom type="box" size="0.1 0.1 0.1" rgba="0.2 0.5 0.9 1"/></body>
    <body pos="0.3 0 0.15"><geom type="sphere" size="0.15" rgba="0.9 0.3 0.2 1"/></body>
  </worldbody>
</mujoco>
"""
모델 = mujoco.MjModel.from_xml_string(xml)
데이터 = mujoco.MjData(모델)
mujoco.mj_forward(모델, 데이터)                    # 렌더 전에 위치를 계산해 둔다

with mujoco.Renderer(모델, 240, 320) as 렌더러:    # 높이 240, 너비 320 픽셀
    렌더러.update_scene(데이터, camera="front")    # 장면에 넣어 둔 카메라로
    앞 = 렌더러.render()
    print("이미지:", 앞.shape, 앞.dtype)

    자유 = mujoco.MjvCamera()                      # 장면 밖에서 자유롭게 움직이는 카메라
    자유.lookat[:] = [0, 0, 0.1]                   # 바라볼 점
    자유.distance = 2.0                            # 그 점에서 떨어진 거리(m)
    자유.azimuth = 45                              # 수평으로 돈 각도(도)
    자유.elevation = -35                           # 내려다보는 각도(도)
    렌더러.update_scene(데이터, camera=자유)
    비스듬히 = 렌더러.render()

제목들 = ["장면 속 카메라 front", "자유 카메라 (방위 45도, 고도 -35도)"]
프레임격자([앞, 비스듬히], 제목들, "render_cameras.png", 열=2)
print("저장: render_cameras.png")
