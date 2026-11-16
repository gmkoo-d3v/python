"""1장 MuJoCo 맛보기: 바닥 위 1m 높이에서 떨어지는 상자 하나."""
상자_xml = """
<mujoco>
  <option timestep="0.002"/>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.7 0.8 0.95" rgb2="1 1 1" width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.9 0.9 0.9" rgb2="0.75 0.75 0.75"
             width="256" height="256"/>
    <material name="grid" texture="grid" texrepeat="20 20"/>
  </asset>
  <worldbody>
    <light pos="0 -1 3" dir="0 0.3 -1"/>
    <geom name="floor" type="plane" size="5 5 0.1" material="grid"/>
    <body name="box" pos="0 0 1">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1" rgba="0.9 0.5 0.1 1"/>
    </body>
    <camera name="side" pos="0 -2 0.55" xyaxes="1 0 0 0 0.05 1"/>
  </worldbody>
</mujoco>
"""
