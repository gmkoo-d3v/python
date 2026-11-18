import mujoco

xml = """
<mujoco>
  <default>
    <geom friction="0.8 0.005 0.0001" rgba="0.6 0.6 0.6 1"/>
    <default class="얼음">
      <geom friction="0.05 0.005 0.0001" rgba="0.7 0.9 1 1"/>
    </default>
  </default>
  <asset>
    <texture name="격자" type="2d" builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.8 0.8 0.8" width="256" height="256"/>
    <material name="바닥재" texture="격자" texrepeat="4 4"/>
  </asset>
  <worldbody>
    <geom name="바닥" type="plane" size="2 2 0.1" material="바닥재"/>
    <body name="나무상자" pos="0 0 0.1"><geom name="나무" type="box" size="0.1 0.1 0.1"/></body>
    <body name="얼음상자" pos="0.5 0 0.1"><geom name="얼음" class="얼음" type="box" size="0.1 0.1 0.1"/></body>
  </worldbody>
</mujoco>
"""
모델 = mujoco.MjModel.from_xml_string(xml)
for 이름 in ["바닥", "나무", "얼음"]:
    g = 모델.geom(이름)
    print(f"{이름}: 마찰 {g.friction[0]:.2f}, 색 {g.rgba.round(2)}, 재질 번호 {int(g.matid[0])}")
