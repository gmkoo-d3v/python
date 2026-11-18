import mujoco

xml = """
<mujoco>
  <worldbody>
    <body name="상자"><geom type="box" size="0.1 0.1 0.1"/></body>
    <body name="공"><geom type="sphere" size="0.1"/></body>
    <body name="막대"><geom type="capsule" fromto="0 0 0 0 0 0.5" size="0.02"/></body>
    <body name="스티로폼"><geom type="box" size="0.1 0.1 0.1" density="20"/></body>
    <body name="쇠공"><geom type="sphere" size="0.1" mass="30"/></body>
  </worldbody>
</mujoco>
"""
모델 = mujoco.MjModel.from_xml_string(xml)
for 이름 in ["상자", "공", "막대", "스티로폼", "쇠공"]:
    print(f"질량 {모델.body(이름).mass[0]:7.3f} kg ← {이름}")
print("상자 geom의 size:", 모델.geom(0).size, "→ 한 변", 2 * 모델.geom(0).size[0], "m")
