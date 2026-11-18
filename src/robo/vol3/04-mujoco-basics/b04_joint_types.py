import mujoco

xml = """
<mujoco>
  <worldbody>
    <body name="문" pos="0 0 1">
      <joint name="경첩" type="hinge" axis="0 0 1"/><geom type="box" size="0.3 0.02 0.5"/></body>
    <body name="서랍" pos="1 0 1">
      <joint name="레일" type="slide" axis="1 0 0"/><geom type="box" size="0.2 0.2 0.1"/></body>
    <body name="눈알" pos="2 0 1">
      <joint name="구관절" type="ball"/><geom type="sphere" size="0.1"/></body>
    <body name="주사위" pos="3 0 1">
      <freejoint name="자유"/><geom type="box" size="0.05 0.05 0.05"/></body>
    <body name="선반" pos="4 0 1"><geom type="box" size="0.3 0.1 0.02"/></body>
  </worldbody>
</mujoco>
"""
모델 = mujoco.MjModel.from_xml_string(xml)
for j in range(모델.njnt):
    관절 = 모델.joint(j)
    종류 = mujoco.mjtJoint(int(관절.type[0])).name          # 예: 'mjJNT_HINGE'
    시작 = int(관절.qposadr[0])
    끝 = int(모델.jnt_qposadr[j + 1]) if j + 1 < 모델.njnt else 모델.nq
    print(f"{종류:12s} qpos[{시작}:{끝}] ({끝 - 시작}칸), qvel {int(관절.dofadr[0])}번부터 ← {관절.name}")
print("전체 nq =", 모델.nq, "/ nv =", 모델.nv)
print("선반의 관절 수:", int(모델.body("선반").jntnum[0]))
