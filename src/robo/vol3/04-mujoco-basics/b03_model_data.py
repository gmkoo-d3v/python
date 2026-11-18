import mujoco

xml = """
<mujoco>
  <worldbody>
    <body name="ball" pos="0 0 1">
      <freejoint/>
      <geom type="sphere" size="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""
모델 = mujoco.MjModel.from_xml_string(xml)       # 설계도: 모양, 질량, 규칙
가 = mujoco.MjData(모델)                          # 판 하나
나 = mujoco.MjData(모델)                          # 같은 설계도로 만든 다른 판

for _ in range(250):                              # 판 '가'만 0.5초(250걸음) 진행
    mujoco.mj_step(모델, 가)
print(f"가: {가.time:.1f}초, 높이 {가.qpos[2]:.3f}m")
print(f"나: {나.time:.1f}초, 높이 {나.qpos[2]:.3f}m")

mujoco.mj_resetData(모델, 가)                     # 판을 처음 상태로
print(f"되돌린 가: {가.time:.1f}초, 높이 {가.qpos[2]:.3f}m")

모델.opt.gravity[2] = -1.62                       # 규칙을 바꾸면(달의 중력) 모든 판에 적용된다
for _ in range(250):
    mujoco.mj_step(모델, 가)
print(f"달에서 0.5초 뒤 가: 높이 {가.qpos[2]:.3f}m")
