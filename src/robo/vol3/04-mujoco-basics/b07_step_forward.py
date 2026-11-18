import mujoco

xml = """
<mujoco>
  <worldbody>
    <body pos="0 0 1"><freejoint/><geom type="sphere" size="0.1"/></body>
  </worldbody>
</mujoco>
"""
모델 = mujoco.MjModel.from_xml_string(xml)
데이터 = mujoco.MjData(모델)

for _ in range(3):
    mujoco.mj_forward(모델, 데이터)              # 지금 상태로 계산만: 시간은 그대로
print(f"mj_forward 3번: 시간 {데이터.time}, 높이 {데이터.qpos[2]}, z 가속도 {데이터.qacc[2]:.2f}")

for _ in range(100):
    mujoco.mj_step(모델, 데이터)                 # 계산하고 한 걸음 나아간다
print(f"mj_step 100번: 시간 {데이터.time:.3f}")
print(f"  상태 qpos z  : {데이터.qpos[2]:.4f}")
print(f"  계산값 xpos z: {데이터.body(1).xpos[2]:.4f}  ← 한 걸음 전 위치")
mujoco.mj_forward(모델, 데이터)                  # 지금 상태로 다시 계산하면
print(f"  forward 뒤   : {데이터.body(1).xpos[2]:.4f}")

데이터.qpos[2] = 0.3                             # 상태를 직접 바꿔도 마찬가지
print("qpos를 바꾼 직후 xpos z:", round(float(데이터.body(1).xpos[2]), 4))
mujoco.mj_forward(모델, 데이터)
print("forward 뒤 xpos z     :", round(float(데이터.body(1).xpos[2]), 4))
