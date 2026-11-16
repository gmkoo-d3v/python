import mujoco

중력 = 9.81
for 마찰 in [1.0, 0.5, 0.2]:
    # 바닥에 놓인 상자를 옆으로 1m/s로 밀어 보낸다. 미끄럼 마찰 계수만 바꾼다
    # (MuJoCo는 닿은 두 물체 중 큰 값을 쓰므로 바닥과 상자에 같은 값을 준다)
    xml = f"""
    <mujoco>
      <worldbody>
        <geom type="plane" size="5 5 0.1" friction="{마찰}"/>
        <body pos="0 0 0.1"><freejoint/>
          <geom type="box" size="0.1 0.1 0.1" friction="{마찰}"/></body>
      </worldbody>
    </mujoco>"""
    모델 = mujoco.MjModel.from_xml_string(xml)
    데이터 = mujoco.MjData(모델)
    데이터.qvel[0] = 1.0
    while 데이터.time < 2.0:
        mujoco.mj_step(모델, 데이터)
    공식 = 1.0 ** 2 / (2 * 마찰 * 중력)             # 미끄러져 멈추는 거리 = 속도² / (2 × 마찰 × 중력)
    print(f"마찰 계수 {마찰}: 미끄러진 거리 {데이터.qpos[0]:.3f}m (공식 {공식:.3f}m)")
