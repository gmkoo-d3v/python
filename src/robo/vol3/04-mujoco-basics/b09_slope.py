import mujoco
import numpy as np

g, 기울기_도 = 9.81, 20
θ = np.deg2rad(기울기_도)
아래쪽 = np.array([np.cos(θ), 0, -np.sin(θ)])      # 비탈을 따라 내려가는 방향
법선 = np.array([np.sin(θ), 0, np.cos(θ)])         # 비탈면에 수직인 방향
print(f"tan {기울기_도}도 = {np.tan(θ):.3f} (마찰 계수가 이보다 크면 미끄러지지 않는다)")

for 마찰 in [0.1, 0.3, 0.4, 0.6, 0.2, 0.35, 0.37]:
    시작 = 0.1 * 법선                                # 상자 중심을 면에서 0.1m(반 길이) 띄운다
    xml = f"""
    <mujoco>
      <default><geom friction="{마찰} 0.005 0.0001"/></default>
      <worldbody>
        <geom type="plane" size="5 5 0.1" euler="0 {기울기_도} 0"/>
        <body pos="{시작[0]} 0 {시작[2]}" euler="0 {기울기_도} 0">
          <freejoint/><geom type="box" size="0.1 0.1 0.1"/>
        </body>
      </worldbody>
    </mujoco>
    """
    모델 = mujoco.MjModel.from_xml_string(xml)
    데이터 = mujoco.MjData(모델)
    while 데이터.time < 1.0 - 1e-9:
        mujoco.mj_step(모델, 데이터)
    간_거리 = float((데이터.qpos[:3] - 시작) @ 아래쪽)
    식_가속도 = max(g * (np.sin(θ) - 마찰 * np.cos(θ)), 0.0)   # 미끄러질 때 a = g(sinθ - μcosθ)
    print(f"마찰 {마찰}: 1초 동안 {간_거리:.3f}m 내려감 (식 ½at² = {0.5 * 식_가속도:.3f}m)")
