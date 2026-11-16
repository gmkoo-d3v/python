import mujoco

from box_scene import 상자_xml

모델 = mujoco.MjModel.from_xml_string(상자_xml)
데이터 = mujoco.MjData(모델)
중력 = 9.81

print("시뮬레이션 한 걸음:", 모델.opt.timestep, "초")
for 목표_시각 in [0.0, 0.2, 0.4, 0.6, 1.0]:
    while 데이터.time < 목표_시각 - 1e-9:
        mujoco.mj_step(모델, 데이터)              # 물리를 한 걸음(0.002초) 진행
    높이 = 데이터.qpos[2]                          # 상자 중심의 높이(z)
    공식 = max(1 - 0.5 * 중력 * 목표_시각 ** 2, 0.1)   # 자유 낙하 공식(바닥에 닿으면 0.1에서 멈춘다)
    print(f"{목표_시각:.1f}초: 상자 중심 높이 {높이:.3f}m (공식 {공식:.3f}m)")
