import time

import mujoco


def 실시간_배수(xml: str, 걸음_수: int = 5000) -> float:
    """시뮬레이션 시간 ÷ 계산에 걸린 실제 시간."""
    모델 = mujoco.MjModel.from_xml_string(xml)
    데이터 = mujoco.MjData(모델)
    시작 = time.perf_counter()
    for _ in range(걸음_수):
        mujoco.mj_step(모델, 데이터)
    return 데이터.time / (time.perf_counter() - 시작)


진자 = """<mujoco><worldbody><body pos="0 0 1"><joint type="hinge" axis="0 1 0"/>
<geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.02"/></body></worldbody></mujoco>"""

# 상자 20개를 쏟아붓는 장면: 접촉이 많아 계산이 무거운 편이다
상자들 = "".join(
    f'<body pos="{(i % 5) * 0.25 - 0.5} {(i // 5) * 0.25 - 0.4} {0.3 + 0.1 * i}"><freejoint/>'
    f'<geom type="box" size="0.08 0.08 0.08" mass="0.5"/></body>'
    for i in range(20)
)
상자_장면 = f'<mujoco><worldbody><geom type="plane" size="3 3 0.1"/>{상자들}</worldbody></mujoco>'

print(f"진자 하나      : 실제 시간의 약 {실시간_배수(진자):,.0f}배")
print(f"상자 20개 쏟기 : 실제 시간의 약 {실시간_배수(상자_장면):,.0f}배")
