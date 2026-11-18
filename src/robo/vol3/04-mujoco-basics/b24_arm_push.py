import mujoco
import numpy as np

from 그림도구 import 프레임격자
from 팔장면 import XML

모델 = mujoco.MjModel.from_xml_string(XML)
데이터 = mujoco.MjData(모델)
상자 = 모델.geom("block").id
바닥_탁자 = {모델.geom("floor").id, 모델.geom("table").id}


def 상자와_닿은_것() -> set[int]:
    """지금 상자와 맞닿아 있는 geom 번호들."""
    짝들 = ({c.geom1, c.geom2} for c in 데이터.contact[:데이터.ncon])
    return {g for 짝 in 짝들 if 상자 in 짝 for g in 짝 - {상자}}


사건 = {}
찍을_시각 = [0.0, 0.4, 0.9, 1.2, 1.5, 2.5]
프레임들 = []
with mujoco.Renderer(모델, 240, 320) as 렌더러:
    while 데이터.time < 2.5 + 1e-9:
        if len(프레임들) < len(찍을_시각) and 데이터.time >= 찍을_시각[len(프레임들)] - 1e-9:
            mujoco.mj_forward(모델, 데이터)
            렌더러.update_scene(데이터, camera="overview")
            프레임들.append(렌더러.render())
        비율 = min(데이터.time / 2.0, 1.0)                    # 2초 동안 목표를 0도 → 100도로 조금씩
        데이터.ctrl[:] = [np.deg2rad(100 * 비율), 0.0]
        mujoco.mj_step(모델, 데이터)
        닿은_것 = 상자와_닿은_것()
        if 닿은_것 - 바닥_탁자:                              # 바닥·탁자가 아닌 것 = 팔
            사건.setdefault("팔", 데이터.time)
        if 모델.geom("floor").id in 닿은_것:
            사건.setdefault("바닥", 데이터.time)

print(f"팔이 상자에 처음 닿은 시각: {사건['팔']:.3f}초")
print(f"상자가 바닥에 처음 닿은 시각: {사건['바닥']:.3f}초")
print("상자 마지막 위치:", np.round(데이터.body("block").xpos, 3))
프레임격자(프레임들, [f"{t:.1f}초" for t in 찍을_시각], "arm_push_frames.png")
print("저장: arm_push_frames.png")
