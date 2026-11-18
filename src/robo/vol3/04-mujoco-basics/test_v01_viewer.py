"""v01_viewer_pendulum.py 테스트 시나리오 (pytest).

시나리오
1. 진자 장면: 처음 각도 90도, 관절 1개, 액추에이터 0개.
2. 진자 1초 뒤 각도가 교안 b01_hello.py 출력(-75.3도)과 같다 → 같은 물리.
3. 팔 장면: 액추에이터 2개(어깨, 팔꿈치), 제어 함수가 2초 뒤 어깨 목표를 100도로 만든다.
4. 팔 장면 2.5초 뒤 상자가 바닥(z ≈ 0.03m)에 떨어져 있다 → 교안 b24 결과와 같다.
"""
import mujoco
import numpy as np

from v01_viewer_pendulum import 장면_만들기, 제어


def _진행(모델, 데이터, 팔, 초):
    """초만큼 제어 + 걸음을 반복한다."""
    while 데이터.time < 초 - 1e-9:
        제어(모델, 데이터, 팔)
        mujoco.mj_step(모델, 데이터)


def test_진자_초기상태():
    모델, 데이터 = 장면_만들기(팔=False)
    assert 모델.njnt == 1 and 모델.nu == 0
    assert np.isclose(np.rad2deg(데이터.qpos[0]), 90.0)


def test_진자_1초뒤_교안과_같다():
    모델, 데이터 = 장면_만들기(팔=False)
    _진행(모델, 데이터, False, 1.0)
    assert abs(np.rad2deg(데이터.qpos[0]) - (-75.3)) < 0.1


def test_팔_제어_목표():
    모델, 데이터 = 장면_만들기(팔=True)
    assert 모델.nu == 2
    데이터.time = 3.0
    제어(모델, 데이터, True)
    assert np.isclose(np.rad2deg(데이터.ctrl[0]), 100.0) and 데이터.ctrl[1] == 0.0


def test_팔_상자_떨어뜨리기():
    모델, 데이터 = 장면_만들기(팔=True)
    _진행(모델, 데이터, True, 2.5)
    mujoco.mj_forward(모델, 데이터)
    상자 = 데이터.body("block").xpos
    assert abs(상자[2] - 0.03) < 0.005 and 상자[1] > 0.35   # 바닥 위, 탁자(y ≤ 0.35) 밖
