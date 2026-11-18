"""v03~v05 보충 예제 테스트 시나리오 (pytest). 슬라이드에 적힌 숫자와 같아야 한다.

1. v03 뼈대: 2초 뒤 관절 [30, 60]도, 손끝 (0.26, 0.40, 0.45).          (21번)
2. v04 관절만: 1초 뒤 어깨 -28.7도, 팔꿈치 -35.4도로 처진다.            (35번)
3. v04 액추에이터: 1초 뒤 어깨 -2.9도, 팔꿈치 -1.6도 (조금만 처짐).     (35번)
4. v05 진자: 0도 공 (0,0,0.5), 90도 공 (-0.5,0,1), 0.2초 뒤 67.5도.     (8번)
"""
import numpy as np

from v03_skeleton import 뼈대_실행
from v04_joint_vs_actuator import 처짐_재기
from v05_pendulum_numbers import 공_위치, 놓고_진행


def test_v03_뼈대_출력():
    관절, 손끝 = 뼈대_실행()
    assert np.allclose(관절, [30, 60], atol=0.1)
    assert np.allclose(손끝, [0.26, 0.40, 0.45], atol=0.005)


def test_v04_관절만_처진다():
    각도 = 처짐_재기(모터=False)
    assert np.allclose(각도, [-28.7, -35.4], atol=0.1)


def test_v04_액추에이터는_버틴다():
    각도 = 처짐_재기(모터=True)
    assert np.allclose(각도, [-2.9, -1.6], atol=0.1)


def test_v05_진자_숫자():
    assert np.allclose(공_위치(0), [0, 0, 0.5], atol=1e-6)
    assert np.allclose(공_위치(90), [-0.5, 0, 1], atol=1e-6)
    각도, _ = 놓고_진행()
    assert abs(각도 - 67.5) < 0.1
