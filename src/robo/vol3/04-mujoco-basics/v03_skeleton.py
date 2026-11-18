"""2일차 보충 예제: 모든 MuJoCo 프로그램의 기본 뼈대 (교안 21번 슬라이드 "기본 뼈대 코드").

팔장면.py의 2관절 팔에 목표 각도(어깨 30도, 팔꿈치 60도)를 넣고 2초 진행한 뒤
관절 각도와 손끝 위치를 읽는다.

실행 방법 (04-mujoco-basics 폴더 안에서):
    python v03_skeleton.py
출력:
    관절 [30. 60.] 도 | 손끝 [0.26  0.4   0.45]
"""
import mujoco
import numpy as np

from 팔장면 import XML                              # 팔장면.py 안의 XML 글자


def 뼈대_실행(목표_도=(30.0, 60.0), 초: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """설계도 → 상태 → 반복(명령·진행) → 읽기 순서로 팔을 움직인다.

    Args:
        목표_도: (어깨, 팔꿈치) 목표 각도(도).
        초: 시뮬레이션 시간(초).

    Returns:
        (관절 각도(도) 2개, 손끝 위치 x·y·z) 튜플.
    """
    모델 = mujoco.MjModel.from_xml_string(XML)      # ② 설계도 (한 번)
    데이터 = mujoco.MjData(모델)                     # ③ 상태 (처음엔 관절 모두 0도)
    명령 = np.deg2rad(목표_도)                        # ④ 목표 각도를 라디안으로
    while 데이터.time < 초 - 1e-9:                    # ⑤ 2초 = 1000걸음
        데이터.ctrl[:] = 명령                         # ⑥ 모터 2개에 목표 넣기
        mujoco.mj_step(모델, 데이터)                  # ⑦ 0.002초 진행
    mujoco.mj_forward(모델, 데이터)                   # 손끝 위치(xpos)를 지금 각도로 다시 계산
    return np.rad2deg(데이터.qpos[:2]), 데이터.site("tip").xpos.copy()   # ⑧ 읽기


if __name__ == "__main__":
    관절, 손끝 = 뼈대_실행()
    print("관절", np.round(관절, 1), "도 | 손끝", np.round(손끝, 3))
