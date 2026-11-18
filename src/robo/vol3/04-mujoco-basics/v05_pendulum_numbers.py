"""2일차 보충 예제: 진자 코드의 숫자 확인 (교안 8번 슬라이드 "진자 코드").

b01_hello.py와 같은 진자에서 슬라이드 그림의 숫자를 직접 출력한다.
  - 0도일 때 공 위치 (0, 0, 0.5) = 회전 중심 (0, 0, 1)에서 0.5m 아래
  - 90도로 들면 공 위치 (-0.5, 0, 1)
  - 놓고 100걸음(0.2초) 뒤 각도 67.5도

실행 방법 (04-mujoco-basics 폴더 안에서):
    python v05_pendulum_numbers.py
"""
import mujoco
import numpy as np

xml = """<mujoco><worldbody>
  <body pos="0 0 1">
    <joint type="hinge" axis="0 1 0"/>
    <geom type="sphere" pos="0 0 -0.5" size="0.05"/>
  </body></worldbody></mujoco>"""


def 공_위치(각도_도: float) -> np.ndarray:
    """진자를 주어진 각도로 놓았을 때 공(geom)의 세계 위치를 돌려준다.

    Args:
        각도_도: 관절 각도(도).

    Returns:
        공 중심의 (x, y, z).
    """
    모델 = mujoco.MjModel.from_xml_string(xml)
    데이터 = mujoco.MjData(모델)
    데이터.qpos[0] = np.deg2rad(각도_도)
    mujoco.mj_forward(모델, 데이터)                   # 위치만 계산 (시간은 그대로)
    return 데이터.geom_xpos[0].copy()


def 놓고_진행(처음_도: float = 90.0, 걸음: int = 100) -> tuple[float, np.ndarray]:
    """처음 각도로 들었다 놓고 걸음 수만큼 진행한다.

    Args:
        처음_도: 처음 각도(도).
        걸음: mj_step 횟수 (1걸음 = 0.002초).

    Returns:
        (각도(도), 공 위치) 튜플.
    """
    모델 = mujoco.MjModel.from_xml_string(xml)
    데이터 = mujoco.MjData(모델)
    데이터.qpos[0] = np.deg2rad(처음_도)
    for _ in range(걸음):
        mujoco.mj_step(모델, 데이터)
    mujoco.mj_forward(모델, 데이터)
    return float(np.rad2deg(데이터.qpos[0])), 데이터.geom_xpos[0].copy()


if __name__ == "__main__":
    print("0도  공 위치:", np.round(공_위치(0), 3))
    print("90도 공 위치:", np.round(공_위치(90), 3))
    각도, 위치 = 놓고_진행()
    print(f"0.2초 뒤 각도 {각도:.1f}도, 공 위치 {np.round(위치, 3)}")
