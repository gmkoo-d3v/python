"""2일차 보충 예제: MuJoCo 대화형 뷰어로 진자와 2관절 팔을 직접 눈으로 보기.

교안 b01_hello.py(진자)와 팔장면.py(탁자 위 2관절 팔)를 창에 띄워
학생들이 마우스로 돌려 보며 관찰하게 한다.

실행 방법:
    python v01_viewer_pendulum.py            # 진자 (창이 열린다)
    python v01_viewer_pendulum.py --arm      # 탁자 위 2관절 팔이 상자를 민다
    python v01_viewer_pendulum.py --headless # 창 없이 5초 계산만 (테스트용)

macOS에서는 python 대신 mjpython 으로 실행해야 창이 열린다.
"""
import argparse
import time

import mujoco
import numpy as np

# 진자 장면: b01_hello.py와 같은 구조에 바닥·조명·줄(capsule)만 더했다
진자_XML = """
<mujoco model="진자">
  <option timestep="0.002"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="1 1 0.1" rgba="0.85 0.85 0.85 1"/>
    <body pos="0 0 1">
      <joint name="pivot" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.008" mass="0" rgba="0.4 0.4 0.4 1"/>  <!-- 줄은 보이기만, 질량 0 -->
      <geom type="sphere" pos="0 0 -0.5" size="0.05" rgba="0.9 0.3 0.2 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def 장면_만들기(팔: bool) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """선택한 장면의 모델과 데이터를 만든다.

    Args:
        팔: True면 팔장면.py의 탁자 위 2관절 팔, False면 진자.

    Returns:
        (MjModel, MjData) 튜플. 진자는 처음 각도를 90도로 들어 둔다.
    """
    if 팔:
        from 팔장면 import XML       # 같은 폴더의 팔장면.py (교안 공용 모듈)
        모델 = mujoco.MjModel.from_xml_string(XML)
        데이터 = mujoco.MjData(모델)
    else:
        모델 = mujoco.MjModel.from_xml_string(진자_XML)
        데이터 = mujoco.MjData(모델)
        데이터.qpos[0] = np.deg2rad(90)      # 옆으로 90도 들었다 놓기
    mujoco.mj_forward(모델, 데이터)           # 첫 화면을 그리기 전에 위치 계산
    return 모델, 데이터


def 제어(모델: mujoco.MjModel, 데이터: mujoco.MjData, 팔: bool) -> None:
    """매 걸음 직전에 부르는 제어 함수. 팔 장면이면 b24처럼 어깨를 천천히 돌린다.

    Args:
        모델: MuJoCo 모델 (액추에이터 수 확인용).
        데이터: 현재 상태. ctrl을 여기서 채운다.
        팔: 팔 장면 여부. 진자는 액추에이터가 없으니 아무것도 하지 않는다.
    """
    if 팔 and 모델.nu == 2:
        비율 = min(데이터.time / 2.0, 1.0)                    # 2초 동안 0 → 1
        데이터.ctrl[:] = [np.deg2rad(100 * 비율), 0.0]       # 어깨 목표 0 → 100도


def 창_없이_실행(모델: mujoco.MjModel, 데이터: mujoco.MjData, 팔: bool, 초: float = 5.0) -> None:
    """뷰어 없이 계산만 한다. 화면이 없는 환경에서 동작을 확인할 때 쓴다.

    Args:
        모델: MuJoCo 모델.
        데이터: MuJoCo 데이터.
        팔: 팔 장면 여부.
        초: 시뮬레이션할 시간(초).
    """
    while 데이터.time < 초 - 1e-9:
        제어(모델, 데이터, 팔)
        mujoco.mj_step(모델, 데이터)
    mujoco.mj_forward(모델, 데이터)                          # 출력 전에 xpos를 지금 상태로
    if 팔:
        print(f"{데이터.time:.1f}초 뒤 상자 위치:", np.round(데이터.body("block").xpos, 3))
    else:
        print(f"{데이터.time:.1f}초 뒤 진자 각도: {np.rad2deg(데이터.qpos[0]):.1f}도")


def 창으로_실행(모델: mujoco.MjModel, 데이터: mujoco.MjData, 팔: bool) -> None:
    """수동(passive) 뷰어를 띄우고, 실제 시간에 맞춰 시뮬레이션을 진행한다.

    창 안에서: 왼쪽 드래그=회전, 오른쪽 드래그=이동, 휠=확대,
    Space=일시정지, Backspace=처음으로. 창을 닫으면 끝난다.

    Args:
        모델: MuJoCo 모델.
        데이터: MuJoCo 데이터.
        팔: 팔 장면 여부.
    """
    import mujoco.viewer                                      # 화면이 있는 컴퓨터에서만 불러온다
    with mujoco.viewer.launch_passive(모델, 데이터) as 뷰어:
        while 뷰어.is_running():
            시작 = time.time()
            제어(모델, 데이터, 팔)
            mujoco.mj_step(모델, 데이터)                       # 한 걸음(0.002초) 진행
            뷰어.sync()                                        # 바뀐 상태를 창에 반영
            남은 = 모델.opt.timestep - (time.time() - 시작)
            if 남은 > 0:
                time.sleep(남은)                               # 너무 빠르지 않게 실제 시간에 맞춤


if __name__ == "__main__":
    파서 = argparse.ArgumentParser(description="MuJoCo 뷰어로 진자 / 2관절 팔 보기")
    파서.add_argument("--arm", action="store_true", help="탁자 위 2관절 팔 장면")
    파서.add_argument("--headless", action="store_true", help="창 없이 5초 계산만")
    인자 = 파서.parse_args()

    모델, 데이터 = 장면_만들기(인자.arm)
    if 인자.headless:
        창_없이_실행(모델, 데이터, 인자.arm)
    else:
        창으로_실행(모델, 데이터, 인자.arm)
