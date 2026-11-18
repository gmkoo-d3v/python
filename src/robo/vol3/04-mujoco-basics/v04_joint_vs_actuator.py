"""2일차 보충 예제: 관절과 액추에이터는 다르다 (교안 35번 슬라이드).

같은 2관절 팔(세로 평면, 중력 있음)을 수평으로 들고 놓는다.
  - 관절만: joint는 "돌 수 있게 허락"만 한다. 버티지 못하고 처진다.
  - 관절 + position 액추에이터: 0도를 지키려 힘을 내서 조금만 처진다.

실행 방법 (04-mujoco-basics 폴더 안에서):
    python v04_joint_vs_actuator.py           # 1초 뒤 각도 출력
    python v04_joint_vs_actuator.py --그림    # 사진도 저장 (v04_joint_vs_actuator.png)
출력:
    관절만      : 1초 뒤 어깨 -28.7도, 팔꿈치 -35.4도
    액추에이터  : 1초 뒤 어깨 -2.9도, 팔꿈치 -1.6도
"""
import argparse

import mujoco
import numpy as np

틀 = """
<mujoco>
  <visual><global offwidth="800" offheight="600"/></visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.97 0.98 0.98" rgb2="0.85 0.9 0.9" width="256" height="256"/>
    <texture name="격자" type="2d" builtin="checker" rgb1="0.8 0.85 0.85" rgb2="0.92 0.94 0.94" width="256" height="256"/>
    <material name="바닥재" texture="격자" texrepeat="6 6"/>
  </asset>
  <option timestep="0.002" integrator="implicitfast"/>
  <worldbody>
    <light pos="0 -1.5 2.5" dir="0 0.6 -1"/>
    <geom type="plane" size="2 2 0.1" material="바닥재"/>
    <geom type="box" pos="0 0 0.4" size="0.04 0.04 0.4" rgba="0.35 0.38 0.4 1"/>   <!-- 기둥 -->
    <body pos="0 0 0.85">
      <joint name="shoulder" type="hinge" axis="0 -1 0" damping="0.05"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.025" mass="0.5" rgba="0.05 0.42 0.44 1"/>
      <body pos="0.3 0 0">
        <joint name="elbow" type="hinge" axis="0 -1 0" damping="0.05"/>
        <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.022" mass="0.3" rgba="0.3 0.65 0.62 1"/>
        <geom type="sphere" pos="0.25 0 0" size="0.035" mass="0.2" rgba="0.95 0.78 0.2 1"/>
      </body>
    </body>
  </worldbody>
  {액추}
</mujoco>"""

# 액추에이터 있는 경우: 두 관절에 position 모터 (목표 0도를 지키려 힘을 낸다)
액추 = ('<actuator><position joint="shoulder" kp="60" dampratio="1"/>'
        '<position joint="elbow" kp="30" dampratio="1"/></actuator>')


def 처짐_재기(모터: bool, 초: float = 1.0, 사진: bool = False):
    """팔을 수평(0도)에서 놓고 '초'만큼 진행한 뒤 관절 각도를 돌려준다.

    Args:
        모터: True면 position 액추에이터 2개를 단다. False면 관절만.
        초: 시뮬레이션 시간(초).
        사진: True면 0초와 마지막 순간의 사진(숫자 배열) 두 장도 돌려준다.

    Returns:
        관절 각도(도) 배열 [어깨, 팔꿈치]. 사진=True면 (각도, [사진0, 사진끝]).
    """
    모델 = mujoco.MjModel.from_xml_string(틀.replace("{액추}", 액추 if 모터 else ""))
    데이터 = mujoco.MjData(모델)
    사진들 = []
    if 사진:
        렌더러 = mujoco.Renderer(모델, 400, 400)
        카메라 = mujoco.MjvCamera()
        카메라.lookat[:] = [0.2, 0, 0.6]
        카메라.distance, 카메라.azimuth, 카메라.elevation = 1.9, 90, -8
        mujoco.mj_forward(모델, 데이터)
        렌더러.update_scene(데이터, camera=카메라)
        사진들.append(렌더러.render())
    while 데이터.time < 초 - 1e-9:
        if 모델.nu:
            데이터.ctrl[:] = [0.0, 0.0]                # 목표: 두 관절 모두 0도(수평)
        mujoco.mj_step(모델, 데이터)
    각도 = np.rad2deg(데이터.qpos.copy())
    if 사진:
        mujoco.mj_forward(모델, 데이터)
        렌더러.update_scene(데이터, camera=카메라)
        사진들.append(렌더러.render())
        렌더러.close()
        return 각도, 사진들
    return 각도


if __name__ == "__main__":
    파서 = argparse.ArgumentParser(description="관절만 vs 관절+액추에이터")
    파서.add_argument("--그림", action="store_true", help="사진 저장")
    인자 = 파서.parse_args()
    for 이름, 모터 in [("관절만", False), ("액추에이터", True)]:
        결과 = 처짐_재기(모터, 사진=인자.그림)
        각도 = 결과[0] if 인자.그림 else 결과
        print(f"{이름:6s}: 1초 뒤 어깨 {각도[0]:.1f}도, 팔꿈치 {각도[1]:.1f}도")
        if 인자.그림:
            if not 모터:
                왼쪽 = 결과[1]                          # [0초, 1초] (관절만)
            else:
                from PIL import Image
                Image.fromarray(np.hstack([왼쪽[0], 왼쪽[1], 결과[1][1]])).save("v04_joint_vs_actuator.png")
                print("사진 저장: v04_joint_vs_actuator.png (왼쪽부터 0초 · 관절만 1초 · 액추에이터 1초)")
