"""2일차 보충 예제: 제어 루프를 직접 짜 본다 (교안 33번 슬라이드 "제어 루프 설계").

장면: 0.5m 막대 끝에 1kg 추, 어깨 경첩에 motor 액추에이터 하나.
      motor는 ctrl 값이 곧 토크(N·m)다. 수평(0도)에서 출발해 목표 60도로 든다.

매 걸음 ① 감지(각도 읽기) → ② 판단(오차) → ③ 명령(토크) → ④ 진행(mj_step).
결과: 0.3초쯤 60도 근처, 2초 뒤 56.1도. 남은 3.9도는 중력 때문이다
      (명령 2.74N·m가 추의 무게를 버티는 데 쓰인다).

실행 방법 (04-mujoco-basics 폴더 안에서):
    python v02_control_loop.py              # 숫자 출력 + 그래프 v02_control_loop.png
    python v02_control_loop.py --kp 200     # kp를 바꿔 처짐 비교
"""
import argparse

import mujoco
import numpy as np

# 어깨 경첩 축을 -y로 두면 +각도 = 막대 끝이 위(+z)로 올라간다
XML = """
<mujoco model="제어 루프">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <light pos="0 -1 2" dir="0 0.5 -1"/>
    <body name="팔" pos="0 0 1">
      <joint name="shoulder" type="hinge" axis="0 -1 0"/>
      <geom type="capsule" fromto="0 0 0 0.5 0 0" size="0.01" mass="0"/>  <!-- 막대: 질량 0 -->
      <geom type="sphere" pos="0.5 0 0" size="0.04" mass="1"/>            <!-- 추 1kg -->
    </body>
  </worldbody>
  <actuator>
    <motor name="어깨모터" joint="shoulder"/>   <!-- ctrl = 토크(N·m) -->
  </actuator>
</mujoco>
"""


def 장면_만들기() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """막대-추 장면의 모델과 데이터를 만든다.

    Returns:
        (MjModel, MjData) 튜플. 처음 각도는 0도(수평)다.
    """
    모델 = mujoco.MjModel.from_xml_string(XML)
    데이터 = mujoco.MjData(모델)
    return 모델, 데이터


def 제어_루프(초: float = 2.0, 목표_도: float = 60.0, kp: float = 40.0, kd: float = 4.0) -> dict:
    """P-D 제어 루프를 돌리고 매 걸음 시각·각도·토크를 기록한다.

    Args:
        초: 시뮬레이션할 시간(초).
        목표_도: 목표 각도(도).
        kp: 1rad 틀릴 때 내는 토크(N·m). 클수록 세게 민다.
        kd: 각속도 1rad/s당 반대로 거는 토크(브레이크).

    Returns:
        "시각", "각도_도", "토크" 키를 가진 사전. 값은 매 걸음의 numpy 배열.
    """
    모델, 데이터 = 장면_만들기()
    목표 = np.deg2rad(목표_도)
    시각, 각도들, 토크들 = [], [], []
    while 데이터.time < 초 - 1e-9:
        각도 = 데이터.joint("shoulder").qpos[0]      # ① 감지: 지금 각도
        각속도 = 데이터.joint("shoulder").qvel[0]
        오차 = 목표 - 각도                           # ② 판단: 목표와의 차이
        데이터.ctrl[0] = kp * 오차 - kd * 각속도      # ③ 명령: 토크 계산해 넣기
        mujoco.mj_step(모델, 데이터)                 # ④ 진행: 0.002초
        시각.append(데이터.time)
        각도들.append(np.rad2deg(데이터.qpos[0]))
        토크들.append(데이터.ctrl[0])
    return {"시각": np.array(시각), "각도_도": np.array(각도들), "토크": np.array(토크들)}


def 그래프_저장(기록: dict, 목표_도: float, 파일: str = "v02_control_loop.png") -> None:
    """각도와 토크를 시간에 따라 그려 PNG로 저장한다.

    Args:
        기록: 제어_루프()가 돌려준 사전.
        목표_도: 그래프에 점선으로 표시할 목표 각도(도).
        파일: 저장할 파일 이름.
    """
    import matplotlib
    matplotlib.use("Agg")                          # 창 없이 파일로만 저장
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    있는_글꼴 = {f.name for f in font_manager.fontManager.ttflist}
    for 글꼴 in ["Malgun Gothic", "AppleGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"]:
        if 글꼴 in 있는_글꼴:                      # 컴퓨터에 있는 한글 글꼴 하나를 고른다
            plt.rcParams["font.family"] = 글꼴
            break
    plt.rcParams["axes.unicode_minus"] = False
    그림, (왼, 오) = plt.subplots(1, 2, figsize=(10, 4))
    왼.plot(기록["시각"], 기록["각도_도"], label="각도")
    왼.axhline(목표_도, ls="--", color="gray", label="목표")
    왼.set(xlabel="시간 (초)", ylabel="각도 (도)", title="각도")
    왼.legend()
    오.plot(기록["시각"], 기록["토크"], color="tab:orange")
    오.set(xlabel="시간 (초)", ylabel="토크 (N·m)", title="모터 명령")
    그림.tight_layout()
    그림.savefig(파일, dpi=120)
    print("그래프 저장:", 파일)


if __name__ == "__main__":
    파서 = argparse.ArgumentParser(description="막대-추 P-D 제어 루프")
    파서.add_argument("--kp", type=float, default=40.0, help="1rad 오차당 토크 (기본 40)")
    파서.add_argument("--kd", type=float, default=4.0, help="브레이크 세기 (기본 4)")
    인자 = 파서.parse_args()

    기록 = 제어_루프(초=2.0, kp=인자.kp, kd=인자.kd)
    for t in [0.3, 1.0, 2.0]:
        i = int(round(t / 0.002)) - 1                # t초에 해당하는 걸음 번호
        print(f"{t:.1f}초 | 각도 {기록['각도_도'][i]:5.1f}도 | 토크 {기록['토크'][i]:6.2f}N·m")
    print(f"처짐: {60 - 기록['각도_도'][-1]:.1f}도 (중력 때문)")
    그래프_저장(기록, 60.0)
