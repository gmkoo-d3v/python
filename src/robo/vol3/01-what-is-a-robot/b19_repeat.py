import time

import mujoco
import numpy as np

from box_scene import 상자_xml

모델 = mujoco.MjModel.from_xml_string(상자_xml)


def 한번_던지기(옆_속도: float) -> np.ndarray:
    """상자를 옆으로 밀면서 떨어뜨리고 3초 뒤 위치를 돌려준다."""
    데이터 = mujoco.MjData(모델)                  # 매번 처음 상태에서 다시 시작
    데이터.qvel[0] = 옆_속도                       # 자유 관절의 속도: 앞 셋이 x, y, z 방향
    while 데이터.time < 3.0:
        mujoco.mj_step(모델, 데이터)
    return 데이터.qpos[:3].copy()


벽시계_시작 = time.perf_counter()
첫번째 = 한번_던지기(1.0)
두번째 = 한번_던지기(1.0)
걸린_시간 = time.perf_counter() - 벽시계_시작
print("첫 번째 3초 뒤 위치:", np.round(첫번째, 3))
print("두 번째와 완전히 같은가?", np.array_equal(첫번째, 두번째))
print("6초어치 시뮬레이션이 실제 1초 안에 끝났나?", 걸린_시간 < 1.0)

for 옆_속도 in [0.5, 1.5, 2.0]:                  # 조건만 바꿔 다시 해 보기도 공짜다
    print(f"옆 속도 {옆_속도}m/s → 3초 뒤 x {한번_던지기(옆_속도)[0]:.3f}m")
