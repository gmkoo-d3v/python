import mujoco
import numpy as np

from 야코비안도구 import 야코비안
from 팔도구 import 순기구학
from 팔모델 import MJCF

모델 = mujoco.MjModel.from_xml_string(MJCF())
데이터 = mujoco.MjData(모델)
print("관절 수(nq):", 모델.nq, "/ 속도 수(nv):", 모델.nv)

생성기 = np.random.default_rng(0)
위치_차이, 자세_차이, 야코비안_차이 = [], [], []
for _ in range(100):                                  # 아무 관절 각도 100개로 비교
    q = 생성기.uniform(-np.pi, np.pi, size=6)
    데이터.qpos[:] = q
    mujoco.mj_forward(모델, 데이터)                    # 순기구학을 포함한 계산
    T = 순기구학(q)[-1]
    위치_차이.append(np.abs(데이터.site("tip").xpos - T[:3, 3]).max())
    자세_차이.append(np.abs(데이터.site("tip").xmat.reshape(3, 3) - T[:3, :3]).max())

    jacp, jacr = np.zeros((3, 모델.nv)), np.zeros((3, 모델.nv))
    mujoco.mj_jacSite(모델, 데이터, jacp, jacr, 데이터.site("tip").id)
    야코비안_차이.append(np.abs(np.vstack([jacp, jacr]) - 야코비안(q)).max())

print(f"손끝 위치 최대 차이  : {max(위치_차이):.1e} m")
print(f"손끝 자세 최대 차이  : {max(자세_차이):.1e}")
print(f"야코비안 최대 차이   : {max(야코비안_차이):.1e}")
