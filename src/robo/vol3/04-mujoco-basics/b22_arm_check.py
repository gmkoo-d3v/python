import mujoco
import numpy as np

from 팔장면 import XML, 손끝_계산
from 회전도구 import 깔끔

모델 = mujoco.MjModel.from_xml_string(XML)
데이터 = mujoco.MjData(모델)
print("관절:", [모델.joint(i).name for i in range(모델.njnt)], "/ nq =", 모델.nq)
print("액추에이터:", [모델.actuator(i).name for i in range(모델.nu)])
어깨 = 모델.actuator("shoulder")
print("어깨 액추에이터 kp:", 어깨.gainprm[0], "/ kv:", round(-float(어깨.biasprm[2]), 3))

for 어깨_도, 팔꿈치_도 in [(0, 0), (30, 60), (-45, 90)]:
    데이터.joint("shoulder").qpos[0] = np.deg2rad(어깨_도)
    데이터.joint("elbow").qpos[0] = np.deg2rad(팔꿈치_도)
    mujoco.mj_forward(모델, 데이터)                       # 시간은 그대로, 위치만 계산
    시뮬 = 데이터.site("tip").xpos[:2] - 데이터.body("base").xpos[:2]
    식 = 손끝_계산(np.deg2rad(어깨_도), np.deg2rad(팔꿈치_도))
    print(f"({어깨_도:3d}도, {팔꿈치_도:2d}도) 손끝: MuJoCo {깔끔(시뮬)}, 순기구학 {깔끔(식)}")
