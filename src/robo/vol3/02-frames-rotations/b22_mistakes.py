import numpy as np
from scipy.spatial.transform import Rotation

from rotation_tools import Rz, 깔끔

# 실수 1: [w, x, y, z] 쿼터니언을 순서 확인 없이 scipy에 넣기
q_wxyz = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])    # z축 90도
틀림 = Rotation.from_quat(q_wxyz)                       # scipy 기본은 [x, y, z, w]
맞음 = Rotation.from_quat(q_wxyz, scalar_first=True)
print("순서를 틀리면 회전 벡터(도):", 깔끔(np.rad2deg(틀림.as_rotvec()), 1))
print("순서를 맞추면 회전 벡터(도):", 깔끔(np.rad2deg(맞음.as_rotvec()), 1))


# 실수 2: 변환을 곱하는 순서 바꾸기
def 변환(R: np.ndarray, p) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


T_방_로봇 = 변환(Rz(90), [1.5, 0.0, 0.0])
T_로봇_손 = 변환(np.eye(3), [0.3, 0.0, 0.4])              # 손은 로봇 앞 0.3m, 높이 0.4m
print("T_방_로봇 @ T_로봇_손:", 깔끔((T_방_로봇 @ T_로봇_손)[:3, 3]))
print("T_로봇_손 @ T_방_로봇:", 깔끔((T_로봇_손 @ T_방_로봇)[:3, 3]))
