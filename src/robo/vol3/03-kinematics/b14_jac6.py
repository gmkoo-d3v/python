import numpy as np
from scipy.spatial.transform import Rotation

from 야코비안도구 import 야코비안
from 팔도구 import 순기구학
from 회전도구 import 깔끔

q = np.deg2rad([30, 45, 60, 0, 45, 0])
J = 야코비안(q)
print("야코비안 모양:", J.shape)
print("허리 관절 열(손끝 속도 | 각속도):", 깔끔(J[:, 0]))

# 수치 야코비안: 관절 하나를 ±h 돌렸을 때 손끝 위치와 자세가 얼마나 변하는가
h = 1e-6
수치 = np.zeros((6, 6))
for i in range(6):
    dq = np.zeros(6)
    dq[i] = h
    T_앞, T_뒤 = 순기구학(q + dq)[-1], 순기구학(q - dq)[-1]
    수치[:3, i] = (T_앞[:3, 3] - T_뒤[:3, 3]) / (2 * h)
    자세_변화 = T_앞[:3, :3] @ T_뒤[:3, :3].T                 # 바닥 기준으로 돈 만큼
    수치[3:, i] = Rotation.from_matrix(자세_변화).as_rotvec() / (2 * h)
print(f"위치 부분 최대 차이: {np.abs(수치[:3] - J[:3]).max():.1e}")
print(f"자세 부분 최대 차이: {np.abs(수치[3:] - J[3:]).max():.1e}")
