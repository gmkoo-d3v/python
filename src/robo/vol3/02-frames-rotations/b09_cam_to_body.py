import numpy as np

from rotation_tools import 깔끔

# 카메라 좌표(x 오른쪽, y 아래, z 앞)로 잰 값을 로봇 몸 좌표(x 앞, y 왼쪽, z 위)로 옮기는 행렬.
# 각 열은 카메라의 x, y, z축이 몸 좌표에서 어느 방향인지를 적은 것이다.
R_몸_카메라 = np.array([[ 0,  0, 1],
                     [-1,  0, 0],
                     [ 0, -1, 0]])
print("행렬식:", round(float(np.linalg.det(R_몸_카메라)), 6))
print("직교 행렬인가?", np.allclose(R_몸_카메라.T @ R_몸_카메라, np.eye(3)))

# 카메라가 본 컵: 화면 가운데에서 오른쪽 0.1m, 위쪽 0.05m, 앞으로 0.8m
컵_카메라 = np.array([0.1, -0.05, 0.8])
print("몸 기준 컵 위치:", 깔끔(R_몸_카메라 @ 컵_카메라))