import numpy as np
from scipy.spatial.transform import Rotation

from quaternion_tools import 곱, 정규화, 축각_쿼터니언

# 아주 작은 회전(0.01도)을 10만 번 이어 붙인다. 32비트 실수로 계산해 오차를 키워 본다
작은_q = 축각_쿼터니언([1, 2, 3], 0.01).astype(np.float32)
작은_R = Rotation.from_quat(작은_q, scalar_first=True).as_matrix().astype(np.float32)

q = np.array([1, 0, 0, 0], dtype=np.float32)
R = np.eye(3, dtype=np.float32)
for _ in range(100_000):
    q = 곱(q, 작은_q).astype(np.float32)
    R = R @ 작은_R

print(f"쿼터니언 길이        : {np.linalg.norm(q):.6f}  (1이어야 함)")
print(f"회전 행렬 직교성 오차: {np.abs(R.T @ R - np.eye(3)).max():.6f}  (0이어야 함)")
print(f"정규화한 뒤 길이     : {np.linalg.norm(정규화(q)):.6f}")
