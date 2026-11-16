import numpy as np
from scipy.spatial.transform import Rotation

from rotation_tools import 깔끔


def 축각_행렬(축: np.ndarray, 각도_도: float) -> np.ndarray:
    """로드리게스 공식: 단위 축 k를 중심으로 각도만큼 돌리는 회전 행렬."""
    k = 축 / np.linalg.norm(축)
    라디안 = np.deg2rad(각도_도)
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])        # K @ v == np.cross(k, v)
    return np.eye(3) + np.sin(라디안) * K + (1 - np.cos(라디안)) * (K @ K)


축 = np.array([1.0, 1.0, 1.0])               # 정육면체의 대각선 방향
R = 축각_행렬(축, 120)
print("(1,1,1)축으로 120도:\n", 깔끔(R))
print("x축은 어디로?", 깔끔(R @ np.array([1.0, 0.0, 0.0])))

회전벡터 = 축 / np.linalg.norm(축) * np.deg2rad(120)    # 방향 = 축, 길이 = 각도(라디안)
print("scipy와 같은가?", np.allclose(R, Rotation.from_rotvec(회전벡터).as_matrix()))

되읽기 = Rotation.from_matrix(R).as_rotvec()
print("행렬에서 읽은 회전 벡터:", 깔끔(되읽기), "/ 길이:", round(float(np.rad2deg(np.linalg.norm(되읽기))), 3), "도")
