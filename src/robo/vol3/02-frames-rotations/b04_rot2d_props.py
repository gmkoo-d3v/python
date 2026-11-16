import numpy as np


def 회전2d(각도_도: float) -> np.ndarray:
    라디안 = np.deg2rad(각도_도)
    c, s = np.cos(라디안), np.sin(라디안)
    return np.array([[c, -s], [s, c]])


R = 회전2d(30)
print("R의 전치 @ R =\n", np.round(R.T @ R, 6) + 0.0)
print("행렬식:", round(float(np.linalg.det(R)), 6))
print("역행렬과 전치가 같은가?", np.allclose(np.linalg.inv(R), R.T))

점 = np.array([3.0, 4.0])
돌린_점 = R @ 점
print("돌리기 전 길이:", np.linalg.norm(점), "/ 돌린 뒤 길이:", round(float(np.linalg.norm(돌린_점)), 6))
print("전치로 되돌리면:", np.round(R.T @ 돌린_점, 6) + 0.0)
print("30도를 두 번 == 60도 한 번?", np.allclose(R @ R, 회전2d(60)))
