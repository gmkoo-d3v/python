import numpy as np


def 회전2d(각도_도: float) -> np.ndarray:
    """평면에서 원점을 중심으로 반시계 방향으로 돌리는 2×2 회전 행렬."""
    라디안 = np.deg2rad(각도_도)
    c, s = np.cos(라디안), np.sin(라디안)
    return np.array([[c, -s],
                     [s,  c]])


점 = np.array([1.0, 0.0])                 # x축 위, 원점에서 1만큼 떨어진 점
for 각도 in [30, 90, 180]:
    돌린_점 = 회전2d(각도) @ 점
    print(f"{각도:3d}도 돌리면 → {np.round(돌린_점, 3) + 0.0}")
