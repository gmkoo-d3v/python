import numpy as np

from 두링크 import 순기구학2, 야코비안2

목표 = np.array([0.5, 0.3])
하한 = np.deg2rad([-90.0, 0.0])            # 어깨 -90~90도, 팔꿈치 0~150도(한쪽으로만 굽는 팔)
상한 = np.deg2rad([90.0, 150.0])


def 역기구학(q: np.ndarray, 한계_지키기: bool) -> np.ndarray:
    for _ in range(50):
        J = 야코비안2(q)
        q = q + J.T @ np.linalg.solve(J @ J.T + 0.01**2 * np.eye(2), 목표 - 순기구학2(q))
        if 한계_지키기:
            q = np.clip(q, 하한, 상한)        # 한계를 넘으면 한계에 붙여 둔다
    return q


for 한계_지키기 in [False, True]:
    q = 역기구학(np.deg2rad([90.0, 10.0]), 한계_지키기)
    거리 = np.linalg.norm(목표 - 순기구학2(q))
    안 = bool(np.all((q >= 하한 - 1e-9) & (q <= 상한 + 1e-9)))
    print(f"한계 지키기 {한계_지키기!s:5s}: 어깨 {np.rad2deg(q[0]):6.1f}도, 팔꿈치 {np.rad2deg(q[1]):6.1f}도, "
          f"남은 거리 {거리:.1e}m, 한계 안? {안}")
