import numpy as np

from 회전도구 import Rz, 깔끔


def 변환(R: np.ndarray, p) -> np.ndarray:
    """2장의 동차 변환: 회전 R과 이동 p를 4×4 하나로."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


손끝_관절 = np.array([0.3, 0.0, 0.0, 1.0])     # 관절 좌표계에서 본 손끝(동차 좌표)

print("회전 관절: z축으로 돌리기")
for 각도_도 in [0, 45, 90]:
    손끝 = 변환(Rz(각도_도), [0, 0, 0]) @ 손끝_관절
    print(f"  {각도_도:3d}도 → 손끝 {깔끔(손끝[:3])}, 관절까지 거리 {np.linalg.norm(손끝[:3]):.2f}m")

print("직선 관절: z축을 따라 밀기")
for 거리 in [0.0, 0.1, 0.2]:
    손끝 = 변환(np.eye(3), [0, 0, 거리]) @ 손끝_관절
    print(f"  {거리:.1f}m → 손끝 {깔끔(손끝[:3])}")
