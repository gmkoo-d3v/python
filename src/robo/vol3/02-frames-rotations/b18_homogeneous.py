import argparse

import numpy as np

from rotation_tools import Rz, 깔끔


def 변환(R: np.ndarray, p) -> np.ndarray:
    """회전 R과 이동 p를 4×4 동차 변환 행렬 하나로 묶는다."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def 역변환(T: np.ndarray) -> np.ndarray:
    """T의 반대 방향 변환. 회전은 전치, 이동은 -Rᵀp."""
    R, p = T[:3, :3], T[:3, 3]
    return 변환(R.T, -R.T @ p)


parser = argparse.ArgumentParser(description="방 좌표의 컵을 로봇 좌표로 변환")
parser.add_argument("--yaw", type=float, default=90.0, help="로봇의 요 각도 (도)")
args = parser.parse_args()
if not np.isfinite(args.yaw):
    parser.error("yaw는 유한한 수여야 합니다")
# 기본값 90도이면 로봇의 앞이 방의 +y 쪽이다.
T_방_로봇 = 변환(Rz(args.yaw), [1.5, 0.0, 0.0])
print("T_방_로봇 =\n", 깔끔(T_방_로봇))

컵_방 = np.array([2.0, 1.0, 0.0, 1.0])           # 동차 좌표: 끝에 1을 붙인다
컵_로봇 = 역변환(T_방_로봇) @ 컵_방
print("로봇 기준 컵:", 깔끔(컵_로봇[:3]))
print("다시 방 기준:", 깔끔((T_방_로봇 @ 컵_로봇)[:3]))
print("np.linalg.inv와 같은가?", np.allclose(역변환(T_방_로봇), np.linalg.inv(T_방_로봇)))
