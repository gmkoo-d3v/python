import numpy as np
from scipy.spatial.transform import Rotation

from 야코비안도구 import 야코비안
from 팔도구 import 순기구학

# 목표 포즈: 어떤 관절 각도에서 손끝이 있던 위치와 자세를 그대로 목표로 삼는다
정답_도 = np.array([30, 45, 60, 0, 45, 0])
T_목표 = 순기구학(np.deg2rad(정답_도))[-1]


def 포즈_오차(T_지금: np.ndarray) -> np.ndarray:
    """[위치 오차(m) 3개, 자세 오차(회전 벡터, rad) 3개]. 둘 다 바닥 기준."""
    위치 = T_목표[:3, 3] - T_지금[:3, 3]
    자세 = Rotation.from_matrix(T_목표[:3, :3] @ T_지금[:3, :3].T).as_rotvec()
    return np.concatenate([위치, 자세])


def 역기구학(최대_걸음: float | None, λ: float, 출력: bool = False) -> np.ndarray:
    q = np.deg2rad([0.0, 20.0, 20.0, 10.0, 20.0, 10.0])       # 아무렇게나 고른 시작 자세
    for 반복 in range(100):
        e = 포즈_오차(순기구학(q)[-1])
        끝 = np.linalg.norm(e) < 1e-8
        if 출력 and (반복 % 4 == 0 or 끝):
            print(f"  {반복:2d}회: 위치 오차 {np.linalg.norm(e[:3]) * 1000:8.3f}mm, "
                  f"자세 오차 {np.rad2deg(np.linalg.norm(e[3:])):7.3f}도")
        if 끝:
            break
        if 최대_걸음 is not None and np.linalg.norm(e) > 최대_걸음:
            e = e * 최대_걸음 / np.linalg.norm(e)                 # 한 번에 너무 멀리 가지 않게
        J = 야코비안(q)
        q = q + J.T @ np.linalg.solve(J @ J.T + λ**2 * np.eye(6), e)   # 감쇠 최소제곱 한 걸음
    return q


print("(가) 걸음 제한 없음, λ = 0.01")
q = 역기구학(None, 0.01)
print("  찾은 관절 각도(도):", np.round(np.rad2deg(q), 1) + 0.0)
print("  -180~180도로 정리:", np.round(np.rad2deg((q + np.pi) % (2 * np.pi) - np.pi), 1) + 0.0)
e = 포즈_오차(순기구학(q)[-1])
print(f"  남은 오차: 위치 {np.linalg.norm(e[:3]) * 1000:.6f}mm, 자세 {np.rad2deg(np.linalg.norm(e[3:])):.6f}도")
print("(나) 걸음 0.2로 제한, λ = 0.05")
q = 역기구학(0.2, 0.05, 출력=True)
print("  찾은 관절 각도(도):", np.round(np.rad2deg(q), 1) + 0.0)
