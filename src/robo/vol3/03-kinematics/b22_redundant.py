import numpy as np

길이 = np.array([0.4, 0.3, 0.2])            # 3링크 평면 팔: 관절 3개로 손끝 위치(2개)만 맞춘다


def 손끝(q: np.ndarray) -> np.ndarray:
    누적 = np.cumsum(q)                        # 각 링크가 바닥에서 본 방향
    return np.array([길이 @ np.cos(누적), 길이 @ np.sin(누적)])


def 야코비안3(q: np.ndarray) -> np.ndarray:
    누적 = np.cumsum(q)
    J = np.zeros((2, 3))
    for i in range(3):                         # 관절 i는 자기 뒤의 링크를 모두 돌린다
        J[0, i] = -(길이[i:] @ np.sin(누적[i:]))
        J[1, i] = 길이[i:] @ np.cos(누적[i:])
    return J


목표 = np.array([0.5, 0.3])
print("같은 목표, 다른 시작 → 다른 답")
for 시작_도 in [(0, 30, 30), (90, -60, -60), (-30, 90, 60)]:
    q = np.deg2rad(시작_도)
    for _ in range(100):
        q = q + np.linalg.pinv(야코비안3(q)) @ (목표 - 손끝(q))
    print(f"  시작 {시작_도} → 관절 {np.round(np.rad2deg(q), 1) + 0.0}도, 손끝 {np.round(손끝(q), 4) + 0.0}")

# 영공간 움직임: 손끝은 그대로 두고 관절만 움직이기
print("손끝을 붙잡은 채 관절만 움직이기")
for 걸음 in range(301):
    if 걸음 % 100 == 0:
        print(f"  {걸음:3d}걸음: 관절 {np.round(np.rad2deg(q), 1) + 0.0}도, 손끝 {np.round(손끝(q), 4) + 0.0}")
    J = 야코비안3(q)
    영공간 = np.eye(3) - np.linalg.pinv(J) @ J        # 손끝을 움직이지 않는 관절 속도만 남기는 행렬
    q = q + 영공간 @ np.array([0.0, 0.0, 0.002])     # 손목(세 번째 관절)을 조금씩 돌리고 싶다
    q = q + np.linalg.pinv(야코비안3(q)) @ (목표 - 손끝(q))   # 쌓이는 작은 오차는 바로 고친다
