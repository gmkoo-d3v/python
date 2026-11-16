import numpy as np

from point_robot import 최대_속도, 움직이기

시작 = np.array([0.0, 0.0])
목표 = np.array([2.0, 1.0])
주기 = 0.1  # 0.1초마다 명령을 보낸다

# 출발 전에 한 번만 계산한다: 어느 방향으로, 몇 번 명령을 보낼까
거리 = np.linalg.norm(목표 - 시작)
방향 = (목표 - 시작) / 거리
명령_횟수 = round(거리 / (최대_속도 * 주기))
print(f"거리 {거리:.3f}m, 최대 속도로 {명령_횟수}번 보내면 된다")

위치 = 시작.copy()
for _ in range(명령_횟수):  # 감지 없이 정해 둔 명령만 보낸다
    위치 = 움직이기(위치, 방향 * 최대_속도, 주기)
print(
    "도착한 곳:",
    np.round(위치, 3),
    "/ 목표까지 남은 거리:",
    round(float(np.linalg.norm(목표 - 위치)), 3),
    "m",
)
