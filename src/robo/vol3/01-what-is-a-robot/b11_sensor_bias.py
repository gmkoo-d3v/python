import numpy as np

from point_robot import 움직이기, 위치_재기

목표 = np.array([2.0, 1.0])
주기, 이득 = 0.1, 1.5
생성기 = np.random.default_rng(0)

for 치우침_x in [0.0, 0.05]:
    위치 = np.array([0.0, 0.0])
    for _ in range(80):
        잰_위치 = 위치_재기(위치, 생성기, 0.02) + np.array([치우침_x, 0.0])   # 늘 x로 치우쳐 잰다
        위치 = 움직이기(위치, 이득 * (목표 - 잰_위치), 주기, 생성기, 0.2)
    print(f"센서 x 치우침 {치우침_x:.2f}m → 멈춘 곳 {np.round(위치, 3)}, 목표와 x 차이 {위치[0] - 목표[0]:+.3f}m")
