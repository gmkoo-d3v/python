import numpy as np

from avoidance_tools import 장애물_감지

for 잰_위치 in [np.array([0.0, 0.0]), np.array([0.5, 0.25]), np.array([0.75, 0.2])]:
    감지 = 장애물_감지(잰_위치)
    if 감지 is None:
        print(f"{잰_위치} → 감지 없음")
    else:
        거리, 바깥 = 감지
        print(f"{잰_위치} → 표면까지 {거리:.3f}m, 장애물에서 나를 향한 방향 {np.round(바깥, 3)}")
