import numpy as np

from avoidance_tools import 달리기, 목표, 장애물_반지름, 장애물_중심, 주기

생성기 = np.random.default_rng(0)
for 이름, 옆으로 in [("2. 밀려나기", False), ("3. 밀려나며 옆으로", True)]:
    for 조건, 미끄러짐 in [("잡음만", 0.0), ("잡음+미끄러짐", 0.2)]:
        도착_시각들, 충돌 = [], 0
        for _ in range(20):
            경로 = 달리기(True, 옆으로, 생성기, 미끄러짐, 센서_잡음=0.02, 걸음_수=300)
            충돌 += bool((np.linalg.norm(경로 - 장애물_중심, axis=1) < 장애물_반지름).any())
            닿음 = np.linalg.norm(경로 - 목표, axis=1) < 0.05
            if 닿음.any():
                도착_시각들.append(np.argmax(닿음) * 주기)       # 처음 목표 5cm 안에 든 시각
        print(f"{이름}, {조건}: 20번 중 도착 {len(도착_시각들)}번, 충돌 {충돌}번, "
              f"걸린 시간 {min(도착_시각들):.1f}~{max(도착_시각들):.1f}초")
