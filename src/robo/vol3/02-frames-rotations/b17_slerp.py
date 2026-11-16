import numpy as np
from scipy.spatial.transform import Rotation, Slerp

시작 = Rotation.from_euler("ZYX", [0, 0, 0], degrees=True)
끝 = Rotation.from_euler("ZYX", [170, 60, -120], degrees=True)
비율 = np.linspace(0, 1, 11)

# 방법 1: 요·피치·롤 숫자를 그냥 나눠서 섞기
오일러_경로 = Rotation.from_euler("ZYX", np.outer(비율, [170, 60, -120]), degrees=True)
# 방법 2: 구면 선형 보간(slerp)
slerp_경로 = Slerp([0, 1], Rotation.concatenate([시작, 끝]))(비율)


def 걸음_각도(경로: Rotation) -> np.ndarray:
    """이웃한 두 자세 사이가 몇 도인지."""
    return np.rad2deg((경로[:-1].inv() * 경로[1:]).magnitude())


print("시작에서 끝까지 최단 회전:", round(float(np.rad2deg((시작.inv() * 끝).magnitude())), 1), "도")
for 이름, 경로 in [("오일러 섞기", 오일러_경로), ("slerp", slerp_경로)]:
    걸음 = 걸음_각도(경로)
    print(f"{이름:6s}: 걸음마다 {걸음.min():5.1f}~{걸음.max():5.1f}도, 모두 더하면 {걸음.sum():6.1f}도")
