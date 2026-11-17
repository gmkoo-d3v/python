import numpy as np

from 두링크 import 순기구학2, 야코비안2
from 회전도구 import 깔끔


def 수치_야코비안(함수, q: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """관절을 하나씩 ±h만큼 움직여 손끝이 얼마나 변하는지 잰다(중앙 차분)."""
    열들 = []
    for i in range(len(q)):
        dq = np.zeros(len(q))
        dq[i] = h
        열들.append((함수(q + dq) - 함수(q - dq)) / (2 * h))
    return np.column_stack(열들)


q = np.deg2rad([30.0, 45.0])
print("수치 야코비안:\n", 깔끔(수치_야코비안(순기구학2, q), 4))
print("공식 야코비안:\n", 깔끔(야코비안2(q), 4))
print("가장 큰 차이:", f"{np.abs(수치_야코비안(순기구학2, q) - 야코비안2(q)).max():.1e}")
