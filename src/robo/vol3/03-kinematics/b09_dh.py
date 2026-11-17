import numpy as np

from 회전도구 import Rx, Rz, 깔끔


def 변환(R: np.ndarray, p) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def DH(세타_도: float, d: float, a: float, 알파_도: float) -> np.ndarray:
    """DH 관례 한 줄: z축으로 θ 돌고 d 올라간 뒤, x축으로 a 나아가고 α 돈다."""
    return (변환(Rz(세타_도), [0, 0, d]) @ 변환(Rx(알파_도), [a, 0, 0]))


# 2링크 평면 팔의 DH 표: (θ, d, a, α)
어깨_도, 팔꿈치_도 = 30, 45
DH_표 = [(어깨_도, 0.0, 0.4, 0.0),
        (팔꿈치_도, 0.0, 0.3, 0.0)]
T = np.eye(4)
for 줄 in DH_표:
    T = T @ DH(*줄)
print("DH로 구한 손끝 위치:", 깔끔(T[:3, 3]))
