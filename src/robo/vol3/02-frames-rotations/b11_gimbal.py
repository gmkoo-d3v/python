import warnings

import numpy as np
from scipy.spatial.transform import Rotation

from rotation_tools import Rx, Ry, Rz


def 자세(요: float, 피치: float, 롤: float) -> np.ndarray:
    return Rz(요) @ Ry(피치) @ Rx(롤)


# 피치가 정확히 90도이면, 롤과 요가 같은 일을 한다
가 = 자세(0, 90, 30)
나 = 자세(-30, 90, 0)
다 = 자세(20, 90, 50)
print("요 0, 롤 30  == 요 -30, 롤 0 ?", np.allclose(가, 나))
print("요 0, 롤 30  == 요 20, 롤 50 ?", np.allclose(가, 다))

# 피치가 89도만 돼도 셋은 다른 자세다
print("피치 89도에서도 같은가?", np.allclose(자세(0, 89, 30), 자세(-30, 89, 0)))

with warnings.catch_warnings(record=True) as 경고들:
    warnings.simplefilter("always")
    되읽기 = Rotation.from_matrix(가).as_euler("ZYX", degrees=True)
print("되읽은 요·피치·롤:", np.round(되읽기, 3) + 0.0)
print("경고:", 경고들[0].message)
