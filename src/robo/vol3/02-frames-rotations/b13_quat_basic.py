import numpy as np
from scipy.spatial.transform import Rotation

from quaternion_tools import 벡터_돌리기, 축각_쿼터니언
from rotation_tools import 깔끔

q = 축각_쿼터니언([0, 0, 1], 90)            # z축으로 90도
print("z축 90도 쿼터니언 [w x y z]:", np.round(q, 4))
print("길이:", round(float(np.linalg.norm(q)), 6))
print("x축을 돌리면:", 깔끔(벡터_돌리기(q, [1, 0, 0])))

r = Rotation.from_rotvec([0, 0, np.pi / 2])
print("scipy as_quat()                 :", np.round(r.as_quat(), 4))
print("scipy as_quat(scalar_first=True):", np.round(r.as_quat(scalar_first=True), 4))
