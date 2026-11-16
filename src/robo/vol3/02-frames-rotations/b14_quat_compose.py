import numpy as np
from scipy.spatial.transform import Rotation

from quaternion_tools import 곱, 벡터_돌리기, 축각_쿼터니언
from rotation_tools import Rx, Rz, 깔끔

q_z = 축각_쿼터니언([0, 0, 1], 90)
q_x = 축각_쿼터니언([1, 0, 0], 90)

합성 = 곱(q_z, q_x)                          # x축 먼저, 그다음 z축
R_합성 = Rotation.from_quat(합성, scalar_first=True).as_matrix()
print("곱(q_z, q_x) == Rz(90) @ Rx(90) ?", np.allclose(R_합성, Rz(90) @ Rx(90)))
print("곱(q_x, q_z) == 곱(q_z, q_x) ?", np.allclose(곱(q_x, q_z), 합성))

앞 = [1.0, 0.0, 0.0]
print("쿼터니언으로 돌린 앞:", 깔끔(벡터_돌리기(합성, 앞)))
print("행렬로 돌린 앞    :", 깔끔(Rz(90) @ Rx(90) @ 앞))
