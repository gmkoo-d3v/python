import numpy as np
from scipy.spatial.transform import Rotation

from rotation_tools import Rx, Ry, Rz

롤, 피치, 요 = 10, 20, 30
직접 = Rz(요) @ Ry(피치) @ Rx(롤)          # 롤 → 피치 → 요 순서로 고정 축을 따라 돌리기

외재_xyz = Rotation.from_euler("xyz", [롤, 피치, 요], degrees=True).as_matrix()
내재_ZYX = Rotation.from_euler("ZYX", [요, 피치, 롤], degrees=True).as_matrix()
내재_XYZ = Rotation.from_euler("XYZ", [롤, 피치, 요], degrees=True).as_matrix()

print("소문자 'xyz'(고정 축)  == Rz@Ry@Rx ?", np.allclose(외재_xyz, 직접))
print("대문자 'ZYX'(움직이는 축) == Rz@Ry@Rx ?", np.allclose(내재_ZYX, 직접))
print("대문자 'XYZ'(움직이는 축) == Rz@Ry@Rx ?", np.allclose(내재_XYZ, 직접))
print("행렬에서 다시 읽은 요·피치·롤:", np.round(Rotation.from_matrix(직접).as_euler("ZYX", degrees=True), 3))
