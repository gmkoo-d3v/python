import numpy as np

from rotation_tools import Rx, Rz, 깔끔

앞 = np.array([1.0, 0.0, 0.0])

# 행렬 곱은 오른쪽부터 적용된다: Rz(90) @ Rx(90) @ v 는 'x축으로 먼저, 그다음 z축으로'
가 = Rz(90) @ Rx(90) @ 앞
나 = Rx(90) @ Rz(90) @ 앞
print("x 먼저, z 나중:", 깔끔(가))
print("z 먼저, x 나중:", 깔끔(나))
print("두 행렬이 같은가?", np.allclose(Rz(90) @ Rx(90), Rx(90) @ Rz(90)))
