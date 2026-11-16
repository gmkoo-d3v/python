import numpy as np

from rotation_tools import Rx, Ry, Rz, 깔끔

앞 = np.array([1.0, 0.0, 0.0])        # 로봇의 '앞' 방향(x축)
print("z축으로 90도:", 깔끔(Rz(90) @ 앞))   # 왼쪽으로 고개 돌리기
print("y축으로 90도:", 깔끔(Ry(90) @ 앞))   # 고개 숙이기
print("x축으로 90도:", 깔끔(Rx(90) @ 앞))   # 제자리에서 갸웃
