import numpy as np

from quaternion_tools import 벡터_돌리기, 사이각, 축각_쿼터니언
from rotation_tools import 깔끔

q = 축각_쿼터니언([0, 0, 1], 90)
print("q  :", np.round(q, 4) + 0.0)
print("-q :", np.round(-q, 4) + 0.0)
print("q로 돌린 앞 :", 깔끔(벡터_돌리기(q, [1, 0, 0])))
print("-q로 돌린 앞:", 깔끔(벡터_돌리기(-q, [1, 0, 0])))
print("내적:", round(float(np.dot(q, -q)), 4), "/ 사이각:", 사이각(q, -q), "도")

q10 = 축각_쿼터니언([0, 0, 1], 10)
q370 = 축각_쿼터니언([0, 0, 1], 370)
print("10도 :", np.round(q10, 4) + 0.0)
print("370도:", np.round(q370, 4) + 0.0)
