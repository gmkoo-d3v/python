import numpy as np

from 두링크 import 야코비안2

print("팔꿈치   행렬식    조건수      특이값")
for 팔꿈치_도 in [90, 45, 10, 1, 0]:
    J = 야코비안2(np.deg2rad([0.0, 팔꿈치_도]))
    특이값 = np.linalg.svd(J, compute_uv=False)
    조건수 = 특이값[0] / 특이값[1] if 특이값[1] > 1e-12 else np.inf
    print(f"{팔꿈치_도:4d}도  {np.linalg.det(J):8.5f}  {조건수:9.1f}   {np.round(특이값, 5) + 0.0}")
