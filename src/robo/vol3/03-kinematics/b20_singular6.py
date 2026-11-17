import numpy as np

from 야코비안도구 import 야코비안

for 이름, 관절각_도 in [("구부린 자세", [30, 45, 60, 0, 45, 0]),
                     ("손목만 편 자세", [30, 45, 60, 0, 0, 0]),
                     ("모두 0도(쭉 편 자세)", [0, 0, 0, 0, 0, 0])]:
    J = 야코비안(np.deg2rad(관절각_도))
    특이값 = np.linalg.svd(J, compute_uv=False)
    print(f"{이름}: 계수 {np.linalg.matrix_rank(J)}, 특이값 {np.round(특이값, 3) + 0.0}")
