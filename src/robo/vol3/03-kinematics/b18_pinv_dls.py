import numpy as np

from 두링크 import 야코비안2
from 회전도구 import 깔끔


def 감쇠_역(J: np.ndarray, λ: float) -> np.ndarray:
    """감쇠 최소제곱(DLS): Jᵀ(JJᵀ + λ²I)⁻¹. λ = 0이면 의사역행렬과 같다."""
    return J.T @ np.linalg.inv(J @ J.T + λ**2 * np.eye(J.shape[0]))


J = 야코비안2(np.deg2rad([0.0, 1.0]))            # 팔을 거의 다 편 자세(팔꿈치 1도)
for 이름, 원하는_속도 in [("옆으로(y) 0.1m/s", [0.0, 0.1]), ("바깥으로(x) 0.1m/s", [0.1, 0.0])]:
    print(f"원하는 손끝 속도: {이름}")
    for 방법, 역 in [("의사역행렬", np.linalg.pinv(J)), ("DLS λ=0.05", 감쇠_역(J, 0.05))]:
        관절_속도 = 역 @ np.array(원하는_속도)
        print(f"  {방법}: 관절 속도 {깔끔(관절_속도, 2)} rad/s → 실제 손끝 속도 {깔끔(J @ 관절_속도)}")
