"""2.11절 실습: 예제용으로 만든 자이로(각속도) 신호와 그 참값 자세."""
import numpy as np

from 쿼터니언도구 import 곱

주기 = 0.01                         # 100Hz: 1초에 100번 잰다
시간 = np.arange(0, 10, 주기)       # 10초 동안 1000번


def 각속도(t: float) -> np.ndarray:
    """몸 좌표 기준 각속도(rad/s). 실제 센서 값이 아니라 예제용으로 만든 움직임이다."""
    return np.array([0.8 * np.sin(1.3 * t),
                     0.6 * np.cos(0.9 * t),
                     0.5 + 0.3 * np.sin(0.5 * t)])


def 증분_쿼터니언(각속도_값: np.ndarray, 간격: float) -> np.ndarray:
    """각속도로 간격(초)만큼 돌았을 때의 회전. 회전 벡터 = 각속도 × 시간."""
    회전벡터 = 각속도_값 * 간격
    각 = np.linalg.norm(회전벡터)
    if 각 < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return np.concatenate([[np.cos(각 / 2)], np.sin(각 / 2) * 회전벡터 / 각])


def 참_자세(세분: int = 100) -> np.ndarray:
    """한 주기를 다시 세분 조각으로 잘게 나눠 적분한 기준 자세. 시작 자세 포함 1001개."""
    q = np.array([1.0, 0.0, 0.0, 0.0])
    결과 = [q]
    작은_간격 = 주기 / 세분
    for t in 시간:
        for i in range(세분):
            q = 곱(q, 증분_쿼터니언(각속도(t + (i + 0.5) * 작은_간격), 작은_간격))
        결과.append(q)
    return np.array(결과)
