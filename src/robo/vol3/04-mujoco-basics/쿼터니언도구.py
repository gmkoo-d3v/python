"""쿼터니언을 [w, x, y, z] 순서(실수부 먼저)로 다루는 작은 도구 모음."""
import numpy as np


def 축각_쿼터니언(축, 각도_도: float) -> np.ndarray:
    """축을 중심으로 각도만큼 돌리는 단위 쿼터니언."""
    k = np.asarray(축, dtype=float)
    k = k / np.linalg.norm(k)
    반각 = np.deg2rad(각도_도) / 2
    return np.concatenate([[np.cos(반각)], np.sin(반각) * k])


def 곱(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """해밀턴 곱. 곱(q1, q2)는 'q2 먼저, 그다음 q1'으로 돌린 것과 같다."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def 켤레(q: np.ndarray) -> np.ndarray:
    """허수부의 부호를 바꾼 것. 단위 쿼터니언에서는 반대 회전이다."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def 벡터_돌리기(q: np.ndarray, v) -> np.ndarray:
    """q로 3차원 벡터 v를 돌린다: q · (0, v) · q의 켤레."""
    순허수 = np.concatenate([[0.0], np.asarray(v, dtype=float)])
    return 곱(곱(q, 순허수), 켤레(q))[1:]


def 사이각(q1: np.ndarray, q2: np.ndarray) -> float:
    """두 자세가 몇 도 차이 나는지. q와 -q는 같은 자세라서 내적의 절댓값을 쓴다."""
    내적 = abs(float(np.dot(q1, q2)))
    return float(np.rad2deg(2 * np.arccos(min(내적, 1.0))))


def 정규화(q: np.ndarray) -> np.ndarray:
    """길이를 1로 맞춘다."""
    return q / np.linalg.norm(q)
