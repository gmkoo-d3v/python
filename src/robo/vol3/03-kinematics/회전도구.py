"""3차원 기본 회전 행렬과 출력 정리 도구."""
import numpy as np


def Rx(각도_도: float) -> np.ndarray:
    """x축을 중심으로 돌리는 회전 행렬."""
    c, s = np.cos(np.deg2rad(각도_도)), np.sin(np.deg2rad(각도_도))
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])


def Ry(각도_도: float) -> np.ndarray:
    """y축을 중심으로 돌리는 회전 행렬."""
    c, s = np.cos(np.deg2rad(각도_도)), np.sin(np.deg2rad(각도_도))
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def Rz(각도_도: float) -> np.ndarray:
    """z축을 중심으로 돌리는 회전 행렬."""
    c, s = np.cos(np.deg2rad(각도_도)), np.sin(np.deg2rad(각도_도))
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])


def 깔끔(배열: np.ndarray, 자리: int = 3) -> np.ndarray:
    """출력용: 반올림하고 -0.0을 0.0으로 바꾼다."""
    return np.round(배열, 자리) + 0.0
