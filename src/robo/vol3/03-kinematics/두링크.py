"""2링크 평면 팔: 3.2절 순기구학, 3.4절 역기구학에 야코비안을 더해 모은 것. 각도는 rad."""
import numpy as np

위팔, 아래팔 = 0.4, 0.3          # 링크 길이(m)


def 팔_점들(q) -> np.ndarray:
    """관절 각도 q = [어깨, 팔꿈치] → 바닥, 팔꿈치, 손끝 세 점(3×2)."""
    팔꿈치_위치 = 위팔 * np.array([np.cos(q[0]), np.sin(q[0])])
    손끝_위치 = 팔꿈치_위치 + 아래팔 * np.array([np.cos(q[0] + q[1]), np.sin(q[0] + q[1])])
    return np.array([[0.0, 0.0], 팔꿈치_위치, 손끝_위치])


def 순기구학2(q) -> np.ndarray:
    """관절 각도 → 손끝 위치."""
    return 팔_점들(q)[2]


def 역기구학2(목표) -> list[np.ndarray]:
    """손끝 목표 → 가능한 관절 각도 목록(0~2개). 3.4절의 해석해."""
    x, y = 목표
    c2 = (x**2 + y**2 - 위팔**2 - 아래팔**2) / (2 * 위팔 * 아래팔)
    if abs(c2) > 1:
        return []
    각 = np.arccos(c2)
    답 = []
    for 팔꿈치 in ([각] if np.isclose(abs(c2), 1) else [-각, 각]):
        어깨 = np.arctan2(y, x) - np.arctan2(아래팔 * np.sin(팔꿈치), 위팔 + 아래팔 * np.cos(팔꿈치))
        답.append(np.array([(어깨 + np.pi) % (2 * np.pi) - np.pi, 팔꿈치]))
    return 답


def 야코비안2(q) -> np.ndarray:
    """2×2 야코비안. 열 i는 '관절 i만 1rad/s로 돌 때의 손끝 속도'."""
    s1, c1 = np.sin(q[0]), np.cos(q[0])
    s12, c12 = np.sin(q[0] + q[1]), np.cos(q[0] + q[1])
    return np.array([[-위팔 * s1 - 아래팔 * s12, -아래팔 * s12],
                     [위팔 * c1 + 아래팔 * c12, 아래팔 * c12]])
