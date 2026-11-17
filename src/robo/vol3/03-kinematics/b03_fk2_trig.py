import numpy as np

위팔, 아래팔 = 0.4, 0.3          # 링크 길이(m)


def 순기구학2(어깨: float, 팔꿈치: float) -> tuple[np.ndarray, np.ndarray]:
    """2링크 평면 팔. 관절 각도(rad) → (팔꿈치 위치, 손끝 위치)."""
    팔꿈치_위치 = 위팔 * np.array([np.cos(어깨), np.sin(어깨)])
    손끝_위치 = 팔꿈치_위치 + 아래팔 * np.array([np.cos(어깨 + 팔꿈치), np.sin(어깨 + 팔꿈치)])
    return 팔꿈치_위치, 손끝_위치


for 어깨_도, 팔꿈치_도 in [(0, 0), (90, 0), (30, 45), (0, 90), (0, 180)]:
    팔꿈치_위치, 손끝_위치 = 순기구학2(np.deg2rad(어깨_도), np.deg2rad(팔꿈치_도))
    print(f"어깨 {어깨_도:3d}도, 팔꿈치 {팔꿈치_도:3d}도 → "
          f"팔꿈치 {np.round(팔꿈치_위치, 3) + 0.0}, 손끝 {np.round(손끝_위치, 3) + 0.0}")
