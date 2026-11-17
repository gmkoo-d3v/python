import numpy as np

from 회전도구 import Rz, 깔끔

위팔, 아래팔 = 0.4, 0.3


def 변환(R: np.ndarray, p) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def 이동(p) -> np.ndarray:
    return 변환(np.eye(3), p)


어깨_도, 팔꿈치_도 = 30, 45

# 관절마다 "돌리고, 링크 길이만큼 x축으로 나아가기"를 곱해 나간다
T_바닥_위팔 = 변환(Rz(어깨_도), [0, 0, 0])                 # 어깨 관절이 돌린 위팔 좌표계
T_위팔_아래팔 = 이동([위팔, 0, 0]) @ 변환(Rz(팔꿈치_도), [0, 0, 0])
T_아래팔_손끝 = 이동([아래팔, 0, 0])
T_바닥_손끝 = T_바닥_위팔 @ T_위팔_아래팔 @ T_아래팔_손끝

print("T_바닥_손끝 =\n", 깔끔(T_바닥_손끝))
a, ab = np.deg2rad(어깨_도), np.deg2rad(어깨_도 + 팔꿈치_도)
삼각함수_답 = [위팔 * np.cos(a) + 아래팔 * np.cos(ab), 위팔 * np.sin(a) + 아래팔 * np.sin(ab)]
print("삼각 함수로 구한 손끝:", 깔끔(np.array(삼각함수_답)))
x축 = T_바닥_손끝[:3, 0]
print("손끝이 향한 각도:", round(float(np.rad2deg(np.arctan2(x축[1], x축[0]))), 3), "도")
