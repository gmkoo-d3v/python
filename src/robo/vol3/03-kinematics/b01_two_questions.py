import numpy as np

위팔, 아래팔 = 0.4, 0.3          # 링크 길이(m)


def 손끝(어깨: float, 팔꿈치: float) -> np.ndarray:
    """관절 각도(rad) → 손끝 위치. 이 공식은 3.2절에서 만든다."""
    return np.array([위팔 * np.cos(어깨) + 아래팔 * np.cos(어깨 + 팔꿈치),
                     위팔 * np.sin(어깨) + 아래팔 * np.sin(어깨 + 팔꿈치)])


# 질문 1: 관절을 이만큼 돌리면 손끝은 어디? → 계산 한 번
print("어깨 30도, 팔꿈치 45도 → 손끝", np.round(손끝(np.deg2rad(30), np.deg2rad(45)), 3))

# 질문 2: 손끝을 (0.5, 0.3)에 두려면? → 1도 간격으로 모든 조합을 다 넣어 보기
목표 = np.array([0.5, 0.3])
각도들 = np.deg2rad(np.arange(-180, 180))
답 = [(a, b) for a in 각도들 for b in 각도들 if np.linalg.norm(손끝(a, b) - 목표) < 0.005]
print("시험한 조합:", len(각도들) ** 2, "개 / 5mm 안에 든 조합:", len(답), "개")
for 이름, 조건 in [("팔꿈치 +", lambda b: b > 0), ("팔꿈치 -", lambda b: b < 0)]:
    무리 = np.rad2deg([(a, b) for a, b in 답 if 조건(b)])
    print(f"{이름} 쪽 답 {len(무리)}개, 평균: 어깨 {무리[:, 0].mean():.1f}도, 팔꿈치 {무리[:, 1].mean():.1f}도")
