"""1장의 점 로봇: 평면 위의 점이 속도 명령을 받아 움직인다. 위치 단위는 m, 시간은 초."""
import numpy as np

최대_속도 = 0.5        # m/s. 모터가 낼 수 있는 가장 빠른 속도


def 속도_제한(속도: np.ndarray, 최대: float = 최대_속도) -> np.ndarray:
    """속도 벡터의 크기가 최대를 넘으면 방향은 두고 크기만 줄인다."""
    크기 = np.linalg.norm(속도)
    if 크기 > 최대:
        return 속도 * (최대 / 크기)
    return 속도


def 움직이기(위치: np.ndarray, 속도_명령: np.ndarray, 주기: float,
          생성기: np.random.Generator | None = None, 미끄러짐: float = 0.0) -> np.ndarray:
    """속도 명령을 한 주기 동안 따른 뒤의 '진짜' 위치(행동).

    미끄러짐이 있으면 바퀴가 헛돌아 명령보다 덜 가고, 진행 방향의 왼쪽으로 조금씩 샌다.
    """
    이동 = 속도_제한(속도_명령) * 주기
    if 생성기 is not None and 미끄러짐 > 0:
        덜_감 = 생성기.uniform(1 - 미끄러짐, 1.0)             # 명령의 80~100%만 간다(미끄러짐 0.2일 때)
        옆_비율 = 생성기.normal(미끄러짐 / 2, 미끄러짐 / 2)    # 평균적으로 왼쪽으로 샌다 한쪽으로 치우치는것을 말함
        왼쪽 = np.array([-이동[1], 이동[0]])                 # 진행 방향을 왼쪽으로 직각만큼 튼 벡터
        이동 = 덜_감 * 이동 + 옆_비율 * 왼쪽
    return 위치 + 이동


def 위치_재기(위치: np.ndarray, 생성기: np.random.Generator | None = None,
           잡음: float = 0.0) -> np.ndarray:
    """센서로 잰 위치(감지). 잡음이 있으면 잴 때마다 조금씩 틀린 값이 나온다."""
    if 생성기 is None or 잡음 == 0:
        return 위치.copy()
    return 위치 + 생성기.normal(0, 잡음, size=2)
