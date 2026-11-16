import numpy as np

from point_robot import 움직이기, 위치_재기


class 고리로봇:
    """감지·판단·행동을 메서드 셋으로 나눈 점 로봇."""

    def __init__(self, 목표: np.ndarray, 생성기: np.random.Generator, 주기: float = 0.1):
        self.위치 = np.array([0.0, 0.0])        # 진짜 위치(로봇 자신은 모른다)
        self.목표, self.생성기, self.주기 = 목표, 생성기, 주기
        self.박자 = 0

    def 감지(self) -> np.ndarray:
        return 위치_재기(self.위치, self.생성기, 잡음=0.02)

    def 판단(self, 잰_위치: np.ndarray) -> np.ndarray:
        return 1.5 * (self.목표 - 잰_위치)

    def 행동(self, 속도_명령: np.ndarray) -> None:
        self.위치 = 움직이기(self.위치, 속도_명령, self.주기, self.생성기, 미끄러짐=0.2)

    def 한_박자(self) -> float:
        잰_위치 = self.감지()
        self.행동(self.판단(잰_위치))
        self.박자 += 1
        return float(np.linalg.norm(self.목표 - 잰_위치))   # 로봇이 '생각하는' 남은 거리


로봇 = 고리로봇(np.array([2.0, 1.0]), np.random.default_rng(0))
while 로봇.한_박자() > 0.05:          # 잰 값으로 5cm 안에 들어올 때까지
    pass
print(f"{로봇.박자}박자({로봇.박자 * 로봇.주기:.1f}초) 만에 멈춤")
print(f"진짜 위치 {np.round(로봇.위치, 3)}, 진짜 남은 거리 {np.linalg.norm(로봇.목표 - 로봇.위치):.3f}m")
