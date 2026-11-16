import argparse  # 명령어 단에서 조절 해준다.

import numpy as np

from point_robot import 최대_속도, 움직이기, 위치_재기

시작, 목표 = np.array([0.0, 0.0]), np.array([2.0, 1.0])
주기, 이득 = 0.1, 1.5
parser = argparse.ArgumentParser(description="열린 고리와 닫힌 고리의 오차 비교")
parser.add_argument("--slip", type=float, default=0.2, help="미끄러짐 비율 (0~1)")
parser.add_argument("--noise", type=float, default=0.02, help="위치 센서 잡음 표준편차 (m)")
args = parser.parse_args()
if not 0 <= args.slip <= 1 or not np.isfinite(args.noise) or args.noise < 0:
    parser.error("slip은 0~1, noise는 0 이상의 유한한 수여야 합니다")
미끄러짐, 센서_잡음 = args.slip, args.noise


def 열린_고리(생성기: np.random.Generator) -> np.ndarray:
    방향 = (목표 - 시작) / np.linalg.norm(목표 - 시작)
    위치 = 시작.copy()
    for _ in range(round(np.linalg.norm(목표 - 시작) / (최대_속도 * 주기))):
        위치 = 움직이기(위치, 방향 * 최대_속도, 주기, 생성기, 미끄러짐)
    return 위치


def 닫힌_고리(생성기: np.random.Generator) -> np.ndarray:
    위치 = 시작.copy()
    for _ in range(80):
        잰_위치 = 위치_재기(위치, 생성기, 센서_잡음)
        위치 = 움직이기(위치, 이득 * (목표 - 잰_위치), 주기, 생성기, 미끄러짐)
    return 위치


생성기 = np.random.default_rng(0)
for 이름, 방법 in [("열린 고리", 열린_고리), ("닫힌 고리", 닫힌_고리)]:
    오차 = np.array([np.linalg.norm(목표 - 방법(생성기)) for _ in range(20)])   # 20번 반복
    print(f"{이름}: 목표까지 남은 거리 평균 {오차.mean():.3f}m, 가장 클 때 {오차.max():.3f}m")
