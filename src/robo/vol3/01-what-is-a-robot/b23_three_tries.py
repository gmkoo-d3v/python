import argparse
import math

import avoidance_tools
from avoidance_tools import 결과, 달리기

parser = argparse.ArgumentParser(description="장애물 위치를 바꾸며 회피 규칙 비교")
parser.add_argument("--obstacle-y", type=float, default=0.5, help="장애물 중심 y 좌표 (m)")
args = parser.parse_args()
if not math.isfinite(args.obstacle_y):
    parser.error("obstacle-y는 유한한 수여야 합니다")
avoidance_tools.장애물_중심[1] = args.obstacle_y

시도들 = [("1. 목표만 보고 간다", False, False),
        ("2. 장애물에서 밀려난다", True, False),
        ("3. 밀려나면서 옆으로 돈다", True, True)]
for 이름, 밀어내기, 옆으로 in 시도들:
    print(이름)
    print("  ", 결과(달리기(밀어내기, 옆으로)))
