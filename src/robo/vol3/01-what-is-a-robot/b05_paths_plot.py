import matplotlib.pyplot as plt
import numpy as np

from point_robot import 최대_속도, 움직이기, 위치_재기

# plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["font.family"] = "D2Coding"
plt.rcParams["axes.unicode_minus"] = False

시작, 목표 = np.array([0.0, 0.0]), np.array([2.0, 1.0])
주기, 이득, 미끄러짐, 센서_잡음 = 0.1, 1.5, 0.2, 0.02
생성기 = np.random.default_rng(1)

# 열린 고리: 정해 둔 명령만 보내며 지나간 자리를 모은다
방향 = (목표 - 시작) / np.linalg.norm(목표 - 시작)
열린_경로 = [시작]
for _ in range(45):
    열린_경로.append(움직이기(열린_경로[-1], 방향 * 최대_속도, 주기, 생성기, 미끄러짐))

# 닫힌 고리: 매번 재고 고치며 지나간 자리와 잰 값을 모은다
닫힌_경로, 잰_값들 = [시작], []
for _ in range(80):
    잰_값들.append(위치_재기(닫힌_경로[-1], 생성기, 센서_잡음))
    닫힌_경로.append(움직이기(닫힌_경로[-1], 이득 * (목표 - 잰_값들[-1]), 주기, 생성기, 미끄러짐))

열린_경로, 닫힌_경로, 잰_값들 = np.array(열린_경로), np.array(닫힌_경로), np.array(잰_값들)
그림, 칸 = plt.subplots(figsize=(7, 4))
칸.plot(*zip(시작, 목표), color="lightgray", linestyle="--", label="곧은 길")
칸.plot(열린_경로[:, 0], 열린_경로[:, 1], "o-", markersize=2.5, color="tab:red", label="열린 고리")
칸.plot(닫힌_경로[:, 0], 닫힌_경로[:, 1], "o-", markersize=2.5, color="tab:blue", label="닫힌 고리")
칸.scatter(잰_값들[:, 0], 잰_값들[:, 1], s=6, color="tab:cyan", alpha=0.6, label="닫힌 고리가 잰 위치")
칸.scatter(*목표, marker="*", s=250, color="gold", edgecolors="black", zorder=5, label="목표")
칸.set_xlabel("x (m)")
칸.set_ylabel("y (m)")
칸.set_aspect("equal")
칸.legend(loc="upper left", fontsize=8)
그림.tight_layout()
그림.savefig("open_closed.png", dpi=200)
print("열린 고리 끝:", np.round(열린_경로[-1], 3), "/ 닫힌 고리 끝:", np.round(닫힌_경로[-1], 3))
print("저장: open_closed.png")
