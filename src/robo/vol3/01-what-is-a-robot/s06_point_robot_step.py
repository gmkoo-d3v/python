import matplotlib.pyplot as plt
import numpy as np

# plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["font.family"] = "D2Coding"
plt.rcParams["axes.unicode_minus"] = False

# point_robot.py 에서 그대로 가져온 부분 (이름과 매개변수 동일)
최대_속도 = 0.5        # m/s


def 속도_제한(속도, 최대=최대_속도):
    크기 = np.linalg.norm(속도)
    return 속도 * (최대 / 크기) if 크기 > 최대 else 속도


def 움직이기(위치, 속도_명령, 주기, 생성기=None, 미끄러짐=0.0):
    이동 = 속도_제한(속도_명령) * 주기
    if 생성기 is not None and 미끄러짐 > 0:
        덜_감 = 생성기.uniform(1 - 미끄러짐, 1.0)
        옆_비율 = 생성기.normal(미끄러짐 / 2, 미끄러짐 / 2)
        왼쪽 = np.array([-이동[1], 이동[0]])
        이동 = 덜_감 * 이동 + 옆_비율 * 왼쪽
    return 위치 + 이동


def 위치_재기(위치, 생성기=None, 잡음=0.0):
    if 생성기 is None or 잡음 == 0:
        return 위치.copy()
    return 위치 + 생성기.normal(0, 잡음, size=2)


# 한 주기: b02 와 같은 방향(목표 [2, 1]), 최대 속도, 주기 0.1초, 미끄러짐 0.2, 센서 잡음 0.02 m
시작, 목표 = np.array([0.0, 0.0]), np.array([2.0, 1.0])
주기, 미끄러짐, 센서_잡음 = 0.1, 0.2, 0.02
속도_명령 = (목표 - 시작) / np.linalg.norm(목표 - 시작) * 최대_속도
생성기 = np.random.default_rng(1)

명령_끝 = 움직이기(시작, 속도_명령, 주기)                       # 미끄러짐 없는 계획
진짜_끝 = 움직이기(시작, 속도_명령, 주기, 생성기, 미끄러짐)     # 이번에 실제로 간 곳
반복 = np.array([움직이기(시작, 속도_명령, 주기, 생성기, 미끄러짐) for _ in range(400)])
잰_점 = np.array([위치_재기(진짜_끝, 생성기, 센서_잡음) for _ in range(6)])
print(f"명령 이동 {(명령_끝 - 시작).round(4)} (길이 {np.linalg.norm(명령_끝 - 시작):.4f} m)")
print(f"실제 이동 {(진짜_끝 - 시작).round(4)} (길이 {np.linalg.norm(진짜_끝 - 시작):.4f} m, "
      f"명령의 {np.linalg.norm(진짜_끝 - 시작) / np.linalg.norm(명령_끝 - 시작):.0%})")
print(f"센서 잡음 표준편차 {센서_잡음} m, 잰 위치 6개 중 첫 값 {잰_점[0].round(4)}")

그림, 칸 = plt.subplots(figsize=(10, 5.6))
칸.scatter(*반복.T, s=10, color="tab:blue", alpha=0.18, edgecolor="none", zorder=2)
칸.add_patch(plt.Circle(진짜_끝, 센서_잡음, fill=False, color="tab:cyan", linestyle="--", linewidth=1.5, zorder=3))
칸.annotate("", xy=명령_끝, xytext=시작, zorder=4,
          arrowprops=dict(arrowstyle="-|>", color="gray", linestyle="--", linewidth=2, shrinkA=0, shrinkB=0))
칸.annotate("", xy=진짜_끝, xytext=시작, zorder=5,
          arrowprops=dict(arrowstyle="-|>", color="tab:blue", linewidth=2.5, shrinkA=0, shrinkB=0))
칸.scatter(*잰_점.T, s=45, color="lightskyblue", edgecolor="tab:cyan", zorder=6)
칸.scatter(*시작, s=60, color="black", zorder=7)
칸.scatter(*명령_끝, s=50, facecolor="white", edgecolor="gray", zorder=7)
칸.scatter(*진짜_끝, s=70, color="tab:blue", zorder=7)

칸.text(시작[0] - 0.004, 시작[1] - 0.005, "출발 위치", fontsize=11, ha="right", va="top")
칸.annotate("계획: 명령 × 주기\n(가려던 곳)", xy=명령_끝, xytext=(0.07, 0.016), fontsize=11, color="dimgray",
          va="center", arrowprops=dict(arrowstyle="-", color="gray"))
칸.annotate("진짜 위치\n로봇 자신은 모른다", xy=진짜_끝, xytext=(0.036, 0.066), fontsize=11, color="tab:blue",
          ha="center", va="center", arrowprops=dict(arrowstyle="-", color="tab:blue"))
가장_먼 = 잰_점[np.argmax(잰_점[:, 0])]
칸.annotate("잰 위치(센서 값):\n같은 자리에서 6번 잰 것\n로봇은 이것만 안다", xy=가장_먼, xytext=(0.07, 0.05), fontsize=11,
          color="tab:cyan", va="center", arrowprops=dict(arrowstyle="-|>", color="tab:cyan"))
칸.annotate("같은 명령을 400번 반복했을 때\n끝나는 곳: 조금 덜 가고\n왼쪽으로 샌다", xy=(0.04, 0.012), xytext=(0.06, -0.012),
          fontsize=10, color="tab:blue", va="center", arrowprops=dict(arrowstyle="-", color="tab:blue", alpha=0.5))
칸.text(진짜_끝[0], 진짜_끝[1] - 센서_잡음 - 0.0035, "센서 잡음 크기\n(표준편차 0.02 m)", fontsize=10,
        color="tab:cyan", ha="center", va="top")

칸.set_xlim(-0.04, 0.125)
칸.set_ylim(-0.04, 0.075)
칸.set_aspect("equal")
칸.set_xticks(np.arange(-0.02, 0.121, 0.02))
칸.set_yticks(np.arange(-0.04, 0.071, 0.02))
칸.set_xlabel("x (m)")
칸.set_ylabel("y (m)")
칸.grid(alpha=0.3)
그림.subplots_adjust(left=0.09, right=0.98, bottom=0.11, top=0.84)
그림.text(0.5, 0.95, "한 주기(0.1초) 동안 일어나는 일: 명령한 이동 ≠ 실제 이동 ≠ 잰 위치", ha="center", fontsize=12)
그림.text(0.5, 0.89, f"명령 이동 {(명령_끝 - 시작).round(3).tolist()} m   →   실제 이동 {(진짜_끝 - 시작).round(3).tolist()} m"
        f"   (최대 속도 {최대_속도} m/s, 미끄러짐 {미끄러짐}, 센서 잡음 {센서_잡음} m)", ha="center", fontsize=10)
그림.savefig("s06_point_robot_step.png", dpi=200)
print("저장: s06_point_robot_step.png")
