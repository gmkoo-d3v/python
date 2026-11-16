import matplotlib.pyplot as plt
import numpy as np

# plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["font.family"] = "D2Coding"
plt.rcParams["axes.unicode_minus"] = False

# b02_open_loop.py, b03_closed_loop.py, point_robot.py와 같은 설정 (완벽한 세상: 미끄러짐·잡음 없음)
최대_속도 = 0.5
시작, 목표 = np.array([0.0, 0.0]), np.array([2.0, 1.0])
주기, 이득 = 0.1, 1.5


def 속도_제한(속도, 최대=최대_속도):
    크기 = np.linalg.norm(속도)
    return 속도 * (최대 / 크기) if 크기 > 최대 else 속도


def 움직이기(위치, 속도_명령, 주기):
    return 위치 + 속도_제한(속도_명령) * 주기


거리 = np.linalg.norm(목표 - 시작)
방향 = (목표 - 시작) / 거리
명령_횟수 = round(거리 / (최대_속도 * 주기))

# 열린 고리: 정해 둔 명령만 45번
위치 = 시작.copy()
열린 = [0.0]                                   # 시작에서 간 거리(경로 위)
for _ in range(명령_횟수):
    위치 = 움직이기(위치, 방향 * 최대_속도, 주기)
    열린.append(float(np.linalg.norm(위치 - 시작)))
열린_끝 = 위치.copy()
# 닫힌 고리: 매 주기 남은 거리를 재고 속도를 정한다
위치 = 시작.copy()
닫힌, 잘림 = [0.0], [True]
for _ in range(80):
    속도_명령 = 이득 * (목표 - 위치)
    잘림.append(bool(np.linalg.norm(속도_명령) > 최대_속도))   # 이 걸음에서 최대 속도로 잘렸나
    위치 = 움직이기(위치, 속도_명령, 주기)
    닫힌.append(float(np.linalg.norm(위치 - 시작)))
열린, 닫힌, 잘림 = np.array(열린), np.array(닫힌), np.array(잘림)   # 잘림[k]: k번째 점에 이른 걸음이 잘렸나
무릎 = 최대_속도 / 이득
잘린_걸음 = int(np.sum(잘림[1:]))
print(f"열린 고리: {명령_횟수}번, 도착 {np.round(열린_끝, 3)}, 남은 거리 {np.linalg.norm(목표 - 열린_끝):.3f}m (지나침)")
print(f"닫힌 고리: 80걸음, 도착 {np.round(위치, 3)}, 남은 거리 {np.linalg.norm(목표 - 위치):.3f}m, 최대 속도로 잘린 걸음 {잘린_걸음}")
print(f"꺾이는 점: 남은 거리 {무릎:.3f}m (= 최대 속도 / 이득), 최대 속도 {최대_속도}m/s")
for 초 in [1, 2, 3, 4, 5]:
    print(f"  {초}초: 남은 거리 {거리 - 닫힌[초 * 10]:.3f}m")

그림 = plt.figure(figsize=(11, 6))
파랑, 주황, 회색 = "tab:blue", "tab:orange", "gray"
점끝 = 거리 + 0.3


def 줄_만들기(칸, 점, 색, 색2=None, 아래=-0.7, 위=1.1, 선끝=0.95):
    칸.set_xlim(-0.1, 2.5)
    칸.set_ylim(아래, 위)
    칸.axhline(0, color=회색, lw=1, zorder=1)
    칸.scatter(점, np.zeros_like(점), s=22, color=색 if 색2 is None else np.where(색2, 색, "tab:green"), zorder=3)
    칸.scatter([0], [0], s=70, color="black", zorder=4)
    칸.axvline(거리, ymin=(-0.05 - 아래) / (위 - 아래), ymax=(선끝 - 아래) / (위 - 아래), color="tab:red", ls="--", lw=1.2, zorder=2)
    칸.text(0, -0.12, "시작", ha="center", va="top", fontsize=11)
    칸.text(거리, -0.12, "목표", ha="center", va="top", fontsize=11, color="tab:red")
    칸.set_yticks([])
    칸.set_xticks([0, 0.5, 1, 1.5, 2])
    for 변 in ["top", "right", "left"]:
        칸.spines[변].set_visible(False)


# 1) 열린 고리
칸1 = 그림.add_axes([0.04, 0.56, 0.60, 0.30])
줄_만들기(칸1, 열린, 파랑, 아래=-0.7, 위=1.5, 선끝=0.35)
그림.text(0.04, 0.965, "열린 고리: 같은 명령을 45번 (간격이 모두 같다)", fontsize=12, fontweight="bold", va="top")
칸1.text(0.0, 0.3, f"점 하나 = 한 걸음 (0.05m), 모두 {명령_횟수}개", ha="left", fontsize=10, color=파랑)
칸1.axvspan(2.14, 2.30, ymin=0.1, ymax=0.4, color="lightgray", alpha=0.6, zorder=0)
확대 = 칸1.inset_axes([0.50, 0.55, 0.48, 0.43])
확대.set_xlim(2.14, 2.30)
확대.set_ylim(-1.5, 0.9)
확대.axhline(0, color=회색, lw=1)
확대.scatter(열린[-4:], np.zeros(4), s=40, color=파랑, zorder=3)
확대.axvline(거리, color="tab:red", ls="--", lw=1.2)
확대.set_yticks([])
확대.set_xticks([2.2, 2.3])
확대.tick_params(labelsize=10)
확대.set_title("회색 구간 확대", fontsize=10, pad=3)
확대.text(거리 - 0.004, 0.2, "목표", ha="right", va="bottom", fontsize=10, color="tab:red")
확대.text(2.30, -0.2, f"멈춘 곳:{chr(10)}{열린[-1] - 거리:.3f}m 지나침", ha="right", va="top", fontsize=10, color=파랑)
for 변 in ["top", "right", "left"]:
    확대.spines[변].set_visible(False)

# 2) 닫힌 고리
칸2 = 그림.add_axes([0.04, 0.12, 0.60, 0.26])
줄_만들기(칸2, 닫힌, 주황, 잘림, 아래=-0.9, 위=1.1)
칸2.set_xlabel("시작에서 간 거리 (m)  (위아래 두 줄 모두)", fontsize=10)
그림.text(0.04, 0.455, "닫힌 고리: 매번 남은 거리를 재고 속도를 정한다", fontsize=12, fontweight="bold", va="top")
꺾임_x = 거리 - 무릎
칸2.axvline(꺾임_x, ymin=(-0.05 + 0.9) / 2.0, color="tab:purple", ls=":", lw=1.5, zorder=2)
칸2.text(꺾임_x - 0.04, 1.05, f"남은 거리 {무릎:.2f}m: 여기부터 간격이 줄어든다", ha="right", va="top", fontsize=10, color="tab:purple")
칸2.text(0.0, 0.3, "주황 점: 최대 속도로 잘림 (간격 같음)", ha="left", fontsize=10, color=주황)
칸2.text(2.45, -0.5, "초록 점: 속도가 거리에 비례해 줄어듦", ha="right", va="top", fontsize=10, color="tab:green")

# 3) 속도 규칙
칸3 = 그림.add_axes([0.76, 0.36, 0.22, 0.48])
남은 = np.linspace(0, 1.0, 200)
칸3.plot(남은, np.minimum(이득 * 남은, 최대_속도), color=주황, lw=2.5)
칸3.plot([0, 1.0], [최대_속도] * 2, color=파랑, lw=2, ls="--")
칸3.scatter([무릎], [최대_속도], s=60, color="tab:purple", zorder=5)
칸3.set_xlim(0, 1.0)
칸3.set_ylim(0, 0.72)
칸3.set_xticks([0, 0.333, 0.5, 1.0])
칸3.set_xticklabels(["0", "0.333", "0.5", "1"], fontsize=10)
칸3.set_yticks([0, 0.25, 0.5])
칸3.tick_params(labelsize=10)
칸3.grid(alpha=0.3)
칸3.set_xlabel("남은 거리 (m)", fontsize=11)
칸3.set_ylabel("속도 명령 (m/s)", fontsize=11)
칸3.text(0.02, 0.55, "열린 고리: 속도는 늘 0.5,\n거리와 무관", ha="left", va="bottom", fontsize=10, color=파랑)
칸3.text(무릎 + 0.04, 0.44, f"꺾이는 점\n거리 {무릎:.3f}m", ha="left", va="top", fontsize=10, color="tab:purple")
칸3.text(0.14, 0.04, "기울기 = 이득 1.5", ha="left", fontsize=10, color=주황)
그림.text(0.68, 0.965, "속도 규칙: 거리를 보고 속도를\n바꾸는 것은 닫힌 고리뿐", fontsize=12, fontweight="bold", va="top")
그림.text(0.68, 0.17, "닫힌 고리: 속도 =\nmin(이득 × 거리, 최대 속도)\n이득 1.5, 최대 속도 0.5\n가까워질수록 속도가 줄어\n멈출 때 거의 지나치지 않는다", fontsize=10, color=주황, va="center")

그림.savefig("s07_open_closed_steps.png", dpi=200)
print("저장: s07_open_closed_steps.png")
