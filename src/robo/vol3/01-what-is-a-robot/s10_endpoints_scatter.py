import matplotlib.pyplot as plt
import numpy as np

# plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["font.family"] = "D2Coding"
plt.rcParams["axes.unicode_minus"] = False

# point_robot.py 의 함수를 그대로 복사 (이름·매개변수·난수 사용 순서 동일)
최대_속도 = 0.5


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


# b04_slip_noise.py 의 설정과 두 고리
시작, 목표 = np.array([0.0, 0.0]), np.array([2.0, 1.0])
주기, 이득, 센서_잡음 = 0.1, 1.5, 0.02


def 열린_고리(생성기, 미끄러짐):
    방향 = (목표 - 시작) / np.linalg.norm(목표 - 시작)
    위치 = 시작.copy()
    for _ in range(round(np.linalg.norm(목표 - 시작) / (최대_속도 * 주기))):
        위치 = 움직이기(위치, 방향 * 최대_속도, 주기, 생성기, 미끄러짐)
    return 위치


def 닫힌_고리(생성기, 미끄러짐):
    위치 = 시작.copy()
    for _ in range(80):
        잰_위치 = 위치_재기(위치, 생성기, 센서_잡음)
        위치 = 움직이기(위치, 이득 * (목표 - 잰_위치), 주기, 생성기, 미끄러짐)
    return 위치


def 끝점_모으기(미끄러짐):          # b04 와 같이 seed 0, 열린 고리 20번 → 닫힌 고리 20번
    생성기 = np.random.default_rng(0)
    return [np.array([방법(생성기, 미끄러짐) - 목표 for _ in range(20)]) for 방법 in (열린_고리, 닫힌_고리)]


그림 = plt.figure(figsize=(10, 6.6))
격자 = 그림.add_gridspec(2, 2, width_ratios=[1.7, 1])
for 줄, 미끄러짐 in enumerate([0.2, 0.4]):
    열린, 닫힌 = 끝점_모으기(미끄러짐)
    열린_오차, 닫힌_오차 = np.linalg.norm(열린, axis=1), np.linalg.norm(닫힌, axis=1)
    print(f"슬립 {미끄러짐}: 열린 고리 평균 {열린_오차.mean():.3f}m (최대 {열린_오차.max():.3f}m), "
          f"닫힌 고리 평균 {닫힌_오차.mean():.3f}m (최대 {닫힌_오차.max():.3f}m)")
    중심 = 열린.mean(axis=0)            # 빨간 점들의 한가운데

    # 왼쪽: 실제 크기로 두 고리를 한 그림에서 비교 (두 줄 모두 같은 범위)
    칸 = 그림.add_subplot(격자[줄, 0])
    칸.scatter(*열린.T, s=28, color="tab:red", alpha=0.75, zorder=3)
    칸.scatter(*닫힌.T, s=14, color="tab:blue", zorder=4)
    칸.scatter(0, 0, marker="*", s=380, color="gold", edgecolor="black", zorder=5)
    칸.annotate("", xy=중심, xytext=(0, 0), arrowprops=dict(arrowstyle="<->", color="tab:red", linewidth=1.5,
                                                         shrinkA=9, shrinkB=0))
    칸.text(중심[0] * 0.45 + 0.06, 중심[1] * 0.45 - 0.05, f"평균 {열린_오차.mean():.3f} m\n빗나감", color="tab:red", fontsize=11, ha="right", va="top")
    칸.text(중심[0], 열린[:, 1].max() + 0.03, "명령보다 덜 가고\n왼쪽으로 샌다", color="tab:red", fontsize=10, ha="center", va="bottom")
    칸.annotate("★ 목표 = 닫힌 고리 20번\n(별에 겹침, 오른쪽 확대)", xy=(0.01, 0.03), xytext=(0.43, 0.2), color="tab:blue", fontsize=10, va="bottom", ha="right",
             arrowprops=dict(arrowstyle="-", color="tab:blue"))
    칸.set(xlim=(-0.8, 0.5), ylim=(-0.12, 0.5))
    칸.set_xlabel("목표에서 x 방향 어긋남 (m)")
    칸.set_ylabel("y 방향 어긋남 (m)")
    칸.set_aspect("equal")
    칸.grid(alpha=0.3)
    칸.set_title("열린 고리(빨강) vs 닫힌 고리(파랑): 20번의 끝점", fontsize=11, pad=8)

    # 오른쪽: 닫힌 고리만 100배 확대
    칸 = 그림.add_subplot(격자[줄, 1])
    칸.add_patch(plt.Circle((0, 0), 닫힌_오차.mean(), fill=False, color="tab:blue", linestyle="--", linewidth=1.5))
    칸.scatter(*닫힌.T, s=28, color="tab:blue", alpha=0.75, zorder=3)
    칸.scatter(0, 0, marker="*", s=380, color="gold", edgecolor="black", zorder=5)
    칸.text(0, 0.026, f"점선 원: 평균 오차 {닫힌_오차.mean():.3f} m", ha="center", va="center", fontsize=10, color="tab:blue")
    칸.text(-0.028, -0.028, "★ = 목표", fontsize=10, va="bottom")
    칸.set(xlim=(-0.03, 0.03), ylim=(-0.03, 0.03))
    칸.set_xticks(np.arange(-0.02, 0.021, 0.01))
    칸.set_yticks(칸.get_xticks())
    칸.tick_params(labelsize=8)
    칸.set_xlabel("x 방향 어긋남 (m)")
    칸.set_ylabel("y 방향 어긋남 (m)")
    칸.set_aspect("equal")
    칸.grid(alpha=0.3)
    칸.set_title("닫힌 고리만 확대 (눈금 1cm)", fontsize=11, pad=8)

그림.subplots_adjust(left=0.08, right=0.98, bottom=0.09, top=0.88, wspace=0.25, hspace=0.5)
for 줄, 미끄러짐 in enumerate([0.2, 0.4]):
    그림.text(0.5, 그림.axes[줄 * 2].get_position().y1 + 0.05, f"미끄러짐 {미끄러짐}  (센서 잡음 0.02 m)", ha="center", fontsize=13)
그림.savefig("s10_endpoints_scatter.png", dpi=200)
print("저장: s10_endpoints_scatter.png")
