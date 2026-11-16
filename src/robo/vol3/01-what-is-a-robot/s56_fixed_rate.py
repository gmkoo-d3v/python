import matplotlib.pyplot as plt
import numpy as np

# plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["font.family"] = "D2Coding"
plt.rcParams["axes.unicode_minus"] = False

주기 = 20                                   # b12_fixed_rate.py: 주기 = 0.02초 = 20ms (50Hz)
일 = [4, 6, 5, 28, 5, 6]                    # 박자마다 일하는 시간(ms). 설명용 값이다. 3번째만 주기보다 길다
박자_수 = len(일)

# 방법 1: 일하고 나서 sleep(주기)
시각, 시작1 = 0, []
for 박자 in range(박자_수):
    시작1.append(시각)
    시각 += 일[박자] + 주기                   # 일 + 주기만큼 쉼 → 박자마다 일하는 시간만큼 밀린다
끝1 = 시각

# 방법 2: b12_fixed_rate.py와 같은 계산 (다음_박자는 주기씩만 늘어난다)
시각, 다음_박자, 시작2, 쉼2, 놓침 = 0, 0, [], [], []
for 박자 in range(박자_수):
    시작2.append(시각)
    시각 += 일[박자]
    다음_박자 += 주기
    남은_시간 = 다음_박자 - 시각
    if 남은_시간 > 0:
        쉼2.append(남은_시간)
        시각 = 다음_박자                       # 그때까지만 쉰다
    else:
        쉼2.append(0)
        놓침.append((박자, 다음_박자))          # 일이 주기보다 오래 걸렸다
끝2 = 시각
print("방법 1 시작(ms):", 시작1, "끝", 끝1, f"→ 이상적인 {박자_수}박자 끝 {박자_수 * 주기}ms보다 {끝1 - 박자_수 * 주기}ms 늦음")
print("방법 2 시작(ms):", 시작2, "끝", 끝2, f"→ {박자_수 * 주기}ms에 맞음")
print("놓친 박자:", [f"{b}번째 박자를 놓쳤다 (박자 시각 {t}ms)" for b, t in 놓침])

그림 = plt.figure(figsize=(10, 6.2))
위 = 그림.add_axes([0.04, 0.55, 0.93, 0.30])
아래 = 그림.add_axes([0.04, 0.10, 0.93, 0.30])
for 칸 in (위, 아래):
    칸.set_xlim(-6, 205)
    칸.set_ylim(-1.5, 1.7)
    for 선 in ["left", "right", "top"]:
        칸.spines[선].set_visible(False)
    칸.set_yticks([])
    칸.set_xticks(range(0, 181, 40))
    칸.tick_params(labelsize=10)
    for k in range(10):                      # 이상적인 박자 격자: 칸 높이만큼만(글자를 지나지 않게)
        칸.plot([k * 주기, k * 주기], [0, 1.05], color="gray", alpha=0.5, lw=1, ls="--", zorder=2.5)
    칸.set_xlabel("시간 (ms)   점선 = 박자 (20ms마다)", fontsize=10)


def 칸_그리기(칸, 시작, 쉼, 색):
    for k in range(박자_수):
        칸.add_patch(plt.Rectangle((시작[k], 0), 일[k], 1, color=색, zorder=3))
        칸.add_patch(plt.Rectangle((시작[k] + 일[k], 0), 쉼[k], 1, color=색, alpha=0.2, zorder=2))
        칸.text(시작[k] + 1, 1.12, str(k), fontsize=10, ha="left", va="bottom", color=색)


칸_그리기(위, 시작1, [주기] * 박자_수, "tab:orange")
칸_그리기(아래, 시작2, 쉼2, "tab:blue")

# 위: 밀린 정도를 박자마다 표시
for k in range(박자_수):
    늦음 = 시작1[k] - k * 주기
    위.text(시작1[k] + 1, -0.12, f"+{늦음}", fontsize=10, color="tab:red", ha="left", va="top")
위.annotate("", xy=(끝1, -0.75), xytext=(박자_수 * 주기, -0.75),
          arrowprops=dict(arrowstyle="<->", color="tab:red", lw=1.5))
위.text(끝1 + 3, -0.75, f"{박자_수}박자 뒤\n{끝1 - 박자_수 * 주기}ms 늦음", fontsize=10, color="tab:red", va="center")
위.text(-6, 1.65, "일하고 나서 sleep(주기): 박자가 계속 밀린다", fontsize=12, va="top", color="tab:orange")

# 아래: 놓친 박자 표시
for 박자, 시각 in 놓침:
    아래.plot([시각, 시각], [-0.1, 1.05], color="tab:red", lw=2, zorder=5)
    아래.text(시각 + 3, -0.35, f"{박자}번째 박자를 놓쳤다\n(일이 {일[박자]}ms > 주기 {주기}ms)\n다음 박자는 쉬지 않고 바로 시작",
            fontsize=10, color="tab:red", va="top")
위.text(-6, -0.55, "붉은 숫자: 박자보다 늦은 시간 (ms)", fontsize=10, color="tab:red", va="top")
아래.text(-6, -0.35, "진한 칸 = 일\n연한 칸 = 쉼", fontsize=10, va="top")
아래.text(143, -0.35, "한 번 놓쳐도 그다음\n박자(100ms)부터 다시\n격자 위에서 시작", fontsize=10, va="top", color="tab:blue")
아래.text(-6, 1.65, "다음 박자의 시각까지만 쉰다: 박자가 밀리지 않는다", fontsize=12, va="top", color="tab:blue")

그림.text(0.5, 0.97, "제어 주기 0.02초 = 20ms (50Hz) 박자에 맞추는 두 가지 방법", ha="center", va="top", fontsize=12)
그림.savefig("s56_fixed_rate.png", dpi=200)
print("저장: s56_fixed_rate.png")
