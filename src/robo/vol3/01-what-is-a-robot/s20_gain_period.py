import matplotlib.pyplot as plt
import numpy as np

# plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["font.family"] = "D2Coding"
plt.rcParams["axes.unicode_minus"] = False

목표 = 1.0                                   # b13_period.py와 같은 1차원 비례 제어
끝_시각 = 4.0


def 기록_만들기(이득, 주기, 걸음수=None):
    걸음수 = round(끝_시각 / 주기) if 걸음수 is None else 걸음수
    x, 기록 = 0.0, [0.0]
    for _ in range(걸음수):
        x = x + 이득 * (목표 - x) * 주기       # 오차가 매번 (1 - 이득×주기)배가 된다
        기록.append(x)
    return np.array(기록)


def 구역(값):
    if 값 < 1:
        return "부드럽게 다가감"
    if 값 < 2:
        return "지나쳤다 수렴"
    return "계속 진동" if np.isclose(값, 2) else "발산"


# 슬라이드에서 쓰는 경우: (이득, 주기, 줄 높이, 꼬리표)
경우 = [(4, 0.02, 1, ""), (4, 0.2, 1, ""), (4, 0.4, 1, ""), (4, 0.6, 1, ""), (4, 0.3, 1, "확인문제"),
        (2, 0.02, 0, ""), (2, 0.2, 0, ""), (2, 0.4, 0, ""), (2, 0.6, 0, "")]
for 이득, 주기, _, _ in 경우:
    기록 = 기록_만들기(이득, 주기)
    print(f"이득 {이득} 주기 {주기:.2f}초: 이득×주기 {이득 * 주기:.2f} → {구역(이득 * 주기)}, "
          f"최대 지나침 {max(기록.max() - 목표, 0):.3f}m")

그림 = plt.figure(figsize=(10, 6.4))
선 = 그림.add_axes([0.09, 0.50, 0.89, 0.40])
선.set_xlim(-0.3, 2.7)
선.set_ylim(-0.75, 1.55)
선.axis("off")

# 수직선 위의 구역(0~1, 1~2, 2, 2 초과)
for 시작, 끝, 색, 글 in [(0, 1, "tab:green", "부드럽게 다가감"), (1, 2, "tab:orange", "지나쳤다 수렴"),
                        (2, 2.7, "tab:red", "발산")]:
    선.add_patch(plt.Rectangle((시작, 0.35), 끝 - 시작, 0.37, color=색, alpha=0.25, lw=0))
    선.text((시작 + 끝) / 2, 0.56, 글, ha="center", va="center", fontsize=11, zorder=3,
         bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5))
선.plot([0, 2.7], [0.35, 0.35], color="black", lw=1.5)
선.plot([2, 2], [0.28, 0.72], color="tab:red", lw=3)
선.text(2, 0.78, "정확히 2: 계속 진동", ha="center", fontsize=11, color="tab:red")
for 눈금 in [0, 1, 2]:
    선.plot([눈금, 눈금], [0.30, 0.35], color="black", lw=1.5)
    선.text(눈금, 0.18, str(눈금), ha="center", va="top", fontsize=11)
선.text(2.35, -0.4, "이득 × 주기\n(한 번에 고치는 비율)", ha="center", fontsize=11)
선.text(-0.28, 1.1, "이득 4", fontsize=11, va="center")
선.text(-0.28, -0.4, "이득 2", fontsize=11, va="center")

for 이득, 주기, 줄, 꼬리표 in 경우:
    값 = 이득 * 주기
    높이 = 1.1 if 줄 else -0.4
    선.plot([값, 값], [0.35, 높이 - (0.17 if 줄 else -0.17)], color="gray", lw=0.8, zorder=1)
    선.scatter(값, 0.35, s=45, color="black" if 줄 else "white", edgecolor="black", zorder=5,
               marker="*" if 꼬리표 else "o")
    글 = f"주기 {주기}\n{값:.2f}" if not 꼬리표 else f"{꼬리표}\n주기 {주기}\n{값:.2f}"
    선.text(값, 높이, 글, ha="center", va="center", fontsize=9 if 꼬리표 else 10,
            color="tab:purple" if 꼬리표 else "black")

# 아래: 구역마다 계단 응답 하나씩(슬라이드의 이득 4 경우에서 고른다)
예 = [(0.8, "tab:green"), (1.6, "tab:orange"), (2.0, "tab:red"), (2.4, "tab:red")]
for 번호, (값, 색) in enumerate(예):
    칸 = 그림.add_axes([0.09 + 번호 * 0.235, 0.12, 0.19, 0.28])
    기록 = 기록_만들기(값, 1.0, 8)              # 주기 1로 두면 이득×주기 = 값
    칸.plot(기록, "o-", color=색, markersize=4)
    칸.axhline(목표, color="gray", linestyle="--", linewidth=1)
    칸.set_ylim(-0.6, 2.6)
    칸.set_xlim(-0.3, 8.3)
    칸.grid(alpha=0.3)
    칸.set_xlabel("걸음", fontsize=10)
    칸.set_xticks(range(0, 9, 2))
    if 번호 == 0:
        칸.set_ylabel("위치 (목표 1)", fontsize=10)
    else:
        칸.set_yticklabels([])
    칸.set_title(f"이득×주기 {값:g}\n{구역(값)}", fontsize=11, color=색)
    print(f"예시 {값:g}: 8걸음 뒤 위치 {기록[-1]:.3f}")
그림.text(0.98, 0.012, "※ 발산하는 그래프는 그림 위쪽 밖으로 나간다", ha="right", fontsize=9, color="gray")
그림.text(0.535, 0.955, "이득 × 주기 한 숫자가 움직임을 정한다 (오차가 매 걸음 1 − 이득×주기 배)",
          ha="center", fontsize=12)
그림.savefig("s20_gain_period.png", dpi=200)
print("저장: s20_gain_period.png")
