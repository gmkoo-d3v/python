import matplotlib.pyplot as plt
import numpy as np

# plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["font.family"] = "D2Coding"
plt.rcParams["axes.unicode_minus"] = False

# b01_washer_vs_robot.py 의 값과 규칙을 그대로 옮긴다
세탁기_순서 = [("물 받기", 3), ("세탁", 20), ("헹굼", 10), ("탈수", 7)]
안전_거리 = 0.2
앞_거리_측정값 = [1.2, 0.8, 0.45, 0.15, 0.9, 0.6]
행동들 = ["앞으로 간다" if 거리 > 안전_거리 else "제자리에서 돈다" for 거리 in 앞_거리_측정값]
for 단계, 분 in 세탁기_순서:
    print(f"[세탁기] {단계} {분}분")
for 거리, 행동 in zip(앞_거리_측정값, 행동들):
    print(f"[청소기] 앞 {거리:.2f}m → {행동}")

그림, (왼, 오) = plt.subplots(1, 2, figsize=(10, 5.2), gridspec_kw={"width_ratios": [1, 1.05]})
그림.subplots_adjust(left=0.04, right=0.98, bottom=0.24, top=0.82, wspace=0.12)

# 왼쪽: 세탁기 = 정해진 시간표 (가로 막대를 이어 붙인다)
색_세탁 = ["#9ecae1", "#6baed6", "#4292c6", "#2171b5"]
시작 = 0
for (단계, 분), 색 in zip(세탁기_순서, 색_세탁):
    왼.barh(0, 분, left=시작, height=0.5, color=색, edgecolor="white", linewidth=2)
    왼.text(시작 + 분 / 2, 0, 단계 if 분 > 5 else "", ha="center", va="center", fontsize=11, color="white" if 분 > 5 else "black")
    왼.text(시작 + 분 / 2, -0.42, f"{분}분", ha="center", va="top", fontsize=11)
    if 분 <= 5:
        왼.annotate(단계, xy=(시작 + 분 / 2, 0.25), xytext=(시작 + 분 / 2, 0.6), ha="center", fontsize=11,
                  arrowprops=dict(arrowstyle="-", color="gray"))
    시작 += 분
왼.annotate("", xy=(시작, 1.0), xytext=(0, 1.0), arrowprops=dict(arrowstyle="->", color="gray"))
왼.text(시작 / 2, 1.08, f"시간 →  (모두 {시작}분)", ha="center", fontsize=11, color="dimgray")
왼.text(시작 / 2, -1.25, "순서도 시간도 미리 정해져 있다\n빨래가 얼마나 더러운지는 재지 않는다",
        ha="center", va="top", fontsize=11, color="tab:blue")
왼.set_xlim(-1, 시작 + 1)
왼.set_ylim(-2.2, 1.6)
왼.set_axis_off()

# 오른쪽: 청소기 = 잰 값에 따라 행동이 바뀐다
색_행동 = {"앞으로 간다": "tab:green", "제자리에서 돈다": "tab:orange"}
번호 = np.arange(1, len(앞_거리_측정값) + 1)
오.bar(번호, 앞_거리_측정값, width=0.6, color=[색_행동[행동] for 행동 in 행동들])
for 순번, 거리, 행동 in zip(번호, 앞_거리_측정값, 행동들):
    if 거리 < 안전_거리:   # 막대가 낮아서 값은 막대 안에 쓴다
        오.text(순번, 거리 / 2, f"{거리:.2f}", ha="center", va="center", fontsize=10, color="white")
    else:
        오.text(순번, 거리 + 0.03, f"{거리:.2f}", ha="center", fontsize=10)
    오.text(순번, -0.17, "전진" if 행동 == "앞으로 간다" else "회전", ha="center", va="top", fontsize=11,
            color=색_행동[행동], fontweight="bold")
오.axhline(안전_거리, color="gray", linestyle="--")
오.text(6.5, 안전_거리 + 0.03, f"안전 거리\n{안전_거리} m", ha="left", va="bottom", fontsize=11, color="dimgray")
오.annotate("안전 거리보다\n가까우니\n제자리에서 돈다", xy=(4, 0.3), xytext=(3.5, 0.88), ha="center",
          va="center", fontsize=11, color="tab:orange",
          arrowprops=dict(arrowstyle="-|>", color="tab:orange", shrinkA=2, connectionstyle="arc3,rad=-0.25"))
오.set_xlim(0.4, 8.0)
오.set_ylim(0, 1.4)
오.set_xticks(번호)
오.set_xticklabels([f"{순번}번째" for 순번 in 번호], fontsize=10)
오.tick_params(axis="x", length=0, pad=4)
오.set_yticks([0, 0.5, 1.0])
오.set_ylabel("잰 앞 거리 (m)")
오.grid(axis="y", alpha=0.3)
for 변 in ("top", "right"):
    오.spines[변].set_visible(False)
그림.text(0.67, 0.035, "잰 값이 바뀌면 행동도 바뀐다\n초록: 앞으로 간다 / 주황: 제자리에서 돈다",
        ha="center", va="bottom", fontsize=11, color="dimgray")

그림.canvas.draw()
for 칸, 제목 in [(왼, "세탁기: 정해진 순서대로"), (오, "로봇 청소기: 재고, 고르고, 움직인다")]:
    자리 = 칸.get_position()
    그림.text((자리.x0 + 자리.x1) / 2, 0.9, 제목, ha="center", fontsize=13)
그림.savefig("s04_washer_vs_robot.png", dpi=200)
print("저장: s04_washer_vs_robot.png")
