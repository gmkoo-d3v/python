import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path

# plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["font.family"] = "D2Coding"
plt.rcParams["axes.unicode_minus"] = False

선색, 흐린색, 감지색 = "dimgray", "lightgray", "tab:blue"   # 쓰는 색은 이 셋뿐

# 칸 하나가 그림 전체를 덮고, 좌표 1칸 = 1인치(72pt)가 되게 한다
그림 = plt.figure(figsize=(10, 6.6))
칸 = 그림.add_axes([0, 0, 1, 1])
칸.set_xlim(0, 10)
칸.set_ylim(0, 6.6)
칸.set_aspect("equal")
칸.axis("off")


def 상자(x, y, 글, 너비=1.6, 높이=0.6, 색=선색):
    칸.add_patch(FancyBboxPatch((x - 너비 / 2, y - 높이 / 2), 너비, 높이, boxstyle="round,pad=0,rounding_size=0.12",
                                facecolor="white", edgecolor=색, linewidth=1.6))
    칸.text(x, y, 글, ha="center", va="center", fontsize=14)
    return {"왼": (x - 너비 / 2, y), "오": (x + 너비 / 2, y), "아래": (x, y - 높이 / 2)}


def 화살표(*점들, 색=선색, 선="-", 굵기=1.6):    # 점들을 차례로 잇고 마지막 점에 화살촉
    칸.add_patch(FancyArrowPatch(path=Path(점들), arrowstyle="-|>", mutation_scale=16, color=색,
                                 linewidth=굵기, linestyle=선, joinstyle="round", capstyle="round"))


# 1) 위: 감지 → 판단 → 행동 → 세상 → 다시 감지 (슬라이드 1.1의 낱말과 청소기 예)
칸.text(5, 6.3, "감지·판단·행동 고리", ha="center", fontsize=13)
단계들 = [(2.0, "감지", "지금 세상이 어떤지 잰다", "청소기가 앞 벽까지 거리를 잰다"),
         (5.0, "판단", "잰 값으로 할 일을 고른다", "0.2m보다 가까우면 돈다"),
         (8.0, "행동", "몸을 움직인다", "바퀴를 굴린다")]
위_높이 = 5.45
상자들 = {}
for x, 이름, 뜻, 예 in 단계들:
    상자들[이름] = 상자(x, 위_높이, 이름, 색=감지색 if 이름 == "감지" else 선색)
    칸.text(x, 위_높이 - 0.6, 뜻, ha="center", va="center", fontsize=11)
    칸.text(x, 위_높이 - 0.92, 예, ha="center", va="center", fontsize=10, color="gray")
화살표(상자들["감지"]["오"], 상자들["판단"]["왼"])
화살표(상자들["판단"]["오"], 상자들["행동"]["왼"])

세상 = 상자(5.0, 3.75, "세상")
오른_팔, 왼_팔 = 9.65, 0.35                       # 바깥으로 돌아가는 길의 x 위치(설명 글과 겹치지 않게)
행동_오, 세상_오, 세상_왼, 감지_왼 = 상자들["행동"]["오"], 세상["오"], 세상["왼"], 상자들["감지"]["왼"]
화살표(행동_오, (오른_팔, 행동_오[1]), (오른_팔, 세상_오[1]), 세상_오)
화살표(세상_왼, (왼_팔, 세상_왼[1]), (왼_팔, 감지_왼[1]), 감지_왼, 색=감지색, 굵기=2.2)
칸.text(오른_팔 - 0.12, 세상_오[1] + 0.1, "세상이 바뀐다", ha="right", va="bottom", fontsize=10)
칸.text(왼_팔 + 0.12, 세상_왼[1] + 0.1, "다시 잰다", ha="left", va="bottom", fontsize=10, color=감지색)

칸.plot([0.3, 9.7], [3.1, 3.1], color=흐린색, linewidth=1)   # 위아래 구분선

# 2) 아래: 열린 고리와 닫힌 고리. 다른 점은 몸에서 판단으로 돌아오는 감지 화살표 하나뿐
for 왼쪽, 제목, 설명, 예, 닫힘 in [(0.0, "열린 고리", "계획한 명령을 그대로 보낸다", "예: 세탁기, 눈 감고 컵 잡기", False),
                                (5.0, "닫힌 고리", "매번 재서 다음 명령을 고친다(피드백)", "예: 눈 뜨고 컵 잡기", True)]:
    가운데 = 왼쪽 + 2.5
    칸.text(가운데, 2.75, 제목, ha="center", fontsize=13)
    칸.text(가운데, 2.38, 설명, ha="center", fontsize=11)
    고리 = [상자(가운데 + 밀기, 1.7, 글, 너비=1.1, 높이=0.55) for 밀기, 글 in [(-1.6, "판단"), (0.0, "명령"), (1.6, "몸")]]
    화살표(고리[0]["오"], 고리[1]["왼"])
    화살표(고리[1]["오"], 고리[2]["왼"])
    # 몸 아래 → 바닥 → 판단 아래: 닫힌 고리는 진하게, 열린 고리는 흐린 점선 위에 × 표시
    바닥 = 1.0
    몸_아래, 판단_아래 = 고리[2]["아래"], 고리[0]["아래"]
    색, 선, 굵기 = (감지색, "-", 2.2) if 닫힘 else (흐린색, "--", 1.4)
    화살표(몸_아래, (몸_아래[0], 바닥), (판단_아래[0], 바닥), 판단_아래, 색=색, 선=선, 굵기=굵기)
    if 닫힘:
        칸.text(가운데, 바닥 - 0.15, "감지: 잰 값", ha="center", va="top", fontsize=11, color=감지색)
    else:
        칸.text(가운데, 바닥, "×", ha="center", va="center", fontsize=22, color=선색,
                bbox=dict(facecolor="white", edgecolor="none", pad=1))
        칸.text(가운데, 바닥 - 0.25, "감지 없음", ha="center", va="top", fontsize=11, color="gray")
    칸.text(가운데, 0.18, 예, ha="center", va="bottom", fontsize=10, color="gray")

그림.savefig("s05_loop_diagram.png", dpi=200)
print("저장: s05_loop_diagram.png")
