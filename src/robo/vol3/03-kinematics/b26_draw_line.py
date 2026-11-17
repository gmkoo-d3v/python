import matplotlib.pyplot as plt
import numpy as np

from 두링크 import 순기구학2, 야코비안2, 역기구학2

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

주기 = 0.01                                    # 0.01초마다 관절 속도를 정한다
시간 = np.arange(0, 5, 주기)
시작, 속도 = np.array([0.3, 0.1]), np.array([0.1, 0.0])    # (0.3, 0.1)에서 x쪽으로 0.1m/s
목표들 = 시작 + np.outer(시간, 속도)                         # 끝에서는 x가 0.8 가까이: 팔이 닿지 않는 곳


def 따라가기(λ: float) -> tuple[np.ndarray, np.ndarray]:
    """매 순간 v = 원하는 속도 + 5 × 남은 오차를 관절 속도로 바꿔 움직인다(λ = 0이면 의사역행렬)."""
    q = 역기구학2(시작)[0]
    손끝들, 관절_빠르기 = [], []
    for 목표 in 목표들:
        v = 속도 + 5.0 * (목표 - 순기구학2(q))
        J = 야코비안2(q)
        관절_속도 = J.T @ np.linalg.solve(J @ J.T + λ**2 * np.eye(2), v)
        q = q + 관절_속도 * 주기
        손끝들.append(순기구학2(q))
        관절_빠르기.append(np.abs(관절_속도).max())
    return np.array(손끝들), np.array(관절_빠르기)


그림, (왼쪽, 오른쪽) = plt.subplots(1, 2, figsize=(10, 3.8))
왼쪽.plot(*목표들.T, "k--", linewidth=1, label="그리려는 선")
왼쪽.add_patch(plt.Circle((0, 0), 0.7, fill=False, color="gray", linestyle=":"))
for 이름, λ in [("의사역행렬", 0.0), ("DLS λ=0.05", 0.05)]:
    손끝들, 빠르기 = 따라가기(λ)
    print(f"{이름}: 가장 빠른 관절 속도 {빠르기.max():7.1f}rad/s, "
          f"선(y = 0.1)에서 가장 멀리 벗어난 거리 {np.abs(손끝들[:, 1] - 0.1).max() * 1000:5.1f}mm")
    왼쪽.plot(*손끝들.T, linewidth=1.5, label=이름)
    오른쪽.plot(시간, 빠르기, label=이름)
왼쪽.set_xlim(0.2, 0.85)
왼쪽.set_ylim(-0.25, 0.3)
왼쪽.set_aspect("equal")
왼쪽.set_xlabel("x (m)")
왼쪽.set_ylabel("y (m)")
왼쪽.legend(fontsize=8, loc="lower left")
오른쪽.set_yscale("log")
오른쪽.set_yticks([0.01, 0.1, 1, 10, 100], ["0.01", "0.1", "1", "10", "100"])     # 눈금 글자를 보통 숫자로
오른쪽.minorticks_off()
오른쪽.set_xlabel("시간(초)")
오른쪽.set_ylabel("가장 빠른 관절 속도(rad/s)")
오른쪽.legend(fontsize=8)
그림.tight_layout()
그림.savefig("draw_line.png", dpi=200)
print("저장: draw_line.png")
