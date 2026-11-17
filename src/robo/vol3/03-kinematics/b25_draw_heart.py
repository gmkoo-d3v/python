import matplotlib.pyplot as plt
import numpy as np

from 두링크 import 순기구학2, 야코비안2, 역기구학2, 팔_점들

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# 그릴 그림: 하트 곡선(가로 약 0.26m)을 팔 앞쪽 (0.4, 0.15)에 놓는다
t = np.linspace(0, 2 * np.pi, 400)
하트 = 0.008 * np.stack([16 * np.sin(t)**3,
                       13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)])
경로 = 하트.T + [0.4, 0.15]

q = 역기구학2(경로[0])[0]                   # 첫 점만 해석해(팔꿈치 위 답)로 시작
관절_기록, 손끝_기록 = [], []
for 목표 in 경로:
    for _ in range(3):                      # 바로 앞 자세에서 출발하니 몇 번이면 충분하다
        J = 야코비안2(q)
        q = q + J.T @ np.linalg.solve(J @ J.T + 0.01**2 * np.eye(2), 목표 - 순기구학2(q))
    관절_기록.append(q.copy())
    손끝_기록.append(순기구학2(q))
관절_기록, 손끝_기록 = np.array(관절_기록), np.array(손끝_기록)

오차 = np.linalg.norm(손끝_기록 - 경로, axis=1)
print(f"경로 점 {len(경로)}개, 가장 큰 손끝 오차 {오차.max():.1e}m")
print(f"어깨 {np.rad2deg(관절_기록[:, 0]).min():.1f}~{np.rad2deg(관절_기록[:, 0]).max():.1f}도, "
      f"팔꿈치 {np.rad2deg(관절_기록[:, 1]).min():.1f}~{np.rad2deg(관절_기록[:, 1]).max():.1f}도")

그림, (왼쪽, 오른쪽) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1, 1.3]})
for k in range(0, len(경로), 80):
    점 = 팔_점들(관절_기록[k])
    왼쪽.plot(점[:, 0], 점[:, 1], "o-", color="gray", alpha=0.5, linewidth=2)
왼쪽.plot(*손끝_기록.T, color="tab:red", linewidth=2, label="손끝이 그린 선")
왼쪽.plot(0, 0, "ks")
왼쪽.set_aspect("equal")
왼쪽.set_xlabel("x (m)")
왼쪽.set_ylabel("y (m)")
왼쪽.legend(fontsize=8, loc="lower right")
오른쪽.plot(np.rad2deg(관절_기록[:, 0]), label="어깨")
오른쪽.plot(np.rad2deg(관절_기록[:, 1]), label="팔꿈치")
오른쪽.set_xlabel("경로 점 번호")
오른쪽.set_ylabel("관절 각도(도)")
오른쪽.legend(fontsize=8)
그림.tight_layout()
그림.savefig("draw_heart.png", dpi=200)
print("저장: draw_heart.png")
