import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

위팔, 아래팔 = 0.4, 0.3


def 팔_점들(어깨: np.ndarray, 팔꿈치: np.ndarray) -> np.ndarray:
    """(바닥, 팔꿈치, 손끝) 세 점. 각도 배열을 넣으면 한꺼번에 계산된다."""
    팔꿈치_위치 = 위팔 * np.stack([np.cos(어깨), np.sin(어깨)])
    손끝_위치 = 팔꿈치_위치 + 아래팔 * np.stack([np.cos(어깨 + 팔꿈치), np.sin(어깨 + 팔꿈치)])
    return np.stack([np.zeros_like(손끝_위치), 팔꿈치_위치, 손끝_위치])


생성기 = np.random.default_rng(0)
표본 = 생성기.uniform(-np.pi, np.pi, size=(2, 20000))      # 관절 공간에서 고르게 뽑은 각도
손끝들 = 팔_점들(표본[0], 표본[1])[2]
거리 = np.linalg.norm(손끝들, axis=0)
print(f"제한 없음: 손끝 거리 {거리.min():.3f} ~ {거리.max():.3f}m")

# 관절 한계: 어깨 0~180도, 팔꿈치 -150~150도
제한 = (표본[0] >= 0) & (np.abs(표본[1]) <= np.deg2rad(150))
print(f"관절 한계 안: 표본 {제한.sum()}개, 손끝 거리 {거리[제한].min():.3f} ~ {거리[제한].max():.3f}m")

그림, (왼쪽, 오른쪽) = plt.subplots(1, 2, figsize=(9, 4.2))
for 어깨_도, 팔꿈치_도 in [(20, 30), (60, -70), (120, 90), (160, 0)]:
    점 = 팔_점들(np.deg2rad(어깨_도), np.deg2rad(팔꿈치_도))
    왼쪽.plot(점[:, 0], 점[:, 1], "o-", linewidth=3, label=f"({어깨_도}도, {팔꿈치_도}도)")
왼쪽.set_title("관절 각도 (어깨, 팔꿈치)에 따른 팔 모양")
왼쪽.legend(fontsize=8, loc="lower left")
오른쪽.scatter(*손끝들, s=1, color="lightgray", label="제한 없음")
오른쪽.scatter(*손끝들[:, 제한], s=1, color="tab:blue", label="관절 한계 안")
for 반지름 in [위팔 - 아래팔, 위팔 + 아래팔]:
    오른쪽.add_patch(plt.Circle((0, 0), 반지름, fill=False, color="k", linestyle="--"))
오른쪽.set_title("손끝이 닿는 곳(작업 공간)")
오른쪽.legend(fontsize=8, loc="lower left", markerscale=6)
for 칸 in (왼쪽, 오른쪽):
    칸.plot(0, 0, "ks")
    칸.set_xlim(-0.8, 0.8)
    칸.set_ylim(-0.8, 0.8)
    칸.set_aspect("equal")
    칸.set_xlabel("x (m)")
    칸.set_ylabel("y (m)")
그림.tight_layout()
그림.savefig("workspace2.png", dpi=200)
print("저장: workspace2.png")
