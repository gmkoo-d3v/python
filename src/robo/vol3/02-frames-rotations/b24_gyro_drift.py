import matplotlib.pyplot as plt
import numpy as np

from gyro_data import 각속도, 시간, 주기, 증분_쿼터니언, 참_자세
from quaternion_tools import 곱, 사이각

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

참값 = 참_자세()
생성기 = np.random.default_rng(0)
잡음 = 생성기.normal(0, 0.02, size=(len(시간), 3))     # 잴 때마다 흔들리는 값(rad/s)
치우침 = np.array([0.0, 0.0, 0.01])                    # 늘 같은 쪽으로 틀리는 값(rad/s)

경우들 = {"깨끗한 센서": (0, 0), "잡음만": (1, 0), "잡음 + 치우침": (1, 1)}
그림, 칸 = plt.subplots(figsize=(7, 3.2))
for 이름, (잡음_켜기, 치우침_켜기) in 경우들.items():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    오차 = []
    for k, t in enumerate(시간):
        잰_값 = 각속도(t) + 잡음_켜기 * 잡음[k] + 치우침_켜기 * 치우침
        q = 곱(q, 증분_쿼터니언(잰_값, 주기))
        오차.append(사이각(q, 참값[k + 1]))
    칸.plot(시간 + 주기, 오차, label=이름)
    print(f"{이름:8s}: " + ", ".join(f"{초}초 {오차[초 * 100 - 1]:.2f}도" for 초 in [1, 2, 5, 10]))
칸.set_xlabel("시간(초)")
칸.set_ylabel("자세 오차(도)")
칸.legend()
그림.tight_layout()
그림.savefig("gyro_drift.png", dpi=200)
print("저장: gyro_drift.png")
