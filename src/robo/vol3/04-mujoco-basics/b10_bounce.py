import matplotlib.pyplot as plt
import mujoco
import numpy as np

plt.rcParams["font.family"] = "D2Coding"
plt.rcParams["axes.unicode_minus"] = False

경우들 = {"기본값 (0.02 1)": "0.02 1", "덜 감쇠 (0.02 0.1)": "0.02 0.1", "스프링 직접 (-10000 -10)": "-10000 -10"}
그림, 칸 = plt.subplots(figsize=(7, 3.2))
for 이름, solref in 경우들.items():
    xml = f"""
    <mujoco>
      <worldbody>
        <geom type="plane" size="2 2 0.1"/>
        <body pos="0 0 1"><freejoint/><geom type="sphere" size="0.1" solref="{solref}"/></body>
      </worldbody>
    </mujoco>
    """
    모델 = mujoco.MjModel.from_xml_string(xml)
    데이터 = mujoco.MjData(모델)
    시각, 높이 = [], []
    while 데이터.time < 2.5:
        mujoco.mj_step(모델, 데이터)
        시각.append(데이터.time)
        높이.append(데이터.qpos[2])
    높이 = np.array(높이)
    처음_닿음 = int(np.argmax(높이 < 0.1))                 # 공 반지름 0.1m 아래로 내려간 첫 순간
    다시_뜬_높이 = 높이[처음_닿음:].max() - 0.1                # 닿은 뒤 공 아랫면이 가장 높이 뜬 높이
    print(f"{이름}: 가장 깊이 파고듦 {0.1 - 높이.min():.3f}m, 다시 뜬 높이 {max(다시_뜬_높이, 0):.3f}m")
    칸.plot(시각, 높이, label=이름)
칸.set_xlabel("시간(초)")
칸.set_ylabel("공 중심 높이(m)")
칸.legend()
그림.tight_layout()
그림.savefig("bounce.png", dpi=200)
print("저장: bounce.png")
