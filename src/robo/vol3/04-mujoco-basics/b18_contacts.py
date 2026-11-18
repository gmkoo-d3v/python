import matplotlib.pyplot as plt
import mujoco
import numpy as np

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

xml = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="box" pos="0 0 0.3">
      <freejoint/>
      <geom name="box_geom" type="box" size="0.1 0.1 0.05" mass="2"/>
    </body>
  </worldbody>
</mujoco>
"""
모델 = mujoco.MjModel.from_xml_string(xml)
데이터 = mujoco.MjData(모델)
힘 = np.zeros(6)
시각, 높이, 수직력 = [], [], []
while 데이터.time < 1.0:
    mujoco.mj_step(모델, 데이터)
    if 데이터.ncon > 0 and not any(수직력):          # 처음 닿은 순간의 접촉을 들여다본다
        print(f"처음 닿은 시각 {데이터.time:.3f}초, 접촉 수 {데이터.ncon}")
        for c in 데이터.contact[:데이터.ncon]:
            이름1 = mujoco.mj_id2name(모델, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
            이름2 = mujoco.mj_id2name(모델, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
            print(f"  {이름1}-{이름2}: 위치 {np.round(c.pos, 3) + 0.0}, 거리 {c.dist:+.4f}m, 법선 {c.frame[:3]}")
    합 = 0.0
    for i in range(데이터.ncon):
        mujoco.mj_contactForce(모델, 데이터, i, 힘)   # 접촉 좌표계 기준 [수직, 접선1, 접선2, 토크 3개]
        합 += 힘[0]
    시각.append(데이터.time)
    높이.append(데이터.qpos[2])
    수직력.append(합)

가장_큰 = int(np.argmax(수직력))
무게 = 2 * 9.81
print(f"가장 큰 수직력 {수직력[가장_큰]:.0f}N ({시각[가장_큰]:.3f}초), 무게의 {수직력[가장_큰] / 무게:.0f}배")
print(f"1초 뒤 접촉 수 {데이터.ncon}, 수직력 합 {수직력[-1]:.3f}N, 무게 mg = {무게:.3f}N")

그림, (위, 아래) = plt.subplots(2, 1, figsize=(7, 4), sharex=True)
위.plot(시각, 높이)
위.set_ylabel("상자 중심 높이(m)")
아래.plot(시각, 수직력, color="tab:red")
아래.axhline(무게, color="gray", linestyle="--", linewidth=1)
아래.set_ylabel("수직력 합(N)")
아래.set_xlabel("시간(초)")
그림.tight_layout()
그림.savefig("impact.png", dpi=200)
print("저장: impact.png")
