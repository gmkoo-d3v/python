import matplotlib.pyplot as plt
import numpy as np

# plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["font.family"] = "D2Coding"
plt.rcParams["axes.unicode_minus"] = False


def 축각_쿼터니언(축, 각도_도: float) -> np.ndarray:
    """[w x y z] = [cos(θ/2), sin(θ/2)·축]  (quaternion_tools.축각_쿼터니언과 같은 식)"""
    k = np.asarray(축, dtype=float) / np.linalg.norm(축)
    반각 = np.deg2rad(각도_도) / 2
    return np.concatenate([[np.cos(반각)], np.sin(반각) * k])


def 글(q):
    return "[" + ", ".join(f"{x:g}" for x in (np.round(q, 4) + 0.0)) + "]"


# 1) 슬라이드 47, 64쪽의 값 (b13_quat_basic.py, b15_quat_double.py): z축 회전
for 각 in [10, 90, 360, 370, 720]:
    print(f"{각:>3}도:", np.round(축각_쿼터니언([0, 0, 1], 각), 4) + 0.0)

각도 = np.linspace(0, 720, 721)
w = np.cos(np.deg2rad(각도) / 2)
z = np.sin(np.deg2rad(각도) / 2)

보라, 파랑, 주황 = "tab:purple", "tab:blue", "tab:orange"
그림 = plt.figure(figsize=(10, 6.2))
칸 = 그림.add_axes([0.08, 0.13, 0.90, 0.72])
칸.set(xlim=(-15, 770), ylim=(-1.75, 1.8), xticks=range(0, 721, 90), yticks=[-1, -0.5, 0, 0.5, 1])
칸.set_xticklabels(["0°", "90°", "180°", "270°", "360°\n(한 바퀴)", "450°", "540°", "630°", "720°\n(두 바퀴)"])
칸.set_xlabel("z축으로 돌린 각도 θ", fontsize=11)
칸.set_ylabel("쿼터니언 성분의 값", fontsize=11)
칸.grid(axis="y", alpha=0.3)
칸.axhline(0, color="gray", linewidth=0.8)
칸.axvspan(360, 720, color="lightgray", alpha=0.35, linewidth=0)
칸.axvline(360, color="gray", linewidth=0.8, linestyle=":")

칸.plot(각도, w, color=보라, linewidth=2.4)
칸.plot(각도, z, color=파랑,linewidth=2.4)
칸.text(480, 0.55, "w = cos(θ/2)", color=보라, ha="center", fontsize=11)
칸.text(610, -1.22, "z 성분 = sin(θ/2)", color=파랑, ha="center", fontsize=11)
칸.text(180, 1.64, "첫 바퀴 (0°~360°)", color="dimgray", ha="center", fontsize=11)
칸.text(540, 1.64, "둘째 바퀴: 같은 자세, 부호만 반대", color="dimgray", ha="center", fontsize=11)


def 점(각, 위치, 글자, va, 색="black", ha="left"):
    """각도 각의 w, z 값에 점을 찍고, 글자는 위치(x, y)에 놓고 가는 선으로 잇는다."""
    q = 축각_쿼터니언([0, 0, 1], 각)
    칸.plot([각, 각], [q[0], q[3]], "o", color=색, markersize=7, zorder=5)
    칸.annotate(글자.format(글(q)), xy=(각, q[0]), xytext=위치, color=색, ha=ha, va=va, fontsize=10,
               arrowprops=dict(arrowstyle="-", color=색, linewidth=0.8, shrinkA=3, shrinkB=4))


점(90, (110, 1.28), "90°  q = {}\n(슬라이드 47쪽)", "bottom")
점(360, (345, -1.3), "360°  q = {}\n= -(처음 q)", "top", ha="right")
점(720, (725, 1.28), "720°  q = {}\n처음과 같은 숫자", "bottom", ha="right")

# 2) 10°와 370°: 슬라이드 64쪽의 "같은 자세에 쿼터니언이 둘"
q10, q370 = 축각_쿼터니언([0, 0, 1], 10), 축각_쿼터니언([0, 0, 1], 370)
칸.plot([10, 370], [q10[0], q370[0]], "--", color=주황, linewidth=1.5, zorder=4)
칸.plot([10, 10, 370, 370], [q10[0], q10[3], q370[0], q370[3]], "o", color=주황, markersize=7, zorder=5)
칸.text(18, q10[0] + 0.08, "10°", color=주황, fontsize=10, ha="left", va="bottom")
칸.text(378, q370[0] - 0.08, "370°", color=주황, fontsize=10, ha="left", va="top")
칸.text(15, -0.55, f"10° : {글(q10)}\n370°: {글(q370)}\n같은 자세, 부호만 반대", color=주황,
       fontsize=10, ha="left", va="center")

그림.text(0.5, 0.945, "쿼터니언은 반각으로 만든다: q = [cos(θ/2), sin(θ/2)·축]", ha="center", fontsize=13)
그림.text(0.5, 0.895, "z축 회전이라 [w, x, y, z]에서 x, y는 0 (보라 = w, 파랑 = z 성분)", ha="center",
        fontsize=10, color="dimgray")
그림.savefig("s47_quaternion_half_angle.png", dpi=200)
print("저장: s47_quaternion_half_angle.png")
