import numpy as np
from scipy.spatial.transform import Rotation

from gyro_data import 각속도, 시간, 주기, 증분_쿼터니언, 참_자세
from quaternion_tools import 곱, 사이각, 정규화

참값 = 참_자세()

롤피치요 = np.zeros(3)                       # (가) 오일러 각에 각속도를 그냥 더하기
q_근사 = np.array([1.0, 0.0, 0.0, 0.0])      # (나) 쿼터니언 미분 식을 한 걸음씩(1차 근사)
q_증분 = np.array([1.0, 0.0, 0.0, 0.0])      # (다) 한 주기 동안 돈 회전을 곱해 나가기
q_가운데 = np.array([1.0, 0.0, 0.0, 0.0])    # (라) (다)와 같되, 주기 한가운데의 각속도를 쓴다면?
오차 = {"(가) 오일러 각에 더하기": [], "(나) 쿼터니언 1차 근사": [],
        "(다) 쿼터니언 증분 곱": [], "(라) 가운데 값으로 증분 곱": []}

for k, t in enumerate(시간):
    w = 각속도(t)                            # 센서가 이번에 알려 준 각속도
    롤피치요 = 롤피치요 + w * 주기
    q_근사 = 정규화(q_근사 + 0.5 * 곱(q_근사, np.concatenate([[0.0], w])) * 주기)
    q_증분 = 곱(q_증분, 증분_쿼터니언(w, 주기))
    q_가운데 = 곱(q_가운데, 증분_쿼터니언(각속도(t + 주기 / 2), 주기))

    q_가 = Rotation.from_euler("ZYX", 롤피치요[::-1]).as_quat(scalar_first=True)
    for 이름, q in zip(오차, [q_가, q_근사, q_증분, q_가운데]):
        오차[이름].append(사이각(q, 참값[k + 1]))

for 이름, 값들 in 오차.items():
    print(f"{이름:16s}: 10초 뒤 {값들[-1]:7.3f}도, 가장 클 때 {max(값들):7.3f}도")
