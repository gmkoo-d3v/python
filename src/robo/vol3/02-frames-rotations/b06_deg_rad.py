import numpy as np

print("np.cos(90)             =", round(float(np.cos(90)), 4))      # 90을 라디안으로 읽는다
print("np.cos(np.deg2rad(90)) =", round(float(np.cos(np.deg2rad(90))), 4))
print("180도 =", np.deg2rad(180), "라디안")
print("1라디안 =", round(float(np.rad2deg(1)), 2), "도")
