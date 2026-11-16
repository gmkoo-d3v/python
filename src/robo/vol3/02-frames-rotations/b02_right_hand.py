import numpy as np

x축 = np.array([1, 0, 0])
y축 = np.array([0, 1, 0])
z축 = np.array([0, 0, 1])

# 오른손 좌표계라면 x에서 y로 감아쥔 엄지가 z를 가리킨다
print("x × y =", np.cross(x축, y축))
print("y × z =", np.cross(y축, z축))
print("z × x =", np.cross(z축, x축))
print("y × x =", np.cross(y축, x축))     # 순서를 바꾸면 반대쪽
