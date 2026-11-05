# [2권 Chapter 5. 자동 미분과 경사하강법 · 5.4 PyTorch에게 미분 맡기기 · manuscript-vol2/chapters/05-autograd.md:366]
import torch

# x = torch.tensor(3.0, requires_grad=True)   # "이 값에 대한 기울기가 필요해요" 표시
x = torch.tensor(2.0, requires_grad=True)   # "이 값에 대한 기울기가 필요해요" 표시
# y = x ** 2                                   # 평소처럼 계산
y = x ** 3                                   # 평소처럼 계산
print(y)

y.backward()                                 # y를 x로 미분해 줘
print("PyTorch가 구한 기울기:", x.grad)
print("손으로 구한 기울기 2x:", 2 * 3.0)

h = 1e-3
print("수치 미분:", ((3.0 + h) ** 2 - (3.0 - h) ** 2) / (2 * h))
"""
tensor(9., grad_fn=<PowBackward0>)
PyTorch가 구한 기울기: tensor(6.)
손으로 구한 기울기 2x: 6.0
수치 미분: 5.999999999999339

tensor(8., grad_fn=<PowBackward0>)
PyTorch가 구한 기울기: tensor(12.)
손으로 구한 기울기 2x: 6.0
수치 미분: 5.999999999999339


"""