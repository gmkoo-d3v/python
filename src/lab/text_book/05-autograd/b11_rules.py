# [2권 Chapter 5. 자동 미분과 경사하강법 · 5.4 PyTorch에게 미분 맡기기 · manuscript-vol2/chapters/05-autograd.md:440]
import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2                        # 결과가 숫자 3개인 벡터
try:
    y.backward()
except RuntimeError as e:
    print("오류 1:", e)

# y.sum().backward()                # 하나로 모으면 된다
y.mean().backward()                # 하나로 모으면 된다
print("x.grad:", x.grad)          # 각 칸의 기울기 2x

try:
    torch.tensor([1, 2, 3], requires_grad=True)
except RuntimeError as e:
    print("오류 2:", e)

print("x는 잎인가?", x.is_leaf, "/ y는 잎인가?", y.is_leaf)
print("y.grad:", y.grad)          # 중간 결과의 기울기는 보관하지 않는다

"""

오류 1: grad can be implicitly created only for scalar outputs
x.grad: tensor([2., 4., 6.])
오류 2: Only Tensors of floating point and complex dtype can require gradients
x는 잎인가? True / y는 잎인가? False
y.grad: None
C:\Users\user\Documents\카카오톡 받은 파일\머신러닝-딥러닝-코드\예제코드\05-autograd\b11_rules.py:21: UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the .grad field to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more information. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\build\aten\src\ATen/core/TensorBody.h:493.)
  print("y.grad:", y.grad)          # 중간 결과의 기울기는 보관하지 않는다

오류 1: grad can be implicitly created only for scalar outputs
x.grad: tensor([0.6667, 1.3333, 2.0000])
오류 2: Only Tensors of floating point and complex dtype can require gradients
x는 잎인가? True / y는 잎인가? False
y.grad: None
C:\Users\user\Documents\카카오톡 받은 파일\머신러닝-딥러닝-코드\예제코드\05-autograd\b11_rules.py:21: UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the .grad field to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more information. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\build\aten\src\ATen/core/TensorBody.h:493.)
  print("y.grad:", y.grad)          # 중간 결과의 기울기는 보관하지 않는다


"""