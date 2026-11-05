# [2권 Chapter 6. nn.Module과 학습 루프: 모든 모델이 따르는 표준형 · 6.1 층 하나: 입력 @ 가중치 + 편향 · manuscript-vol2/chapters/06-nn-module.md:81]
import torch
from torch import nn

torch.manual_seed(0)
층 = nn.Linear(3, 5)                 # 입력 3개 → 출력 2개
print(층)
print("weight:", 층.weight.shape, "bias:", 층.bias.shape)

x = torch.rand(5, 3)
print(층(x))
직접 = x @ 층.weight.T + 층.bias      # 손으로 같은 계산 5*3 + 5 = 20
print("같은가:", torch.allclose(층(x), 직접))
print("학습 대상인가:", 층.weight.requires_grad)

"""
Linear(in_features=3, out_features=5, bias=True)
weight: torch.Size([5, 3]) bias: torch.Size([5])
tensor([[ 0.1132, -0.2034,  0.7372, -0.4920, -1.0718],
        [-0.1152, -0.1508,  0.5001, -0.3938, -1.0255],
        [-0.0595, -0.1559,  0.3426, -0.2732, -0.8348],
        [-0.3286,  0.1460,  0.4341, -0.4934, -0.7959],
        [ 0.0326,  0.1168,  0.4602, -0.4287, -0.4875]],
       grad_fn=<AddmmBackward0>)
같은가: True
학습 대상인가: True
"""