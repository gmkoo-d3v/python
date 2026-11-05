# [2권 Chapter 6. nn.Module과 학습 루프: 모든 모델이 따르는 표준형 · 6.2 활성화 함수: 직선을 구부리는 장치 · manuscript-vol2/chapters/06-nn-module.md:164]
import torch
from torch import nn

x = torch.linspace(-3, 3, 7)
print("입력      :", x)
print("ReLU      :", torch.relu(x))
print("시그모이드:", torch.sigmoid(x).round(decimals=3))
print("tanh      :", torch.tanh(x).round(decimals=3))

활성화 = nn.ReLU()                    # 층처럼 쓰는 모양
print("nn.ReLU   :", 활성화(x))

"""
입력      : tensor([-3., -2., -1.,  0.,  1.,  2.,  3.])
ReLU      : tensor([0., 0., 0., 0., 1., 2., 3.])
시그모이드: tensor([0.0470, 0.1190, 0.2690, 0.5000, 0.7310, 0.8810, 0.9530])
tanh      : tensor([-0.9950, -0.9640, -0.7620,  0.0000,  0.7620,  0.9640,  0.9950])
nn.ReLU   : tensor([0., 0., 0., 0., 1., 2., 3.])
"""