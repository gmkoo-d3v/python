# [2권 Chapter 10. 전이 학습: 남이 학습한 모델 빌려 쓰기 · 10.1 사전학습 모델 불러오기 · manuscript-vol2/chapters/10-transfer-learning.md:111]
import torch
from torchvision import models

모델 = models.resnet18(weights=None)     # 구조만 볼 때는 가중치를 받지 않아도 된다

# x = torch.zeros(1, 3, 224, 224)           # 사진 1장, 3색, 224 × 224
x = torch.zeros(1, 3, 160, 160)           # 사진 1장, 3색, 224 × 224
for 이름, 부품 in 모델.named_children():
    if 이름 == "fc":
        x = torch.flatten(x, 1)            # resnet18의 forward가 fc 바로 앞에서 하는 일
    x = 부품(x)
    개수 = sum(p.numel() for p in 부품.parameters())
    print(f"{이름:8} {type(부품).__name__:17} {개수:>10,}개 → {tuple(x.shape)}")
print(모델.fc)
"""
conv1    Conv2d                 9,408개 → (1, 64, 112, 112)
bn1      BatchNorm2d              128개 → (1, 64, 112, 112)
relu     ReLU                       0개 → (1, 64, 112, 112)
maxpool  MaxPool2d                  0개 → (1, 64, 56, 56)
layer1   Sequential           147,968개 → (1, 64, 56, 56)
layer2   Sequential           525,568개 → (1, 128, 28, 28)
layer3   Sequential         2,099,712개 → (1, 256, 14, 14)
layer4   Sequential         8,393,728개 → (1, 512, 7, 7)
avgpool  AdaptiveAvgPool2d          0개 → (1, 512, 1, 1)
fc       Linear               513,000개 → (1, 1000)
Linear(in_features=512, out_features=1000, bias=True)

conv1    Conv2d                 9,408개 → (1, 64, 80, 80)
bn1      BatchNorm2d              128개 → (1, 64, 80, 80)
relu     ReLU                       0개 → (1, 64, 80, 80)
maxpool  MaxPool2d                  0개 → (1, 64, 40, 40)
layer1   Sequential           147,968개 → (1, 64, 40, 40)
layer2   Sequential           525,568개 → (1, 128, 20, 20)
layer3   Sequential         2,099,712개 → (1, 256, 10, 10)
layer4   Sequential         8,393,728개 → (1, 512, 5, 5)
avgpool  AdaptiveAvgPool2d          0개 → (1, 512, 1, 1)
fc       Linear               513,000개 → (1, 1000)
Linear(in_features=512, out_features=1000, bias=True)
"""