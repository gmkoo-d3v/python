# [2권 Chapter 9. CNN과 이미지 분류: 돋보기를 옮기며 무늬 찾기 · 9.5 첫 CNN으로 옷 사진 분류하기 · manuscript-vol2/chapters/09-cnn.md:400]
import torch
from torch import nn

모델 = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(),   # 무늬 찾기 32가지
    nn.MaxPool2d(2),                                          # 28×28 → 14×14
    nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),  # 무늬의 조합 64가지
    nn.MaxPool2d(2),
    nn.AdaptiveAvgPool2d(1), nn.Flatten(),nn.Linear(64, 128), nn.ReLU(),
    # 14×14 → 7×7
    # nn.Flatten(),                                             # 64×7×7 = 3,136개를 한 줄로
    # nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
    nn.Linear(128, 10),
)

x = torch.zeros(64, 1, 28, 28)                               # 옷 사진 64장 묶음
print(f"{'입력':10} {str(tuple(x.shape)):17}")
for 층 in 모델:
    x = 층(x)
    개수 = sum(p.numel() for p in 층.parameters())
    print(f"{type(층).__name__:10} {str(tuple(x.shape)):17} {개수:>9,}")
print(f"전체 파라미터 {sum(p.numel() for p in 모델.parameters()):,}개")
"""
입력         (64, 1, 28, 28)  
Conv2d     (64, 32, 28, 28)        320
ReLU       (64, 32, 28, 28)          0
MaxPool2d  (64, 32, 14, 14)          0
Conv2d     (64, 64, 14, 14)     18,496
ReLU       (64, 64, 14, 14)          0
MaxPool2d  (64, 64, 7, 7)            0
Flatten    (64, 3136)                0
Linear     (64, 128)           401,536
ReLU       (64, 128)                 0
Linear     (64, 10)              1,290
전체 파라미터 421,642개

입력         (64, 1, 28, 28)  
Conv2d     (64, 32, 28, 28)        320
ReLU       (64, 32, 28, 28)          0
MaxPool2d  (64, 32, 14, 14)          0
Conv2d     (64, 64, 14, 14)     18,496
ReLU       (64, 64, 14, 14)          0
MaxPool2d  (64, 64, 7, 7)            0
AdaptiveAvgPool2d (64, 64, 1, 1)            0
Flatten    (64, 64)                  0
Linear     (64, 128)             8,320
ReLU       (64, 128)                 0
Linear     (64, 10)              1,290
전체 파라미터 28,426개

"""