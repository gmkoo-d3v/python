# [2권 Chapter 9. CNN과 이미지 분류: 돋보기를 옮기며 무늬 찾기 · 9.3 채널: 겹겹이 쌓인 그림 · manuscript-vol2/chapters/09-cnn.md:281]
import torch
from torch import nn

torch.manual_seed(0)
컬러 = torch.rand(1, 3, 5, 5)                    # 컬러 사진 1장: 빨강·초록·파랑 3채널, 5×5
층 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)
# 층 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5)

print("필터 묶음:", 층.weight.shape, " 편향:", 층.bias.shape)
print("파라미터:", sum(p.numel() for p in 층.parameters()), "=", 8 * 3 * 3 * 3 + 8)

출력 = 층(컬러)
print("출력:", 출력.shape)

# 출력 0번 채널의 왼쪽 위 칸을 손으로: 세 채널의 3×3을 각자 곱해 모두 더하고 편향
조각 = 컬러[0, :, 0:3, 0:3]                      # (3, 3, 3): 세 채널의 왼쪽 위 3×3
# 조각 = 컬러[0, :, 0:5, 0:5]                      # (3, 3, 3): 세 채널의 왼쪽 위 3×3
손으로 = (조각 * 층.weight[0]).sum() + 층.bias[0]
print(f"손으로 {손으로.item():.6f}, 층이 낸 값 {출력[0, 0, 0, 0].item():.6f}")

"""
필터 묶음: torch.Size([8, 3, 3, 3])  편향: torch.Size([8])
파라미터: 224 = 224
출력: torch.Size([1, 8, 3, 3])
손으로 0.111630, 층이 낸 값 0.111630

필터 묶음: torch.Size([16, 3, 5, 5])  편향: torch.Size([16])
파라미터: 1216 = 224
출력: torch.Size([1, 16, 1, 1])
손으로 -0.254027, 층이 낸 값 -0.254027
"""