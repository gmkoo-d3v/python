"""4차시 실습: YOLO v3 뼈대를 쌓고 출력 크기 확인 (6강 노트북 4.5절, 실습 6 뒷부분)."""

import torch
import torch.nn as nn

LEAKY_SLOPE = 0.1
BOX_AND_CONFIDENCE = 5
STAGE_BLOCKS = (1, 2, 8, 8, 4)   # Darknet-53의 단계별 잔차 블록 수
ANCHORS = [[(10, 13), (16, 30), (33, 23)], [(30, 61), (62, 45), (59, 119)], [(116, 90), (156, 198), (373, 326)]]


def conv_unit(in_channels: int, out_channels: int, kernel_size: int, stride: int = 1) -> nn.Sequential:
    """합성곱, BatchNorm, LeakyReLU 묶음입니다."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(LEAKY_SLOPE),
    )


class DarknetResidual(nn.Module):
    """1x1로 채널을 반으로 줄였다가 3x3으로 되돌리고, 입력을 더합니다."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(conv_unit(channels, channels // 2, 1), conv_unit(channels // 2, channels, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


def darknet_stage(out_channels: int, num_residuals: int) -> nn.Sequential:
    """stride 2 합성곱으로 크기를 반으로 줄인 뒤 잔차 블록을 쌓습니다."""
    residuals = [DarknetResidual(out_channels) for _ in range(num_residuals)]
    return nn.Sequential(conv_unit(out_channels // 2, out_channels, 3, stride=2), *residuals)


def neck_block(in_channels: int, channels: int) -> nn.Sequential:
    """이어 붙인 feature를 1x1과 3x3을 번갈아 섞습니다."""
    wide = channels * 2
    return nn.Sequential(conv_unit(in_channels, channels, 1), conv_unit(channels, wide, 3), conv_unit(wide, channels, 1),
                         conv_unit(channels, wide, 3), conv_unit(wide, channels, 1))


class YoloHead(nn.Module):
    """feature map 하나에서 박스를 예측합니다."""

    def __init__(self, in_channels: int, anchors: list[tuple[int, int]], num_classes: int, image_size: int) -> None:
        super().__init__()
        self.num_anchors = len(anchors)
        self.num_outputs = BOX_AND_CONFIDENCE + num_classes
        self.image_size = image_size
        self.register_buffer("anchors", torch.tensor(anchors, dtype=torch.float32).view(1, self.num_anchors, 1, 1, 2))
        self.predict = nn.Sequential(conv_unit(in_channels, in_channels * 2, 3),
                                     nn.Conv2d(in_channels * 2, self.num_anchors * self.num_outputs, 1))

    def decode_boxes(self, raw: torch.Tensor) -> torch.Tensor:
        """(배치, 앵커, S, S, 출력)을 픽셀 단위 (중심 x, 중심 y, 너비, 높이)로 풉니다."""
        grid_size = raw.shape[2]
        stride = self.image_size / grid_size
        cell_y, cell_x = torch.meshgrid(torch.arange(grid_size, device=raw.device), torch.arange(grid_size, device=raw.device), indexing="ij")
        cells = torch.stack((cell_x, cell_y), dim=-1).float()
        centers = (torch.sigmoid(raw[..., :2]) + cells) * stride
        sizes = self.anchors * torch.exp(raw[..., 2:4])
        return torch.cat((centers, sizes), dim=-1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = features.shape
        raw = self.predict(features).view(batch, self.num_anchors, self.num_outputs, height, width).permute(0, 1, 3, 4, 2)
        decoded = torch.cat((self.decode_boxes(raw), torch.sigmoid(raw[..., 4:5]), torch.sigmoid(raw[..., 5:])), dim=-1)
        return decoded.flatten(1, 3)


class YoloV3(nn.Module):
    """Darknet-53, FPN 방식의 neck, 세 크기의 예측부를 잇습니다."""

    def __init__(self, anchors: list[list[tuple[int, int]]], num_classes: int = 20, image_size: int = 416) -> None:
        super().__init__()
        self.stem = conv_unit(3, 32, 3)
        self.stage1, self.stage2 = darknet_stage(64, STAGE_BLOCKS[0]), darknet_stage(128, STAGE_BLOCKS[1])
        self.stage3, self.stage4 = darknet_stage(256, STAGE_BLOCKS[2]), darknet_stage(512, STAGE_BLOCKS[3])
        self.stage5 = darknet_stage(1024, STAGE_BLOCKS[4])
        self.neck5, self.shrink5 = neck_block(1024, 512), conv_unit(512, 256, 1)
        self.neck4, self.shrink4 = neck_block(256 + 512, 256), conv_unit(256, 128, 1)
        self.neck3 = neck_block(128 + 256, 128)
        self.upsample = nn.Upsample(scale_factor=2)
        # 격자가 성긴 층일수록 큰 물체를 맡으므로 큰 앵커를 준다
        self.head_large = YoloHead(512, anchors[2], num_classes, image_size)
        self.head_medium = YoloHead(256, anchors[1], num_classes, image_size)
        self.head_small = YoloHead(128, anchors[0], num_classes, image_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        c3 = self.stage3(self.stage2(self.stage1(self.stem(x))))
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        p5 = self.neck5(c5)
        p4 = self.neck4(torch.cat((self.upsample(self.shrink5(p5)), c4), dim=1))
        p3 = self.neck3(torch.cat((self.upsample(self.shrink4(p4)), c3), dim=1))
        per_scale = [self.head_large(p5), self.head_medium(p4), self.head_small(p3)]
        return torch.cat(per_scale, dim=1), per_scale


with torch.no_grad():
    for size, classes in ((416, 20), (608, 20), (416, 3)):
        model = YoloV3(ANCHORS, num_classes=classes, image_size=size).eval()
        merged, per_scale = model(torch.randn(1, 3, size, size))
        print(f"입력 {size}, 클래스 {classes:2d}: 전체 {tuple(merged.shape)}, 크기별 {[s.shape[1] for s in per_scale]}")
    print(f"파라미터(20클래스): {sum(p.numel() for p in YoloV3(ANCHORS).parameters()) / 1e6:.1f}M")

# 10x10 픽셀 자동차가 격자 한 칸에서 차지하는 넓이
for grid in (13, 52):
    print(f"{grid}x{grid} 격자 한 칸 {416 // grid}px, 10x10 물체의 넓이 비율 {100 / (416 / grid) ** 2:.3f}")
