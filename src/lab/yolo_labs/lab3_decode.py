"""3차시 실습: 격자와 앵커, 박스 디코딩 직접 계산 (6강 노트북 4.2~4.3절, 실습 6 앞부분)."""
import math

GRID_V1 = 7
BOXES_PER_CELL_V1 = 2
VOC_CLASSES = 20
BOX_AND_CONFIDENCE = 5


def sigmoid(t: float) -> float:
    """0~1 사이로 누릅니다."""
    return 1 / (1 + math.exp(-t))


def decode_box(tx: float, ty: float, tw: float, th: float, cx: int, cy: int,
               anchor_w: float, anchor_h: float, stride: float) -> tuple[float, float, float, float]:
    """네트워크 출력을 픽셀 단위 (중심 x, 중심 y, 너비, 높이)로 풉니다."""
    center_x = (sigmoid(tx) + cx) * stride
    center_y = (sigmoid(ty) + cy) * stride
    # 앵커가 이미 픽셀 단위이므로 stride를 다시 곱하지 않는다
    return center_x, center_y, anchor_w * math.exp(tw), anchor_h * math.exp(th)


def responsible_cell(box: list[float], width: int, height: int, grid: int) -> tuple[int, int]:
    """박스 중심이 들어간 칸 (x칸, y칸)을 돌려줍니다. 이 칸이 물체를 책임진다."""
    center_x, center_y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return int(center_x // (width / grid)), int(center_y // (height / grid))


print("YOLO v1 출력 크기:", (GRID_V1, GRID_V1, BOXES_PER_CELL_V1 * BOX_AND_CONFIDENCE + VOC_CLASSES),
      "/ 박스 수:", GRID_V1 * GRID_V1 * BOXES_PER_CELL_V1)
print("448x448 이미지, 사람 박스 (100, 50, 220, 400)의 책임 칸:", responsible_cell([100, 50, 220, 400], 448, 448, GRID_V1))

# 13x13 격자(stride 32), 앵커 (156, 198), y칸 4, x칸 7
box = decode_box(0.4, -0.6, 0.3, 0.5, cx=7, cy=4, anchor_w=156, anchor_h=198, stride=32)
print("디코딩:", tuple(round(v, 2) for v in box))
print("칸 범위 x:", 7 * 32, "~", 8 * 32, "/ y:", 4 * 32, "~", 5 * 32)
# 출력이 아무리 커도 중심은 자기 칸 안에 머문다
for extreme in (-50.0, 50.0):
    cx_px, cy_px, _, _ = decode_box(extreme, extreme, 0, 0, cx=7, cy=4, anchor_w=156, anchor_h=198, stride=32)
    assert 7 * 32 <= cx_px <= 8 * 32 and 4 * 32 <= cy_px <= 5 * 32
    print(f"t={extreme:+.0f}일 때 중심: ({cx_px:.1f}, {cy_px:.1f})")
