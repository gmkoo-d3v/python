"""1차시 실습: 박스 형식 변환과 IoU (6강 노트북 실습 1)."""
import torch
from torchvision.ops import box_iou


def xywh_to_xyxy(box: list[float]) -> list[float]:
    """COCO의 [x, y, w, h]를 [x1, y1, x2, y2]로 바꿉니다."""
    x, y, w, h = box
    return [x, y, x + w, y + h]


def iou(box_a: list[float], box_b: list[float]) -> float:
    """xyxy 박스 두 개의 IoU를 계산합니다."""
    left, top = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    right, bottom = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    # 겹치지 않으면 너비나 높이가 음수가 되므로 0으로 자른다
    inter = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter)


def to_yolo(box: list[float], width: int, height: int) -> list[float]:
    """xyxy 픽셀 박스를 YOLO 형식(중심과 크기, 0~1)으로 바꿉니다."""
    x1, y1, x2, y2 = box
    return [(x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height]


assert xywh_to_xyxy([10, 20, 30, 40]) == [10, 20, 40, 60]
assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0

gt = [20.0, 20.0, 70.0, 70.0]
for shift in (25, 12, 4):
    pred = [gt[0] + shift, gt[1] + shift, gt[2] + shift, gt[3] + shift]
    mine = iou(gt, pred)
    ref = box_iou(torch.tensor([gt]), torch.tensor([pred])).item()
    assert abs(mine - ref) < 1e-6
    print(f"대각선으로 {shift:2d}픽셀 밀림: IoU {mine:.3f}")

print("같은 박스 세 형식:")
print("  xyxy :", gt)
print("  xywh :", [gt[0], gt[1], gt[2] - gt[0], gt[3] - gt[1]])
print("  YOLO :", [round(v, 3) for v in to_yolo(gt, 100, 100)], "(100x100 이미지)")
