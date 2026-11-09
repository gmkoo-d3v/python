"""2차시 실습: NMS 직접 구현 (6강 노트북 실습 4)."""
import torch
from torchvision.ops import box_iou, nms


def my_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """점수 순으로 박스를 고르며 많이 겹치는 박스를 버립니다."""
    order = scores.argsort(descending=True)
    keep = []
    while len(order) > 0:
        best = order[0]
        keep.append(best.item())
        rest = order[1:]
        ious = box_iou(boxes[best].unsqueeze(0), boxes[rest]).squeeze(0)
        # 기준값 이하로 겹치는 박스만 다음 차례로 넘긴다
        order = rest[ious <= iou_threshold]
    return torch.tensor(keep, dtype=torch.int64)


boxes_demo = torch.tensor([
    [10.0, 10.0, 60.0, 60.0],
    [12.0, 12.0, 62.0, 62.0],
    [14.0, 8.0, 64.0, 58.0],
    [100.0, 100.0, 150.0, 150.0],
])
scores_demo = torch.tensor([0.9, 0.8, 0.7, 0.6])
print("첫 박스와의 IoU:", [round(v, 3) for v in box_iou(boxes_demo[:1], boxes_demo).squeeze(0).tolist()])
assert my_nms(boxes_demo, scores_demo, 0.5).tolist() == [0, 3]
print("남은 박스 번호(기준 0.5):", my_nms(boxes_demo, scores_demo, 0.5).tolist())

torch.manual_seed(0)
top_left = torch.rand(300, 2) * 200
random_boxes = torch.cat([top_left, top_left + 20 + torch.rand(300, 2) * 60], dim=1)
random_scores = torch.rand(300)
for threshold in (0.3, 0.5, 0.7):
    mine = my_nms(random_boxes, random_scores, threshold)
    assert mine.tolist() == nms(random_boxes, random_scores, threshold).tolist()
    print(f"무작위 박스 300개, 기준 {threshold}: {len(mine)}개 남음 (torchvision과 같음)")
