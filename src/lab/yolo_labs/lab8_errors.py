"""7차시 실습: 틀린 것을 종류별로 센다 — 위치 어긋남, 중복, 배경 오탐, 놓침(크기별)."""
import sys
from collections import Counter

import torch
from torchvision.ops import box_iou
from ultralytics import YOLO

from common_yolo import BEST, IMAGE_SIZE, pick_device, read_truth, require_lab5, valid_images

MATCH_IOU = 0.5     # 이 이상 겹치면 맞음
NEAR_IOU = 0.1      # 맞음은 아니어도 이 이상 겹치면 '위치 어긋남', 아래면 '배경 오탐'
CONFS = (0.25, 0.5, 0.646)   # 0.646은 lab7에서 F1이 가장 높던 conf
SMALL, MEDIUM = 32 * 32, 96 * 96   # COCO의 크기 구분(넓이, 픽셀²)


def size_name(box: list[float]) -> str:
    area = (box[2] - box[0]) * (box[3] - box[1])
    return "작음" if area < SMALL else "중간" if area < MEDIUM else "큼"


def classify(pred: torch.Tensor, truth: torch.Tensor) -> tuple[Counter, list[int]]:
    """점수 순으로 정답과 짝짓는다. 한 정답에는 예측 하나만. 남은 예측은 오류 종류로, 남은 정답은 놓침으로."""
    kinds: Counter = Counter()
    matched = [False] * len(truth)
    ious = box_iou(pred, truth) if len(pred) and len(truth) else torch.zeros(len(pred), len(truth))
    for i in range(len(pred)):   # pred는 이미 점수 내림차순
        best = int(ious[i].argmax()) if len(truth) else -1
        best_iou = float(ious[i, best]) if len(truth) else 0.0
        if best_iou >= MATCH_IOU and not matched[best]:
            matched[best] = True
            kinds["맞음"] += 1
        elif best_iou >= MATCH_IOU:
            kinds["중복"] += 1          # 이미 짝지어진 정답에 또 낸 박스: NMS가 못 지운 것
        elif best_iou >= NEAR_IOU:
            kinds["위치 어긋남"] += 1   # 사람 근처지만 박스가 덜 맞음
        else:
            kinds["배경 오탐"] += 1     # 정답 사람과 거의 안 겹침(라벨 없는 사람일 수도)
    return kinds, [k for k, m in enumerate(matched) if not m]


def main() -> int:
    require_lab5()
    device = pick_device()
    model = YOLO(str(BEST))
    images = valid_images()
    truth_by_image = {p: read_truth(p) for p in images}
    all_truth = [b for boxes in truth_by_image.values() for b in boxes]
    print("정답 크기:", dict(Counter(size_name(b) for b in all_truth)), "합계", len(all_truth))
    for conf in CONFS:
        total, missed_sizes, worst = Counter(), Counter(), []
        for path in images:
            result = model.predict(str(path), imgsz=IMAGE_SIZE, conf=conf, device=device, verbose=False)[0]
            truth = torch.tensor(truth_by_image[path]).reshape(-1, 4)
            kinds, missed = classify(result.boxes.xyxy.cpu(), truth)
            total += kinds
            missed_sizes.update(size_name(truth_by_image[path][k]) for k in missed)
            errors = len(missed) + sum(v for k, v in kinds.items() if k != "맞음")
            worst.append((errors, path.name, len(missed), len(result.boxes), len(truth)))
        n_missed = sum(missed_sizes.values())
        print(f"conf {conf}: {dict(total)}, 놓침 {n_missed} {dict(missed_sizes)}")
        top = sorted(worst, reverse=True)[:3]
        print("   오류가 많은 사진:", ", ".join(f"{name}(오류 {e}, 놓침 {m}, 탐지 {d}/정답 {t})" for e, name, m, d, t in top))
    return 0


if __name__ == "__main__":
    sys.exit(main())
