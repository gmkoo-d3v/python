"""6차시 실습: 숫자 하나 뒤에 있는 것 — IoU 기준별 AP, F1이 가장 좋은 conf, 도구마다 다른 mAP."""
import json
import sys
import tempfile

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO

from common_yolo import BEST, IMAGE_SIZE, YOLO_ROOT, pick_device, read_truth, require_lab5, valid_images

EVAL_CONF = 0.001   # AP를 재려면 점수가 낮은 예측까지 모두 있어야 한다


def ultralytics_view(model: YOLO, device: int | str) -> None:
    """ultralytics의 val이 내는 숫자를 한 겹씩 펼친다."""
    box = model.val(data=str(YOLO_ROOT / "data.yaml"), imgsz=IMAGE_SIZE, plots=False, device=device, verbose=False).box
    ious = np.linspace(0.5, 0.95, 10)
    print("IoU 기준별 AP:", "  ".join(f"{t:.2f}:{ap:.3f}" for t, ap in zip(ious, box.all_ap[0])))
    print(f"mAP50 {box.map50:.3f} = 첫 칸 / mAP50-95 {box.map:.3f} = 열 칸의 평균 / mAP75 {box.map75:.3f}")
    best = int(box.f1_curve[0].argmax())
    print(f"F1이 가장 높은 conf {box.px[best]:.3f}: 정밀도 {box.p_curve[0][best]:.3f}, 재현율 {box.r_curve[0][best]:.3f}, F1 {box.f1_curve[0][best]:.3f}")
    for conf in (0.25, 0.5, 0.9):
        k = int(np.searchsorted(box.px, conf))
        print(f"  conf {conf}: 정밀도 {box.p_curve[0][k]:.3f}, 재현율 {box.r_curve[0][k]:.3f}, F1 {box.f1_curve[0][k]:.3f}")


def coco_view(model: YOLO, device: int | str) -> None:
    """같은 모델을 pycocotools(COCO 공식 평가 코드)로 다시 잰다."""
    images, annotations, detections = [], [], []
    for image_id, path in enumerate(valid_images(), start=1):
        result = model.predict(str(path), imgsz=IMAGE_SIZE, conf=EVAL_CONF, device=device, verbose=False)[0]
        height, width = result.orig_shape
        images.append({"id": image_id, "file_name": path.name, "width": width, "height": height})
        for x1, y1, x2, y2 in read_truth(path):
            annotations.append({"id": len(annotations) + 1, "image_id": image_id, "category_id": 1,
                                "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
        for (x1, y1, x2, y2), score in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
            detections.append({"image_id": image_id, "category_id": 1, "bbox": [x1, y1, x2 - x1, y2 - y1], "score": score})
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "person"}]}, f)
    truth = COCO(f.name)
    evaluator = COCOeval(truth, truth.loadRes(detections), "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    print(f"pycocotools  AP50 {evaluator.stats[1]:.3f} / AP50-95 {evaluator.stats[0]:.3f} / 예측 {len(detections)}개, 정답 {len(annotations)}명")
    sizes = sorted((a["bbox"][3] for a in annotations))
    print(f"정답 박스 높이(px): 가장 작은 {sizes[0]:.0f}, 가운데 {sizes[len(sizes) // 2]:.0f}, 가장 큰 {sizes[-1]:.0f}")


def main() -> int:
    require_lab5()
    device = pick_device()
    model = YOLO(str(BEST))
    print("== ultralytics val ==")
    ultralytics_view(model, device)
    print("== pycocotools ==")
    coco_view(model, device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
