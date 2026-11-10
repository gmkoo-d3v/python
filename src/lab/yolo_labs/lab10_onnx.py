"""9차시 실습: best.pt를 ONNX로 내보내고, ultralytics 없이 onnxruntime만으로 박스를 낸다.

출력 (1, 5, 3549)을 손으로 풀어 본다: 3549 = 52² + 26² + 13² 칸, 5 = 박스 4 + 사람 점수 1(앵커 없음, objectness 없음).
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from torchvision.ops import box_iou, nms
from ultralytics import YOLO

from common_yolo import BEST, IMAGE_SIZE, pick_device, require_lab5, valid_images

CONF = 0.5
NMS_IOU = 0.7      # ultralytics 기본값과 같게
PAD_VALUE = 114    # ultralytics가 빈 곳을 채우는 회색


def letterbox(path: Path) -> tuple[np.ndarray, float, tuple[int, int]]:
    """긴 변을 416에 맞춰 줄이고, 남는 곳을 가운데 정렬로 회색으로 채운다(ultralytics와 같은 cv2 방식)."""
    bgr = cv2.imread(str(path))
    height, width = bgr.shape[:2]
    scale = IMAGE_SIZE / max(width, height)
    new_w, new_h = round(width * scale), round(height * scale)
    left, top = round((IMAGE_SIZE - new_w) / 2 - 0.1), round((IMAGE_SIZE - new_h) / 2 - 0.1)
    canvas = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), PAD_VALUE, dtype=np.uint8)
    canvas[top:top + new_h, left:left + new_w] = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    rgb = canvas[..., ::-1]   # 모델은 RGB를 받는다
    array = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0   # (1, 3, 416, 416), 0~1
    return array, scale, (left, top)


def decode(output: np.ndarray, scale: float, pad: tuple[float, float]) -> tuple[torch.Tensor, torch.Tensor]:
    """(1, 5, N) → 점수로 거르고, 중심·크기를 xyxy로, 패딩과 크기 조절을 되돌리고, NMS."""
    pred = torch.from_numpy(output[0]).T          # (N, 5): cx, cy, w, h, 사람 점수
    pred = pred[pred[:, 4] >= CONF]
    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad[0]) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad[1]) / scale
    keep = nms(boxes, pred[:, 4], NMS_IOU)
    return boxes[keep], pred[keep, 4]


def main() -> int:
    require_lab5()
    onnx_path = Path(YOLO(str(BEST)).export(format="onnx", imgsz=IMAGE_SIZE, dynamic=False, simplify=True, verbose=False))
    # best.pt는 가중치를 float16으로, ONNX는 float32로 저장해서 크기가 두 배쯤 된다
    print(f"파일 크기: best.pt {BEST.stat().st_size / 2**20:.1f}MiB → {onnx_path.name} {onnx_path.stat().st_size / 2**20:.1f}MiB")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp, out = session.get_inputs()[0], session.get_outputs()[0]
    print(f"입력 {inp.name} {inp.shape} / 출력 {out.name} {out.shape}")

    reference = YOLO(str(onnx_path), task="detect")   # 같은 ONNX를 ultralytics가 돌린 결과와 견준다
    pt_model = YOLO(str(BEST))
    total_mine = total_ref = total_pt = 0
    shifts, unmatched = [], 0
    for path in valid_images():
        array, scale, pad = letterbox(path)
        boxes, _ = decode(session.run(None, {inp.name: array})[0], scale, pad)
        ref = reference.predict(str(path), imgsz=IMAGE_SIZE, conf=CONF, iou=NMS_IOU, device="cpu", verbose=False)[0].boxes.xyxy.cpu()
        pt = pt_model.predict(str(path), imgsz=IMAGE_SIZE, conf=CONF, iou=NMS_IOU, device=pick_device(), verbose=False)[0].boxes
        total_mine, total_ref, total_pt = total_mine + len(boxes), total_ref + len(ref), total_pt + len(pt)
        if len(boxes) and len(ref):
            # 내 박스마다 가장 많이 겹치는 ultralytics 박스를 찾아 꼭짓점 차이(픽셀)를 잰다
            best_iou, nearest = box_iou(boxes, ref).max(dim=1)
            same = best_iou >= 0.9
            shifts += (boxes[same] - ref[nearest[same]]).abs().amax(dim=1).tolist()
            unmatched += int((~same).sum())
        else:
            unmatched += len(boxes)
    print(f"검증 34장, conf {CONF}: 직접 푼 ONNX {total_mine}개 / ultralytics의 ONNX {total_ref}개 / ultralytics의 best.pt {total_pt}개")
    print(f"짝지은 박스 {len(shifts)}개의 꼭짓점 차이: 가장 큰 {max(shifts):.3f}px, 평균 {np.mean(shifts):.4f}px / 짝 없는 박스 {unmatched}개")

    array, _, _ = letterbox(valid_images()[0])
    for _ in range(3):
        session.run(None, {inp.name: array})   # 처음 몇 번은 준비 시간이 섞인다
    start = time.perf_counter()
    for _ in range(50):
        session.run(None, {inp.name: array})
    print(f"onnxruntime CPU 추론만: {(time.perf_counter() - start) / 50 * 1000:.1f}ms/장 (기본 설정)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
