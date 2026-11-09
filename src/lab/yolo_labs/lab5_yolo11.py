"""5차시 실습: ultralytics YOLO11로 사람 탐지 전이학습 (6강 노트북 5.4절, 실습 8).

라이선스: ultralytics는 AGPL-3.0이고, 약관상 직접 학습한 모델에도 적용됩니다.
이 파일은 교안 코드의 MIT 라이선스에서 제외하며 AGPL-3.0 조건을 따릅니다.
"""
import csv
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("YOLO_VERBOSE", "False")   # import 전에 정해야 적용된다
# MPS에서 결정적 구현이 없다는 경고가 배치마다 나와 출력을 덮으므로 그것만 끈다
warnings.filterwarnings("ignore", message=".*does not have a deterministic implementation.*")

import torch
import yaml
from ultralytics import YOLO
from ultralytics.engine.trainer import BaseTrainer

from common import load_split

WEIGHTS = "yolo11s.pt"
IMAGE_SIZE = 416
BATCH = 16
EPOCHS = int(os.environ.get("YOLO_EPOCHS", "50"))   # 수업 시간이 짧으면 YOLO_EPOCHS=10
PRINT_EVERY = 10
SEED = 1729
HERE = Path(__file__).resolve().parent
YOLO_ROOT = HERE / "data" / "PennFudan_yolo"
RUN_DIR = HERE / "runs"


def to_yolo_label(box: list[int], width: int, height: int) -> tuple[float, float, float, float]:
    """xyxy 픽셀 박스를 YOLO 형식(중심과 크기, 0~1)으로 바꿉니다."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height)


def yolo_to_xyxy(label: list[float], width: int, height: int) -> list[float]:
    """YOLO 형식(0~1)을 xyxy 픽셀 박스로 되돌립니다."""
    center_x, center_y, w, h = label
    return [(center_x - w / 2) * width, (center_y - h / 2) * height, (center_x + w / 2) * width, (center_y + h / 2) * height]


def write_dataset(train: list[dict], valid: list[dict]) -> Path:
    """images/labels 폴더와 data.yaml을 만듭니다. 라벨 폴더는 images를 labels로 바꾼 경로여야 한다."""
    shutil.rmtree(YOLO_ROOT, ignore_errors=True)   # 이전 실행의 파일이 섞이지 않게 한다
    for split, split_samples in (("train", train), ("valid", valid)):
        (YOLO_ROOT / split / "images").mkdir(parents=True)
        (YOLO_ROOT / split / "labels").mkdir(parents=True)
        for s in split_samples:
            shutil.copy(s["path"], YOLO_ROOT / split / "images" / s["path"].name)
            rows = [" ".join(["0"] + [f"{v:.6f}" for v in to_yolo_label(b, s["width"], s["height"])]) for b in s["boxes"]]
            (YOLO_ROOT / split / "labels" / f"{s['path'].stem}.txt").write_text("\n".join(rows) + "\n")
    config = {"path": str(YOLO_ROOT), "train": "train/images", "val": "valid/images", "nc": 1, "names": ["person"]}
    config_path = YOLO_ROOT / "data.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


def print_progress(trainer: BaseTrainer) -> None:
    """정해진 주기마다 검증 성능을 한 줄로 출력합니다."""
    epoch = trainer.epoch + 1
    if epoch % PRINT_EVERY == 0 or epoch == 1:
        print(f"epoch {epoch:3d} | mAP50 {trainer.metrics['metrics/mAP50(B)']:.3f} | mAP50-95 {trainer.metrics['metrics/mAP50-95(B)']:.3f}", flush=True)


def main() -> None:
    device = 0 if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    train, valid = load_split()
    print("train:", len(train), "/ valid:", len(valid), "/ device:", device)

    # 1) 라벨 한 줄 확인과 되돌리기 검사
    config_path = write_dataset(train, valid)
    first = train[0]
    lines = (YOLO_ROOT / "train" / "labels" / f"{first['path'].stem}.txt").read_text().strip().splitlines()
    print("라벨 파일", first["path"].stem + ".txt:", *lines, sep="\n  ")
    for line, box in zip(lines, first["boxes"]):
        restored = yolo_to_xyxy([float(v) for v in line.split()[1:]], first["width"], first["height"])
        assert max(abs(a - b) for a, b in zip(restored, box)) < 1.0, (restored, box)
    print("되돌린 박스가 원래 박스와 1픽셀 이내: 통과")

    # 2) 사전학습 모델(COCO 80클래스)로 검증 이미지 한 장
    pretrained = YOLO(WEIGHTS)
    print(f"사전학습 {WEIGHTS}: 클래스 {len(pretrained.names)}개, 파라미터 {sum(p.numel() for p in pretrained.model.parameters()) / 1e6:.1f}M")
    sample_image = str(YOLO_ROOT / "valid" / "images" / valid[0]["path"].name)
    result = pretrained(sample_image, verbose=False)[0]
    names = [result.names[int(c)] for c in result.boxes.cls]
    print(f"사전학습으로 {valid[0]['path'].name}: {dict((n, names.count(n)) for n in sorted(set(names)))} / 정답 사람 {len(valid[0]['boxes'])}명")

    # 3) 전이학습
    start = time.time()
    model = YOLO(WEIGHTS)
    model.add_callback("on_fit_epoch_end", print_progress)
    model.train(data=str(config_path), imgsz=IMAGE_SIZE, batch=BATCH, epochs=EPOCHS, device=device,
                project=str(RUN_DIR), name="person", exist_ok=True, seed=SEED, plots=False)
    print(f"학습 시간: {time.time() - start:.0f}초 ({EPOCHS} epoch)")

    # 4) best.pt를 명시적으로 불러 평가
    best = YOLO(str(RUN_DIR / "person" / "weights" / "best.pt"))
    metrics = best.val(data=str(config_path), imgsz=IMAGE_SIZE, plots=False, device=device, verbose=False)
    print(f"best.pt  mAP50 = {metrics.box.map50:.3f} / mAP50-95 = {metrics.box.map:.3f} / 클래스 {best.names}")
    with open(RUN_DIR / "person" / "results.csv") as f:
        rows = list(csv.DictReader(f))
    best_row = max(rows, key=lambda r: float(r["metrics/mAP50-95(B)"].strip()))
    print("가장 좋은 epoch:", int(float(best_row["epoch"])))

    # 5) 기준 점수와 탐지 수
    counts = {conf: len(best.predict(sample_image, conf=conf, verbose=False)[0].boxes) for conf in (0.25, 0.5, 0.9)}
    print("conf별 탐지 수:", counts, "/ 정답", len(valid[0]["boxes"]))
    assert counts[0.25] >= counts[0.5] >= counts[0.9]   # 기준을 높이면 줄어들기만 한다
    total = sum(len(r.boxes) for r in best.predict([str(YOLO_ROOT / "valid" / "images" / s["path"].name) for s in valid], conf=0.5, verbose=False, stream=True))
    print("검증 34장 conf 0.5 탐지 합계:", total, "/ 정답", sum(len(s["boxes"]) for s in valid))


if __name__ == "__main__":
    sys.exit(main())
