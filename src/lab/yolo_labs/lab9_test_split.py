"""8차시 실습: 고르는 데 쓴 데이터와 성적을 매기는 데이터를 나눈다(학습 / 검증 / 시험).

lab5의 학습 136장을 다시 학습 102장, 검증 34장으로 나누고, lab5의 검증 34장은 '시험'으로만 쓴다.
best.pt는 검증으로 고르므로 검증 점수는 낙관적이다. 시험 점수와 얼마나 다른지 잰다.
"""
import random
import shutil
import sys
import time

import yaml
from ultralytics import YOLO

from common import load_split
from common_yolo import HERE, IMAGE_SIZE, pick_device
from lab5_yolo11 import BATCH, SEED, WEIGHTS, to_yolo_label

SPLIT_ROOT = HERE / "data" / "PennFudan_3way"
RUN = HERE / "runs" / "person_3way"
EPOCHS = 50
VALID_SHARE = 0.25   # 학습 136장 가운데 검증으로 뗄 몫


def write_split(name: str, samples: list[dict]) -> None:
    for sub in ("images", "labels"):
        (SPLIT_ROOT / name / sub).mkdir(parents=True)
    for s in samples:
        shutil.copy(s["path"], SPLIT_ROOT / name / "images" / s["path"].name)
        rows = [" ".join(["0"] + [f"{v:.6f}" for v in to_yolo_label(b, s["width"], s["height"])]) for b in s["boxes"]]
        (SPLIT_ROOT / name / "labels" / f"{s['path'].stem}.txt").write_text("\n".join(rows) + "\n")


def main() -> int:
    device = pick_device()
    train_all, test = load_split()
    shuffled = train_all.copy()
    random.Random(SEED).shuffle(shuffled)
    n_valid = int(len(shuffled) * VALID_SHARE)
    valid, train = shuffled[:n_valid], shuffled[n_valid:]
    shutil.rmtree(SPLIT_ROOT, ignore_errors=True)
    for name, part in (("train", train), ("valid", valid), ("test", test)):
        write_split(name, part)
    names = {n: {s["path"].name for s in p} for n, p in (("train", train), ("valid", valid), ("test", test))}
    assert not (names["train"] & names["valid"]) and not (names["train"] & names["test"]) and not (names["valid"] & names["test"])
    print(f"학습 {len(train)} / 검증 {len(valid)} / 시험 {len(test)}  (겹치는 사진 없음)")
    config = {"path": str(SPLIT_ROOT), "train": "train/images", "val": "valid/images", "test": "test/images", "nc": 1, "names": ["person"]}
    (SPLIT_ROOT / "data.yaml").write_text(yaml.safe_dump(config))

    start = time.time()
    YOLO(WEIGHTS).train(data=str(SPLIT_ROOT / "data.yaml"), imgsz=IMAGE_SIZE, batch=BATCH, epochs=EPOCHS, device=device,
                        project=str(RUN.parent), name=RUN.name, exist_ok=True, seed=SEED, plots=False, verbose=False)
    print(f"학습 {time.time() - start:.0f}초")
    for weights in ("best.pt", "last.pt"):
        model = YOLO(str(RUN / "weights" / weights))
        row = []
        for split in ("val", "test"):
            box = model.val(data=str(SPLIT_ROOT / "data.yaml"), split=split, imgsz=IMAGE_SIZE, plots=False, device=device, verbose=False).box
            row.append(f"{'검증' if split == 'val' else '시험'} mAP50 {box.map50:.3f} / mAP50-95 {box.map:.3f}")
        print(f"{weights:8s}: " + " | ".join(row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
