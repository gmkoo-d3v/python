"""5차시 더 해 보기: 학습한 best.pt로 동영상 탐지 (6강 노트북 5.4절 '동영상 탐지').

lab5_yolo11.py를 먼저 돌려 runs/person/weights/best.pt가 있어야 합니다.
라이선스: ultralytics(AGPL-3.0)를 쓰므로 교안 코드의 MIT에서 제외합니다.
"""
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("YOLO_VERBOSE", "False")   # import 전에 정해야 적용된다
warnings.filterwarnings("ignore", message=".*does not have a deterministic implementation.*")

import cv2
import torch
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
BEST = HERE / "runs" / "person" / "weights" / "best.pt"
VALID = HERE / "data" / "PennFudan_yolo" / "valid"
VIDEO = HERE / "runs" / "demo.mp4"
VIDEO_SIZE = (640, 480)   # 영상 프레임 크기. 원래 사진과 가로세로 비율이 다르다
VIDEO_FPS = 5
VIDEO_FRAMES = 20
CONF = 0.4
SIZES = (416, 1280)       # 학습 크기 그대로, 그리고 노트북처럼 크게


def make_video(images: list[Path]) -> None:
    """검증 사진을 이어 붙여 짧은 동영상을 만듭니다. 내 동영상이 없어도 돌아가게 하려는 것."""
    writer = cv2.VideoWriter(str(VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, VIDEO_SIZE)
    for path in images:
        writer.write(cv2.resize(cv2.imread(str(path)), VIDEO_SIZE))
    writer.release()


def label_count(image: Path) -> int:
    """그 사진의 YOLO 라벨 줄 수 = 정답 사람 수."""
    return len((VALID / "labels" / f"{image.stem}.txt").read_text().strip().splitlines())


def main() -> int:
    if not BEST.exists():
        print("best.pt가 없습니다. lab5_yolo11.py를 먼저 돌리세요:", BEST)
        return 1
    device = 0 if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    images = sorted((VALID / "images").glob("*.png"))[:VIDEO_FRAMES]
    make_video(images)
    truth = [label_count(p) for p in images]
    print(f"동영상: {len(images)}프레임, {VIDEO_SIZE[0]}x{VIDEO_SIZE[1]}, {VIDEO_FPS}fps / device: {device}")
    print("프레임별 정답 수:", truth, "합계", sum(truth))

    model = YOLO(str(BEST))
    for size in SIZES:
        # 첫 프레임은 준비 시간이 섞이므로 한 번 미리 돌린다
        next(iter(model.predict(source=str(VIDEO), imgsz=size, conf=CONF, stream=True, verbose=False, device=device)))
        start = time.time()
        # stream=True: 프레임 결과를 쌓지 않고 하나씩 받는다. 긴 동영상에서도 메모리가 넘치지 않는다
        counts = [len(r.boxes) for r in model.predict(source=str(VIDEO), imgsz=size, conf=CONF, stream=True, verbose=False, device=device)]
        elapsed = time.time() - start
        exact = sum(c == t for c, t in zip(counts, truth))
        print(f"imgsz {size:4d}: 프레임별 탐지 수 {counts} 합계 {sum(counts)} / 정답과 같은 프레임 {exact}/{len(truth)} / {len(counts) / elapsed:.1f} 프레임/초")
    return 0


if __name__ == "__main__":
    sys.exit(main())
