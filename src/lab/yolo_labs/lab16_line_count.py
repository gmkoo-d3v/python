"""15차시: 선 넘기 세기, 그리고 내 영상.

트랙 ID가 있으면 '세로선을 왼쪽→오른쪽으로 몇 명, 오른쪽→왼쪽으로 몇 명 넘었나'를 셀 수 있다.
합성 영상에서는 정답과 견주고, ByteTrack 결과를 그려 runs/track/tracked.mp4로 남긴다.

내 영상에 돌리기(조별로 휴대폰이나 Jetson 카메라로 찍은 사람이 지나가는 영상):
    uv run --no-project --with ultralytics --with scipy python lab16_line_count.py 내영상.mp4
"""
import sys
from pathlib import Path

import cv2
import numpy as np

from common_track import (OUT, PEOPLE, detect_frames, evaluate, load_video, run_tracker, save_video, show, synth,
                          synth_detections)
from lab13_iou_track import IoUTracker
from lab14_sort import SortTracker
from lab15_bytetrack import ByteTracker

LINE_X = 200              # 합성 영상의 세는 선(기둥 왼쪽)
COLORS = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180),
          (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (170, 110, 40)]


def count_crossings(frames_of_items: list[list[tuple[int, list[float]]]], line_x: float) -> tuple[int, int]:
    """ID마다 마지막으로 본 가운데 x를 기억해 두고, 선의 반대편에서 다시 보이면 한 번 넘은 것으로 센다."""
    last_x: dict[int, float] = {}
    right, left = 0, 0
    for items in frames_of_items:
        for track_id, box in items:
            cx = (box[0] + box[2]) / 2
            if track_id in last_x:
                if last_x[track_id] < line_x <= cx:
                    right += 1
                elif cx < line_x <= last_x[track_id]:
                    left += 1
            last_x[track_id] = cx
    return right, left


def draw(frames: list[np.ndarray], tracks: list[list[tuple[int, list[float]]]], line_x: int) -> list[np.ndarray]:
    """상자와 ID, 세는 선, 지금까지 넘은 수를 그린다."""
    drawn, seen = [], []
    for frame, items in zip(frames, tracks):
        seen.append(items)
        right, left = count_crossings(seen, line_x)
        canvas = frame.copy()
        cv2.line(canvas, (line_x, 0), (line_x, canvas.shape[0]), (255, 255, 255), 2)
        for track_id, box in items:
            x1, y1, x2, y2 = (int(v) for v in box)
            color = COLORS[track_id % len(COLORS)]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(canvas, str(track_id), (x1, max(y1 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(canvas, f"-> {right}   <- {left}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        drawn.append(canvas)
    return drawn


def synthetic() -> None:
    frames, truth = synth()
    dets = synth_detections()
    truth_items = [[(g["id"], g["box"]) for g in boxes] for boxes in truth]
    right, left = count_crossings(truth_items, LINE_X)
    print(f"선 x={LINE_X}을 넘은 사람(정답): → {right}명, ← {left}명 (영상 속 사람 {PEOPLE}명)")
    for name, tracker in [("IoU 추적", IoUTracker()), ("SORT", SortTracker()), ("ByteTrack", ByteTracker())]:
        tracks = run_tracker(tracker, dets)
        r, l = count_crossings(tracks, LINE_X)
        print(f"  {name:<10} → {r}명, ← {l}명   ", end="")
        show("", evaluate(truth, tracks))
    tracks = run_tracker(ByteTracker(), dets)
    save_video(draw(frames, tracks, LINE_X), OUT / "tracked.mp4")
    print(f"\nByteTrack 결과 영상: {OUT / 'tracked.mp4'}")


def my_video(path: Path) -> None:
    frames, fps = load_video(path)
    line_x = frames[0].shape[1] // 2
    print(f"{path.name}: {len(frames)}장면, {fps:.0f}fps, 크기 {frames[0].shape[1]}x{frames[0].shape[0]}, 선 x={line_x}(가운데)")
    dets = detect_frames(frames)
    tracks = run_tracker(ByteTracker(), dets)
    right, left = count_crossings(tracks, line_x)
    ids = {i for items in tracks for i, _ in items}
    print(f"ByteTrack: 만든 ID {len(ids)}개, 선을 → {right}명, ← {left}명 넘음")
    out = OUT / f"{path.stem}_tracked.mp4"
    save_video(draw(frames, tracks, line_x), out, fps=int(round(fps)))
    print(f"결과 영상: {out}  (정답이 없으니 눈으로 세어 견줘 보세요)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        my_video(Path(sys.argv[1]))
    else:
        synthetic()
