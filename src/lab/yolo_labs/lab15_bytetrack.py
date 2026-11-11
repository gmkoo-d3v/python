"""14차시: ByteTrack, 그리고 세 추적기 견주기.

SORT는 점수가 낮은 검출을 버린다. 그런데 반쯤 가려진 사람은 점수가 낮게 나온다.
ByteTrack은 짝짓기를 두 번 한다.
  1차: 점수 높은 검출(≥ HIGH)과 모든 트랙
  2차: 1차에서 짝을 못 찾은 트랙과 점수 낮은 검출(LOW ~ HIGH)
새 트랙은 점수가 아주 높은 검출(≥ NEW)에서만 만든다. 낮은 검출은 '이미 따라가던 사람'을 잇는 데만 쓴다.
"""
import numpy as np

from common_track import evaluate, iou_matrix, match, run_tracker, show, synth, synth_detections
from lab13_iou_track import IoUTracker
from lab14_sort import SortTracker

HIGH = 0.5
LOW = 0.1
NEW = 0.6
FIRST_IOU = 0.2       # 1차 짝짓기(원 논문의 match_thresh 0.8 = 1 - IoU)
SECOND_IOU = 0.5      # 2차 짝짓기는 더 엄격하게(낮은 검출에는 헛검출이 섞여 있다)
MAX_AGE = 30          # 원 논문의 track_buffer(30장면)


class ByteTracker(SortTracker):
    def __init__(self, high: float = HIGH, low: float = LOW, new: float = NEW, max_age: int = MAX_AGE):
        super().__init__(conf=high, max_age=max_age, min_hits=1)
        self.high, self.low, self.new = high, low, new

    def update(self, dets: np.ndarray) -> list[tuple[int, list[float]]]:
        high = dets[dets[:, 4] >= self.high]
        low = dets[(dets[:, 4] >= self.low) & (dets[:, 4] < self.high)]
        predicted = np.array([t.predict() for t in self.tracks]).reshape(-1, 4)
        pairs, left_tracks, left_high = match(iou_matrix(predicted, high[:, :4]), FIRST_IOU)
        for r, c in pairs:
            self.tracks[r].correct(high[c, :4])
        # 2차: 바로 앞 장면까지 보던 트랙만(오래 잃은 트랙에 낮은 검출을 붙이면 엉뚱한 사람과 이어진다)
        recent = [r for r in left_tracks if self.tracks[r].missed == 1]
        pairs2, _, _ = match(iou_matrix(predicted[recent], low[:, :4]), SECOND_IOU)
        for r, c in pairs2:
            self.tracks[recent[r]].correct(low[c, :4])
        for c in left_high:
            if high[c, 4] >= self.new:
                self.new_track(high[c, :4])
        self.tracks = [t for t in self.tracks if t.missed <= self.max_age]
        return self.output()


def main() -> None:
    _, truth = synth()
    dets = synth_detections()
    results = {
        "IoU 추적": evaluate(truth, run_tracker(IoUTracker(), dets)),
        "SORT": evaluate(truth, run_tracker(SortTracker(), dets)),
        "ByteTrack": evaluate(truth, run_tracker(ByteTracker(), dets)),
    }
    for name, m in results.items():
        show(name, m)
    print()
    print("2차 짝짓기를 끄면(낮은 검출을 쓰지 않으면)")
    show("  LOW=HIGH", evaluate(truth, run_tracker(ByteTracker(low=HIGH), dets)))


if __name__ == "__main__":
    main()
