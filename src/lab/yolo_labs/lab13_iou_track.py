"""12차시: IoU로 이어 붙이는 가장 단순한 추적기.

앞 장면의 상자와 이번 장면의 상자 가운데 IoU 합이 가장 큰 짝(헝가리안 방법)을 같은 사람으로 본다.
짝이 없는 검출은 새 사람, 짝이 없는 트랙은 그 자리에서 버린다(기억 없음).
"""
import numpy as np

from common_track import evaluate, iou_matrix, match, run_tracker, show, synth, synth_detections

CONF = 0.5        # 이 점수 이상의 검출만 쓴다
MIN_IOU = 0.3     # 이보다 덜 겹치면 같은 사람으로 보지 않는다


class IoUTracker:
    def __init__(self, conf: float = CONF, min_iou: float = MIN_IOU):
        self.conf, self.min_iou = conf, min_iou
        self.tracks: dict[int, np.ndarray] = {}     # ID → 마지막 상자
        self.next_id = 1

    def update(self, dets: np.ndarray) -> list[tuple[int, list[float]]]:
        boxes = dets[dets[:, 4] >= self.conf, :4]
        ids = list(self.tracks)
        old = np.array([self.tracks[i] for i in ids]).reshape(-1, 4)
        pairs, _, new = match(iou_matrix(old, boxes), self.min_iou)
        self.tracks = {ids[r]: boxes[c] for r, c in pairs}          # 짝이 없는 트랙은 사라진다
        for c in new:
            self.tracks[self.next_id] = boxes[c]
            self.next_id += 1
        return [(i, b.tolist()) for i, b in self.tracks.items()]


def main() -> None:
    _, truth = synth()
    dets = synth_detections()
    tracks = run_tracker(IoUTracker(), dets)
    show("IoU 추적", evaluate(truth, tracks))
    for min_iou in (0.1, 0.5):
        show(f"  IoU≥{min_iou}", evaluate(truth, run_tracker(IoUTracker(min_iou=min_iou), dets)))
    lost = sum(1 for a, b in zip(tracks, tracks[1:]) for i, _ in a if i not in {j for j, _ in b})
    print(f"\n트랙이 끊긴 횟수 {lost}번. 한 장면만 못 찾아도, 기둥 뒤로 잠깐 가려져도 그 사람은 새 ID로 다시 태어난다.")


if __name__ == "__main__":
    main()
