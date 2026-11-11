"""13차시: 예측하며 따라가기(SORT).

트랙마다 칼만 필터로 '다음 장면에 어디쯤 있을지'를 예측하고, 예측한 상자와 검출을 IoU로 짝짓는다.
짝을 못 찾은 트랙은 바로 버리지 않고 max_age 장면 동안 예측만으로 기다린다.
그래서 잠깐 놓치거나 기둥 뒤로 가려져도 같은 ID로 다시 잡을 수 있다.
"""
import numpy as np

from common_track import evaluate, iou_matrix, match, run_tracker, show, synth, synth_detections

CONF = 0.5
MIN_IOU = 0.3
MAX_AGE = 15      # 이만큼(1.5초) 못 찾으면 트랙을 버린다
MIN_HITS = 2      # 이만큼 잇달아 찾아야 내보낸다(한 번 튄 헛검출은 ID를 받지 않게)


def to_state(box: np.ndarray) -> np.ndarray:
    """xyxy 상자 → (가운데 x, 가운데 y, 너비, 높이)."""
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[2] - box[0], box[3] - box[1]])


def to_box(state: np.ndarray) -> np.ndarray:
    cx, cy, w, h = state[:4]
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


class KalmanTrack:
    """상태 8칸: 가운데·크기 4칸과 그 빠르기 4칸. 한 장면에 빠르기만큼 움직인다고 본다(등속)."""

    F = np.eye(8) + np.eye(8, k=4)                 # 다음 위치 = 지금 위치 + 빠르기
    H = np.eye(4, 8)                               # 재는 것은 위치·크기 4칸뿐
    Q = np.diag([1, 1, 1, 1, 0.5, 0.5, 0.1, 0.1]) ** 2     # 움직임이 등속에서 벗어나는 정도
    R = np.diag([4, 4, 8, 8]) ** 2                         # 검출 상자의 흔들림(픽셀)

    def __init__(self, box: np.ndarray, track_id: int):
        self.x = np.r_[to_state(box), np.zeros(4)]
        self.P = np.diag([10, 10, 10, 10, 100, 100, 100, 100]) ** 2.0
        self.id, self.hits, self.missed = track_id, 1, 0

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.missed += 1
        return to_box(self.x)

    def correct(self, box: np.ndarray) -> None:
        y = to_state(box) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P
        self.hits += 1
        self.missed = 0

    def box(self) -> np.ndarray:
        return to_box(self.x)


class SortTracker:
    def __init__(self, conf: float = CONF, min_iou: float = MIN_IOU, max_age: int = MAX_AGE, min_hits: int = MIN_HITS):
        self.conf, self.min_iou, self.max_age, self.min_hits = conf, min_iou, max_age, min_hits
        self.tracks: list[KalmanTrack] = []
        self.next_id = 1

    def new_track(self, box: np.ndarray) -> None:
        self.tracks.append(KalmanTrack(box, self.next_id))
        self.next_id += 1

    def output(self) -> list[tuple[int, list[float]]]:
        """이번 장면에 검출로 확인했고, 충분히 여러 번 본 트랙만 내보낸다."""
        return [(t.id, t.box().tolist()) for t in self.tracks if t.missed == 0 and t.hits >= self.min_hits]

    def update(self, dets: np.ndarray) -> list[tuple[int, list[float]]]:
        boxes = dets[dets[:, 4] >= self.conf, :4]
        predicted = np.array([t.predict() for t in self.tracks]).reshape(-1, 4)
        pairs, _, new = match(iou_matrix(predicted, boxes), self.min_iou)
        for r, c in pairs:
            self.tracks[r].correct(boxes[c])
        for c in new:
            self.new_track(boxes[c])
        self.tracks = [t for t in self.tracks if t.missed <= self.max_age]
        return self.output()


def main() -> None:
    from lab13_iou_track import IoUTracker
    _, truth = synth()
    dets = synth_detections()
    show("IoU 추적", evaluate(truth, run_tracker(IoUTracker(), dets)))
    show("SORT", evaluate(truth, run_tracker(SortTracker(), dets)))
    print()
    for max_age in (0, 5, 15, 30):
        show(f"  max_age {max_age}", evaluate(truth, run_tracker(SortTracker(max_age=max_age), dets)))


if __name__ == "__main__":
    main()
