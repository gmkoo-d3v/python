"""19차시: 겉모습으로 같은 사람 찾기(수요일 추적 영상에 DINOv3 더하기).

3일차 추적기는 위치와 움직임만 봤다. 그래서 기둥 뒤로 오래 가렸다 나온 사람은 새 ID를 받았다.
1) 알아보기: 사람마다 처음 몇 장면의 모습을 '사진첩'에 넣고, 나중 장면의 모습이 사진첩의 누구와 가장
   닮았는지 맞혀 본다. DINOv3 특징과 색 히스토그램(HSV)을 견준다. 나중 모습을 그늘에 들어간 것처럼
   어둡고 푸르게 바꿔서도 해 본다(조명이 바뀌는 실제 현장).
2) 이어 붙이기: 추적기가 끊어 놓은 트랙 가운데, 앞 트랙이 끝나고 GAP 장면 안에, 걷던 빠르기로 갔을
   자리 근처(REACH 픽셀)에서 시작한 트랙이 겉모습도 THRESHOLD보다 닮았으면 같은 ID로 잇는다.
lab5(best.pt)와 3일차 공통 코드(common_track.py, lab14·15)를 씁니다.
"""
import cv2
import numpy as np
from PIL import Image

from common_dino import load
from common_dino import embed as dino_embed
from common_track import PEOPLE, WIDTH, evaluate, run_tracker, show, synth, synth_detections
from lab14_sort import SortTracker
from lab15_bytetrack import ByteTracker

GALLERY_SHOTS = 3        # 사진첩에 넣는 처음 모습 수(사람마다)
LATER = 30               # 사진첩 마지막 장면에서 이만큼 지난 뒤의 모습을 질문으로
QUERIES = 10             # 사람마다 질문 수
GAP = 50                 # 앞 트랙이 끝나고 이 장면 수 안에 시작한 트랙만 이을 후보
REACH = 60               # 걷던 대로 갔을 자리에서 이만큼(픽셀) 안에서 시작해야 후보
THRESHOLD = 0.8          # 트랙 평균 특징의 코사인 유사도가 이보다 크면 잇는다
CROPS_PER_TRACK = 10
PERSON_SIZE = (256, 128) # 사람 조각은 세로로 길게(정사각으로 늘이지 않는다)
SHADE = (0.55, 0.6, 0.75)    # 그늘: R·G·B에 곱하는 값(어둡고 푸르게)


def crop(frame: np.ndarray, box: list[float]) -> Image.Image:
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
    return Image.fromarray(frame[y1:max(y2, y1 + 2), x1:max(x2, x1 + 2)])


def hsv_hist(images: list[Image.Image]) -> np.ndarray:
    """견줄 거리: 색상·채도·명도 8×8×8칸 히스토그램(길이 1로 맞춤)."""
    out = []
    for im in images:
        hsv = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2HSV)
        h = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256]).ravel()
        out.append(np.sqrt(h / max(h.sum(), 1)))
    return np.array(out)


def shade(image: Image.Image) -> Image.Image:
    a = np.asarray(image, np.float32) * np.array(SHADE)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def recognition(frames: list[np.ndarray], truth: list[list[dict]], embed, shaded: bool = False) -> tuple[float, int]:
    inside = lambda g: g["box"][0] > 2 and g["box"][2] < WIDTH - 2      # 화면 가장자리에 잘린 모습은 뺀다
    gallery, gallery_id, queries, query_id = [], [], [], []
    for pid in range(1, PEOPLE + 1):
        seen = [(t, g) for t, boxes in enumerate(truth) for g in boxes if g["id"] == pid and inside(g)]
        clear = [(t, g) for t, g in seen if g["visible"] >= 0.8] or seen   # 늘 가려진 사람은 보이는 대로
        if not clear:
            continue
        first = clear[:GALLERY_SHOTS]
        later = [(t, g) for t, g in seen if t >= first[-1][0] + LATER and g["visible"] >= 0.5]  # 없으면 질문도 없다
        for t, g in first:
            gallery.append(crop(frames[t], g["box"]))
            gallery_id.append(pid)
        for t, g in later[::max(1, len(later) // QUERIES)][:QUERIES]:
            view = crop(frames[t], g["box"])
            queries.append(shade(view) if shaded else view)
            query_id.append(pid)
    gal, qry = embed(gallery), embed(queries)
    guess = np.array(gallery_id)[np.argmax(qry @ gal.T, axis=1)]
    return float((guess == np.array(query_id)).mean()), len(queries)


def stitch(frames: list[np.ndarray], tracks: list[list[tuple[int, list[float]]]], embed) -> list[list[tuple[int, list[float]]]]:
    """끊긴 트랙을 겉모습으로 잇는다. 시작이 이른 트랙부터, 가장 닮은 '이미 끝난' 트랙에 붙인다."""
    spans: dict[int, list[tuple[int, list[float]]]] = {}
    for t, items in enumerate(tracks):
        for tid, box in items:
            spans.setdefault(tid, []).append((t, box))
    feature = {}
    for tid, seen in spans.items():
        picks = seen[::max(1, len(seen) // CROPS_PER_TRACK)][:CROPS_PER_TRACK]
        f = embed([crop(frames[t], b) for t, b in picks]).mean(axis=0)
        feature[tid] = f / np.linalg.norm(f)
    start = {tid: s[0][0] for tid, s in spans.items()}
    end = {tid: s[-1][0] for tid, s in spans.items()}
    cx = lambda box: (box[0] + box[2]) / 2

    def reachable(old: int, new: int) -> bool:
        """앞 트랙이 마지막 빠르기로 계속 걸었다면 새 트랙이 시작한 자리 근처(REACH 픽셀 안)에 와 있어야 한다."""
        seen = spans[old]
        back = seen[-min(len(seen), 5)]
        speed = (cx(seen[-1][1]) - cx(back[1])) / max(seen[-1][0] - back[0], 1)
        guess = cx(seen[-1][1]) + speed * (start[new] - end[old])
        return abs(cx(spans[new][0][1]) - guess) < REACH

    root, taken = {tid: tid for tid in spans}, set()
    for tid in sorted(spans, key=start.get):
        candidates = [p for p in spans if p not in taken and end[p] < start[tid] <= end[p] + GAP and reachable(p, tid)]
        if not candidates:
            continue
        best = max(candidates, key=lambda p: feature[p] @ feature[tid])
        if feature[best] @ feature[tid] > THRESHOLD:
            root[tid] = root[best]
            taken.add(best)
    return [[(root[tid], box) for tid, box in items] for items in tracks]


def main() -> None:
    frames, truth = synth()
    dets = synth_detections()
    model = load()
    dino = lambda ims: dino_embed(model, ims, size=PERSON_SIZE)
    print("1) 알아보기: 처음 모습(사람마다 3장)으로 만든 사진첩에서 나중 모습의 주인 맞히기")
    print(f"   {'':<8}{'그대로':>8}{'그늘에서':>10}   (아무렇게나 찍으면 {1 / PEOPLE:.3f})")
    for name, fn in [("DINOv3", dino), ("색", hsv_hist)]:
        same, n = recognition(frames, truth, fn)
        dark, _ = recognition(frames, truth, fn, shaded=True)
        print(f"   {name:<8}{same:>8.3f}{dark:>10.3f}   (질문 {n}장)")
    print(f"\n2) 이어 붙이기(앞 트랙이 끝나고 {GAP}장면 안, 걷던 자리에서 {REACH}픽셀 안, 코사인 유사도 > {THRESHOLD})")
    for name, tracker in [("SORT", SortTracker()), ("ByteTrack", ByteTracker())]:
        tracks = run_tracker(tracker, dets)
        show(name, evaluate(truth, tracks))
        show("  +DINOv3", evaluate(truth, stitch(frames, tracks, dino)))
        show("  +색", evaluate(truth, stitch(frames, tracks, hsv_hist)))


if __name__ == "__main__":
    main()
