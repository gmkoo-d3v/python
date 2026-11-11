"""3일차(추적) 공통 부분: 정답 ID가 있는 합성 영상, 검출 캐시, 추적 지표(MOTA·IDF1·ID 바뀜).

합성 영상은 Penn-Fudan 검증 사진의 사람을 마스크로 오려 내 배경 위를 걷게 만든다.
누가 누구인지 우리가 알고 있으니, 추적기가 ID를 몇 번 바꿨는지 숫자로 셀 수 있다.
"""
import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("YOLO_VERBOSE", "False")   # ultralytics import 전에 정해야 적용된다
warnings.filterwarnings("ignore", message=".*does not have a deterministic implementation.*")

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from common_yolo import BEST, IMAGE_SIZE, YOLO_ROOT, pick_device, require_lab5

HERE = Path(__file__).resolve().parent
PENN = HERE / "data" / "PennFudanPed"
OUT = HERE / "runs" / "track"
VIDEO = OUT / "synth.mp4"
TRUTH = OUT / "truth.json"
DETS = OUT / "dets.npz"

WIDTH, HEIGHT = 640, 360
FPS = 10
FRAMES = 160
PEOPLE = 12
SEED = 11
PILLAR = (392, 432)        # 기둥의 x 범위. 그 뒤를 지나는 사람은 잠깐 완전히 가려진다
MIN_VISIBLE = 0.3          # 보이는 부분이 이보다 적으면 정답에서 뺀다(사람도 못 알아볼 만큼 가려짐)
MIN_TRUTH_HEIGHT = 40
DET_CONF = 0.05            # 낮은 점수 검출도 남겨 둔다(ByteTrack이 쓴다)
MATCH_IOU = 0.5            # 지표에서 "같은 사람을 짚었다"로 칠 IoU


# ---------- 합성 영상 ----------

def _sprites() -> list[np.ndarray]:
    """검증 사진의 사람을 마스크로 오려 RGBA 조각으로 만든다. 사진 가장자리에 잘린 사람은 뺀다."""
    sprites = []
    for image in sorted((YOLO_ROOT / "valid" / "images").glob("*.png")):
        rgb = np.array(Image.open(PENN / "PNGImages" / image.name).convert("RGB"))
        mask = np.array(Image.open(PENN / "PedMasks" / f"{image.stem}_mask.png"))
        for k in range(1, int(mask.max()) + 1):
            ys, xs = np.nonzero(mask == k)
            if len(ys) == 0:
                continue
            y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            touches = x1 == 0 or y1 == 0 or x2 == mask.shape[1] or y2 == mask.shape[0]
            if touches or y2 - y1 < 150:
                continue
            alpha = (mask[y1:y2, x1:x2] == k).astype(np.uint8) * 255
            sprites.append(np.dstack([rgb[y1:y2, x1:x2], alpha]))
    return sprites


def _background() -> np.ndarray:
    """보도처럼 보이는 배경: 위는 벽, 아래는 바닥, 약한 잡음."""
    rng = np.random.default_rng(SEED)
    bg = np.zeros((HEIGHT, WIDTH, 3), np.float32)
    rows = np.linspace(0, 1, HEIGHT)[:, None]
    bg[:] = (np.where(rows < 0.55, 150 + 30 * rows, 110 + 40 * rows))[..., None]
    bg[int(HEIGHT * 0.55):, :, 2] -= 15                     # 바닥은 조금 누렇게
    bg += rng.normal(0, 6, bg.shape)
    return np.clip(bg, 0, 255).astype(np.uint8)


def _plan_walkers(rng: np.random.Generator, count: int) -> list[dict]:
    """사람마다 걷는 방향, 빠르기, 발 높이(가까울수록 아래·크게), 나타나는 시각을 정한다."""
    walkers = []
    for i in range(count):
        foot = int(rng.choice([250, 275, 300, 330, 355]))     # 같은 줄을 마주 걷는 사람끼리 겹친다
        height = int(110 + (foot - 250) * 1.0)
        walkers.append({"id": i + 1, "dir": 1 if i % 2 == 0 else -1, "speed": float(rng.uniform(5, 11)),
                        "foot": foot, "height": height, "start": int(rng.integers(0, FRAMES - 70)),
                        "phase": float(rng.uniform(0, 6.28))})
    return walkers


def _place(walker: dict, sprite: np.ndarray, t: int) -> tuple[int, int, np.ndarray] | None:
    """t번째 장면에서 조각을 어디에 그릴지. 아직 안 나왔거나 이미 지나갔으면 None."""
    steps = t - walker["start"]
    if steps < 0:
        return None
    scale = walker["height"] / sprite.shape[0]
    piece = cv2.resize(sprite, (max(1, int(sprite.shape[1] * scale)), walker["height"]))
    w = piece.shape[1]
    x = -w + steps * walker["speed"] if walker["dir"] > 0 else WIDTH - steps * walker["speed"]
    if x > WIDTH or x < -w:
        return None
    if walker["dir"] < 0:
        piece = piece[:, ::-1]                               # 왼쪽으로 걷는 사람은 좌우를 뒤집는다
    bob = int(2 * np.sin(walker["phase"] + steps * 0.9))      # 걸을 때 오르내림
    return int(x), walker["foot"] - walker["height"] + bob, piece


def make_video() -> tuple[list[np.ndarray], list[list[dict]]]:
    """합성 영상의 장면들과 장면마다의 정답(ID, 보이는 부분의 상자, 보이는 비율)을 만든다."""
    sprites = _sprites()
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(sprites))[:PEOPLE]
    walkers = _plan_walkers(rng, PEOPLE)
    background = _background()
    frames, truth = [], []
    for t in range(FRAMES):
        frame = background.copy()
        owner = np.zeros((HEIGHT, WIDTH), np.int16)           # 픽셀마다 맨 앞에 보이는 사람의 ID
        full = {}
        for walker in sorted(walkers, key=lambda w: w["foot"]):    # 먼 사람부터 그린다
            placed = _place(walker, sprites[order[walker["id"] - 1]], t)
            if placed is None:
                continue
            x, y, piece = placed
            fx1, fy1 = max(x, 0), max(y, 0)
            fx2, fy2 = min(x + piece.shape[1], WIDTH), min(y + piece.shape[0], HEIGHT)
            if fx2 <= fx1 or fy2 <= fy1:
                continue
            crop = piece[fy1 - y:fy2 - y, fx1 - x:fx2 - x]
            alpha = crop[..., 3:4] / 255.0
            region = frame[fy1:fy2, fx1:fx2]
            frame[fy1:fy2, fx1:fx2] = (alpha * crop[..., :3] + (1 - alpha) * region).astype(np.uint8)
            owner[fy1:fy2, fx1:fx2][crop[..., 3] > 127] = walker["id"]
            full[walker["id"]] = int((crop[..., 3] > 127).sum())
        frame[:, PILLAR[0]:PILLAR[1]] = (70, 72, 78)          # 기둥은 사람 앞에 있다
        owner[:, PILLAR[0]:PILLAR[1]] = 0
        boxes = []
        for pid, total in full.items():
            ys, xs = np.nonzero(owner == pid)
            if len(ys) == 0 or len(ys) / total < MIN_VISIBLE or ys.max() - ys.min() < MIN_TRUTH_HEIGHT:
                continue
            boxes.append({"id": pid, "box": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                          "visible": round(len(ys) / total, 3)})
        frames.append(frame)
        truth.append(boxes)
    return frames, truth


def save_video(frames: list[np.ndarray], path: Path, fps: int = FPS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def load_video(path: Path) -> tuple[list[np.ndarray], float]:
    """동영상을 RGB 장면 목록으로 읽는다."""
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS) or FPS
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise SystemExit(f"동영상을 읽지 못했습니다: {path}")
    return frames, fps


def synth() -> tuple[list[np.ndarray], list[list[dict]]]:
    """합성 영상과 정답. 씨앗이 정해져 있어 매번 같은 장면이 나온다.

    mp4는 손실 압축이라 다시 읽으면 원본과 조금 다르므로, 검출은 늘 다시 그린 원본 장면에 한다.
    runs/track의 synth.mp4와 truth.json은 눈으로 보고 열어 보라고 남기는 사본이다.
    """
    frames, truth = make_video()
    if not VIDEO.exists():
        save_video(frames, VIDEO)
        TRUTH.write_text(json.dumps(truth))
    return frames, truth


# ---------- 검출 ----------

def detect_frames(frames: list[np.ndarray], conf: float = DET_CONF) -> list[np.ndarray]:
    """장면마다 2일차 best.pt로 사람을 찾는다. 결과는 (N, 5) = x1, y1, x2, y2, 점수."""
    require_lab5()
    from ultralytics import YOLO
    model = YOLO(str(BEST))
    device = pick_device()
    results = []
    for frame in frames:
        r = model.predict(frame, imgsz=IMAGE_SIZE, conf=conf, device=device, verbose=False)[0]
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()[:, None]
        results.append(np.hstack([boxes, scores]).astype(np.float32))
    return results


def synth_detections() -> list[np.ndarray]:
    """합성 영상의 검출 결과. 한 번 돌려 runs/track/dets.npz에 두고 다시 쓴다."""
    if DETS.exists():
        saved = np.load(DETS)
        return [saved[f"f{t}"] for t in range(FRAMES)]
    frames, _ = synth()
    dets = detect_frames(frames)
    np.savez(DETS, **{f"f{t}": d for t, d in enumerate(dets)})
    return dets


# ---------- 상자와 매칭 ----------

def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N,4)와 (M,4) xyxy 상자의 IoU 표(1일차 lab1과 같은 계산)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter)


def match(iou: np.ndarray, min_iou: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """IoU 합이 가장 큰 짝을 헝가리안 방법으로 찾고, min_iou 아래인 짝은 버린다."""
    if iou.size == 0:
        return [], list(range(iou.shape[0])), list(range(iou.shape[1]))
    rows, cols = linear_sum_assignment(-iou)
    pairs = [(r, c) for r, c in zip(rows, cols) if iou[r, c] >= min_iou]
    left_rows = sorted(set(range(iou.shape[0])) - {r for r, _ in pairs})
    left_cols = sorted(set(range(iou.shape[1])) - {c for _, c in pairs})
    return pairs, left_rows, left_cols


def run_tracker(tracker, dets: list[np.ndarray]) -> list[list[tuple[int, list[float]]]]:
    """추적기에 장면을 차례로 넣고, 장면마다 (트랙 ID, 상자) 목록을 모은다."""
    return [tracker.update(d) for d in dets]


# ---------- 지표 ----------

def evaluate(truth: list[list[dict]], tracks: list[list[tuple[int, list[float]]]], thr: float = MATCH_IOU) -> dict:
    """CLEAR MOT(MOTA, ID 바뀜)과 IDF1을 계산한다.

    MOTA: 장면마다 정답과 트랙을 짝짓는다. 앞 장면의 짝이 아직 겹치면 그대로 두고, 나머지를 헝가리안으로.
          정답의 짝이 앞에서 짝이던 트랙과 다르면 ID 바뀜 1번. MOTA = 1 - (놓침 + 헛짚음 + ID 바뀜) / 정답 수.
    IDF1: 영상 전체에서 정답 한 사람과 트랙 하나를 1:1로 묶어(겹친 장면 수가 가장 많게), 그 짝이 맞힌 장면 비율.
    """
    last, fn, fp, switches, total = {}, 0, 0, 0, 0
    overlap: dict[tuple[int, int], int] = {}
    gt_count: dict[int, int] = {}
    trk_count: dict[int, int] = {}
    for gt, trk in zip(truth, tracks):
        g_ids = [g["id"] for g in gt]
        g_boxes = np.array([g["box"] for g in gt], float).reshape(-1, 4)
        t_ids = [i for i, _ in trk]
        t_boxes = np.array([b for _, b in trk], float).reshape(-1, 4)
        iou = iou_matrix(g_boxes, t_boxes)
        total += len(gt)
        for gi in g_ids:
            gt_count[gi] = gt_count.get(gi, 0) + 1
        for ti in t_ids:
            trk_count[ti] = trk_count.get(ti, 0) + 1
        for r, c in zip(*np.nonzero(iou >= thr)):
            overlap[(g_ids[r], t_ids[c])] = overlap.get((g_ids[r], t_ids[c]), 0) + 1
        pairs = []
        free_g, free_t = set(range(len(gt))), set(range(len(trk)))
        for r, gi in enumerate(g_ids):                      # 앞 장면의 짝을 먼저 이어 간다
            if gi in last and last[gi] in t_ids:
                c = t_ids.index(last[gi])
                if c in free_t and iou[r, c] >= thr:
                    pairs.append((r, c))
                    free_g.discard(r)
                    free_t.discard(c)
        rows, cols = sorted(free_g), sorted(free_t)
        sub_pairs, _, _ = match(iou[np.ix_(rows, cols)], thr)
        for r, c in sub_pairs:
            gi, ti = g_ids[rows[r]], t_ids[cols[c]]
            if gi in last and last[gi] != ti:
                switches += 1
            pairs.append((rows[r], cols[c]))
        for r, c in pairs:
            last[g_ids[r]] = t_ids[c]
        fn += len(gt) - len(pairs)
        fp += len(trk) - len(pairs)
    g_list, t_list = sorted(gt_count), sorted(trk_count)
    score = np.zeros((len(g_list), len(t_list)))
    for (gi, ti), n in overlap.items():
        score[g_list.index(gi), t_list.index(ti)] = n
    rows, cols = linear_sum_assignment(-score) if score.size else ([], [])
    idtp = float(score[rows, cols].sum()) if score.size else 0.0
    idf1 = 2 * idtp / (sum(gt_count.values()) + sum(trk_count.values())) if total else 0.0
    return {"MOTA": 1 - (fn + fp + switches) / total, "IDF1": idf1, "ID바뀜": switches,
            "놓침": fn, "헛짚음": fp, "트랙수": len(trk_count), "정답": total}


def show(name: str, m: dict) -> None:
    print(f"{name:<10} MOTA {m['MOTA']:6.3f}  IDF1 {m['IDF1']:.3f}  ID 바뀜 {m['ID바뀜']:3d}  "
          f"놓침 {m['놓침']:4d}  헛짚음 {m['헛짚음']:4d}  만든 ID {m['트랙수']:3d}개(실제 {PEOPLE}명)")
