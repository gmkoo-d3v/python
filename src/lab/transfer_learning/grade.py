"""
제출 채점 — submission.csv 의 id·label 을 answers.csv 정답과 비교한다.
 
  uv run grade.py 해바라기_03.csv                 # 전체 정답률
  uv run grade.py 해바라기_03.csv --private       # 공개·비공개 보드로 나눈 점수까지
  uv run grade.py 해바라기_03.csv --answers 경로   # answers.csv 위치를 따로 줄 때
 
answers.csv 는 id,label,board 세 열이다 (board = public · private).
"""
 
import argparse
import csv
from pathlib import Path
 
HERE = Path(__file__).resolve().parent
 
 
def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))
 
 
def load_submission(path: Path) -> dict[str, int]:
    rows = read_csv(path)
    if not rows:
        raise SystemExit(f"{path} 에 예측이 없다 (머리줄만 있음) — data/test 를 확인할 것")
    if set(rows[0]) != {"id", "label"}:
        raise SystemExit(f"{path} 의 열은 id,label 이어야 한다 (지금: {','.join(rows[0])})")
    pred: dict[str, int] = {}
    for r in rows:
        if r["id"] in pred:
            raise SystemExit(f"id {r['id']} 가 두 번 나온다")
        pred[r["id"]] = int(r["label"])
    return pred
 
 
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument("--answers", type=Path, default=HERE / "answers.csv")
    parser.add_argument("--private", action="store_true", help="공개·비공개 보드로 나눈 점수도 보인다")
    args = parser.parse_args()
 
    answers = read_csv(args.answers)
    pred = load_submission(args.submission)
 
    answer_ids = {a["id"] for a in answers}
    extra = set(pred) - answer_ids
    if extra:
        raise SystemExit(f"정답에 없는 id 가 {len(extra)}개 있다 — 다른 데이터의 제출 파일인지 확인할 것")
    missing = answer_ids - set(pred)
 
    hits = {"public": 0, "private": 0}
    total = {"public": 0, "private": 0}
    for a in answers:
        if a["id"] not in pred:
            continue
        total[a["board"]] += 1
        hits[a["board"]] += pred[a["id"]] == int(a["label"])
 
    n_hit, n_sub = sum(hits.values()), sum(total.values())
    print(f"제출 파일: {args.submission.name}  (답한 사진 {n_sub}/{len(answers)}장)")
    print(f"정답률      {n_hit / n_sub:.4f}  ({n_hit}/{n_sub})  ← 답한 사진 기준")
    if missing:
        print(f"전체 기준   {n_hit / len(answers):.4f}  ({n_hit}/{len(answers)})  ← 빠진 {len(missing)}장은 오답 처리")
    if args.private:
        for board in ("public", "private"):
            if total[board]:
                print(f"{board:<8}    {hits[board] / total[board]:.4f}  ({hits[board]}/{total[board]})")
 
 
if __name__ == "__main__":
    main()
 