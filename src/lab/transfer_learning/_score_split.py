import csv
ans = {}
with open("answers.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        ans[r["id"]] = (int(r["label"]), r.get("board", ""))
def score(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    tot = {k: [0, 0] for k in ("public", "private", "all")}
    for r in rows:
        lab, board = ans[r["id"]]
        ok = int(r["label"]) == lab
        for k in (board, "all"):
            tot[k][0] += ok; tot[k][1] += 1
    return tot
for p in ("submission_v1_9094.csv", "submission_v5_9406.csv", "submission.csv"):
    t = score(p)
    out = " | ".join(f"{k}: {v[0]}/{v[1]} = {v[0]/v[1]:.4f}" for k, v in t.items())
    print(f"{p:28s} {out}")
