"""Keyword groups for ordering records before manual title/abstract screening.

Usage:
  python3 -I screen.py DATA_DIR titles            -> DATA_DIR/titles.tsv (all records, sorted by score)
  python3 -I screen.py DATA_DIR abstracts RID...  -> prints abstracts of selected records
"""
import json
import re
import sys
from pathlib import Path

GROUPS = {
    "robotics": r"\brobot|slam\b|mapping|\bmap\b|maps\b|locali[sz]|navigation|lidar|rgb-?d|odometry|drone|uav|vehicle",
    "object": r"\bobject|segment|instance|landmark|scene graph|semantic map|open-(set|vocabulary)|centroid",
    "uncertainty": r"uncertain|covarian|partial(ly)? (observ|view|visib)|occlu|truncat|noise bound|bounded noise|"
                   r"calibrat|conformal|\bbias|heteroscedastic|probabilistic",
    "association": r"associat|alignment|align\b|registration|relocali|loop closure|correspondence|matching|merg",
}


def load(data):
    return [json.loads(l) for l in open(Path(data) / "records.jsonl")]


def score(r):
    text = f"{r['title']} {r.get('abstract') or ''}".lower()
    hits = {g: bool(re.search(p, text)) for g, p in GROUPS.items()}
    return sum(hits.values()), hits


def main():
    data, mode = sys.argv[1], sys.argv[2]
    recs = load(data)
    if mode == "titles":
        rows = []
        for r in recs:
            s, hits = score(r)
            rows.append((s, r["rid"], (r.get("date") or "")[:7], (r.get("venue") or "")[:30],
                         "".join(g[0] for g, h in hits.items() if h), ",".join(r["sources"]), r["title"]))
        rows.sort(key=lambda x: (-x[0], x[1]))
        with open(Path(data) / "titles.tsv", "w") as f:
            f.write("score\trid\tdate\tvenue\tgroups\tsources\ttitle\n")
            for row in rows:
                f.write("\t".join(str(x) for x in row) + "\n")
        from collections import Counter
        print("records", len(rows), "by score", sorted(Counter(r[0] for r in rows).items()))
    elif mode == "abstracts":
        want = set(sys.argv[3:])
        for r in recs:
            if r["rid"] in want:
                print(f"## {r['rid']} | {r.get('date')} | {r.get('venue')} | arXiv {r.get('arxiv')} | doi {r.get('doi')}")
                print(r["title"])
                print((r.get("abstract") or "(no abstract)")[:1800])
                print()


if __name__ == "__main__":
    main()
