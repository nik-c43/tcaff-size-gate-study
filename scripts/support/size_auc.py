"""T2.0: does the soft size similarity separate 'same object' from 'different objects' labels? (protocol T2.0)

s = sqrt(min(wi/wj, wj/wi) * min(hi/hj, hj/hi)) from the map sizes of both sides of each labelled pair.
ROC AUC with a 95% cluster bootstrap over the 24 episodes (2000 resamples, seed 0). The criterion was
written down before the first computation (2026-10-17T17:40:34Z) and is quoted in the protocol.

Usage: python -I size_auc.py OUT_JSON
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
tasks = json.loads((REPO / "annotations/task_metadata.json").read_text())["tasks"]
tasks = tasks if isinstance(tasks, list) else list(tasks.values())
label = {r["task_id"]: r["label"] for r in csv.DictReader(open(REPO / "annotations/labels.csv"))}


def ratio(a, b):
    return min(a / b, b / a)


rows = []
for t in tasks:
    y = label[t["task_id"]]
    if y not in ("same_object", "different_objects"):
        continue
    a, b = t["sides"]
    s = np.sqrt(ratio(a["map_width_m"], b["map_width_m"]) * ratio(a["map_height_m"], b["map_height_m"]))
    rows.append(dict(episode=t["episode_id"], y=int(y == "same_object"), s=float(s), admitted=t["gate"]["admitted"]))


def auc(sub):
    pos = np.array([x["s"] for x in sub if x["y"]])
    neg = np.array([x["s"] for x in sub if not x["y"]])
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def with_ci(sub):
    by = defaultdict(list)
    for x in sub:
        by[x["episode"]].append(x)
    eps = sorted(by)
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(2000):
        pick = [x for e in rng.choice(eps, len(eps)) for x in by[e]]
        if any(x["y"] for x in pick) and not all(x["y"] for x in pick):
            boot.append(auc(pick))
    return dict(n_same=sum(x["y"] for x in sub), n_different=sum(1 - x["y"] for x in sub), episodes=len(eps),
                auc=auc(sub), ci95=np.quantile(boot, [0.025, 0.975]).tolist())


res = {"test": "T2.0 size-similarity AUC on labelled pairs (configs/support_hypotheses.json)",
       "all": with_ci(rows), "gate_rejected": with_ci([x for x in rows if not x["admitted"]]),
       "gate_admitted": with_ci([x for x in rows if x["admitted"]])}
res["passed"] = res["all"]["ci95"][0] > 0.5 and res["gate_rejected"]["ci95"][0] > 0.5
Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True)
Path(sys.argv[1]).write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
