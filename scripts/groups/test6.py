"""Test 6 (protocol: test6_protocol.json): colour as an identity cue.

Object level: annotated pairs, object colour vs enlarged-neighbourhood colour, AUC with episode bootstrap
(same code path as the repository's size_auc.py; size AUC is recomputed here as the reference).
Group level: oracle groups from detections of the last 20 s, test-3 ranking metrics.
Usage: python -I test6.py COLORS_DIR MAPS_NPZ OUT_JSON
"""
import csv
import json
import os
import sys
from collections import defaultdict
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
import yaml
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.stats import rankdata

LAB = Path(os.environ["LAB_ROOT"])
REPO = Path(__file__).resolve().parents[2]
CDIR, MAPS, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path[:0] = [str(LAB / "vendor/tcaff")]
ROBOTS = ["RR01", "RR04", "RR06", "RR08"]
WEIGHTS = ["core", "full", "centre"]
rng = np.random.default_rng(0)


def to_lab(rgb):
    c = np.clip(np.asarray(rgb, float) / 255.0, 0, 1)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = lin @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def chroma(rgb):
    rgb = np.asarray(rgb, float)
    s = rgb.sum(-1, keepdims=True)
    return (rgb / np.where(s > 0, s, np.nan))[..., :2]


SPACES = {"rg": chroma, "ab": lambda x: to_lab(x)[..., 1:]}

# ---------------- detections with colours ----------------
det = {}
for R in ROBOTS:
    f = np.load(CDIR / f"{R}_colors.npz")
    det[R] = {k: f[k] for k in f.files}
    det[R]["key"] = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(f["frame"], f["det"]))}
    by_frame = defaultdict(list)
    for i, a in enumerate(f["frame"]):
        by_frame[int(a)].append(i)
    det[R]["by_frame"] = by_frame


def colour_of(R, frame, d, weighting, neighbourhood):
    D = det[R]
    i = D["key"].get((frame, d))
    if i is None:
        return None
    if not neighbourhood:
        return D[weighting][i]
    idx = np.array(D["by_frame"][frame])
    near = idx[np.linalg.norm(D["c"][idx] - D["c"][i], axis=1) <= 0.7]
    w = D["npix_full"][near].astype(float)
    cols = D[weighting][near]
    ok = np.isfinite(cols).all(1) & (w > 0)
    if not ok.any():
        return None
    return (cols[ok] * w[ok, None]).sum(0) / w[ok].sum()


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return None
    r = rankdata(np.r_[pos, neg])
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def with_ci(rows, key):
    by = defaultdict(list)
    for x in rows:
        by[x["episode"]].append(x)
    eps = sorted(by)
    a = auc([x[key] for x in rows if x["y"]], [x[key] for x in rows if not x["y"]])
    boot = []
    r = np.random.default_rng(0)
    for _ in range(2000):
        pick = [x for e in r.choice(eps, len(eps)) for x in by[e]]
        p, n = [x[key] for x in pick if x["y"]], [x[key] for x in pick if not x["y"]]
        if p and n:
            boot.append(auc(p, n))
    return dict(n_same=sum(x["y"] for x in rows), n_different=sum(1 - x["y"] for x in rows), auc=a,
                ci95=np.quantile(boot, [0.025, 0.975]).round(3).tolist())


# ---------------- object level ----------------
tasks = json.loads((REPO / "annotations/task_metadata.json").read_text())["tasks"]
tasks = tasks if isinstance(tasks, list) else list(tasks.values())
label = {r["task_id"]: r["label"] for r in csv.DictReader(open(REPO / "annotations/labels.csv"))}
rows, missing = [], 0
for t in tasks:
    y = label[t["task_id"]]
    if y not in ("same_object", "different_objects"):
        continue
    a, b = t["sides"]
    ratio = lambda p, q: min(p / q, q / p)
    row = dict(episode=t["episode_id"], y=int(y == "same_object"), admitted=t["gate"]["admitted"],
               size=float(np.sqrt(ratio(a["map_width_m"], b["map_width_m"]) * ratio(a["map_height_m"], b["map_height_m"]))))
    for wgt in WEIGHTS:
        for nb in (False, True):
            sides = []
            for side in (a, b):
                cols = [colour_of(side["robot"], v["frame_index"], v["detection_index"], wgt, nb) for v in side["views"]]
                cols = [c for c in cols if c is not None and np.isfinite(c).all()]
                sides.append(np.median(np.array(cols), axis=0) if cols else None)
            for sp, fn in SPACES.items():
                k = f"{'nbhd' if nb else 'object'}_{wgt}_{sp}"
                row[k] = None if sides[0] is None or sides[1] is None else -float(np.linalg.norm(fn(sides[0]) - fn(sides[1])))
    rows.append(row)
keys = [k for k in rows[0] if k.startswith(("object_", "nbhd_"))]
complete = [r for r in rows if all(r[k] is not None for k in keys)]
obj = {"pairs_total": len(rows), "pairs_with_all_colours": len(complete),
       "size_reference": {"all": with_ci(complete, "size"), "gate_rejected": with_ci([r for r in complete if not r["admitted"]], "size")}}
for k in keys:
    obj[k] = {"all": with_ci(complete, k), "gate_rejected": with_ci([r for r in complete if not r["admitted"]], k)}
best = max(keys, key=lambda k: obj[k]["all"]["auc"])
obj["best"] = best
obj["rule_met"] = bool(obj[best]["all"]["auc"] >= 0.75 and obj[best]["all"]["ci95"][0] > 0.66)
# colour + size: rank-average of the two similarities
for k in (best,):
    rs, rc = rankdata([r["size"] for r in complete]), rankdata([r[k] for r in complete])
    for r, a_, b_ in zip(complete, rs, rc):
        r["size_plus_" + k] = (a_ + b_) / 2
    obj["size_plus_" + k] = {"all": with_ci(complete, "size_plus_" + k)}

# ---------------- group level (oracle groups from detections) ----------------
from robotdatapy.data import PoseData  # noqa: E402
params = yaml.safe_load((LAB / "vendor/tcaff/demo/params/tcaff_mot_dataset.yaml").read_text())
boxes = np.loadtxt(LAB / "data/tcaff_mot_data/objects.txt", delimiter=",")
box_group = fcluster(linkage(boxes, "single"), 0.7, "distance") - 1
world_group = {}
for R in ROBOTS:
    gt = PoseData.from_bag(path=str(LAB / f"data/tcaff_mot_data/data/kimera_odom/{R}.bag"), topic=f"/{R}/world", time_tol=10., interp=True)
    T_BC = np.asarray(params[R]["T_BC"], float)
    D = det[R]
    g = np.full(len(D["time"]), -1)
    for i, (t, c) in enumerate(zip(D["time"], D["c"])):
        try:
            p = (gt.T_WB(t) @ T_BC @ np.r_[c, 1.0])[:3]
        except Exception:
            continue
        d = np.linalg.norm(boxes - p[:2], axis=1)
        if d.min() < 0.5 and 0.08 <= p[2] < 0.9:
            g[i] = box_group[d.argmin()]
    world_group[R] = g
step_times = [f["time"] for f in json.loads((LAB / "data/tcaff_baseline_release/RR01_RR04.json").read_text())["frames"]]

SIGS = ["area_mean", "equal_mean", "set"]


def signatures(R, t, wgt, sp):
    D, g = det[R], world_group[R]
    sel = (D["time"] > t - 20.0) & (D["time"] <= t) & (g >= 0) & np.isfinite(D[wgt]).all(1)
    out = {}
    for k in np.unique(g[sel]):
        m = sel & (g == k)
        if m.sum() < 3:
            continue
        cols = SPACES[sp](D[wgt][m])
        w = D["npix_full"][m].astype(float)
        sub = cols if len(cols) <= 60 else cols[np.linspace(0, len(cols) - 1, 60).astype(int)]
        out[int(k)] = dict(area_mean=(cols * w[:, None]).sum(0) / w.sum(), equal_mean=cols.mean(0), set=sub)
    return out


def chamfer(a, b):
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return 0.5 * (d.min(1).mean() + d.min(0).mean())


def sig_dist(A, B, sig):
    return chamfer(A["set"], B["set"]) if sig == "set" else float(np.linalg.norm(A[sig] - B[sig]))


grp = {}
for wgt in WEIGHTS:
    for sp in SPACES:
        S = {(R, k): signatures(R, t, wgt, sp) for R in ROBOTS for k, t in enumerate(step_times)}
        for sig in SIGS:
            top1, top2, chance, pos, neg, uniq = [], [], [], [], [], set()
            for a, b in permutations(ROBOTS, 2):
                for k in range(len(step_times)):
                    GA, GB = S[(a, k)], S[(b, k)]
                    if not GA or not GB:
                        continue
                    ids = np.array(list(GB))
                    for ga, A in GA.items():
                        dist = np.array([sig_dist(A, GB[gb], sig) for gb in ids])
                        for gb, dd in zip(ids, dist):
                            (pos if gb == ga else neg).append(-dd)
                        if ga not in ids:
                            continue
                        order = np.argsort(dist, kind="stable")
                        top1.append(ids[order[0]] == ga)
                        top2.append(ga in ids[order[:2]])
                        chance.append(1 / len(ids))
                        uniq.add((frozenset((a, b)), ga))
            grp[f"{sig}_{wgt}_{sp}"] = dict(cases=len(top1), unique_robotpair_group_cases=len(uniq),
                                            top1=float(np.mean(top1)), chance_top1=float(np.mean(chance)),
                                            top2_recall=float(np.mean(top2)), auc=auc(pos, neg))
best_g = max(grp, key=lambda k: grp[k]["top1"])
res = {"object_level": obj, "group_level": grp, "group_best": best_g,
       "group_rule_met": bool(grp[best_g]["top1"] >= 0.80 and grp[best_g]["top2_recall"] >= 0.95)}
OUT.write_text(json.dumps(res, indent=1, default=float))
print(json.dumps({"object_best": best, "object_best_auc": obj[best]["all"], "size": obj["size_reference"]["all"],
                  "object_rule_met": obj["rule_met"], "group_best": best_g, "group_best_metrics": grp[best_g],
                  "group_rule_met": res["group_rule_met"]}, indent=1, default=float))
