"""Calibration feasibility: do shape-agnostic mask features narrow the same-object discrepancy bound?

Usage: python -I analyze.py DETECTIONS_JSONL OBJECTS_TXT OUT_JSON
"""
import json
import math
import sys
from collections import Counter, defaultdict

import numpy as np

det_path, obj_path, out_path = sys.argv[1:4]
ALPHA = 0.10          # target miscoverage of the bound
EPS_CLIPPER = 0.25    # release clipper_epsilon, m
RATIO, DIFF = 1.35, 0.10  # release size gate
THIN_S, MIN_DT_S, MAX_PAIRS, MIN_TRACK = 0.5, 2.0, 400, 10
NEAR_BOX, UNAMBIG_MARGIN = 0.35, 0.10
rng = np.random.default_rng(0)

rows = [json.loads(l) for l in open(det_path)]
boxes = np.loadtxt(obj_path, delimiter=",")

# Box groups by single linkage at 0.6 m.
group = list(range(len(boxes)))
def find(i):
    while group[i] != i:
        group[i] = group[group[i]]
        i = group[i]
    return i
for i in range(len(boxes)):
    for j in range(i + 1, len(boxes)):
        if np.linalg.norm(boxes[i] - boxes[j]) < 0.6:
            group[find(i)] = find(j)
gid = {g: k for k, g in enumerate(sorted({find(i) for i in range(len(boxes))}))}
box_group = np.array([gid[find(i)] for i in range(len(boxes))])
group_center = np.array([boxes[box_group == g].mean(axis=0) for g in range(len(gid))])

for d in rows:
    dist = np.linalg.norm(boxes - [d["x"], d["y"]], axis=1)
    order = np.argsort(dist)
    d["box"] = int(order[0]) if dist[order[0]] < NEAR_BOX else None
    d["box_dist"] = float(dist[order[0]])
    d["box_unambiguous"] = bool(dist[order[1]] - dist[order[0]] > UNAMBIG_MARGIN)
    d["group"] = int(box_group[order[0]]) if dist[order[0]] < 0.5 else None
    d["trunc"] = d["border_margin_px"] < 5
    d["partial"] = d["trunc"] or d["occluded"]

tracks = defaultdict(list)
for d in rows:
    if "track" in d:
        tracks[d["track"]].append(d)
tracks = {k: sorted(v, key=lambda d: d["t"]) for k, v in tracks.items() if len(v) >= MIN_TRACK}
box_track = {k: np.mean([d["box"] is not None for d in v]) >= 0.8 for k, v in tracks.items()}


def bearing(d, at):
    return math.atan2(at[1] - d["cam_y"], at[0] - d["cam_x"])


def ang_diff(a, b):
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


pairs = []
for k, v in tracks.items():
    thin, last = [], -1e9
    for d in v:
        if d["t"] - last >= THIN_S:
            thin.append(d)
            last = d["t"]
    cand = [(i, j) for i in range(len(thin)) for j in range(i + 1, len(thin)) if thin[j]["t"] - thin[i]["t"] >= MIN_DT_S]
    if len(cand) > MAX_PAIRS:
        cand = [cand[i] for i in rng.choice(len(cand), MAX_PAIRS, replace=False)]
    for i, j in cand:
        a, b = thin[i], thin[j]
        mid = ((a["x"] + b["x"]) / 2, (a["y"] + b["y"]) / 2)
        wmax, wmin = max(a["w"], b["w"]), min(a["w"], b["w"])
        hmax, hmin = max(a["h"], b["h"]), min(a["h"], b["h"])
        pairs.append({
            "robot": a["robot"], "track": k, "box_track": box_track[k],
            "r_xy": math.hypot(a["x"] - b["x"], a["y"] - b["y"]), "r_z": abs(a["z"] - b["z"]),
            "dpsi": math.degrees(ang_diff(bearing(a, mid), bearing(b, mid))),
            "gate_pass": wmax <= wmin * RATIO and hmax <= hmin * RATIO and wmax - wmin <= DIFF and hmax - hmin <= DIFF,
            "trunc_any": a["trunc"] or b["trunc"], "occl_any": a["occluded"] or b["occluded"],
            "partial_any": a["partial"] or b["partial"], "range_max": max(a["range"], b["range"]),
            "mask_min": min(a["mask_scale_px"], b["mask_scale_px"]),
            "same_box": a["box"] is not None and a["box"] == b["box"],
            "diff_box": a["box"] is not None and b["box"] is not None and a["box"] != b["box"],
        })


def q(xs, p):
    xs = np.sort(np.asarray(xs))
    return float(xs[min(len(xs) - 1, int(math.ceil(p * len(xs))) - 1)]) if len(xs) else float("nan")


def conformal_q(xs, alpha=ALPHA):
    xs = np.sort(np.asarray(xs))
    n = len(xs)
    k = math.ceil((n + 1) * (1 - alpha))
    return float(xs[k - 1]) if 0 < k <= n else float("inf")


def summary(sel, name):
    r = [p["r_xy"] for p in sel]
    return {"subset": name, "n_pairs": len(sel), "n_tracks": len({p["track"] for p in sel}),
            "q50": q(r, .5), "q90": q(r, .9), "q95": q(r, .95),
            "share_r_gt_eps": float(np.mean(np.array(r) > EPS_CLIPPER)) if r else None,
            "size_gate_pass": float(np.mean([p["gate_pass"] for p in sel])) if sel else None}


res = {"settings": {"alpha": ALPHA, "clipper_epsilon": EPS_CLIPPER, "size_gate": [RATIO, DIFF], "thin_s": THIN_S,
                    "min_dt_s": MIN_DT_S, "max_pairs_per_track": MAX_PAIRS, "min_track_len": MIN_TRACK}}
res["counts"] = {"detections": len(rows), "tracked_detections": sum("track" in d for d in rows),
                 "tracks_ge_min": len(tracks), "box_tracks": int(sum(box_track.values())), "pairs": len(pairs),
                 "detections_trunc": int(sum(d["trunc"] for d in rows)), "detections_occluded": int(sum(d["occluded"] for d in rows)),
                 "box_groups": len(gid), "box_spacing_median": float(np.median(np.sort(np.linalg.norm(boxes[:, None] - boxes[None], axis=2), axis=1)[:, 1]))}

subsets = {"all": pairs, "box_tracks": [p for p in pairs if p["box_track"]]}
res["global"] = [summary(v, k) for k, v in subsets.items()]

dpsi_bins = [(0, 30), (30, 90), (90, 181)]
res["conditional"] = {}
for sname, sel in subsets.items():
    rr = np.array([p["range_max"] for p in sel])
    mm = np.array([p["mask_min"] for p in sel])
    rt, mt = np.quantile(rr, [1 / 3, 2 / 3]), np.quantile(mm, [1 / 3, 2 / 3])
    cond = []
    for key in ["partial_any", "trunc_any", "occl_any"]:
        for val in (False, True):
            cond.append(summary([p for p in sel if p[key] == val], f"{key}={val}"))
    for lo, hi in dpsi_bins:
        cond.append(summary([p for p in sel if lo <= p["dpsi"] < hi], f"dpsi {lo}-{hi} deg"))
    for i, (lo, hi) in enumerate(zip([-np.inf, *rt], [*rt, np.inf])):
        cond.append(summary([p for p in sel if lo <= p["range_max"] < hi], f"range_max tercile {i + 1} [{lo:.2f},{hi:.2f})"))
    for i, (lo, hi) in enumerate(zip([-np.inf, *mt], [*mt, np.inf])):
        cond.append(summary([p for p in sel if lo <= p["mask_min"] < hi], f"mask_min tercile {i + 1} [{lo:.0f},{hi:.0f})"))
    res["conditional"][sname] = cond


def cell(p):
    b = next(i for i, (lo, hi) in enumerate(dpsi_bins) if lo <= p["dpsi"] < hi)
    return (p["partial_any"], b)


# Leave-one-robot-out: calibrate on three robots, test on the fourth.
res["leave_one_robot_out"] = {}
for sname, sel in subsets.items():
    out = []
    for h in sorted({p["robot"] for p in sel}):
        cal = [p for p in sel if p["robot"] != h]
        test = [p for p in sel if p["robot"] == h]
        if not test:
            continue
        g = conformal_q([p["r_xy"] for p in cal])
        by = defaultdict(list)
        for p in cal:
            by[cell(p)].append(p["r_xy"])
        cq = {c: conformal_q(v) for c, v in by.items()}
        b_cell = [cq.get(cell(p), g) for p in test]
        out.append({"held_out": h, "n_test": len(test), "global_bound": g,
                    "global_coverage": float(np.mean([p["r_xy"] <= g for p in test])),
                    "cell_mean_bound": float(np.mean(b_cell)),
                    "cell_coverage": float(np.mean([p["r_xy"] <= b for p, b in zip(test, b_cell)])),
                    "cell_bounds": {f"partial={c[0]},dpsi_bin={dpsi_bins[c[1]]}": round(v, 3) for c, v in sorted(cq.items())}})
    res["leave_one_robot_out"][sname] = out

# Spearman correlations of discrepancy with pair features.
def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])
res["spearman_r_xy"] = {s: {f: spearman([p[f] for p in sel], [p["r_xy"] for p in sel]) for f in ["dpsi", "range_max", "mask_min", "r_z"]}
                        for s, sel in subsets.items()}

# GT box control: does a track stay on one physical box?
sw = []
for k, v in tracks.items():
    if not box_track[k]:
        continue
    bx = [d["box"] for d in v if d["box"] is not None]
    gx = [d["group"] for d in v if d["group"] is not None]
    sw.append((1 - Counter(bx).most_common(1)[0][1] / len(bx), 1 - Counter(gx).most_common(1)[0][1] / len(gx)))
sw = np.array(sw)
bp = subsets["box_tracks"]
res["gt_box_control"] = {
    "box_tracks": len(sw), "median_share_off_modal_box": float(np.median(sw[:, 0])),
    "tracks_with_gt20pct_off_modal_box": float(np.mean(sw[:, 0] > 0.2)),
    "median_share_off_modal_group": float(np.median(sw[:, 1])),
    "pairs_same_box": summary([p for p in bp if p["same_box"]], "same nearest box"),
    "pairs_diff_box": summary([p for p in bp if p["diff_box"]], "different nearest box"),
}

# Systematic component: residual to an unambiguous nearest box, split along the camera ray.
along, perp = [], []
for d in rows:
    if d["box"] is None or not d["box_unambiguous"]:
        continue
    c = boxes[d["box"]]
    ray = np.array([c[0] - d["cam_x"], c[1] - d["cam_y"]])
    ray /= np.linalg.norm(ray)
    e = np.array([d["x"] - c[0], d["y"] - c[1]])
    along.append(float(e @ ray))
    perp.append(float(abs(e[0] * ray[1] - e[1] * ray[0])))
res["ray_decomposition_unambiguous_box_detections"] = {
    "n": len(along), "along_ray_q10_q50_q90": [q(along, .1), q(along, .5), q(along, .9)],
    "share_toward_camera": float(np.mean(np.array(along) < 0)),
    "perp_q50_q90": [q(perp, .5), q(perp, .9)],
    "note": "negative along-ray = observed centroid lies between camera and GT box centre"}

# Viewpoint change: within-robot same-track pairs vs inter-robot pairs on the same box group.
by_group = defaultdict(list)
for d in rows:
    if d["group"] is not None:
        by_group[d["group"]].append(d)
inter = []
for g, v in by_group.items():
    c = group_center[g]
    for _ in range(4000):
        a, b = v[rng.integers(len(v))], v[rng.integers(len(v))]
        if a["robot"] != b["robot"]:
            inter.append(math.degrees(ang_diff(bearing(a, c), bearing(b, c))))
within = [p["dpsi"] for p in bp]
res["viewpoint_change_deg"] = {"within_robot_box_track_pairs_q50_q90": [q(within, .5), q(within, .9)],
                               "inter_robot_same_group_q50_q90": [q(inter, .5), q(inter, .9)],
                               "share_inter_robot_above_within_q90": float(np.mean(np.array(inter) > q(within, .9)))}

json.dump(res, open(out_path, "w"), indent=1)
print(json.dumps(res, indent=1)[:12000])
