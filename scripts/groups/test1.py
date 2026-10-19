"""Test 1 (exploratory): can a robot find floor-standing groups of objects in its own TCAFF map,
how pure are the groups, how noisy are group centres, and how often do two robots share groups?

On-board rules (odometry frame only):
  candidate floor obstacle: w <= W_MAX, H_MIN <= h <= H_MAX, |z - z_robot| <= DZ
  wall: inlier (<= DELTA) of a line in XY with >= N_WALL inliers, span >= SPAN, and at most 5% of the
        map objects more than 0.5 m beyond the line on the side opposite to the robot
  groups: single-linkage clusters of floor obstacles (not on walls) with radius R;
          accepted if SIZE_MIN <= size <= SIZE_MAX and max distance to centre <= RAD_MAX
Evaluation only (GT): world positions, place classes, GT box / GT group of each object.
Usage: [NOWALL=1] python -I test1.py MAPS_NPZ OUT_JSON [R]   (NOWALL=1 skips the wall-line search)
"""
import json
import sys
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

import os
W_MAX, H_MIN, H_MAX, DZ = 0.5, 0.08, 0.6, 0.35
DELTA, N_WALL, SPAN = 0.3, 8, 3.0
DELTA=float(os.environ.get("DELTA",DELTA)); N_WALL=int(os.environ.get("N_WALL",N_WALL)); SPAN=float(os.environ.get("SPAN",SPAN)); BEYOND=float(os.environ.get("BEYOND",0.05)); NOWALL=os.environ.get("NOWALL")=="1"
SIZE_MIN, SIZE_MAX, RAD_MAX = 2, 10, 1.0
R = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7
rng = np.random.default_rng(0)

f = np.load(sys.argv[1])
D, cols = f["data"], list(f["cols"])
boxes, gt_group = f["boxes"], f["gt_group"]
G = {k: D[:, i] for i, k in enumerate(cols)}
n_groups = gt_group.max() + 1
group_centre = np.array([boxes[gt_group == g].mean(0) for g in range(n_groups)])

wz, db = G["wz"], G["dbox"]
place = np.full(len(D), "other", dtype=object)
place[(np.abs(G["wx"]) > 4.5) | (np.abs(G["wy"]) > 4.5)] = "room_edge"
place[wz > 1.2] = "high"
place[(db < 0.35) & (wz < 0.9)] = "gt_obstacle"
place[wz < 0.08] = "floor"
dynamic = G["dmover"] < 0.3


def wall_mask(xy, robot_xy):
    n = len(xy)
    on_wall = np.zeros(n, bool)
    free = np.ones(n, bool)
    for _ in range(6):
        idx = np.flatnonzero(free)
        if len(idx) < N_WALL:
            break
        pairs = np.array(list(combinations(idx, 2)))
        if len(pairs) > 3000:
            pairs = pairs[rng.choice(len(pairs), 3000, replace=False)]
        d = xy[pairs[:, 1]] - xy[pairs[:, 0]]
        L = np.linalg.norm(d, axis=1)
        keep = L >= 0.5
        pairs, d, L = pairs[keep], d[keep], L[keep]
        if not len(pairs):
            break
        u = d / L[:, None]
        nrm = np.c_[-u[:, 1], u[:, 0]]
        rel = xy[None, :, :] - xy[pairs[:, 0]][:, None, :]          # P x n x 2
        off = np.einsum("pnk,pk->pn", rel, nrm)
        proj = np.einsum("pnk,pk->pn", rel, u)
        inl = free[None, :] & (np.abs(off) <= DELTA)
        k = inl.sum(1)
        span = np.where(inl, proj, np.inf).min(1)
        span = np.where(inl, proj, -np.inf).max(1) - span
        side = np.sign(np.einsum("pk,pk->p", robot_xy[None, :] - xy[pairs[:, 0]], nrm))
        side[side == 0] = 1.0
        beyond = ((-side[:, None] * off) > 0.5).mean(1)
        ok = (k >= N_WALL) & (span >= SPAN) & (beyond <= BEYOND)
        if not ok.any():
            break
        best = np.flatnonzero(ok)[np.argmax(k[ok])]
        on_wall |= inl[best]
        free &= ~inl[best]
    return on_wall


def groups(xy):
    if len(xy) < 2:
        return np.full(len(xy), -1)
    lab = fcluster(linkage(xy, "single"), R, "distance") - 1
    out = np.full(len(xy), -1)
    k = 0
    for c in np.unique(lab):
        m = lab == c
        if not (SIZE_MIN <= m.sum() <= SIZE_MAX):
            continue
        if np.linalg.norm(xy[m] - xy[m].mean(0), axis=1).max() > RAD_MAX:
            continue
        out[m] = k
        k += 1
    return out


role = np.full(len(D), "rest", dtype=object)   # wall / floor_obstacle / rest
cluster_id = np.full(len(D), -1)
keys = defaultdict(list)
for i, (r, s) in enumerate(zip(G["robot"].astype(int), G["step"].astype(int))):
    keys[(r, s)].append(i)
steps_by_robot = defaultdict(dict)
for (r, s), idx in keys.items():
    idx = np.array(idx)
    xy = np.c_[G["x"][idx], G["y"][idx]]
    wall = np.zeros(len(idx), bool) if NOWALL else wall_mask(xy, np.array([G["rx"][idx[0]], G["ry"][idx[0]]]))
    cand = (G["w"][idx] <= W_MAX) & (G["h"][idx] >= H_MIN) & (G["h"][idx] <= H_MAX) & \
           (np.abs(G["z"][idx] - G["rz"][idx]) <= DZ)
    role[idx[wall]] = "wall"
    fo = cand & ~wall
    role[idx[fo]] = "floor_obstacle"
    lab = groups(xy[fo])
    cluster_id[idx[fo]] = lab
    cl = {}
    for c in np.unique(lab[lab >= 0]):
        members = idx[fo][lab == c]
        gg = G["ggroup"][members].astype(int)
        valid = gg[gg >= 0]
        dom = Counter(valid.tolist()).most_common(1)[0][0] if len(valid) else -1
        cl[int(c)] = dict(members=members, dom=dom,
                          odom=np.c_[G["x"][members], G["y"][members]].mean(0),
                          world=np.c_[G["wx"][members], G["wy"][members]].mean(0),
                          n_groups=len(set(valid.tolist())), share_dom=float((gg == dom).mean()) if dom >= 0 else 0.0)
    steps_by_robot[r][s] = cl

res = {"parameters": dict(W_MAX=W_MAX, H_MIN=H_MIN, H_MAX=H_MAX, DZ=DZ, DELTA=DELTA, N_WALL=N_WALL, SPAN=SPAN,
                          R=R, SIZE_MIN=SIZE_MIN, SIZE_MAX=SIZE_MAX, RAD_MAX=RAD_MAX),
       "objects": int(len(D))}

# (a) classification
conf = {p: dict(Counter(role[place == p])) for p in ["gt_obstacle", "room_edge", "floor", "high", "other"]}
fo = role == "floor_obstacle"
res["classification"] = dict(
    by_place=conf,
    obstacle_recall=float((role[place == "gt_obstacle"] == "floor_obstacle").mean()),
    floor_obstacle_precision=float((place[fo] == "gt_obstacle").mean()),
    floor_obstacle_composition={k: float(v / fo.sum()) for k, v in Counter(place[fo]).items()},
    floor_obstacle_dynamic_share=float(dynamic[fo].mean()),
    room_edge_on_wall=float((role[place == "room_edge"] == "wall").mean()),
    obstacle_on_wall=float((role[place == "gt_obstacle"] == "wall").mean()))

# (b) purity
cls = [c for r in steps_by_robot for s in steps_by_robot[r] for c in steps_by_robot[r][s].values()]
n_cl = len(cls)
res["purity"] = dict(
    clusters=n_cl,
    clusters_without_gt_group=float(np.mean([c["dom"] < 0 for c in cls])),
    clusters_spanning_two_gt_groups=float(np.mean([c["n_groups"] >= 2 for c in cls])),
    member_share_of_dominant_group_q10_q50=np.quantile([c["share_dom"] for c in cls if c["dom"] >= 0], [0.1, 0.5]).round(3).tolist(),
    members_q50_q90=np.quantile([len(c["members"]) for c in cls], [0.5, 0.9]).tolist())
split = []
for r in steps_by_robot:
    for s, cl in steps_by_robot[r].items():
        cnt = Counter(c["dom"] for c in cl.values() if c["dom"] >= 0)
        split += [v >= 2 for v in cnt.values()]
res["purity"]["gt_group_split_into_several_clusters"] = float(np.mean(split)) if split else None
obst_rows = place == "gt_obstacle"
res["purity"]["obstacle_objects_in_accepted_cluster"] = float((cluster_id[obst_rows] >= 0).mean())

# (c) centre noise: world error vs GT group centre, and pairwise distance consistency between robots
err = [np.linalg.norm(c["world"] - group_centre[c["dom"]]) for c in cls if c["dom"] >= 0 and c["share_dom"] >= 0.5]
obj_err = db[obst_rows]
res["centre_error_m"] = dict(group_q50_q90=np.quantile(err, [0.5, 0.9]).round(3).tolist(),
                             object_to_nearest_box_q50_q90=np.quantile(obj_err, [0.5, 0.9]).round(3).tolist())
robots = sorted(steps_by_robot)
dG, shared_counts = [], defaultdict(list)
d_obj = {"same_group": [], "cross_group": []}
for a, b in combinations(robots, 2):
    for s in sorted(set(steps_by_robot[a]) & set(steps_by_robot[b])):
        ca = {c["dom"]: c for c in steps_by_robot[a][s].values() if c["dom"] >= 0 and c["share_dom"] >= 0.5}
        cb = {c["dom"]: c for c in steps_by_robot[b][s].values() if c["dom"] >= 0 and c["share_dom"] >= 0.5}
        common = sorted(set(ca) & set(cb))
        shared_counts[(a, b)].append(len(common))
        for g1, g2 in combinations(common, 2):
            dG.append(abs(np.linalg.norm(ca[g1]["odom"] - ca[g2]["odom"]) - np.linalg.norm(cb[g1]["odom"] - cb[g2]["odom"])))
    # object-level correct correspondences: one representative map object per GT box per robot (closest), every 5th step
    for s in sorted(set(steps_by_robot[a]) & set(steps_by_robot[b]))[::5]:
        rep = []
        for r in (a, b):
            idx = np.array(keys[(r, s)])
            m = idx[obst_rows[idx]]
            best = {}
            for i in m:
                bx = int(G["nbox"][i])
                if bx not in best or db[i] < db[best[bx]]:
                    best[bx] = i
            rep.append(best)
        common = sorted(set(rep[0]) & set(rep[1]))
        for b1, b2 in combinations(common, 2):
            i1, i2, j1, j2 = rep[0][b1], rep[0][b2], rep[1][b1], rep[1][b2]
            da = np.hypot(G["x"][i1] - G["x"][i2], G["y"][i1] - G["y"][i2])
            dbb = np.hypot(G["x"][j1] - G["x"][j2], G["y"][j1] - G["y"][j2])
            key = "same_group" if gt_group[b1] == gt_group[b2] else "cross_group"
            d_obj[key].append(abs(da - dbb))
q = lambda v: np.quantile(v, [0.5, 0.9, 0.99]).round(3).tolist() if len(v) else None
res["pairwise_consistency_of_correct_matches"] = dict(
    group_centres_dG_q50_q90_q99=q(dG), group_pairs=len(dG),
    objects_same_group_d_q50_q90_q99=q(d_obj["same_group"]), objects_same_group_pairs=len(d_obj["same_group"]),
    objects_same_group_share_d_below_eps025=float(np.mean(np.array(d_obj["same_group"]) < 0.25)) if d_obj["same_group"] else None,
    objects_cross_group_d_q50_q90_q99=q(d_obj["cross_group"]), objects_cross_group_pairs=len(d_obj["cross_group"]),
    objects_cross_group_share_d_below_eps025=float(np.mean(np.array(d_obj["cross_group"]) < 0.25)) if d_obj["cross_group"] else None)

# (d) coverage of shared groups per undirected robot pair and step
res["shared_groups"] = {}
for (a, b), v in shared_counts.items():
    v = np.array(v)
    res["shared_groups"][f"{a}-{b}"] = dict(steps=len(v), share_ge1=float((v >= 1).mean()), share_ge2=float((v >= 2).mean()),
                                            share_ge3=float((v >= 3).mean()))
allv = np.concatenate([np.array(v) for v in shared_counts.values()])
res["shared_groups"]["all"] = dict(share_ge1=float((allv >= 1).mean()), share_ge2=float((allv >= 2).mean()),
                                   share_ge3=float((allv >= 3).mean()))
fo_other = fo & (place == "other")
res["diag"] = dict(fo_other_dbox_q25_q50_q75=np.quantile(db[fo_other], [0.25, 0.5, 0.75]).round(2).tolist() if fo_other.any() else None,
                   fo_other_within_07=float((db[fo_other] < 0.7).mean()) if fo_other.any() else None,
                   obstacle_on_wall_by_group={int(g): float((role[(place == "gt_obstacle") & (G["ggroup"] == g)] == "wall").mean()) for g in range(n_groups)})
json.dump(res, open(sys.argv[2], "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
