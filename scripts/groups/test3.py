"""Test 3 (protocol: test3_protocol.json): group matching by on-board descriptors.
Usage: python -I test3.py MAPS_NPZ OUT_JSON"""
import json
import sys
from collections import defaultdict
from itertools import combinations, permutations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.stats import rankdata

f = np.load(sys.argv[1])
D, cols = f["data"], list(f["cols"])
G = {k: D[:, i] for i, k in enumerate(cols)}
gt_group = f["gt_group"]
oracle_label = np.where((G["dbox"] < 0.5) & (G["wz"] >= 0.08) & (G["wz"] < 0.9), gt_group[G["nbox"].astype(int)], -1)
keys = defaultdict(list)
for i, (r, s) in enumerate(zip(G["robot"].astype(int), G["step"].astype(int))):
    keys[(r, s)].append(i)
keys = {k: np.array(v) for k, v in keys.items()}


def describe(idx, identity):
    xy = np.c_[G["x"][idx], G["y"][idx]]
    c = xy.mean(0)
    pd = np.array([np.linalg.norm(p - q) for p, q in combinations(xy, 2)])
    sz = np.c_[np.log(np.clip(G["w"][idx], 0.01, None)), np.log(np.clip(G["h"][idx], 0.01, None))]
    return dict(identity=int(identity), n=len(idx), radius=float(np.linalg.norm(xy - c, axis=1).max()), pd=pd, sz=sz)


def oracle_groups(idx):
    lab = oracle_label[idx]
    return [describe(idx[lab == g], g) for g in np.unique(lab[lab >= 0]) if (lab == g).sum() >= 2]


def onboard_groups(idx):
    cand = (G["w"][idx] <= 0.5) & (G["h"][idx] >= 0.08) & (G["h"][idx] <= 0.6) & (np.abs(G["z"][idx] - G["rz"][idx]) <= 0.35)
    m = idx[cand]
    if len(m) < 2:
        return []
    xy = np.c_[G["x"][m], G["y"][m]]
    lab = fcluster(linkage(xy, "single"), 0.7, "distance")
    out = []
    for c in np.unique(lab):
        mem = m[lab == c]
        if not (2 <= len(mem) <= 10):
            continue
        p = np.c_[G["x"][mem], G["y"][mem]]
        if np.linalg.norm(p - p.mean(0), axis=1).max() > 1.0:
            continue
        ol = oracle_label[mem]
        vals, cnt = np.unique(ol[ol >= 0], return_counts=True)
        ident = vals[cnt.argmax()] if len(vals) and cnt.max() >= 0.5 * len(mem) else -1
        out.append(describe(mem, ident))
    return out


def chamfer(a, b):
    if a.ndim == 1:
        d = np.abs(a[:, None] - b[None, :])
    else:
        d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return 0.5 * (d.min(1).mean() + d.min(0).mean())


COMP = ["size", "arrangement", "count", "radius"]


def dists(A, B):
    return np.array([chamfer(A["sz"], B["sz"]), chamfer(A["pd"], B["pd"]),
                     abs(A["n"] - B["n"]) / max(A["n"], B["n"]), abs(A["radius"] - B["radius"])])


def auc(pos, neg):
    if not len(pos) or not len(neg):
        return None
    r = rankdata(np.r_[pos, neg])
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def evaluate(builder):
    groups = {k: builder(v) for k, v in keys.items()}
    robots = sorted({k[0] for k in keys})
    steps = sorted({k[1] for k in keys})
    cand = []        # (case_id, a, b, step, idA, idB, dists)
    cases = []
    for a, b in permutations(robots, 2):
        for s in steps:
            GA, GB = groups.get((a, s), []), groups.get((b, s), [])
            if not GA or not GB:
                continue
            for ia, A in enumerate(GA):
                rows = [(A["identity"], B["identity"], dists(A, B)) for B in GB]
                cases.append(dict(a=a, b=b, step=s, idA=A["identity"], rows=rows))
    allD = np.array([r[2] for c in cases for r in c["rows"]])
    med = np.median(allD, axis=0)
    med[med == 0] = 1.0
    out = dict(map_groups=int(sum(len(v) for v in groups.values())),
               false_group_share=float(np.mean([g["identity"] < 0 for v in groups.values() for g in v])) if groups else None,
               median_distances=dict(zip(COMP, med.round(4).tolist())))
    for name, w in [("combined", np.ones(4) / 4)] + [(c, np.eye(4)[i]) for i, c in enumerate(COMP)]:
        top1, top2, chance, pruned, pos, neg, uniq = [], [], [], [], [], [], set()
        for c in cases:
            ids = np.array([r[1] for r in c["rows"]])
            sc = -np.array([(r[2] / med) @ w for r in c["rows"]])
            for i_, s_ in zip(ids, sc):
                if c["idA"] >= 0 and i_ >= 0:
                    (pos if i_ == c["idA"] else neg).append(s_)
                elif c["idA"] >= 0 or i_ >= 0:
                    neg.append(s_)
            if c["idA"] < 0 or c["idA"] not in ids:
                continue
            order = np.argsort(-sc, kind="stable")
            top1.append(ids[order[0]] == c["idA"])
            top2.append(c["idA"] in ids[order[:2]])
            chance.append(1.0 / len(ids))
            wrong = [k for k in range(len(ids)) if ids[k] != c["idA"]]
            pruned += [k not in order[:2] for k in wrong]
            uniq.add((frozenset((c["a"], c["b"])), c["idA"]))
        out[name] = dict(cases=len(top1), unique_robotpair_group_cases=len(uniq),
                         top1=float(np.mean(top1)) if top1 else None, chance_top1=float(np.mean(chance)) if chance else None,
                         top2_recall=float(np.mean(top2)) if top2 else None,
                         top2_wrong_pruned=float(np.mean(pruned)) if pruned else None,
                         candidates_q50=float(np.median([1 / c for c in chance])) if chance else None,
                         auc=auc(np.array(pos), np.array(neg)))
    return out


res = {"oracle": evaluate(oracle_groups), "onboard": evaluate(onboard_groups)}
o = res["oracle"]["combined"]
if o["top1"] >= 0.80 and o["top2_recall"] >= 0.95:
    verdict = "proceed to an on-board group gate"
elif o["top1"] >= o["chance_top1"] + 0.20:
    verdict = "size and arrangement are not enough: add colour"
else:
    verdict = "descriptors without colour fail"
res["verdict"] = verdict
json.dump(res, open(sys.argv[2], "w"), indent=1)
print(json.dumps(res, indent=1))
