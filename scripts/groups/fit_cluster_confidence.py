"""Fit the cluster-confidence logistic model of test 9 (variant B), leave one robot out.
Features of an on-board cluster (test-1 rules): log(members), radius (m), median member height h (m).
Label (GT): real if one GT group holds >= 50% of members. Usage: python -I fit_cluster_confidence.py MAPS_NPZ OUT_JSON"""
import json
import sys
from collections import defaultdict

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

f = np.load(sys.argv[1])
D, cols = f["data"], list(f["cols"])
robots = [str(r) for r in f["robots"]]
G = {k: D[:, i] for i, k in enumerate(cols)}
lab = np.where((G["dbox"] < 0.5) & (G["wz"] >= 0.08) & (G["wz"] < 0.9), f["gt_group"][G["nbox"].astype(int)], -1)
keys = defaultdict(list)
for i, (r, s) in enumerate(zip(G["robot"].astype(int), G["step"].astype(int))):
    keys[(r, s)].append(i)
X, y, rob = [], [], []
for (r, s), idx in keys.items():
    idx = np.array(idx)
    cand = idx[(G["w"][idx] <= 0.5) & (G["h"][idx] >= 0.08) & (G["h"][idx] <= 0.6) & (np.abs(G["z"][idx] - G["rz"][idx]) <= 0.35)]
    if len(cand) < 2:
        continue
    L = fcluster(linkage(np.c_[G["x"][cand], G["y"][cand]], "single"), 0.7, "distance")
    for c in np.unique(L):
        m = cand[L == c]
        if not (2 <= len(m) <= 10):
            continue
        xy = np.c_[G["x"][m], G["y"][m]]
        rad = np.linalg.norm(xy - xy.mean(0), axis=1).max()
        if rad > 1.0:
            continue
        l = lab[m]
        v, cn = np.unique(l[l >= 0], return_counts=True)
        X.append([np.log(len(m)), rad, np.median(G["h"][m])])
        y.append(float(len(v) > 0 and cn.max() >= 0.5 * len(m)))
        rob.append(r)
X, y, rob = np.array(X), np.array(y), np.array(rob)


def fit(X, y):
    Xb = np.c_[np.ones(len(X)), X]
    w = np.zeros(Xb.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-Xb @ w))
        H = Xb.T @ (Xb * (p * (1 - p))[:, None]) + 1e-6 * np.eye(len(w))
        w += np.linalg.solve(H, Xb.T @ (y - p))
    return w


models = {}
for r in np.unique(rob):
    tr = rob != r
    mu, sd = X[tr].mean(0), X[tr].std(0)
    models[robots[r]] = dict(mean=mu.tolist(), std=sd.tolist(), coef=fit((X[tr] - mu) / sd, y[tr]).tolist(),
                             trained_on=[robots[k] for k in np.unique(rob[tr])], n_train=int(tr.sum()))
json.dump(dict(features=["log(members)", "radius_m", "median_member_h_m"], models=models), open(sys.argv[2], "w"), indent=1)
print(json.dumps(models, indent=1))
