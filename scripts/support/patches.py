"""Runtime variants of the H2/H3 protocol (configs/support_hypotheses.json).

TCAFF classes are patched in the running process; vendor code is unchanged.
  S2: size similarity fused into the CLIPPER affinity by geometric mean (ROMAN eq. 3).
  R3: robust (GNC Geman-McClure) re-estimation of each CLIPPER solution before the transform is returned.
apply(variant) also times TCAFFManager.update; G2 and B0 need no patch (G2 is a parameter override).
"""
import atexit
import json
import os
import time

import clipperpy
import numpy as np
from tcaff.realign.frame_aligner import FrameAligner, FrameAlignSolution
from tcaff.realign.wls import wls, wls_residual
from tcaff.tcaff.tcaff_manager import TCAFFManager
from tcaff.utils.transform import T2d_2_T3d
from tcaff.utils.transform import transform as perform_transform

GM_C = 0.5
GNC_FACTOR = 1.4
GNC_MAX_ITER = 200

_orig_put = TCAFFManager.get_putative_assoc
_orig_compose = FrameAligner.compose_solution
_orig_update = TCAFFManager.update


def size_similarity(map1, map2, A):
    if len(A) == 0:
        return np.zeros(0)
    w1, w2 = np.asarray(map1.widths).reshape(-1), np.asarray(map2.widths).reshape(-1)
    h1, h2 = np.asarray(map1.heights).reshape(-1), np.asarray(map2.heights).reshape(-1)
    wa, wb, ha, hb = w1[A[:, 0]], w2[A[:, 1]], h1[A[:, 0]], h2[A[:, 1]]
    return np.sqrt(np.minimum(wa, wb) / np.maximum(wa, wb) * np.minimum(ha, hb) / np.maximum(ha, hb))


def soft_affinity(M, s):
    out = M.copy()
    off = M > 0
    np.fill_diagonal(off, False)
    i, j = np.nonzero(off)
    out[i, j] = np.cbrt(M[i, j] * s[i] * s[j])
    return out


def _put_with_similarity(self, map1, map2):
    A = _orig_put(self, map1, map2)
    self.fa._size_similarity = (np.asarray(A).copy(), size_similarity(map1, map2, np.asarray(A)))
    return A


def _realign_soft(self, detections1, detections2, downweight_nodes=None, putative_assoc=None):
    """realign_static_downweight of TCAFF with the affinity replaced by its soft version."""
    iparams = clipperpy.invariants.EuclideanDistanceParams()
    iparams.sigma = self.sigma
    iparams.epsilon = self.epsilon
    invariant = clipperpy.invariants.EuclideanDistance(iparams)
    params = clipperpy.Params()
    clipper = clipperpy.CLIPPER(invariant, params)
    assert putative_assoc is not None, "S2 needs the gated putative associations"
    A = putative_assoc.astype(np.int32)
    stored_A, s = self._size_similarity
    assert np.array_equal(stored_A, putative_assoc), "size similarity computed for other associations"
    clipper.score_pairwise_consistency(detections1.T, detections2.T, A)
    assert np.array_equal(np.asarray(clipper.get_initial_associations()), A)
    M_orig = soft_affinity(clipper.get_affinity_matrix(), s)
    C = clipper.get_constraint_matrix()
    M = M_orig.copy()
    for idx, downweight in (downweight_nodes or []):
        rows, cols = np.meshgrid(idx, idx, indexing='ij')
        if len(rows) != 0 and len(cols) != 0:
            M[rows, cols] *= downweight
    clipper = clipperpy.CLIPPER(invariant, params)
    clipper.set_matrix_data(M=M, C=C)
    clipper.solve()
    _timing["soft_solves"] += 1
    nodes = clipper.get_solution().nodes
    Ain = np.zeros((len(nodes), 2)).astype(np.int64)
    for i in range(len(nodes)):
        Ain[i, :] = A[nodes[i], :]
    u_sol = clipper.get_solution().u.copy()
    for i in range(u_sol.shape[0]):
        u_sol[i] = u_sol[i] if i in nodes else 0.0
    score = 0 if len(nodes) == 0 else u_sol.T @ M_orig @ u_sol / (u_sol.T @ u_sol)
    return Ain, score, nodes


def robust_transform(p1, p2, w):
    """GNC with Geman-McClure loss around WLS (Yang et al., RA-L 2020); returns (T 3x3, robust weights).

    Weights (mu c^2 / (r^2 + mu c^2))^2 multiply the release recency weights; mu starts at
    2 max(r^2) / c^2 and is divided by GNC_FACTOR until it reaches 1, where a last fit is made.
    """
    T = wls(p1, p2, w)
    r2 = np.sum((p1 - perform_transform(T, p2, stacked_axis=0)) ** 2, axis=1)
    mu = max(2 * r2.max() / GM_C ** 2, 1.0)
    g = np.ones(len(p1))
    for _ in range(GNC_MAX_ITER):
        g = (mu * GM_C ** 2 / (r2 + mu * GM_C ** 2)) ** 2
        T = wls(p1, p2, w * g)
        r2 = np.sum((p1 - perform_transform(T, p2, stacked_axis=0)) ** 2, axis=1)
        if mu == 1.0:
            break
        mu = max(mu / GNC_FACTOR, 1.0)
    return T, g


def _compose_robust(self, objs1_corres, objs2_corres, weights_corres):
    num_objs = len(objs1_corres)
    if num_objs < self.num_objs_req:
        return FrameAlignSolution(success=False)
    p1, p2 = np.asarray(objs1_corres), np.asarray(objs2_corres)
    w = np.asarray(weights_corres).reshape(-1)
    T_new, _ = robust_transform(p1, p2, w)
    _timing["robust_fits"] += 1
    if T_new.shape[0] == 3 and T_new.shape[1] == 3:
        T_new = T2d_2_T3d(T_new)
    residual = wls_residual(objs1_corres, objs2_corres, weights_corres, T_new)
    return FrameAlignSolution(success=True, transform=T_new, transform_residual=residual, num_objs_associated=num_objs)


_timing = {"calls": 0, "seconds": 0.0, "soft_solves": 0, "robust_fits": 0}


def _timed_update(self, *args, **kwargs):
    began = time.perf_counter()
    try:
        return _orig_update(self, *args, **kwargs)
    finally:
        _timing["calls"] += 1
        _timing["seconds"] += time.perf_counter() - began


def apply(variant):
    assert variant in ("B0", "G2", "S2", "R3"), variant
    if variant == "S2":
        TCAFFManager.get_putative_assoc = _put_with_similarity
        FrameAligner.realign_static_downweight = _realign_soft
    if variant == "R3":
        FrameAligner.compose_solution = _compose_robust
    TCAFFManager.update = _timed_update
    out = os.environ.get("SUPPORT_TIMING_FILE")
    if out:
        atexit.register(lambda: open(out, "w").write(json.dumps(dict(variant=variant, **_timing))))


GATE_OVERRIDE = {"B0": None, "R3": None, "G2": {"wh_scale_diff": 1.7, "h_diff": 0.2}, "S2": {"wh_scale_diff": 1.7, "h_diff": 0.2}}
