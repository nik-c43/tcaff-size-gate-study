"""Confidence-weighted on-board cluster candidates for TCAFF frame alignment (test10_protocol.json).

Applied to a running demo process; vendor and demo code are unchanged. No ground truth is used.
apply() patches:
  Robot.__init__ / update_mapping     registry of robots and their latest mapping time
  Robot.update_frame_alignments       passes both robots' odometry heights to the manager update
  TCAFFManager.update                 builds clusters of floor obstacles, appends them as pseudo-objects
  TCAFFManager.get_putative_assoc     size gate on real objects + all cluster pairs
  FrameAligner.realign_static_downweight  affinity with cluster tolerances and M' = D M D
"""
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

SIGMA_G, EPS_G = 0.30, 0.50
STATE = {"active": False, "z": (None, None)}
REGISTRY = {}


def clusters(m, z_robot):
    C, W, H = np.asarray(m.centroids, float), np.asarray(m.widths, float).reshape(-1), np.asarray(m.heights, float).reshape(-1)
    cand = np.flatnonzero((W <= 0.5) & (H >= 0.08) & (H <= 0.6) & (np.abs(C[:, 2] - z_robot) <= 0.35))
    if len(cand) < 2:
        return []
    lab = fcluster(linkage(C[cand, :2], "single"), 0.7, "distance")
    out = []
    for c in np.unique(lab):
        mem = cand[lab == c]
        if not (2 <= len(mem) <= 10):
            continue
        r = float(np.linalg.norm(C[mem, :2] - C[mem, :2].mean(0), axis=1).max())
        if r > 1.0:
            continue
        h = float(np.median(H[mem]))
        q = min(1.0, (len(mem) - 1) / 3) * min(1.0, r / 0.3) * float(np.clip((h - 0.1) / 0.1, 0, 1))
        out.append((mem, q))
    return out


def extend(m, cl):
    from tcaff.realign.object_map import ObjectMap
    if not cl:
        return m
    C = np.asarray(m.centroids, float)
    A = np.asarray(m.ages, float).reshape(-1)
    W, H = np.asarray(m.widths, float), np.asarray(m.heights, float)
    newC = np.array([C[mem].mean(0) for mem, _ in cl])
    newA = np.array([2.0 * max(A[mem].min(), 0.05) / np.sqrt(max(q, 1e-6)) for mem, q in cl])
    pad = lambda X: np.concatenate([X, np.zeros((len(cl),) + X.shape[1:])])
    return ObjectMap(np.concatenate([C, newC]), pad(W), pad(H), np.concatenate([A, newA]))


def apply():
    import clipperpy
    import robot as demo_robot
    from tcaff.realign.frame_aligner import FrameAligner
    from tcaff.tcaff.tcaff_manager import TCAFFManager

    Robot = demo_robot.Robot
    orig_init, orig_map = Robot.__init__, Robot.update_mapping
    orig_update, orig_put = TCAFFManager.update, TCAFFManager.get_putative_assoc
    orig_realign = FrameAligner.realign_static_downweight

    def init(self, *a, **k):
        orig_init(self, *a, **k)
        REGISTRY[self.name] = self
        self._t_map = None

    def update_mapping(self, t):
        self._t_map = t
        return orig_map(self, t)

    def z_of(rb):
        try:
            return float(rb.pose_est_data.T_WB(rb._t_map)[2, 3])
        except Exception:
            return None

    def update_frame_alignments(self):
        for robot in self.neighbors:
            STATE["z"] = (z_of(self), z_of(REGISTRY[robot]) if robot in REGISTRY else None)
            self.frame_align_filters[robot].update(self.get_map(), self.neighbor_maps[robot])
            if self.mot is not None:
                self.mot.neighbor_frame_align[robot] = self.frame_align_filters[robot].T
                self.mot.neighbor_frame_align_cov[robot] = self.frame_align_filters[robot].P

    def update(self, ego_map, other_map):
        za, zb = STATE["z"]
        if len(ego_map) == 0 or len(other_map) == 0 or za is None or zb is None:
            return orig_update(self, ego_map, other_map)
        ca, cb = clusters(ego_map, za), clusters(other_map, zb)
        STATE.update(active=True, real1=ego_map, real2=other_map, n1=len(ego_map), n2=len(other_map),
                     k1=len(ca), k2=len(cb), qa=[q for _, q in ca], qb=[q for _, q in cb])
        try:
            return orig_update(self, extend(ego_map, ca), extend(other_map, cb))
        finally:
            STATE["active"] = False

    def put(self, map1, map2):
        if not STATE["active"]:
            return orig_put(self, map1, map2)
        A = np.asarray(orig_put(self, STATE["real1"], STATE["real2"]), dtype=np.int64).reshape(-1, 2)
        n1, n2 = STATE["n1"], STATE["n2"]
        extra = np.array([[n1 + K, n2 + L] for K in range(STATE["k1"]) for L in range(STATE["k2"])], dtype=np.int64).reshape(-1, 2)
        return np.concatenate([A, extra])

    def realign(self, detections1, detections2, downweight_nodes=None, putative_assoc=None):
        if not STATE["active"]:
            return orig_realign(self, detections1, detections2, downweight_nodes, putative_assoc)
        A = putative_assoc.astype(np.int64)
        P1, P2 = np.asarray(detections1)[:, :2], np.asarray(detections2)[:, :2]
        l1 = np.linalg.norm(P1[A[:, 0]][:, None, :] - P1[A[:, 0]][None, :, :], axis=2)
        l2 = np.linalg.norm(P2[A[:, 1]][:, None, :] - P2[A[:, 1]][None, :, :], axis=2)
        c = np.abs(l1 - l2)
        cl = A[:, 0] >= STATE["n1"]
        anyc = cl[:, None] | cl[None, :]
        sig = np.where(anyc, SIGMA_G, self.sigma)
        eps = np.where(anyc, EPS_G, self.epsilon)
        M_orig = np.where(c < eps, np.exp(-0.5 * c ** 2 / sig ** 2), 0.0)
        shared = (A[:, 0][:, None] == A[:, 0][None, :]) | (A[:, 1][:, None] == A[:, 1][None, :])
        M_orig[shared] = 0.0
        np.fill_diagonal(M_orig, 1.0)
        if cl.any():
            qn = np.ones(len(A))
            qn[cl] = np.sqrt(np.asarray(STATE["qa"])[A[cl, 0] - STATE["n1"]] * np.asarray(STATE["qb"])[A[cl, 1] - STATE["n2"]])
            M_orig = qn[:, None] * M_orig * qn[None, :]
        iparams = clipperpy.invariants.EuclideanDistanceParams()
        iparams.sigma, iparams.epsilon = self.sigma, self.epsilon
        invariant = clipperpy.invariants.EuclideanDistance(iparams)
        cparams = clipperpy.Params()
        C = (M_orig > 0).astype(float)
        M = M_orig.copy()
        for idx, downweight in (downweight_nodes or []):
            rows, cols = np.meshgrid(idx, idx, indexing="ij")
            if len(rows) and len(cols):
                M[rows, cols] *= downweight
        clipper = clipperpy.CLIPPER(invariant, cparams)
        clipper.set_matrix_data(M=M, C=C)
        clipper.solve()
        nodes = clipper.get_solution().nodes
        Ain = np.zeros((len(nodes), 2)).astype(np.int64)
        for i in range(len(nodes)):
            Ain[i, :] = A[nodes[i], :]
        u = clipper.get_solution().u.copy()
        for i in range(u.shape[0]):
            u[i] = u[i] if i in nodes else 0.0
        score = 0 if len(nodes) == 0 else u.T @ M_orig @ u / (u.T @ u)
        return Ain, score, nodes

    Robot.__init__, Robot.update_mapping, Robot.update_frame_alignments = init, update_mapping, update_frame_alignments
    TCAFFManager.update, TCAFFManager.get_putative_assoc = update, put
    FrameAligner.realign_static_downweight = realign
