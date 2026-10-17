"""Within-robot self-supervised pairs for calibration feasibility.

Runs the TCAFF map tracker (release parameters) on saved FastSAM detections with GT poses,
records which track each detection updates, and writes per-detection features.

Usage: python -I extract.py LAB_ROOT OUT_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np
import yaml

LAB, OUT = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(LAB / "vendor" / "tcaff"))

from robotdatapy.data import PoseData  # noqa: E402
from tcaff.mot.motion_model import MotionModel  # noqa: E402
from tcaff.mot.multi_object_tracker import MultiObjectTracker  # noqa: E402
from tcaff.mot import track as track_module  # noqa: E402

DATA = LAB / "data" / "tcaff_mot_data"
ROBOTS = ["RR01", "RR04", "RR06", "RR08"]
DYN_RADIUS = 0.6  # m, exclude detections near pedestrians or other robots
params = yaml.safe_load(open(LAB / "vendor/tcaff/demo/params/tcaff_mot_dataset.yaml"))
mp = params["mapping"]

# Record every track update: (track id, measurement vector).
LOG = []
_orig_update = track_module.Track.update


def logged_update(self, measurements, R):
    LOG.append((self._id, np.array(measurements[0]).reshape(-1).copy()))
    return _orig_update(self, measurements, R)


track_module.Track.update = logged_update


def load_pose(path, topic):
    return PoseData.from_bag(path=str(path), topic=topic, time_tol=10., interp=True)


gt = {r: load_pose(DATA / f"data/kimera_odom/{r}.bag", f"/{r}/world") for r in ROBOTS}
peds = []
for k in range(1, 7):
    try:
        peds.append(load_pose(DATA / "data/pedestrian_gt.bag", f"/PED{k}/world"))
    except Exception as e:  # missing pedestrian topic is reported, not hidden
        print(f"PED{k}: not loaded ({e!r})")


def positions_at(t, poses):
    out = []
    for p in poses:
        try:
            out.append(p.T_WB(t)[:2, 3])
        except Exception:
            pass
    return np.array(out).reshape(-1, 2)


img_w = img_h = 0
for r in ROBOTS:
    for fr in json.load(open(DATA / f"fastsam_data/{r}.json")):
        for u, v in fr["pts_img"]:
            img_w, img_h = max(img_w, u), max(img_h, v)
W, H = (1280, 720) if img_w < 1280 and img_h < 720 else (int(np.ceil(img_w)), int(np.ceil(img_h)))
print("pixel centroid max", round(img_w), round(img_h), "-> image size assumed", W, H)

all_rows = []
for r in ROBOTS:
    T_BC = np.array(params[r]["T_BC"]).reshape(4, 4)
    dim = mp["dim"]
    Q = np.diag([*[mp["Q_el"]] * (dim - 2), mp["Q_el_w"], mp["Q_el_h"]])
    P0 = np.diag([*[mp["P0_el"]] * (dim - 2), mp["P0_el_w"], mp["P0_el_h"]])
    R = np.diag([mp["R_el"]] * dim)
    model = MotionModel(A=np.eye(dim), H=np.eye(dim), Q=Q, R=np.array([]), P0=P0)
    mot = MultiObjectTracker(camera_id=r, connected_cams=[], track_motion_model=model, tau_global=0.,
                             tau_local=mp["tau"], alpha=2000, kappa=mp["kappa"], nu=mp["nu"],
                             track_storage_size=1, dim_association=dim)
    others = [gt[o] for o in ROBOTS if o != r]
    frames = json.load(open(DATA / f"fastsam_data/{r}.json"))
    n_dyn = 0
    for fi, fr in enumerate(frames):
        t = fr["time"]
        try:
            T_WB = gt[r].T_WB(t)
        except Exception:
            continue
        T_WC = T_WB @ T_BC
        dyn = np.vstack([positions_at(t, peds), positions_at(t, others)])
        zs, rows = [], []
        cams = [np.array(c, float) for c in fr["detections"]]
        rng = [np.linalg.norm(c) for c in cams]
        boxes2d = []
        for (u, v), cov in zip(fr["pts_img"], fr["covs_img"]):
            su, sv = np.sqrt(max(cov[0][0], 0)), np.sqrt(max(cov[1][1], 0))
            boxes2d.append((u - 3 ** .5 * su, u + 3 ** .5 * su, v - 3 ** .5 * sv, v + 3 ** .5 * sv))
        for di, (c, w, h) in enumerate(zip(cams, fr["widths"], fr["heights"])):
            zb = T_BC @ np.array([*c, 1.0])
            if not (mp["zmin"] <= zb[0] <= mp["zmax"]):
                continue
            pw = T_WB @ zb
            if len(dyn) and np.min(np.linalg.norm(dyn - pw[:2], axis=1)) < DYN_RADIUS:
                n_dyn += 1
                continue
            u0, u1, v0, v1 = boxes2d[di]
            margin = min(u0, W - u1, v0, H - v1)
            cov = np.array(fr["covs_img"][di])
            occluded = False
            for dj, (a0, a1, b0, b1) in enumerate(boxes2d):
                if dj == di:
                    continue
                ov = max(0, min(u1, a1) - max(u0, a0)) * max(0, min(v1, b1) - max(v0, b0))
                area = max((u1 - u0) * (v1 - v0), 1e-9)
                if ov / area > 0.2 and rng[dj] < rng[di] - 0.1:
                    occluded = True
                    break
            cam = T_WC[:3, 3]
            rows.append({"robot": r, "frame": fi, "det": di, "t": t, "x": pw[0], "y": pw[1], "z": pw[2],
                         "w": w, "h": h, "cam_x": cam[0], "cam_y": cam[1], "range": rng[di],
                         "off_axis_deg": float(np.degrees(np.arctan2(np.hypot(c[0], c[1]), c[2]))),
                         "border_margin_px": float(margin), "mask_scale_px": float(np.linalg.det(cov) ** 0.25),
                         "occluded": occluded})
            zs.append(np.array([pw[0], pw[1], pw[2], w, h]).reshape(-1, 1))
        LOG.clear()
        if zs:
            mot.local_data_association(zs, np.arange(len(zs)), [R.copy() for _ in zs])
            mot.dkf()
            mot.track_manager()
        for tid, z in LOG:
            for k, zk in enumerate(zs):
                if np.allclose(zk.reshape(-1), z):
                    rows[k]["track"] = f"{tid[0]}:{tid[1]}"
                    break
        all_rows.extend(rows)
    print(r, "frames", len(frames), "static detections", sum(1 for x in all_rows if x["robot"] == r),
          "excluded near pedestrians/robots", n_dyn, flush=True)

OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / "detections.jsonl", "w") as f:
    for row in all_rows:
        f.write(json.dumps(row) + "\n")
print("written", len(all_rows), "rows; without track", sum(1 for x in all_rows if "track" not in x))
