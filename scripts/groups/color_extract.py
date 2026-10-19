"""Mean colour of every saved FastSAM detection of one robot (test6_protocol.json, extraction).

One sequential pass over the robot's L515 colour stream. For each detection the mask is approximated
by its pixel mean (pts_img) and covariance (covs_img); m = Mahalanobis distance of a pixel.
  core   : uniform mean over m <= 1
  full   : uniform mean over m <= 2
  centre : mean with weights exp(-m^2/2) over m <= 2
No depth-based trimming. Usage: python -I color_extract.py LAB_ROOT OUT_DIR ROBOT
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

LAB, OUT, R = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
DATA = LAB / "data/tcaff_mot_data"
frames = json.load(open(DATA / f"fastsam_data/{R}.json"))
need = {round(f["time"] * 1e6): i for i, f in enumerate(frames)}
rows = []
done = set()
bag = next((DATA / "data").glob(f"{R}_compressed*.bag"))
with AnyReader([bag], default_typestore=get_typestore(Stores.ROS1_NOETIC)) as reader:
    conn = {c.topic: c for c in reader.connections}
    c_rgb = conn[f"/{R}/l515/color/image_raw/compressed"]
    t_end = max(f["time"] for f in frames) + 1.0
    for con, _, raw in reader.messages(connections=[c_rgb], stop=int(t_end * 1e9)):
        msg = reader.deserialize(raw, con.msgtype)
        key = round((msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9) * 1e6)
        i = need.get(key)
        if i is None or i in done:
            continue
        done.add(i)
        bgr = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)
        rgb = bgr[:, :, ::-1].astype(np.float64)
        H, W = rgb.shape[:2]
        fr = frames[i]
        for j, (c, mu, cov) in enumerate(zip(fr["detections"], fr["pts_img"], fr["covs_img"])):
            S = np.asarray(cov, float) + np.eye(2) * 1e-6
            Si = np.linalg.inv(S)
            hx, hy = 2.0 * np.sqrt(max(S[0, 0], 1e-6)), 2.0 * np.sqrt(max(S[1, 1], 1e-6))
            x0, x1 = int(max(0, np.floor(mu[0] - hx))), int(min(W - 1, np.ceil(mu[0] + hx)))
            y0, y1 = int(max(0, np.floor(mu[1] - hy))), int(min(H - 1, np.ceil(mu[1] + hy)))
            rec = dict(frame=i, det=j, time=fr["time"], c=c, npix_core=0, npix_full=0,
                       core=[np.nan] * 3, full=[np.nan] * 3, centre=[np.nan] * 3)
            if x1 >= x0 and y1 >= y0:
                xs, ys = np.meshgrid(np.arange(x0, x1 + 1) - mu[0], np.arange(y0, y1 + 1) - mu[1])
                m2 = Si[0, 0] * xs ** 2 + 2 * Si[0, 1] * xs * ys + Si[1, 1] * ys ** 2
                patch = rgb[y0:y1 + 1, x0:x1 + 1]
                core, full = m2 <= 1.0, m2 <= 4.0
                if core.any():
                    rec["core"] = patch[core].mean(0).tolist()
                    rec["npix_core"] = int(core.sum())
                if full.any():
                    rec["full"] = patch[full].mean(0).tolist()
                    rec["npix_full"] = int(full.sum())
                    w = np.exp(-0.5 * m2[full])
                    rec["centre"] = (patch[full] * w[:, None]).sum(0) / w.sum()
                    rec["centre"] = rec["centre"].tolist()
            rows.append(rec)
        if len(done) % 500 == 0:
            print(R, len(done), "/", len(frames), flush=True)
OUT.mkdir(parents=True, exist_ok=True)
np.savez_compressed(OUT / f"{R}_colors.npz",
                    frame=np.array([r["frame"] for r in rows]), det=np.array([r["det"] for r in rows]),
                    time=np.array([r["time"] for r in rows]), c=np.array([r["c"] for r in rows], float),
                    npix_core=np.array([r["npix_core"] for r in rows]), npix_full=np.array([r["npix_full"] for r in rows]),
                    core=np.array([r["core"] for r in rows], float), full=np.array([r["full"] for r in rows], float),
                    centre=np.array([r["centre"] for r in rows], float))
print(R, "frames", len(done), "of", len(frames), "detections", len(rows), flush=True)
