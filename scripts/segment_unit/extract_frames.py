"""Extract a time-uniform sample of L515 frames that produced the saved FastSAM JSON.

For each sampled JSON frame: the aligned depth message with the same stamp (raw RVL payload),
the nearest colour JPEG, camera K, GT body pose, T_BC and the saved detections.

Usage: python -I extract_frames.py LAB_ROOT OUT_DIR FRAMES_PER_ROBOT
"""
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from robotdatapy.data import PoseData
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

LAB, OUT, N = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
DATA = LAB / "data/tcaff_mot_data"
ROBOTS = ["RR01", "RR04", "RR06", "RR08"]
params = yaml.safe_load(open(LAB / "vendor/tcaff/demo/params/tcaff_mot_dataset.yaml"))
(OUT / "frames").mkdir(parents=True, exist_ok=True)
manifest = []


def stamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


for r in ROBOTS:
    frames = json.load(open(DATA / f"fastsam_data/{r}.json"))
    gt = PoseData.from_bag(path=str(DATA / f"data/kimera_odom/{r}.bag"), topic=f"/{r}/world", time_tol=10., interp=True)
    valid = [i for i, f in enumerate(frames) if gt.t0 <= f["time"] <= gt.tf]
    picks = [valid[int(k)] for k in np.linspace(0, len(valid) - 1, N)]
    bag = next((DATA / "data").glob(f"{r}_compressed*.bag"))
    with AnyReader([bag], default_typestore=get_typestore(Stores.ROS1_NOETIC)) as reader:
        conn = {c.topic: c for c in reader.connections}
        c_rgb = conn[f"/{r}/l515/color/image_raw/compressed"]
        c_dep = conn[f"/{r}/l515/aligned_depth_to_color/image_raw/compressedDepth"]
        c_inf = conn[f"/{r}/l515/color/camera_info"]
        _, _, raw = next(reader.messages(connections=[c_inf]))
        info = reader.deserialize(raw, c_inf.msgtype)
        K = np.asarray(info.K).reshape(3, 3).tolist()
        for i in picks:
            t = frames[i]["time"]
            win = dict(start=int((t - 0.2) * 1e9), stop=int((t + 0.2) * 1e9))
            deps = [(abs(stamp(m) - t), m) for m in (reader.deserialize(x, c_dep.msgtype) for _, _, x in reader.messages(connections=[c_dep], **win))]
            rgbs = [(abs(stamp(m) - t), m) for m in (reader.deserialize(x, c_rgb.msgtype) for _, _, x in reader.messages(connections=[c_rgb], **win))]
            if not deps or not rgbs:
                continue
            dd, dm = min(deps, key=lambda x: x[0])
            dr, rm = min(rgbs, key=lambda x: x[0])
            key = f"{r}_{i:05d}"
            (OUT / "frames" / f"{key}.jpg").write_bytes(bytes(rm.data))
            (OUT / "frames" / f"{key}.rvl").write_bytes(bytes(dm.data))
            manifest.append({"key": key, "robot": r, "json_index": i, "json_time": t, "depth_dt": dd, "rgb_dt": dr,
                             "depth_format": dm.format, "K": K, "image_size": [info.width, info.height],
                             "T_WB": gt.T_WB(t).tolist(), "T_BC": params[r]["T_BC"],
                             "saved": {k: frames[i][k] for k in ("detections", "pts_img", "widths", "heights")}})
    print(r, "frames", sum(m["robot"] == r for m in manifest), flush=True)

json.dump(manifest, open(OUT / "manifest.json", "w"))
dd = [m["depth_dt"] for m in manifest]
dr = [m["rgb_dt"] for m in manifest]
print("total", len(manifest), "max depth stamp diff %.4f s, rgb diff median %.4f max %.4f" % (max(dd), np.median(dr), max(dr)))
