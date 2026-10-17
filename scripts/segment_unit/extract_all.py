"""Extract every L515 frame used by the saved FastSAM JSON for one robot (sequential bag read).

Usage: python -I extract_all.py LAB_ROOT OUT_DIR ROBOT [MAX_FRAMES]
"""
import json
import sys
from pathlib import Path

import yaml
from robotdatapy.data import PoseData
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

LAB, OUT, R = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
MAX = int(sys.argv[4]) if len(sys.argv) > 4 else None
DATA = LAB / "data/tcaff_mot_data"
params = yaml.safe_load(open(LAB / "vendor/tcaff/demo/params/tcaff_mot_dataset.yaml"))
frames = json.load(open(DATA / f"fastsam_data/{R}.json"))
if MAX:
    frames = frames[:MAX]
gt = PoseData.from_bag(path=str(DATA / f"data/kimera_odom/{R}.bag"), topic=f"/{R}/world", time_tol=10., interp=True)
need = {round(f["time"] * 1e6): i for i, f in enumerate(frames)}
fdir = OUT / R / "frames"
fdir.mkdir(parents=True, exist_ok=True)
got_d, got_c = set(), set()
bag = next((DATA / "data").glob(f"{R}_compressed*.bag"))
with AnyReader([bag], default_typestore=get_typestore(Stores.ROS1_NOETIC)) as reader:
    conn = {c.topic: c for c in reader.connections}
    c_rgb = conn[f"/{R}/l515/color/image_raw/compressed"]
    c_dep = conn[f"/{R}/l515/aligned_depth_to_color/image_raw/compressedDepth"]
    c_inf = conn[f"/{R}/l515/color/camera_info"]
    _, _, raw = next(reader.messages(connections=[c_inf]))
    info = reader.deserialize(raw, c_inf.msgtype)
    K = [list(info.K[0:3]), list(info.K[3:6]), list(info.K[6:9])]
    t_end = max(f["time"] for f in frames) + 1.0
    for con, _, raw in reader.messages(connections=[c_rgb, c_dep], stop=int(t_end * 1e9)):
        msg = reader.deserialize(raw, con.msgtype)
        key = round((msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9) * 1e6)
        if key not in need:
            continue
        i = need[key]
        if con is c_dep and i not in got_d:
            (fdir / f"{i:05d}.rvl").write_bytes(bytes(msg.data))
            got_d.add(i)
        elif con is c_rgb and i not in got_c:
            (fdir / f"{i:05d}.jpg").write_bytes(bytes(msg.data))
            got_c.add(i)
manifest = []
for i, f in enumerate(frames):
    rec = {"index": i, "time": f["time"], "has_frame": i in got_d and i in got_c}
    if rec["has_frame"]:
        try:
            rec["T_WB"] = gt.T_WB(f["time"]).tolist()
        except Exception:
            rec["T_WB"] = None
    manifest.append(rec)
json.dump({"robot": R, "K": K, "T_BC": params[R]["T_BC"], "frames": manifest}, open(OUT / R / "manifest.json", "w"))
print(R, "json frames", len(frames), "with depth+colour", sum(m["has_frame"] for m in manifest),
      "with GT pose", sum(1 for m in manifest if m.get("T_WB")), flush=True)
