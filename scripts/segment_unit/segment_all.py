"""FastSAM (TCAFF wrapper defaults) on all extracted frames of one robot; per-mask attributes only.

Writes one JSON line per frame with, for every mask: TCAFF measurement, border contact,
depth-layer statistics, containment of other masks, and GT-box composition.

Usage: python -I segment_all.py WEIGHTS ROBOT_DIR OBJECTS_TXT
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from fastsam import FastSAM, FastSAMPrompt

WEIGHTS, RDIR, OBJ = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
CONF, IOU, IMGSZ = 0.5, 0.9, 1024
FLOOR_Z, BOX_ZMAX, BOX_R, LAYER, CONTAIN = 0.03, 0.8, 0.25, 0.30, 0.8
torch.set_num_threads(12)
boxes = np.loadtxt(OBJ, delimiter=",")


def rvl_decode(buf, n):
    words = np.frombuffer(buf[: len(buf) // 4 * 4], dtype="<u4").astype(np.uint64)
    shifts = np.arange(28, -1, -4, dtype=np.uint64)
    nib = ((words[:, None] >> shifts[None, :]) & 0xF).astype(np.int64).ravel()
    end = (nib & 8) == 0
    ends = np.flatnonzero(end)
    gid = np.concatenate([[0], np.cumsum(end)[:-1]])
    starts = np.concatenate([[0], ends[:-1] + 1])
    pos = np.arange(len(nib)) - starts[gid]
    vals = np.bincount(gid, weights=((nib & 7) << (3 * pos)).astype(np.float64)).astype(np.int64)
    i = p = 0
    pos_chunks, val_chunks = [], []
    while p < n:
        z, k = vals[i], vals[i + 1]
        i += 2
        p += z
        if k:
            pos_chunks.append(np.arange(p, p + k))
            val_chunks.append(vals[i:i + k])
            p += k
            i += k
    out = np.zeros(n, np.int64)
    if pos_chunks:
        pz = np.concatenate(val_chunks)
        out[np.concatenate(pos_chunks)] = np.cumsum((pz >> 1) ^ -(pz & 1))
    return out.astype(np.uint16)


man = json.load(open(RDIR / "manifest.json"))
K, T_BC = man["K"], np.array(man["T_BC"])
fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
model = FastSAM(WEIGHTS)
out = open(RDIR / "masks.jsonl", "a")
done = set()
if (RDIR / "masks.jsonl").stat().st_size:
    done = {json.loads(l)["index"] for l in open(RDIR / "masks.jsonl")}
t_start = time.time()
for n_done, fr in enumerate(man["frames"]):
    if fr["index"] in done:
        continue
    rec = {"index": fr["index"], "time": fr["time"], "masks": None}
    if fr["has_frame"]:
        stem = RDIR / "frames" / f"{fr['index']:05d}"
        bgr = cv2.imdecode(np.frombuffer(stem.with_suffix(".jpg").read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        H, W = bgr.shape[:2]
        depth = rvl_decode(stem.with_suffix(".rvl").read_bytes()[20:], H * W).reshape(H, W).astype(np.float64) * 1e-3
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = model(rgb, retina_masks=True, device="cpu", imgsz=IMGSZ, conf=CONF, iou=IOU, verbose=False)
        masks = FastSAMPrompt(rgb, res, device="cpu").everything_prompt()
        masks = masks.cpu().numpy() > 0.5 if len(masks) else np.zeros((0, H, W), bool)
        confs = res[0].boxes.conf.cpu().numpy().tolist() if len(masks) else []
        T_WC = np.array(fr["T_WB"]) @ T_BC if fr.get("T_WB") else None
        small = masks[:, ::4, ::4]
        areas = small.reshape(len(small), -1).sum(axis=1) if len(small) else np.zeros(0, int)  # frames without masks
        items = []
        for mi, m in enumerate(masks):
            ys, xs = np.nonzero(m)
            if xs.size < 10:
                continue
            mean = (float(xs.mean()), float(ys.mean()))
            cov = np.cov(np.vstack((xs, ys)))
            d_all = depth[ys, xs]
            valid = d_all >= 1e-3
            it = {"i": mi, "conf": confs[mi] if mi < len(confs) else None, "n_pix": int(xs.size),
                  "border": bool(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any()),
                  "pts_img": list(mean), "cov_img": cov.tolist(), "valid_frac": float(valid.mean()),
                  "contains": [int(j) for j in range(len(small)) if j != mi and 0 < areas[j] < areas[mi]
                               and (small[j] & small[mi]).sum() / areas[j] >= CONTAIN]}
            if valid.any():
                dv = d_all[valid]
                dmed = float(np.median(dv))
                c = [(mean[0] - cx) / fx * dmed, (mean[1] - cy) / fy * dmed, dmed]
                it.update({"c_cam": c, "w": float((xs.max() - xs.min()) / fx * dmed), "h": float((ys.max() - ys.min()) / fy * dmed),
                           "frac_behind": float(np.mean(dv > dmed + LAYER)), "frac_front": float(np.mean(dv < dmed - LAYER))})
                if T_WC is not None:
                    sel = np.flatnonzero(valid)
                    if sel.size > 2000:
                        sel = sel[np.linspace(0, sel.size - 1, 2000).astype(int)]
                    u, v, d = xs[sel], ys[sel], d_all[sel]
                    Pw = (T_WC @ np.stack([(u - cx) / fx * d, (v - cy) / fy * d, d, np.ones_like(d)]))[:3].T
                    dist = np.linalg.norm(Pw[:, None, :2] - boxes[None], axis=2)
                    nb, nd = dist.argmin(axis=1), dist.min(axis=1)
                    is_box = (Pw[:, 2] >= FLOOR_Z) & (Pw[:, 2] <= BOX_ZMAX) & (nd < BOX_R)
                    counts = np.bincount(nb[is_box], minlength=len(boxes))
                    cov_boxes = np.flatnonzero(counts >= max(1, 0.1 * is_box.sum())) if is_box.any() else np.array([], int)
                    it.update({"share_box": float(is_box.mean()), "boxes": cov_boxes.tolist()})
            items.append(it)
        rec["masks"] = items
    out.write(json.dumps(rec) + "\n")
    out.flush()
    if n_done % 200 == 0:
        print(f"{RDIR.name} {n_done + 1}/{len(man['frames'])} elapsed {time.time() - t_start:.0f}s", flush=True)
print(f"{RDIR.name} done in {time.time() - t_start:.0f}s", flush=True)
