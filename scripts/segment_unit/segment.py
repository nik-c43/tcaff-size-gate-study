"""Run FastSAM (TCAFF wrapper defaults) on sampled L515 frames and record per-mask diagnostics.

Per mask: TCAFF measurement (pixel mean/cov, centroid at median depth, width/height),
depth statistics inside and around the mask, and GT-box composition of the back-projected pixels.

Usage: python -I segment.py WEIGHTS DATA_DIR OBJECTS_TXT [MAX_FRAMES]
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from fastsam import FastSAM, FastSAMPrompt

WEIGHTS, DATA, OBJ = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
MAX_FRAMES = int(sys.argv[4]) if len(sys.argv) > 4 else None
CONF, IOU, IMGSZ = 0.5, 0.9, 1024          # TCAFF FastSAM3D defaults
FLOOR_Z, BOX_ZMAX, BOX_R = 0.03, 0.8, 0.25  # m, GT box assignment of back-projected points
LAYER = 0.30                                # m, depth layer threshold inside the mask
RING = 0.15                                 # m, boundary ring threshold
torch.set_num_threads(12)
boxes = np.loadtxt(OBJ, delimiter=",")


def rvl_decode(buf, n):
    """Vectorised RVL (Wilson 2017) decoder for the payload after the 20-byte compressedDepth header."""
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
        delta = (pz >> 1) ^ -(pz & 1)
        out[np.concatenate(pos_chunks)] = np.cumsum(delta)
    return out.astype(np.uint16)


def measure(mask, depth_m, K):
    ys, xs = np.nonzero(mask)
    mean = (xs.mean(), ys.mean())
    cov = np.cov(np.vstack((xs, ys)))
    d = depth_m[ys, xs]
    d = d[d >= 1e-3]  # TCAFF: raw depth < 1 (mm) is invalid
    if d.size == 0:
        return mean, cov, None, None, None
    dmed = float(np.median(d))
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    c = np.array([(mean[0] - cx) / fx * dmed, (mean[1] - cy) / fy * dmed, dmed])
    w = ((xs.max() - cx) / fx - (xs.min() - cx) / fx) * dmed
    h = ((ys.max() - cy) / fy - (ys.min() - cy) / fy) * dmed
    return mean, cov, c, float(w), float(h)


model = FastSAM(WEIGHTS)
manifest = json.load(open(DATA / "manifest.json"))
if MAX_FRAMES:
    manifest = manifest[:MAX_FRAMES]
out_masks = open(DATA / "masks.jsonl", "w")
out_frames = open(DATA / "frames.jsonl", "w")
out_boxes = open(DATA / "boxes.jsonl", "w")
kernel = np.ones((3, 3), np.uint8)

for fi, fr in enumerate(manifest):
    t0 = time.time()
    bgr = cv2.imdecode(np.frombuffer((DATA / "frames" / f"{fr['key']}.jpg").read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    H, W = bgr.shape[:2]
    raw = (DATA / "frames" / f"{fr['key']}.rvl").read_bytes()
    depth_m = rvl_decode(raw[20:], H * W).reshape(H, W).astype(np.float64) * 1e-3
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = model(rgb, retina_masks=True, device="cpu", imgsz=IMGSZ, conf=CONF, iou=IOU, verbose=False)
    masks = FastSAMPrompt(rgb, res, device="cpu").everything_prompt()
    masks = masks.cpu().numpy() > 0.5 if len(masks) else np.zeros((0, H, W), bool)
    confs = res[0].boxes.conf.cpu().numpy().tolist() if len(masks) else []
    K = fr["K"]
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    T_WC = np.array(fr["T_WB"]) @ np.array(fr["T_BC"])
    seg_time = time.time() - t0

    records, box_pts_by_mask = [], []
    for mi, m in enumerate(masks):
        if m.sum() < 10:
            continue
        mean, cov, c, w, h = measure(m, depth_m, K)
        ys, xs = np.nonzero(m)
        d_all = depth_m[ys, xs]
        valid = d_all >= 1e-3
        rec = {"key": fr["key"], "robot": fr["robot"], "mask": mi, "conf": confs[mi] if mi < len(confs) else None,
               "n_pix": int(m.sum()), "valid_frac": float(valid.mean()),
               "touches_border": bool(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any()),
               "pts_img": [float(mean[0]), float(mean[1])], "cov_img": cov.tolist(),
               "tcaff_kept": c is not None}
        cu, cv_ = int(round(mean[0])), int(round(mean[1]))
        rec["centroid_pixel_in_mask"] = bool(0 <= cv_ < H and 0 <= cu < W and m[cv_, cu])
        if c is not None:
            dv = d_all[valid]
            dmed = c[2]
            q25, q75 = np.percentile(dv, [25, 75])
            rec.update({"centroid_cam": c.tolist(), "width": w, "height": h, "d_med": dmed,
                        "iqr_rel": float((q75 - q25) / dmed), "frac_behind": float(np.mean(dv > dmed + LAYER)),
                        "frac_front": float(np.mean(dv < dmed - LAYER))})
            dc = depth_m[cv_, cu] if 0 <= cv_ < H and 0 <= cu < W else 0.0
            rec["centroid_pixel_depth_err"] = float(abs(dc - dmed)) if dc >= 1e-3 else None
            mu8 = m.astype(np.uint8)
            ring = (cv2.dilate(mu8, kernel, iterations=4) > 0) & ~m
            rd = depth_m[ring]
            rd = rd[rd >= 1e-3]
            rec["ring_front"] = float(np.mean(rd < dmed - RING)) if rd.size else None
            rec["ring_behind"] = float(np.mean(rd > dmed + RING)) if rd.size else None
            # Back-project valid mask pixels (subsampled) to the world and classify by GT boxes.
            sel = np.flatnonzero(valid)
            if sel.size > 3000:
                sel = sel[np.linspace(0, sel.size - 1, 3000).astype(int)]
            u, v, d = xs[sel], ys[sel], d_all[sel]
            P = np.stack([(u - cx) / fx * d, (v - cy) / fy * d, d, np.ones_like(d)])
            Pw = (T_WC @ P)[:3].T
            dist = np.linalg.norm(Pw[:, None, :2] - boxes[None], axis=2)
            nb, nd = dist.argmin(axis=1), dist.min(axis=1)
            is_floor = Pw[:, 2] < FLOOR_Z
            is_box = (~is_floor) & (Pw[:, 2] <= BOX_ZMAX) & (nd < BOX_R)
            ids = nb[is_box]
            counts = np.bincount(ids, minlength=len(boxes))
            covered = np.flatnonzero(counts >= max(1, 0.1 * is_box.sum())) if is_box.any() else np.array([], int)
            share_box = float(is_box.mean())
            cat = ("single_box" if len(covered) == 1 else "multi_box") if share_box >= 0.6 else ("box_plus_other" if share_box >= 0.2 else "non_box")
            rec.update({"share_box": share_box, "share_floor": float(is_floor.mean()), "n_boxes_covered": int(len(covered)),
                        "boxes_covered": covered.tolist(), "dominant_box_share": float(counts.max() / max(1, is_box.sum())),
                        "category": cat, "world_centroid": (T_WC @ np.array([*c, 1.0]))[:3].tolist()})
            box_pts_by_mask.append((mi, counts))
        records.append(rec)

    # Saved (authors') detections vs reproduced TCAFF measurements.
    saved_px = np.array(fr["saved"]["pts_img"]).reshape(-1, 2)
    rep = [r for r in records if r["tcaff_kept"]]
    rep_px = np.array([r["pts_img"] for r in rep]).reshape(-1, 2)
    matches = []
    if len(saved_px) and len(rep_px):
        dd = np.linalg.norm(saved_px[:, None] - rep_px[None], axis=2)
        for si in range(len(saved_px)):
            ri = int(dd[si].argmin())
            if dd[si, ri] < 15:
                sc = np.array(fr["saved"]["detections"][si])
                rep[ri]["matched_saved"] = True
                matches.append({"px": float(dd[si, ri]), "c3d": float(np.linalg.norm(sc - np.array(rep[ri]["centroid_cam"]))),
                                "dw": abs(fr["saved"]["widths"][si] - rep[ri]["width"]), "dh": abs(fr["saved"]["heights"][si] - rep[ri]["height"])})
    # Nested masks (smaller mask >= 80% inside a larger one), on 1/4 resolution.
    small = masks[:, ::4, ::4] if len(masks) else masks
    areas = small.reshape(len(small), -1).sum(axis=1) if len(small) else []
    nested = []
    for i in range(len(small)):
        for j in range(len(small)):
            if i != j and areas[i] < areas[j] and areas[i] > 0:
                if (small[i] & small[j]).sum() / areas[i] >= 0.8:
                    nested.append((i, j))
    cat_by_mask = {r["mask"]: r.get("category") for r in records}
    out_frames.write(json.dumps({"key": fr["key"], "robot": fr["robot"], "n_masks": int(len(masks)), "n_kept": len(rep),
                                 "n_saved": int(len(saved_px)), "matches": matches, "seg_time_s": seg_time,
                                 "total_time_s": time.time() - t0,
                                 "nested": [[i, j, cat_by_mask.get(i), cat_by_mask.get(j)] for i, j in nested]}) + "\n")
    for r in records:
        out_masks.write(json.dumps(r) + "\n")
    # Per GT box in this frame: how many masks contain its points, and the category of the dominant mask.
    if box_pts_by_mask:
        allc = np.stack([c for _, c in box_pts_by_mask])
        for b in np.flatnonzero(allc.sum(axis=0) >= 40):
            col = allc[:, b]
            best = int(col.argmax())
            mi = box_pts_by_mask[best][0]
            out_boxes.write(json.dumps({"key": fr["key"], "robot": fr["robot"], "box": int(b),
                                        "n_masks_with_box": int((col >= 0.2 * col.max()).sum()),
                                        "best_mask_category": cat_by_mask.get(mi),
                                        "best_mask_n_boxes": next(r["n_boxes_covered"] for r in records if r["mask"] == mi)}) + "\n")
    out_frames.flush(); out_masks.flush(); out_boxes.flush()
    if fi % 20 == 0:
        print(f"{fi + 1}/{len(manifest)} {fr['key']} masks={len(masks)} kept={len(rep)} saved={len(saved_px)} "
              f"matched={len(matches)} t={time.time() - t0:.1f}s", flush=True)
print("done")
