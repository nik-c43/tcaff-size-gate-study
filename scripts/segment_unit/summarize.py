"""Summaries with frame-cluster bootstrap 95% intervals.

Usage: python -I summarize.py DATA_DIR OUT_JSON
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

DATA, OUT = Path(sys.argv[1]), sys.argv[2]
rng = np.random.default_rng(0)
masks = [json.loads(l) for l in open(DATA / "masks.jsonl")]
frames = [json.loads(l) for l in open(DATA / "frames.jsonl")]
boxes = [json.loads(l) for l in open(DATA / "boxes.jsonl")]
keys = sorted({f["key"] for f in frames})


def boot(items, stat, n=1000):
    """Point estimate and 95% interval, resampling frames."""
    by = defaultdict(list)
    for it in items:
        by[it["key"]].append(it)
    ks = [k for k in keys if k in by]
    est = stat([x for k in ks for x in by[k]])
    vals = []
    for _ in range(n):
        s = rng.choice(len(ks), len(ks), replace=True)
        vals.append(stat([x for i in s for x in by[ks[i]]]))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return [round(float(est), 4), round(float(lo), 4), round(float(hi), 4)]


def share(pred):
    return lambda xs: np.mean([pred(x) for x in xs]) if xs else np.nan


res = {"n_frames": len(frames), "n_masks": len(masks)}

# 1. Reproduction of the authors' saved detections.
m_all = [m for f in frames for m in f["matches"]]
res["reproduction"] = {
    "saved_detections": int(sum(f["n_saved"] for f in frames)),
    "reproduced_kept": int(sum(f["n_kept"] for f in frames)),
    "matched_saved": len(m_all),
    "recall_saved": round(len(m_all) / max(1, sum(f["n_saved"] for f in frames)), 4),
    "precision_reproduced": round(len(m_all) / max(1, sum(f["n_kept"] for f in frames)), 4),
    "matched_centroid_3d_m_q50_q90": [round(float(np.percentile([m["c3d"] for m in m_all], p)), 4) for p in (50, 90)],
    "matched_width_abs_m_q50_q90": [round(float(np.percentile([m["dw"] for m in m_all], p)), 4) for p in (50, 90)],
    "masks_per_frame_q10_q50_q90": np.percentile([f["n_masks"] for f in frames], [10, 50, 90]).tolist(),
    "seg_time_s_median": round(float(np.median([f["seg_time_s"] for f in frames])), 2),
}

kept = [m for m in masks if m["tcaff_kept"]]
res["tcaff_dropped_no_valid_depth"] = boot(masks, share(lambda m: not m["tcaff_kept"]))

# 2. What a segment is, by GT boxes.
cats = ["single_box", "multi_box", "box_plus_other", "non_box"]
res["category_share_all_kept"] = {c: boot(kept, share(lambda m, c=c: m["category"] == c)) for c in cats}
box_rel = [m for m in kept if m["category"] != "non_box"]
res["category_share_box_related"] = {c: boot(box_rel, share(lambda m, c=c: m["category"] == c)) for c in cats[:3]}
mb = [m["n_boxes_covered"] for m in kept if m["category"] == "multi_box"]
res["multi_box_n_boxes_q50_q90_max"] = [float(np.percentile(mb, 50)), float(np.percentile(mb, 90)), int(max(mb))] if mb else None

# 3. Per visible GT box: dominant mask type, duplicates, temporal switching between sampled frames.
res["per_box_observation"] = {
    "n": len(boxes),
    "best_mask_single_box": boot(boxes, share(lambda b: b["best_mask_category"] == "single_box")),
    "best_mask_multi_box": boot(boxes, share(lambda b: b["best_mask_category"] == "multi_box")),
    "box_in_two_or_more_masks": boot(boxes, share(lambda b: b["n_masks_with_box"] >= 2)),
}
order = {f["key"]: i for i, f in enumerate(frames)}
seq = defaultdict(list)
for b in boxes:
    seq[(b["robot"], b["box"])].append((order[b["key"]], b))
changes = []
for v in seq.values():
    v.sort(key=lambda x: x[0])
    for (i, a), (j, b) in zip(v, v[1:]):
        if j == i + 1:  # consecutive sampled frames of the same robot
            changes.append({"key": b["key"], "unit_changed": (a["best_mask_n_boxes"] > 1) != (b["best_mask_n_boxes"] > 1)})
res["per_box_unit_switch_between_consecutive_samples"] = boot(changes, share(lambda c: c["unit_changed"])) if changes else None
res["n_consecutive_box_pairs"] = len(changes)

# 4. Nested masks.
res["frames_with_nested_pair"] = boot([{"key": f["key"], "v": bool(f["nested"])} for f in frames], share(lambda x: x["v"]))
pair_types = defaultdict(int)
for f in frames:
    for i, j, ci, cj in f["nested"]:
        pair_types[f"{ci} inside {cj}"] += 1
res["nested_pair_types_top"] = sorted(pair_types.items(), key=lambda x: -x[1])[:8]

# 5. Depth inside and around masks (masks TCAFF keeps).
res["depth"] = {
    "valid_frac_q10_q50_q90": np.percentile([m["valid_frac"] for m in kept], [10, 50, 90]).round(3).tolist(),
    "valid_frac_below_0.5": boot(kept, share(lambda m: m["valid_frac"] < 0.5)),
    "multi_layer_10pct": boot(kept, share(lambda m: m["frac_behind"] >= 0.1 or m["frac_front"] >= 0.1)),
    "behind_layer_10pct": boot(kept, share(lambda m: m["frac_behind"] >= 0.1)),
    "iqr_rel_q50_q90": np.percentile([m["iqr_rel"] for m in kept], [50, 90]).round(3).tolist(),
    "centroid_pixel_outside_mask": boot(kept, share(lambda m: not m["centroid_pixel_in_mask"])),
    "centroid_pixel_depth_off_0.3m": boot([m for m in kept if m["centroid_pixel_depth_err"] is not None],
                                          share(lambda m: m["centroid_pixel_depth_err"] > 0.3)),
    "touches_image_border": boot(kept, share(lambda m: m["touches_border"])),
    "occluded_boundary_ring_front_20pct": boot([m for m in kept if m["ring_front"] is not None], share(lambda m: m["ring_front"] >= 0.2)),
}
res["depth_by_category"] = {}
for c in cats:
    sel = [m for m in kept if m["category"] == c]
    if not sel:
        continue
    res["depth_by_category"][c] = {
        "n": len(sel),
        "multi_layer_10pct": boot(sel, share(lambda m: m["frac_behind"] >= 0.1 or m["frac_front"] >= 0.1)),
        "touches_border": boot(sel, share(lambda m: m["touches_border"])),
        "occluded_ring": boot([m for m in sel if m["ring_front"] is not None], share(lambda m: m["ring_front"] >= 0.2)),
        "valid_frac_q50": round(float(np.median([m["valid_frac"] for m in sel])), 3),
        "conf_q50": round(float(np.median([m["conf"] for m in sel if m["conf"] is not None])), 3),
    }
# Would a depth-consistency rule separate single boxes from merged segments?
box_masks = [m for m in kept if m["category"] in ("single_box", "multi_box", "box_plus_other")]
res["depth_rule_vs_unit"] = {
    "flag_multi_layer_given_single_box": boot([m for m in box_masks if m["category"] == "single_box"], share(lambda m: m["frac_behind"] >= 0.1 or m["frac_front"] >= 0.1)),
    "flag_multi_layer_given_merged": boot([m for m in box_masks if m["category"] != "single_box"], share(lambda m: m["frac_behind"] >= 0.1 or m["frac_front"] >= 0.1)),
}
# 6. Which reproduced masks the authors' saved data lacks (infers their extra filters).
def elong(m):
    ev = np.linalg.eigvalsh(np.array(m["cov_img"]))
    return float(np.sqrt(ev[-1] / max(ev[0], 1e-9)))


def describe(sel):
    return {"n": len(sel),
            "touches_border": round(float(np.mean([m["touches_border"] for m in sel])), 3),
            "n_pix_q10_q50_q90": np.percentile([m["n_pix"] for m in sel], [10, 50, 90]).round(0).tolist(),
            "d_med_q10_q50_q90": np.percentile([m["d_med"] for m in sel], [10, 50, 90]).round(2).tolist(),
            "width_q50_q90": np.percentile([m["width"] for m in sel], [50, 90]).round(2).tolist(),
            "elongation_q50_q90": np.percentile([elong(m) for m in sel], [50, 90]).round(2).tolist(),
            "conf_q10_q50": np.percentile([m["conf"] for m in sel], [10, 50]).round(3).tolist(),
            "world_z_q10_q50_q90": np.percentile([m["world_centroid"][2] for m in sel], [10, 50, 90]).round(2).tolist(),
            "category": {c: round(float(np.mean([m["category"] == c for m in sel])), 3) for c in cats}}


matched = [m for m in kept if m.get("matched_saved")]
unmatched = [m for m in kept if not m.get("matched_saved")]
res["authors_filter_inference"] = {"matched": describe(matched), "unmatched": describe(unmatched)}
res["authors_filter_rules"] = {
    name: {"removes_unmatched": round(float(np.mean([f(m) for m in unmatched])), 3),
           "would_remove_matched": round(float(np.mean([f(m) for m in matched])), 3)}
    for name, f in {"touches_border": lambda m: m["touches_border"], "elongation>4": lambda m: elong(m) > 4,
                    "conf<0.6": lambda m: m["conf"] < 0.6, "centroid >0.5 m above camera": lambda m: m["centroid_cam"][1] < -0.5,
                    "n_pix>20000": lambda m: m["n_pix"] > 20000}.items()}

# 7. The subset the authors' pipeline actually used (masks matched to saved detections).
mb_rel = [m for m in matched if m["category"] != "non_box"]
cover = defaultdict(int)
for m in matched:
    for b in m.get("boxes_covered", []):
        cover[(m["key"], b)] += 1
layered = lambda m: m["frac_behind"] >= 0.1 or m["frac_front"] >= 0.1
res["tcaff_input_subset"] = {
    "masks": len(matched),
    "box_related": len(mb_rel),
    "category_share_box_related": {c: boot(mb_rel, share(lambda m, c=c: m["category"] == c)) for c in cats[:3]},
    "box_frame_observations": len(cover),
    "box_in_two_or_more_saved_segments": round(float(np.mean([v >= 2 for v in cover.values()])), 4),
    "valid_frac_q10_q50": np.percentile([m["valid_frac"] for m in matched], [10, 50]).round(3).tolist(),
    "valid_frac_below_0.5": boot(matched, share(lambda m: m["valid_frac"] < 0.5)),
    "multi_layer": boot(matched, share(layered)),
    "multi_layer_single_box": boot([m for m in matched if m["category"] == "single_box"], share(layered)),
    "multi_layer_merged": boot([m for m in matched if m["category"] in ("multi_box", "box_plus_other")], share(layered)),
}
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
