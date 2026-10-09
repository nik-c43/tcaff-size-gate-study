"""Inspect size-gate loss and same-map proximity without claiming object labels."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vendor/tcaff'))
from tcaff.tcaff.tcaff_manager import TCAFFManager
from tcaff.realign.object_map import ObjectMap


def object_map(rows):
    a = np.asarray(rows, float).reshape(-1, 6)
    return ObjectMap(a[:, :3], a[:, 3], a[:, 4], a[:, 5])


def allowed_sizes(a, b, ratio, difference):
    """Vectorized copy of the upstream width/height gate, checked against it."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), bool)
    lo = np.minimum(a[:, None, 3:5], b[None, :, 3:5])
    hi = np.maximum(a[:, None, 3:5], b[None, :, 3:5])
    return np.all((hi <= lo * ratio) & (hi - lo <= difference), axis=2)


def near_matching(distance, allowed):
    """Maximize cardinality of allowed proxy pairs, then minimize distance."""
    if not distance.size:
        return []
    n = min(distance.shape)
    scale = max(float(distance.max()), 1.)
    cost = (~allowed) * (n + 1.) + distance / (scale + 1.)
    i, j = linear_sum_assignment(cost)
    return [[int(x), int(y)] for x, y in zip(i, j) if allowed[x, y]]


def percentile(values, q):
    return None if not values else float(np.percentile(values, q))


def summarize(rows):
    return dict(
        frames=len(rows),
        frames_with_proxy_support_at_least_required=sum(r['near_support'] >= r['required'] for r in rows),
        frames_dropped_below_required_by_size_gate=sum(
            r['near_support'] >= r['required'] > r['size_feasible_support'] for r in rows),
        proxy_near_pairs=sum(r['near_pairs'] for r in rows),
        proxy_near_pairs_rejected_by_size=sum(r['near_pairs'] - r['near_allowed_pairs'] for r in rows),
        median_proxy_support=percentile([r['near_support'] for r in rows], 50),
        median_size_feasible_support=percentile([r['size_feasible_support'] for r in rows], 50),
    )


def main():
    config = json.loads((ROOT / 'configs/h1_protocol.json').read_text())
    rows, maps, provenance = [], {}, []
    for pair in config['real_pairs']:
        path = ROOT / 'data/real_maps' / f'{pair}.json'
        data = json.loads(path.read_text())
        kwargs = {k: v for k, v in data['params'].items() if k != 'ts'}
        manager = TCAFFManager(**kwargs)
        provenance.append(dict(path=str(path.relative_to(ROOT)), params=data['params']))
        robots = pair.split('_')
        for frame in data['frames']:
            a, b = (np.asarray(frame[k], float).reshape(-1, 6) for k in ['map_a', 'map_b'])
            for robot, array in zip(robots, [a, b]):
                key = (robot, frame['step'])
                if key in maps:
                    assert np.array_equal(maps[key]['array'], array), 'Robot map differs across pairs'
                else:
                    maps[key] = dict(array=array, time=frame['time'])
            allowed = allowed_sizes(a, b, manager.wh_scale_diff, manager.h_diff)
            upstream = manager.get_putative_assoc(object_map(a), object_map(b))
            expected = np.argwhere(allowed)
            assert np.array_equal(upstream, expected), 'Vectorized gate differs from upstream'
            if frame['truth_se3'] is None or not len(a) or not len(b):
                continue
            T = np.asarray(frame['truth_se3'])
            transformed_b = b[:, :3] @ T[:3, :3].T + T[:3, 3]
            for dimension in [2, 3]:
                distance = cdist(a[:, :dimension], transformed_b[:, :dimension])
                for radius in config['real_centroid_proxy_radii_m']:
                    near = distance < radius
                    before = near_matching(distance, near)
                    after = near_matching(distance, near & allowed)
                    rejected = [[i, j] for i, j in before if not allowed[i, j]]
                    rows.append(dict(
                        pair=pair, step=frame['step'], time=frame['time'], dimension=dimension,
                        radius_m=radius, n_a=len(a), n_b=len(b), required=manager.fa.num_objs_req,
                        near_support=len(before), size_feasible_support=len(after),
                        near_pairs=int(near.sum()), near_allowed_pairs=int((near & allowed).sum()),
                        selected_proxy_pairs=before, selected_rejected_proxy_pairs=rejected,
                        selected_proxy_errors_m=[float(distance[i, j]) for i, j in before],
                        rejected_size_differences_m=[(a[i, 3:5] - b[j, 3:5]).tolist() for i, j in rejected],
                    ))
    proximity = []
    for (robot, step), record in sorted(maps.items()):
        a = record['array']
        distance = cdist(a[:, :3], a[:, :3])
        for radius in config['same_map_proximity_radii_m']:
            i, j = np.where(np.triu(distance < radius, k=1))
            proximity.append(dict(robot=robot, step=step, time=record['time'], radius_m=radius,
                                  n=len(a), close_pairs=np.c_[i, j].tolist(), count=len(i)))
    summaries = []
    for horizon in [60, 337]:
        for dimension in [2, 3]:
            for radius in config['real_centroid_proxy_radii_m']:
                selected = [r for r in rows if r['step'] < horizon and r['dimension'] == dimension
                            and r['radius_m'] == radius]
                summaries.append(dict(horizon_frames=horizon, dimension=dimension, radius_m=radius,
                                      **summarize(selected)))
    same_map_summary = []
    for radius in config['same_map_proximity_radii_m']:
        selected = [r for r in proximity if r['radius_m'] == radius]
        same_map_summary.append(dict(radius_m=radius, unique_robot_frames=len(selected),
                                     frames_with_close_pairs=sum(r['count'] > 0 for r in selected),
                                     close_pair_observations=sum(r['count'] for r in selected)))
    result = dict(scope=config['scope'], warnings=config['limitations'], provenance=provenance,
                  summaries=summaries, same_map_summary=same_map_summary, rows=rows, proximity=proximity)
    out = ROOT / 'results/h1_real_map_audit.json'
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(summaries=summaries, same_map_summary=same_map_summary), indent=2), flush=True)


if __name__ == '__main__':
    main()
