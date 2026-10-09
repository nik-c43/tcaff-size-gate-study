"""Prepare a blind, traceable annotation bundle without changing upstream/B0."""
import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'vendor/tcaff'), str(ROOT / 'src/demo_cpu'), str(ROOT / 'scripts')]
from detections import DetectionData
from robot import Robot
from tcaff.mot.multi_object_tracker import MultiObjectTracker
from tcaff.mot.motion_model import MotionModel
from tcaff.mot.track import Track
from tcaff.utils.transform import transform
from tcaff_baseline_audit import load_poses, pose_error, digest
from h1_map_audit import allowed_sizes


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def check_pins(config):
    for directory, key in [('vendor/tcaff', 'tcaff_commit'), ('clipper', 'clipper_commit')]:
        actual = subprocess.check_output(['git', '-C', str(ROOT / directory), 'rev-parse', 'HEAD'], text=True).strip()
        assert actual == config[key], (key, actual)
        # Untracked user-generated CMake build artefacts do not change upstream sources.
        assert not subprocess.check_output(['git', '-C', str(ROOT / directory), 'status', '--porcelain', '--untracked-files=no'], text=True), directory
    original = json.loads((ROOT / 'results/tcaff_baseline_release/audit.json').read_text())
    for binary in original['environment']['clipperpy_files']:
        assert digest(Path(binary['path'])) == binary['sha256'], 'CLIPPER binary changed since frozen B0'
    for record in original['input_hashes']:
        assert digest(ROOT / record['path']) == record['sha256'], f'Frozen B0 input changed: {record["path"]}'


def gate_details(a, b, ratio=1.35, diff=.1):
    checks = dict(width_ratio=max(a[3], b[3]) > min(a[3], b[3]) * ratio,
                  height_ratio=max(a[4], b[4]) > min(a[4], b[4]) * ratio,
                  width_difference=abs(a[3]-b[3]) > diff,
                  height_difference=abs(a[4]-b[4]) > diff)
    failed = [k for k, v in checks.items() if bool(v)]
    return dict(admitted=not failed, failed_checks=failed,
                first_exclusion_reason=failed[0] if failed else None,
                width_difference_m=float(abs(a[3]-b[3])), height_difference_m=float(abs(a[4]-b[4])),
                ratio_limit=ratio, absolute_difference_limit_m=diff)


class TracedDetectionData(DetectionData):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.raw = json.loads(Path(kwargs['data_file']).read_text())
        self.latest_sources = []

    def detections(self, t):
        self.latest_sources = []
        zs, Rs = super().detections(t)
        index = int(self.idx(t))
        frame = self.raw[index]
        points = np.asarray(frame['detections']).reshape(-1, 3)
        if not len(points):
            return zs, Rs
        body = transform(self.T_BC, points, stacked_axis=0)
        keep = (body[:, 0] <= self.zmax) & (body[:, 0] >= self.zmin)
        expected = np.c_[body[keep], np.asarray(frame['widths'])[keep], np.asarray(frame['heights'])[keep]]
        np.testing.assert_allclose(zs, expected, atol=1e-12, rtol=0)
        for j in np.flatnonzero(keep):
            self.latest_sources.append(dict(frame_index=index, detection_index=int(j), time=frame['time'],
                                            camera_xyz=points[j].tolist(), uv=frame['pts_img'][j],
                                            image_covariance=frame['covs_img'][j],
                                            observed_width_m=frame['widths'][j], observed_height_m=frame['heights'][j]))
        return zs, Rs


def select(config):
    audit_path = ROOT / config['audit']
    audit = json.loads(audit_path.read_text())
    p = config['proxy']
    rows = {(r['pair'], r['step']): r for r in audit['rows']
            if r['dimension'] == p['dimension'] and r['radius_m'] == p['radius_m']}
    episodes, maps, sources = [], {}, {}
    for pair, anchors in config['anchors'].items():
        path = ROOT / config['baseline'] / f'{pair}.json'
        baseline = json.loads(path.read_text())
        old = json.loads((ROOT / 'data/real_maps' / f'{pair}.json').read_text())
        assert baseline['params'] == old['params']
        for f, o in zip(baseline['frames'], old['frames']):
            assert f['time'] == o['time'] and f['step'] == o['step']
            for key in ['map_a', 'map_b', 'truth_se3']:
                assert f[key] == o[key], (pair, f['step'], key)
            for name, key in zip(pair.split('_'), ['map_a', 'map_b']):
                k = (name, f['step'])
                value = dict(time=f['time'], array=f[key])
                if k in maps:
                    assert maps[k] == value
                maps[k] = value
        sources[pair] = dict(path=str(path.relative_to(ROOT)), sha256=digest(path))
        first = next((f for f in baseline['frames'] if f['baseline_estimate'] is not None), None)
        first_error = None if first is None else pose_error(first['baseline_estimate'], first['truth'])
        for k, anchor in enumerate(anchors):
            row = rows[pair, anchor]
            assert row['near_support'] >= 5 > row['size_feasible_support'], (pair, anchor)
            start = max(0, anchor-config['window']['lead_updates'])
            end = start + config['window']['length_updates']-1
            assert end < len(baseline['frames'])
            f = baseline['frames'][anchor]
            error = None if f['baseline_estimate'] is None else pose_error(f['baseline_estimate'], f['truth'])
            correct = lambda e: e[0] <= 1. and e[1] <= 5.
            episodes.append(dict(pair=pair, anchor=anchor, time=f['time'], split='development' if k < 2 else 'holdout',
                                 start=start, end=end, temporal_block=anchor//45,
                                 proxy=row,
                                 anchor_status='no_estimate' if error is None else ('correct_pose' if correct(error) else 'incorrect_pose'),
                                 anchor_error=error,
                                 global_first_step=None if first is None else first['step'],
                                 global_first_error=first_error,
                                 contains_wrong_global_first=first is not None and start <= first['step'] <= end and not correct(first_error)))
    assert len(episodes) == 24
    dev = [e for e in episodes if e['split'] == 'development']
    held = [e for e in episodes if e['split'] == 'holdout']
    assert max(e['end'] for e in dev) < min(e['start'] for e in held)
    rng = np.random.default_rng(config['seed'])
    rng.shuffle(episodes)
    for i, e in enumerate(episodes):
        e['episode_id'] = f'E{i+1:02d}'
    return episodes, maps, sources


def trace_maps(config, expected, wanted):
    """Passive hooks read associations; original mapping computations run intact."""
    os.environ['TCAFF_MOT_DATASET'] = str(ROOT / 'data/tcaff_mot_data')
    params = yaml.safe_load((ROOT / 'vendor/tcaff/demo/params/tcaff_mot_dataset.yaml').read_text())
    poses = load_poses(params)
    robots = []
    for name in params['robots']:
        p = params['mapping']; dim = p['dim']
        Q = np.diag([*[p['Q_el']]*(dim-2), p['Q_el_w'], p['Q_el_h']])
        P0 = np.diag([*[p['P0_el']]*(dim-2), p['P0_el_w'], p['P0_el_h']])
        R = np.eye(dim)*p['R_el']
        model = MotionModel(A=np.eye(dim), H=np.eye(dim), Q=Q, R=np.array([]), P0=P0)
        others = [n for n in params['robots'] if n != name]
        mapper = MultiObjectTracker(camera_id=name, connected_cams=others, track_motion_model=model,
                                    tau_global=0., tau_local=p['tau'], alpha=2000, kappa=p['kappa'],
                                    nu=p['nu'], track_storage_size=1, dim_association=dim)
        detector = TracedDetectionData(data_file=str(ROOT / f'data/tcaff_mot_data/fastsam_data/{name}.json'),
                                       file_type='json', time_tol=.1, T_BC=np.asarray(params[name]['T_BC']),
                                       zmin=p['zmin'], zmax=p['zmax'], R=R, dim=dim)
        robots.append(Robot(name, others, poses[name, 'pose_estimate'], mapper, {}, None, detector, None))
    detectors = {r.name: r.fastsam3d_detections for r in robots}
    histories = defaultdict(list)
    refs = {}
    association, update = MultiObjectTracker.local_data_association, Track.update

    def traced_association(self, Zs, feature_vecs, Rs):
        assert self.merge_range_m == 0.
        sources = detectors[self.camera_id].latest_sources
        assert len(Zs) == len(sources)
        refs.clear()
        refs.update({id(z): dict(source, robot=self.camera_id) for z, source in zip(Zs, sources)})
        return association(self, Zs, feature_vecs, Rs)

    def traced_update(self, measurements, R):
        assert len(measurements) == 1 and id(measurements[0]) in refs
        source = refs[id(measurements[0])]
        assert source['robot'] == self.id[0]
        history = histories[self.id]
        if not history or (history[-1]['frame_index'], history[-1]['detection_index']) != (source['frame_index'], source['detection_index']):
            history.append(deepcopy(source))
        return update(self, measurements, R)

    snapshots, max_error, checked = {}, 0., 0
    t0 = max(r.pose_est_data.t0 for r in robots)
    tf = min(r.pose_est_data.tf for r in robots)
    last_mapping = last_alignment = -np.inf
    step = 0
    try:
        MultiObjectTracker.local_data_association = traced_association
        Track.update = traced_update
        for t in np.arange(t0, tf, params['mapping']['ts']/10):
            if np.round(t-last_mapping, 4) >= params['mapping']['ts']:
                last_mapping = t
                for r in robots:
                    r.update_mapping(t)
            if np.round(t-last_alignment, 4) >= params['tcaff']['ts']:
                last_alignment = t
                for r in robots:
                    array = r.get_map().as_array().reshape(-1, 6)
                    reference = expected[r.name, step]
                    assert abs(t-reference['time']) < 1e-9
                    target = np.asarray(reference['array']).reshape(-1, 6)
                    assert array.shape == target.shape, (r.name, step, array.shape, target.shape)
                    error = float(np.max(abs(array-target), initial=0.))
                    assert error < 1e-10, (r.name, step, error)
                    max_error = max(max_error, error); checked += 1
                    if (r.name, step) in wanted:
                        records = []
                        for i, tr in enumerate(r.mapper.tracks):
                            h = histories[tr.id]
                            assert h
                            # Complete source index chain; image review uses first/last and spaced history.
                            records.append(dict(map_index=i, track_id=list(tr.id), map_row=array[i].tolist(),
                                                source_chain=[[x['frame_index'], x['detection_index'], x['time']] for x in h],
                                                last_observation=deepcopy(h[-1]), first_observation=deepcopy(h[0]),
                                                stale_s=float(t-h[-1]['time'])))
                        snapshots[r.name, step] = records
                if step % 30 == 0:
                    print(f'mapping replay: {step}/336, checked {checked} maps, max difference {max_error:.3g}', flush=True)
                step += 1
    finally:
        MultiObjectTracker.local_data_association = association
        Track.update = update
    assert checked == len(expected) == 1348
    return snapshots, dict(unique_robot_maps_checked=checked, max_absolute_difference=max_error,
                           passive_hooks='Read raw source indices at original Track.update calls; no numeric input or state mutation',
                           track_id_meaning='Original mapper track ID, not physical-object truth',
                           masks_available=False, exact_detection_boundaries_available=False)


def prepare(config, out, config_path=None):
    config_path = config_path or ROOT / 'configs/size_gate_study.json'
    check_pins(config)
    if out.exists():
        raise RuntimeError(f'Refuse to overwrite study directory: {out}')
    out.mkdir(parents=True)
    episodes, maps, source_hashes = select(config)
    wanted = {(r, e['anchor']) for e in episodes for r in e['pair'].split('_')}
    snapshots, checks = trace_maps(config, maps, wanted)
    tasks = []
    for e in episodes:
        a, b = e['pair'].split('_')
        ma = np.asarray(maps[a, e['anchor']]['array']).reshape(-1, 6)
        mb = np.asarray(maps[b, e['anchor']]['array']).reshape(-1, 6)
        source = json.loads((ROOT / config['baseline'] / f'{e["pair"]}.json').read_text())['frames'][e['anchor']]
        T = np.asarray(source['truth_se3'])
        delta = ma[:, None, :3] - (mb[:, :3] @ T[:3, :3].T + T[:3, 3])[None, :, :]
        distances = np.linalg.norm(delta, axis=2)
        proxy_pairs = set(map(tuple, e['proxy']['selected_proxy_pairs']))
        for i, j in np.argwhere(distances < config['proxy']['radius_m']):
            left, right = deepcopy(snapshots[a, e['anchor']][i]), deepcopy(snapshots[b, e['anchor']][j])
            details = gate_details(ma[i], mb[j])
            tasks.append(dict(episode_id=e['episode_id'], pair=e['pair'], step=e['anchor'], time=e['time'],
                              left=left, right=right, gate=details, distance_m=float(distances[i, j]),
                              maximum_proxy_matching=(int(i), int(j)) in proxy_pairs,
                              control_roles=(['size_admitted'] if details['admitted'] else []) +
                              ([] if (int(i), int(j)) in proxy_pairs else ['near_alternative_candidate'])))
    rng = np.random.default_rng(config['seed']+1)
    rng.shuffle(tasks)
    for i, task in enumerate(tasks):
        task['task_id'] = f'P{i+1:04d}'
    assert len(tasks) == sum(e['proxy']['near_pairs'] for e in episodes)
    frozen_hashes = dict(source_hashes=source_hashes, audit_sha256=digest(ROOT / config['audit']),
                         protocol_sha256=digest(config_path), configuration_path=str(config_path.relative_to(ROOT)),
                         preparation_command=sys.argv, python=sys.version,
                         thread_environment={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','MPLCONFIGDIR']})
    fingerprint = hashlib.sha256(json.dumps(dict(episodes=episodes, tasks=tasks, inputs=frozen_hashes), sort_keys=True).encode()).hexdigest()
    private = dict(fingerprint=fingerprint, protocol=config, episodes=episodes, tasks=tasks, checks=checks,
                   provenance=frozen_hashes, source_json_hashes={r:digest(ROOT / f'data/tcaff_mot_data/fastsam_data/{r}.json')
                                                               for r in ['RR01','RR04','RR06','RR08']})
    dump(out / 'analysis_private/manifest.json', private)
    dump(out / 'analysis_private/map_source_chains.json',
         {f'{r}:{s}': records for (r,s), records in snapshots.items()})
    dump(out / 'protocol.json', config)
    from size_gate_images import build_images_and_review
    summary = build_images_and_review(private, out)
    dump(out / 'preparation_summary.json', dict(**checks, **summary, fingerprint=fingerprint,
                                               episodes=len(episodes), development=12, holdout=12,
                                               anchor_status_counts={v:sum(e['anchor_status']==v for e in episodes)
                                                                     for v in ['no_estimate','correct_pose','incorrect_pose']},
                                               wrong_global_first_windows=sum(e['contains_wrong_global_first'] for e in episodes),
                                               gate_admitted_tasks=sum(t['gate']['admitted'] for t in tasks),
                                               gate_rejected_tasks=sum(not t['gate']['admitted'] for t in tasks),
                                               manual_labels_received=False, relaxation_run=False))
    check_pins(config)
    for pair, record in source_hashes.items():
        assert digest(ROOT / record['path']) == record['sha256']
    from size_gate_finalize import finalize
    finalize(out)
    print(json.dumps(json.loads((out / 'preparation_summary.json').read_text()), ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/size_gate_study.json')
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text())
    prepare(config, ROOT / config['output'], ROOT / args.config)


if __name__ == '__main__':
    main()
