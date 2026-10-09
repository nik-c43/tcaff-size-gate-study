"""Audit the released TCAFF baseline, its ground truth and deterministic replay.

Uses the original TCAFFManager factory. RGB-D inference and MOT are outside this
baseline. The paper's tau=8 is a labeled filter-only sensitivity control; the
primary result uses the release YAML unchanged (tau=9).
"""
import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml
import clipperpy

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'vendor/tcaff'), str(ROOT / 'src/demo_cpu')]
from robotdatapy.data import PoseData
from robotdatapy.transform import T_RDFFLU
from tcaff.tcaff.tcaff_manager import TCAFFManager
from tcaff.utils.transform import transform_2_xypsi, xypsi_2_transform


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ast_item(path, class_name, method=None):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    node = cls if method is None else next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method)
    return ast.dump(node, include_attributes=False)


def source_audit():
    vendor, local = ROOT / 'vendor/tcaff/demo', ROOT / 'src/demo_cpu'
    checks = {}
    for name in ['robot.py', 'tcaff_data_processory.py']:
        checks[f'unchanged_{name}'] = digest(vendor / name) == digest(local / name)
    checks['unchanged_DetectionData_AST'] = ast_item(vendor / 'detections.py', 'DetectionData') == ast_item(local / 'detections.py', 'DetectionData')
    checks['unchanged_GT_method_AST'] = ast_item(vendor / 'alignment_results.py', 'AlignmentResults', 'get_Tij_gt') == ast_item(local / 'alignment_results.py', 'AlignmentResults', 'get_Tij_gt')
    status = subprocess.check_output(['git', '-C', str(ROOT / 'vendor/tcaff'), 'status', '--porcelain'], text=True)
    checks['upstream_tcaff_clean'] = status == ''
    assert all(checks.values()), checks
    return checks


def load_poses(params):
    output = {}
    for name in params['robots']:
        for kind in ['pose_estimate', 'pose_gt']:
            p = params[kind]
            post = None
            if p.get('frame') == 'RDF':
                post = T_RDFFLU
            elif 'T_postmultiply' in p:
                post = np.linalg.inv(np.asarray(p['T_postmultiply'][name]))
            path = Path(os.path.expandvars(p['root'])) / f'{name}.bag'
            output[(name, kind)] = PoseData.from_bag(path=str(path), topic=f'/{name}/{p["topic"]}',
                                                   time_tol=10., interp=True, T_postmultiply=post)
    return output


def pose_error(estimate, truth):
    # The released evaluator forms inv(T_GT_SE2) @ T_EST_SE2.
    delta = np.linalg.inv(xypsi_2_transform(*truth)) @ xypsi_2_transform(*estimate)
    dx, dy, yaw = transform_2_xypsi(delta)
    return float(np.hypot(dx, dy)), float(abs(np.degrees(yaw)))


def replay(data, params, poses):
    kwargs = {k: v for k, v in params['tcaff'].items() if k != 'ts'}
    primary = TCAFFManager(**kwargs).faf
    sensitivity = TCAFFManager(**dict(kwargs, main_tree_obj_req=8.)).faf
    a, b = data['pair'].split('_')
    frames, mismatch, max_gt_error, max_replay_error = [], [], 0., 0.
    assert data['params'] == params['tcaff'], 'Captured parameters differ from the released YAML'
    for frame in data['frames']:
        z = [np.asarray(x).reshape(3, 1) for x in frame['zs']]
        R = np.asarray(frame['R'])
        primary.update(z, [R.copy() for _ in z])
        sensitivity.update(z, [R.copy() for _ in z])
        estimate = None if primary.main_tree is None else primary.main_tree.optimal.xhat.ravel()
        control = None if sensitivity.main_tree is None else sensitivity.main_tree.optimal.xhat.ravel()
        reference = frame['baseline_estimate']
        if (estimate is None) != (reference is None):
            mismatch.append(frame['step'])
        elif estimate is not None:
            err = np.asarray(estimate) - np.asarray(reference)
            err[2] = np.arctan2(np.sin(err[2]), np.cos(err[2]))
            max_replay_error = max(max_replay_error, float(np.max(abs(err))))
            if np.max(abs(err)) > 1e-10:
                mismatch.append(frame['step'])
        truth = frame['truth']
        if truth is not None:
            t = frame['time']
            Ai = poses[(a, 'pose_estimate')].T_WB(t)
            Bi = poses[(a, 'pose_gt')].T_WB(t)
            Aj = poses[(b, 'pose_estimate')].T_WB(t)
            Bj = poses[(b, 'pose_gt')].T_WB(t)
            # Independent product equivalent to the author's get_Tij_gt.
            T = Ai @ np.linalg.inv(Bi) @ Bj @ np.linalg.inv(Aj)
            independent_gt = np.asarray(transform_2_xypsi(T))
            gt_error = independent_gt - truth
            gt_error[2] = np.arctan2(np.sin(gt_error[2]), np.cos(gt_error[2]))
            max_gt_error = max(max_gt_error, float(np.max(abs(gt_error))))
            assert np.max(abs(gt_error)) < 1e-9, (data['pair'], frame['step'], gt_error)
            if 'truth_se3' in frame and frame['truth_se3'] is not None:
                assert np.allclose(T, frame['truth_se3'], rtol=0, atol=1e-9)
        row = dict(step=frame['step'], time=frame['time'], gt_available=truth is not None)
        for name, value in [('release_tau9', estimate), ('paper_tau8_control', control)]:
            error = None if value is None or truth is None else pose_error(value, truth)
            row[name] = dict(accepted=value is not None,
                             translation_m=None if error is None else error[0],
                             yaw_deg=None if error is None else error[1])
        frames.append(row)
    assert not mismatch, (data['pair'], mismatch)
    return dict(pair=data['pair'], frames=frames, replay_mismatches=mismatch,
                max_replay_coordinate_error=max_replay_error, max_GT_coordinate_error=max_gt_error)


def summarize(traces, variant):
    pairs, all_errors = [], []
    for trace in traces:
        frames = trace['frames']
        valid = [f[variant] for f in frames if f[variant]['translation_m'] is not None]
        first = next((f for f in frames if f[variant]['accepted']), None)
        all_errors.extend(valid)
        pairs.append(dict(pair=trace['pair'], steps=len(frames),
                          accepted_steps=sum(f[variant]['accepted'] for f in frames),
                          valid_metric_steps=len(valid), gt_steps=sum(f['gt_available'] for f in frames),
                          mean_translation_m=float(np.mean([f['translation_m'] for f in valid])) if valid else None,
                          mean_yaw_deg=float(np.mean([f['yaw_deg'] for f in valid])) if valid else None,
                          first_acceptance_s=None if first is None else first['time'] - frames[0]['time']))
    pair_errors = [p for p in pairs if p['mean_translation_m'] is not None]
    total = sum(p['steps'] for p in pairs)
    return dict(variant=variant, directed_pairs=len(pairs), pair_updates=total,
                accepted_pair_updates=sum(p['accepted_steps'] for p in pairs),
                gt_valid_accepted_pair_updates=len(all_errors),
                availability=sum(p['accepted_steps'] for p in pairs) / total,
                author_mean_of_pair_means_translation_m=float(np.mean([p['mean_translation_m'] for p in pair_errors])),
                author_mean_of_pair_means_yaw_deg=float(np.mean([p['mean_yaw_deg'] for p in pair_errors])),
                pooled_mean_translation_m=float(np.mean([f['translation_m'] for f in all_errors])),
                pooled_mean_yaw_deg=float(np.mean([f['yaw_deg'] for f in all_errors])), pairs=pairs)


def plot(traces, out):
    length = min(len(t['frames']) for t in traces)
    time = np.array([f['time'] for f in traces[0]['frames'][:length]])
    time -= time[0]
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True, layout='constrained')
    for axis, field, label in zip(axes[:2], ['translation_m', 'yaw_deg'], ['Translation error (m)', 'Yaw error (deg)']):
        values = np.array([[np.nan if f['release_tau9'][field] is None else f['release_tau9'][field]
                            for f in t['frames'][:length]] for t in traces])
        available = np.any(np.isfinite(values), axis=0)
        middle = np.full(length, np.nan)
        low, high = middle.copy(), middle.copy()
        middle[available] = np.nanmedian(values[:, available], axis=0)
        low[available], high[available] = np.nanquantile(values[:, available], [.1, .9], axis=0)
        axis.plot(time, middle, color='#255b92', label='Median across directed pairs with an estimate')
        axis.fill_between(time, low, high, color='#255b92', alpha=.2, label='10–90% descriptive range')
        axis.set_ylabel(label)
        axis.grid(alpha=.25)
    accepted = np.array([[f['release_tau9']['accepted'] for f in t['frames'][:length]] for t in traces])
    axes[2].step(time, accepted.mean(axis=0) * 100, where='post', color='#255b92')
    axes[2].set_ylabel('Pairs with estimate (%)')
    axes[2].set_ylim(-2, 102)
    axes[2].set_xlabel('Time since first alignment update (s)')
    axes[2].grid(alpha=.25)
    axes[0].legend(fontsize=8)
    axes[0].set_title('Released TCAFF baseline: one four-robot recording, saved FastSAM detections')
    fig.savefig(out / 'baseline_errors.png', dpi=180)
    fig.savefig(out / 'baseline_errors.svg')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/tcaff_baseline_release')
    parser.add_argument('--output', default='results/tcaff_baseline_release')
    args = parser.parse_args()
    os.environ['TCAFF_MOT_DATASET'] = str(ROOT / 'data/tcaff_mot_data')
    yaml_path = ROOT / 'vendor/tcaff/demo/params/tcaff_mot_dataset.yaml'
    params = yaml.safe_load(yaml_path.read_text())
    checks = source_audit()
    poses = load_poses(params)
    traces = []
    paths = sorted((ROOT / args.input).glob('*.json'))
    assert len(paths) == 12, 'Expected all 12 directed pairs'
    for path in paths:
        traces.append(replay(json.loads(path.read_text()), params, poses))
        print(f'{path.name}: ground truth and original-factory replay match', flush=True)
    out = ROOT / args.output
    out.mkdir(exist_ok=True, parents=True)
    used_inputs = paths + [yaml_path]
    for name in params['robots']:
        used_inputs += [ROOT / f'data/tcaff_mot_data/fastsam_data/{name}.json',
                        ROOT / f'data/tcaff_mot_data/data/kimera_odom/{name}.bag']
    summaries = [summarize(traces, v) for v in ['release_tau9', 'paper_tau8_control']]
    result = dict(
        scope='Author-released TCAFF alignment pipeline on saved author detections; no RGB-D inference or MOT evaluation',
        primary_configuration='Released tcaff_mot_dataset.yaml unchanged, tau=9; paper section IV states default tau=8',
        paper_comparison=dict(section='IV-C', translation_m=.43, yaw_deg=2.3, MOTA=.761,
                              status='Published reference only; MOTA not reproduced and the entire paper is not replicated',
                              url='https://arxiv.org/html/2405.05210v3'),
        code_checks=checks,
        tcaff_commit=subprocess.check_output(['git', '-C', str(ROOT / 'vendor/tcaff'), 'rev-parse', 'HEAD'], text=True).strip(),
        clipper_commit=subprocess.check_output(['git', '-C', str(ROOT / 'clipper'), 'rev-parse', 'HEAD'], text=True).strip(),
        environment=dict(python=sys.version, versions={n: importlib.metadata.version(n) for n in ['numpy', 'scipy', 'robotdatapy', 'rosbags']},
                         clipperpy_path=clipperpy.__file__,
                         clipperpy_files=[dict(path=str(p), sha256=digest(p)) for p in sorted(Path(clipperpy.__file__).parent.glob('*.so'))],
                         stochastic_frontend='Upstream CLIPPER random_device retained; stored proposals make filter replay deterministic'),
        params=params, summaries=summaries,
        replay=dict(pair_updates=sum(len(t['frames']) for t in traces), mismatches=0,
                    max_coordinate_error=max(t['max_replay_coordinate_error'] for t in traces)),
        ground_truth_audit=dict(max_coordinate_error=max(t['max_GT_coordinate_error'] for t in traces),
                               composition='T_odom_i_robot_i @ inverse(T_world_robot_i) @ T_world_robot_j @ inverse(T_odom_j_robot_j)'),
        input_hashes=[dict(path=str(p.relative_to(ROOT)), sha256=digest(p), bytes=p.stat().st_size) for p in used_inputs],
        limits=['One recording; directed pairs and frames are correlated',
                'RGB-D segmentation is provided by the authors, not rerun',
                'No segmentation-change or merge variant is in the primary baseline',
                'Tau=8 control changes only the temporal acceptance threshold on saved proposals'],
    )
    (out / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    (out / 'error_traces.json').write_text(json.dumps(traces, indent=2) + '\n')
    plot(traces, out)
    print(json.dumps({s['variant']: {k: v for k, v in s.items() if k != 'pairs'} for s in summaries}, indent=2), flush=True)


if __name__ == '__main__':
    main()
