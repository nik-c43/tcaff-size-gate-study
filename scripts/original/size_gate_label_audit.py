"""Passive label and graph audit; never runs frame alignment or chooses a gate.

Reports five-correspondence support in the frozen annotation pool. A witness
clique is not a claim that MNO-CLIPPER selects it, or that temporal TCAFF accepts
the resulting pose. Physical identity and geometric usefulness remain distinct.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import clipperpy
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from size_gate_evaluate import validate_labels
from size_gate_prepare import check_pins


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clique_witness(adjacency, required=5):
    """Find any clique of the required size, with exact exhaustive pruning."""
    n = len(adjacency)
    neighbors = [sum(1 << int(j) for j in np.flatnonzero(row)) for row in adjacency]

    def search(candidates, chosen):
        if len(chosen) == required:
            return chosen
        if candidates.bit_count() < required - len(chosen):
            return None
        while candidates.bit_count() >= required - len(chosen):
            bit = candidates & -candidates
            i = bit.bit_length() - 1
            candidates ^= bit
            result = search(candidates & neighbors[i], chosen + [i])
            if result is not None:
                return result
        return None

    return search((1 << n) - 1, [])


def support(tasks, ma, mb, params):
    if not tasks:
        return dict(candidate_edges=0, one_to_one_support=0, clique_at_least_5=False, witness_task_ids=[])
    associations = np.array([[t['left']['map_index'], t['right']['map_index']] for t in tasks], dtype=np.int32)
    allowed = np.zeros((len(ma), len(mb)), dtype=bool)
    allowed[associations[:, 0], associations[:, 1]] = True
    matching = maximum_bipartite_matching(csr_matrix(allowed), perm_type='column')
    p, q = ma[associations[:, 0], :2], mb[associations[:, 1], :2]
    d1 = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
    d2 = np.linalg.norm(q[:, None, :] - q[None, :, :], axis=2)
    discrepancy = np.abs(d1 - d2)
    distinct = ((associations[:, None, 0] != associations[None, :, 0]) &
                (associations[:, None, 1] != associations[None, :, 1]))
    cpp_params = clipperpy.Params()
    weights = np.exp(-.5 * discrepancy**2 / params['clipper_sigma']**2)
    adjacency = distinct & (discrepancy < params['clipper_epsilon']) & (weights > cpp_params.affinityeps)
    np.fill_diagonal(adjacency, False)
    invariant_params = clipperpy.invariants.EuclideanDistanceParams()
    invariant_params.sigma = params['clipper_sigma']
    invariant_params.epsilon = params['clipper_epsilon']
    invariant_params.mindist = 0.
    cpp = clipperpy.CLIPPER(clipperpy.invariants.EuclideanDistance(invariant_params), cpp_params)
    cpp.set_parallelize(False)
    cpp.score_pairwise_consistency(ma[:, :2].T.copy(), mb[:, :2].T.copy(), associations)
    reference = np.asarray(cpp.get_constraint_matrix()) > 0
    np.fill_diagonal(reference, False)
    if not np.array_equal(adjacency, reference):
        raise ValueError('Graph audit differs from pinned CLIPPER constraint matrix')
    witness = clique_witness(adjacency)
    if witness is not None:
        subgraph = adjacency[np.ix_(witness, witness)]
        if len(witness) != 5 or not np.all(subgraph | np.eye(5, dtype=bool)):
            raise ValueError('Invalid five-clique witness')
    return dict(candidate_edges=len(tasks), one_to_one_support=int(np.sum(matching >= 0)),
                clique_at_least_5=witness is not None,
                witness_task_ids=[] if witness is None else [tasks[i]['task_id'] for i in witness],
                graph_matches_pinned_CLIPPER=True)


def distribution(values):
    a = np.asarray(values)
    return dict(n=len(a), median_m=None if not len(a) else float(np.median(a)),
                p90_m=None if not len(a) else float(np.percentile(a, 90)),
                min_m=None if not len(a) else float(np.min(a)), max_m=None if not len(a) else float(np.max(a)),
                descriptive_count_below_0_1m=int(np.sum(a < .1)),
                descriptive_count_below_0_25m=int(np.sum(a < .25)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', default='data/size_gate_study/labels.json')
    parser.add_argument('--output', default='results/size_gate_full_annotation_audit')
    args = parser.parse_args()
    labels_path, out = ROOT / args.labels, ROOT / args.output
    if out.exists() or not out.resolve().is_relative_to(ROOT / 'results'):
        raise ValueError('Use a new results directory')
    study = ROOT / 'data/size_gate_study'
    frozen = json.loads((study / 'bundle_frozen.json').read_text())
    if digest(study / 'analysis_private/manifest.json') != frozen['manifest_sha256']:
        raise ValueError('Manifest changed after freeze')
    manifest = json.loads((study / 'analysis_private/manifest.json').read_text())
    for name, expected in manifest['provenance']['code_hashes'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Frozen code changed: ' + name)
    check_pins(manifest['protocol'])
    source_hash = digest(labels_path)
    labels = validate_labels(manifest, json.loads(labels_path.read_text()))
    episodes = manifest['episodes']
    source = {pair: json.loads((ROOT / manifest['protocol']['baseline'] / (pair + '.json')).read_text())
              for pair in {e['pair'] for e in episodes}}
    groups = {
        'same_object_all_confidences': {'high', 'medium', 'low'},
        'same_object_high_or_medium': {'high', 'medium'},
        'same_object_high_only': {'high'},
    }
    rows, distances = [], defaultdict(lambda: defaultdict(list))
    for episode in episodes:
        tasks = [t for t in manifest['tasks'] if t['episode_id'] == episode['episode_id']]
        frame = source[episode['pair']]['frames'][episode['anchor']]
        ma, mb = np.asarray(frame['map_a']).reshape(-1, 6), np.asarray(frame['map_b']).reshape(-1, 6)
        truth = np.asarray(frame['truth_se3'])
        params = source[episode['pair']]['params']
        row = dict(episode_id=episode['episode_id'], pair=episode['pair'], split=episode['split'], groups={})
        for task in tasks:
            p = ma[task['left']['map_index'], :3]
            q = mb[task['right']['map_index'], :3] @ truth[:3, :3].T + truth[:3, 3]
            xyz = float(np.linalg.norm(p - q))
            if abs(xyz - task['distance_m']) > 1e-10:
                raise ValueError('GT distance differs from frozen selection')
            label = labels[task['task_id']]['label']
            distances[label]['xy'].append(float(np.linalg.norm((p - q)[:2])))
            distances[label]['xyz'].append(xyz)
        for name, confidences in groups.items():
            candidates = [t for t in tasks if labels[t['task_id']]['label'] == 'same_object' and
                          labels[t['task_id']]['confidence'] in confidences]
            row['groups'][name] = dict(before=support(candidates, ma, mb, params),
                                       after=support([t for t in candidates if t['gate']['admitted']], ma, mb, params))
        rows.append(row)
    summaries = {}
    for name in groups:
        summaries[name] = dict(
            episodes=24,
            before_one_to_one_at_least_5=sum(r['groups'][name]['before']['one_to_one_support'] >= 5 for r in rows),
            after_one_to_one_at_least_5=sum(r['groups'][name]['after']['one_to_one_support'] >= 5 for r in rows),
            before_clique_at_least_5=sum(r['groups'][name]['before']['clique_at_least_5'] for r in rows),
            after_clique_at_least_5=sum(r['groups'][name]['after']['clique_at_least_5'] for r in rows),
            clique_drop_5_to_below_5=sum(r['groups'][name]['before']['clique_at_least_5'] and
                                       not r['groups'][name]['after']['clique_at_least_5'] for r in rows))
    summary = dict(fingerprint=manifest['fingerprint'], labels_sha256=source_hash,
                   script_sha256=digest(Path(__file__)), command=sys.argv,
                   scope='Passive diagnostic of annotated near-edge pool; no alignment runs or parameter selection',
                   support=summaries,
                   anchor_map_GT_distances_by_label={label: {dim: distribution(values) for dim, values in axes.items()}
                                                     for label, axes in distances.items()},
                   graph_semantics='Distinct associations; absolute difference of intra-map XY distances < original epsilon=0.25m; sigma=0.15m and original affinityeps; checked against pinned CLIPPER',
                   coordinate_scope='Saved map centroids at anchor transformed by GT; not per-observation physical-object positions; 3D-near selection radius=0.6m',
                   class_caveat='same_object is the recorded human category; mixed box/group units and staticness require review. No automatic relabeling.',
                   inferential_caveat='One recording with correlated episodes. Witness existence is not selected solver output or evidence of correct final pose.',
                   distance_threshold_caveat='0.1m and 0.25m point-residual bins are descriptive, not the CLIPPER pairwise-discrepancy test or identity truth.',
                   relaxation_run=False, baseline_or_labels_modified=False)
    if digest(labels_path) != source_hash:
        raise ValueError('Labels changed during audit')
    check_pins(manifest['protocol'])
    out.mkdir(parents=True)
    (out / 'analysis_private').mkdir()
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    (out / 'analysis_private/per_episode.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'support': summaries, 'all_graphs_verified_against_original_CLIPPER': True,
                      'baseline_or_labels_modified': False}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
