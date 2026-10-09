"""Passive comparison of frozen map dimensions and last source dimensions.

This is a descriptor diagnostic, not a new candidate generator or baseline.
All object IDs, assignments, labels, maps and pose estimates stay unchanged.
"""
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from size_gate_candidate_stage_audit import read, digest, check_pins
from size_gate_evaluate import validate_labels
from size_gate_prepare import gate_details
from size_gate_source_size_stability import history_key


def last_row(side):
    source = side['last_observation']
    return [0., 0., 0., source['observed_width_m'], source['observed_height_m'], 1.]


def main():
    out = ROOT / 'results/size_gate_smoothing_diagnosis_20261009'
    if out.exists():
        raise ValueError('Refuse overwrite')
    manifest_path = ROOT / 'data/size_gate_study/analysis_private/manifest.json'
    manifest = read(manifest_path)
    label_path = ROOT / 'data/size_gate_study/labels.json'
    labels = validate_labels(manifest, read(label_path))
    primary = read(ROOT / 'results/size_gate_study_20261009/summary.json')
    assert digest(label_path) == primary['labels_sha256']
    check_pins(manifest['protocol'])
    histories = {}
    rows = []
    for task in manifest['tasks']:
        left, right = task['left'], task['right']
        map_gate = gate_details(left['map_row'], right['map_row'])
        assert map_gate == task['gate']
        source_gate = gate_details(last_row(left), last_row(right))
        answer = labels[task['task_id']]
        row = dict(task_id=task['task_id'], episode_id=task['episode_id'],
                   label=answer['label'], confidence=answer['confidence'],
                   map_pair_size_admitted=map_gate['admitted'],
                   last_source_pair_size_admitted=source_gate['admitted'],
                   map_gate=map_gate, last_source_gate=source_gate,
                   source_observation_gap_s=abs(left['last_observation']['time'] - right['last_observation']['time']),
                   source_stale_s=[left['stale_s'], right['stale_s']],
                   both_histories_at_least_5=all(len(s['source_chain']) >= 5 for s in [left, right]),
                   last_source_exceeds_max_object_width=any(last_row(s)[3] > 2.5 for s in [left, right]))
        rows.append(row)
        for side in [left, right]:
            key = history_key(side)
            value = dict(track_id=side['track_id'],
                         map_width_height_m=side['map_row'][3:5],
                         last_source_width_height_m=last_row(side)[3:5],
                         source_observations=len(side['source_chain']),
                         absolute_map_last_difference_m=np.abs(np.asarray(side['map_row'][3:5])-last_row(side)[3:5]).tolist(),
                         map_last_pair_size_rejected=not gate_details(side['map_row'], last_row(side))['admitted'])
            if key in histories:
                assert histories[key] == value
            histories[key] = value
    def group_stats(group):
        return dict(pairs=len(group),
                    map_admitted_last_admitted=sum(r['map_pair_size_admitted'] and r['last_source_pair_size_admitted'] for r in group),
                    map_admitted_last_rejected=sum(r['map_pair_size_admitted'] and not r['last_source_pair_size_admitted'] for r in group),
                    map_rejected_last_admitted=sum(not r['map_pair_size_admitted'] and r['last_source_pair_size_admitted'] for r in group),
                    map_rejected_last_rejected=sum(not r['map_pair_size_admitted'] and not r['last_source_pair_size_admitted'] for r in group))
    all_groups = {label: group_stats([r for r in rows if r['label'] == label]) for label in manifest['protocol']['annotation']['labels']}
    high_groups = {label: group_stats([r for r in rows if r['label'] == label and r['confidence'] == 'high']) for label in all_groups}
    values = list(histories.values())
    differences = np.asarray([s['absolute_map_last_difference_m'] for s in values])
    totals = dict(unique_history_snapshots=len(values),
                  map_last_size_rejected=sum(s['map_last_pair_size_rejected'] for s in values),
                  map_last_difference_quantiles_m={str(q): np.percentile(differences, q, axis=0).tolist() for q in [50, 90, 95, 100]},
                  last_source_width_exceeds_2_5m_pairs=sum(r['last_source_exceeds_max_object_width'] for r in rows))
    reasons = {version: dict(Counter(c for r in rows for c in r[version]['failed_checks'])) for version in ['map_gate', 'last_source_gate']}
    summary = dict(totals=totals, all_confidence_pair_groups=all_groups, high_confidence_pair_groups=high_groups,
                   failed_check_counts=reasons, labels_sha256=primary['labels_sha256'],
                   manifest_sha256=digest(manifest_path), script_sha256=digest(Path(__file__)), command=sys.argv,
                   scope='Passive pairwise ratio/difference checks on map state versus last source observation; no alternative solver/filter evaluation',
                   caveats=['Last observations are asynchronous and can be stale.',
                            'Same track ID and a last-source observation do not certify earlier physical identity.',
                            'Map-versus-source differences cannot isolate harmful lag from beneficial denoising.',
                            'Pair checks exclude the separate max-object-width criterion; raw width violations are recorded separately.',
                            'Selected gate-loss episodes and correlated histories are not population samples.'])
    check_pins(manifest['protocol'])
    assert digest(label_path) == primary['labels_sha256']
    out.mkdir()
    (out / 'analysis_private').mkdir()
    for name, value in [('summary.json', summary), ('analysis_private/pairs.json', rows), ('analysis_private/history_snapshots.json', histories)]:
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    lines = ['# Размеры карты и последнего исходного наблюдения', '',
             'Пассивный разбор фиксированных соответствий. Для каждой пары повторно вычислены только исходные размерные проверки: '
             'по размерам карты и по размерам последних связанных детекций. Карты, трекер, MNO и временной фильтр не запускались с заменой размеров.', '',
             f"В {totals['map_last_size_rejected']}/{len(values)} уникальных снимков истории размеры карты и последнего наблюдения сами не проходят парный допуск B0. "
             f"Медиана абсолютной разницы ширины/высоты: {np.median(differences[:,0]):.4f}/{np.median(differences[:,1]):.4f} м; "
             f"p90: {np.percentile(differences[:,0],90):.4f}/{np.percentile(differences[:,1],90):.4f} м.", '',
             '| Категория, высокая уверенность | Пар | Карта + / источник + | Карта + / источник − | Карта − / источник + | Карта − / источник − |',
             '|---|---:|---:|---:|---:|---:|']
    for label, group in high_groups.items():
        lines.append(f"| {label} | {group['pairs']} | {group['map_admitted_last_admitted']} | {group['map_admitted_last_rejected']} | {group['map_rejected_last_admitted']} | {group['map_rejected_last_rejected']} |")
    lines += ['', '«Источник +» не означает достоверное или лучшее соответствие: размеры относятся к последним видимым сегментам, '
              'наблюдения двух роботов могут быть разновременными. Это не абляция сглаживания и не оценка альтернативной позы. '
              'Расхождение может отражать полезное подавление шума, задержку состояния или изменение сегмента; причины здесь не разделены. '
              'Таблица учитывает парные проверки отношения и разности; отдельное ограничение максимальной ширины не применяется к альтернативному набору кандидатов. '
              f"Пар с последней исходной шириной >2.5 м: {totals['last_source_width_exceeds_2_5m_pairs']}.", '',
              '```bash', 'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/tcaff-mpl \\',
              '  .venv/bin/python scripts/size_gate_smoothing_diagnosis.py', '```', '']
    (out / 'report.md').write_text('\n'.join(lines))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
