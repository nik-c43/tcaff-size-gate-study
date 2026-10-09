"""Exploratory extension of the unchanged candidate audit to holdout windows.

All 12 windows, all 29 updates, five repetitions and both frozen configurations.
Manual labels are used only at their exact annotated anchor, never transferred
to a different step's map row indices. Temporal TCAFF is not rerun.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from size_gate_candidate_stage_audit import inspect,read,digest
from size_gate_evaluate import validate_labels
from size_gate_prepare import check_pins


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-results',default='results/size_gate_study_20261009')
    parser.add_argument('--output',default='results/size_gate_candidate_window_audit_20261009')
    args = parser.parse_args()
    primary_dir,out = ROOT/args.main_results,ROOT/args.output
    if out.exists() or not out.resolve().is_relative_to(ROOT/'results'):
        raise ValueError('Use a new separate results directory')
    primary = read(primary_dir/'summary.json')
    assert read(primary_dir/'final_checks.json')['all_checks_passed']
    label_path = ROOT/'data/size_gate_study/labels.json'
    assert digest(label_path) == primary['labels_sha256']
    manifest = read(ROOT/'data/size_gate_study/analysis_private/manifest.json')
    labels = validate_labels(manifest,read(label_path))
    check_pins(manifest['protocol'])
    episodes = [e for e in manifest['episodes'] if e['split']=='holdout']
    sources = {pair:read(ROOT/'data/tcaff_baseline_release'/f'{pair}.json') for pair in {e['pair'] for e in episodes}}
    quality_fractions = {data['params']['max_opt_fraction'] for data in sources.values()}
    assert len(quality_fractions)==1
    quality_fraction = next(iter(quality_fractions))
    out.mkdir(parents=True)
    (out/'runs').mkdir()
    rows = defaultdict(list)
    total_calls = 0
    for episode in episodes:
        task_by_edge = {(t['left']['map_index'],t['right']['map_index']):t for t in manifest['tasks'] if t['episode_id']==episode['episode_id']}
        for repeat in range(1,6):
            for alpha in [1.,primary['alpha']]:
                run = []
                for step in range(episode['start'],episode['end']+1):
                    description = dict(episode,anchor=step)
                    local_tasks = task_by_edge if step==episode['anchor'] else {}
                    run.append(inspect(sources[episode['pair']],description,alpha,repeat,local_tasks,labels))
                total_calls += len(run)
                metrics = {key:float(np.mean([r['metrics'][key] for r in run])) for key in run[0]['metrics']}
                rows[str(alpha)].append(dict(episode_id=episode['episode_id'],repeat=repeat,metrics=metrics))
                (out/'runs'/f'{episode["episode_id"]}_a{alpha}_r{repeat}.json').write_text(json.dumps(dict(
                    episode_id=episode['episode_id'],pair=episode['pair'],start=episode['start'],end=episode['end'],
                    alpha=alpha,repeat=repeat,metrics=metrics,frames=run),ensure_ascii=False,indent=2)+'\n')
        print(f'completed {episode["episode_id"]}; {total_calls}/3480 calls',flush=True)
    summaries = {}
    for alpha,values in rows.items():
        grouped = defaultdict(list)
        for row in values:
            grouped[row['episode_id']].append(row['metrics'])
        by_episode = [dict(episode_id=eid,metrics={k:float(np.mean([r[k] for r in repeats])) for k in repeats[0]})
                      for eid,repeats in sorted(grouped.items())]
        summaries[alpha] = dict(per_episode=by_episode,metrics={k:float(np.mean([e['metrics'][k] for e in by_episode]))
                                                               for k in by_episode[0]['metrics']})
    assert total_calls == 3480
    check_pins(manifest['protocol'])
    assert digest(label_path) == primary['labels_sha256']
    summary = dict(alpha=primary['alpha'],summaries=summaries,candidate_calls=total_calls,
                   run_files=sum(map(len,rows.values())),primary_summary_sha256=digest(primary_dir/'summary.json'),
                   labels_sha256=primary['labels_sha256'],script_sha256=digest(Path(__file__)),
                   candidate_helper_sha256=digest(ROOT/'scripts/size_gate_candidate_stage_audit.py'),command=sys.argv,
                   scope='New exploratory candidate solves for all unchanged holdout windows; no temporal updates or tuning',
                   manual_label_scope='Only original exact anchor; other row indices not assigned semantic labels',
                   inference='Repeated windows on one recording, not independent experiments; results cannot reconstruct the original random associations')
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    b,r = summaries['1.0']['metrics'],summaries[str(primary['alpha'])]['metrics']
    lines = ['# Аудит относительного качества на всех отложенных окнах', '',
             'Проверено 3480 новых вызовов генератора на неизменённых картах: 12 окон × 29 обновлений × '
             '5 повторов × 2 фиксированные конфигурации. Это дополнительная диагностика после основного сравнения. '
             'Временной фильтр не запускался; пороги и стоимость не менялись.', '',
             '| На обновлениях отложенных окон | B0 | Ослабление |', '|---|---:|---:|']
    for key,title in [('good_pre_quality','Есть правильное решение до относительного порога'),
                      ('good_post_quality','Есть правильное решение после порога'),
                      ('good_lost_entirely_at_quality','Последнее правильное решение потеряно на пороге')]:
        lines.append(f'| {title} | {100*b[key]:.2f}% | {100*r[key]:.2f}% |')
    lines += ['', f"Среднее число правильных кандидатов, исключённых порогом: {b['good_candidates_pruned_at_quality']:.4f} → {r['good_candidates_pruned_at_quality']:.4f}.", '',
              'GT применяется только для постфактум оценки ≤1 м / ≤5°. Проверяется действующее условие '
              f'`objective_score > {quality_fraction:g} * max_opt` после геометрического ограничения менеджера. '
              'Записаны те же объекты решений, которые исходный менеджер фактически сохранил; обёртка не меняет выход генератора.', '',
              'Ручные метки привязаны только к точным якорям исходной разметки. Индексы строк карты меняются между кадрами; '
              'метки не переносились на соседние кадры. Все остальные выбранные пары считаются неразмеченными.', '',
              'Новые решения используют исходную случайную инициализацию CLIPPER. Это не точные скрытые соответствия '
              'предыдущих 420 прогонов. Доли усредняются сначала по кадрам и повторам внутри эпизода, затем по 12 зависимым эпизодам '
              'одной записи. Повторной настройки по holdout и новой независимой проверки здесь нет.', '',
              '```bash',
              'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/tcaff-mpl \\',
              '  .venv/bin/python scripts/size_gate_candidate_window_audit.py', '```', '']
    (out/'report.md').write_text('\n'.join(lines))
    print(json.dumps(dict(report=str(out/'report.md'),summaries=summaries),indent=2))


if __name__ == '__main__':
    main()
