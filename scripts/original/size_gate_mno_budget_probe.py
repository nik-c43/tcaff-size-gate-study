"""Exploratory MNO search-budget probe on development anchors only.

Within each new eight-attempt call compare the first four solutions with all
eight, including the unchanged manager relative-score rule. No holdout tuning,
temporal updates, baseline rewrites or new choice of size parameters.
"""
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from size_gate_candidate_stage_audit import inspect,read,digest,check_pins
from size_gate_evaluate import validate_labels


def prefix_metrics(candidates,budget,fraction):
    candidates = candidates[:budget]
    scores = [c['score'] for c in candidates if c['score'] is not None]
    best = max(scores) if scores else 0.
    geometric = [c for c in candidates if c['success'] and c['passes_manager_geometry']]
    retained = [c for c in geometric if c['score']>fraction*best]
    return dict(good_pre_quality=float(any(c['correct'] for c in geometric)),
                good_post_quality=float(any(c['correct'] for c in retained)),
                wrong_retained=float(sum(not c['correct'] for c in retained)))


def main():
    out = ROOT/'results/size_gate_mno_budget_probe_20261009'
    if out.exists():raise ValueError('Refuse overwrite')
    primary = read(ROOT/'results/size_gate_study_20261009/summary.json')
    manifest = read(ROOT/'data/size_gate_study/analysis_private/manifest.json')
    labels_path = ROOT/'data/size_gate_study/labels.json'
    assert digest(labels_path)==primary['labels_sha256']
    labels = validate_labels(manifest,read(labels_path))
    check_pins(manifest['protocol'])
    episodes = [e for e in manifest['episodes'] if e['split']=='development']
    sources = {pair:read(ROOT/'data/tcaff_baseline_release'/f'{pair}.json') for pair in {e['pair'] for e in episodes}}
    out.mkdir();(out/'runs').mkdir()
    rows = defaultdict(list)
    for episode in episodes:
        tasks = {(t['left']['map_index'],t['right']['map_index']):t for t in manifest['tasks'] if t['episode_id']==episode['episode_id']}
        original = sources[episode['pair']]
        assert original['params']['clipper_mult_repeats']==4
        data = dict(original,params=dict(original['params'],clipper_mult_repeats=8))
        for repeat in range(1,6):
            for alpha in [1.,primary['alpha']]:
                run = inspect(data,episode,alpha,repeat,tasks,labels)
                fraction = run['params']['max_opt_fraction']
                four,eight = prefix_metrics(run['candidates'],4,fraction),prefix_metrics(run['candidates'],8,fraction)
                assert eight['good_pre_quality']>=four['good_pre_quality']
                assert sum(c['retained'] for c in run['candidates'])==sum(
                    c['success'] and c['passes_manager_geometry'] and c['score']>fraction*max(
                        [r['score'] for r in run['candidates'] if r['score'] is not None],default=0.) for c in run['candidates'])
                run['prefix_comparison']={'4':four,'8':eight}
                rows[str(alpha)].append(dict(episode_id=episode['episode_id'],comparison=run['prefix_comparison']))
                (out/'runs'/f'{episode["episode_id"]}_a{alpha}_r{repeat}.json').write_text(json.dumps(run,ensure_ascii=False,indent=2)+'\n')
        print('completed '+episode['episode_id'],flush=True)
    summaries = {}
    for alpha,values in rows.items():
        grouped = defaultdict(list)
        for row in values:grouped[row['episode_id']].append(row['comparison'])
        summaries[alpha] = {budget:{key:float(np.mean([np.mean([r[budget][key] for r in repeated])
                                                     for repeated in grouped.values()])) for key in ['good_pre_quality','good_post_quality','wrong_retained']}
                            for budget in ['4','8']}
    check_pins(manifest['protocol'])
    (out/'summary.json').write_text(json.dumps(dict(summaries=summaries,anchors=12,calls=120,split='development',
        primary_summary_sha256=digest(ROOT/'results/size_gate_study_20261009/summary.json'),script_sha256=digest(Path(__file__)),
        helper_sha256=digest(ROOT/'scripts/size_gate_candidate_stage_audit.py'),command=sys.argv,
        scope='Exploratory prefix comparison of eight new MNO attempts on development anchors only; no temporal-filter or held-out effectiveness evaluation',
        limits='The eight-attempt call changes only the diagnostic search budget. First-four prefix shares actual candidate solves with the full-eight result; it is not the hidden random first-four output of earlier runs.'),ensure_ascii=False,indent=2)+'\n')
    lines = ['# Бюджет поиска MNO: диагностическая проверка на development', '',
             'В 120 новых вызовах на 12 ранних якорях сравнен фактический префикс из четырёх решений с восемью решениями того же вызова. '
             'Единственное дополнительное изменение — диагностический бюджет MNO 4 → 8. Размерный α остаётся выбранным ранее. '
             'Стоимость, слияние объектов и временной фильтр не менялись; временные обновления не запускались.', '',
             '| Размерный вариант | Правильное решение до порога, 4 → 8 | Правильное решение после порога, 4 → 8 | Неправильные сохранённые кандидаты, 4 → 8 |',
             '|---|---:|---:|---:|']
    for alpha,values in summaries.items():
        a,b=values['4'],values['8']
        lines.append(f"| α={alpha} | {100*a['good_pre_quality']:.2f}% → {100*b['good_pre_quality']:.2f}% | "
                     f"{100*a['good_post_quality']:.2f}% → {100*b['good_post_quality']:.2f}% | {a['wrong_retained']:.3f} → {b['wrong_retained']:.3f} |")
    lines += ['', 'Правильность: ≤1 м / ≤5°. Это проверка ограниченности поиска на уже использованной ранней части, '
              'не новое отложенное сравнение и не основание выбрать новый алгоритм. '
              'Доли усреднены по пяти повторам внутри якоря, затем по 12 зависимым якорям одной записи. '
              'Первичные 420 прогонов и выбор α не пересчитывались. Цена увеличения бюджета и конечная инициализация здесь не оценены.', '',
              '```bash',
              'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/tcaff-mpl \\',
              '  .venv/bin/python scripts/size_gate_mno_budget_probe.py', '```', '']
    (out/'report.md').write_text('\n'.join(lines))
    print(json.dumps(summaries,indent=2))


if __name__ == '__main__':main()
