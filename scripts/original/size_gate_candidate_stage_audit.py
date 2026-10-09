"""Exploratory anchor reruns recording candidates before manager rejection.

Wraps align_objects without modifying its returned objects, order or content.
Runs unchanged B0 and the already selected relaxation at all 12 holdout anchors,
five repetitions. No new tuning or temporal updates. These newly solved
associations cannot reconstruct the exact random solutions of earlier runs.
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from h1_map_audit import object_map
from size_gate_prepare import check_pins
from size_gate_evaluate import validate_labels
from tcaff_baseline_audit import pose_error
from tcaff.tcaff.tcaff_manager import TCAFFManager
from tcaff.utils.transform import transform_2_xypsi


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect(data,episode,alpha,repeat,task_by_edge,labels):
    kwargs = {k:v for k,v in data['params'].items() if k!='ts'}
    kwargs.update(wh_scale_diff=1+alpha*(kwargs['wh_scale_diff']-1),h_diff=alpha*kwargs['h_diff'])
    manager = TCAFFManager(**kwargs)
    frame = data['frames'][episode['anchor']]
    captured = []
    original = manager.fa.align_objects
    def capture(*args,**arguments):
        result = original(*args,**arguments)
        captured.extend(result)
        return result
    manager.fa.align_objects = capture
    ma,mb = np.asarray(frame['map_a']).reshape(-1,6),np.asarray(frame['map_b']).reshape(-1,6)
    measurements = manager.get_frame_align_measurements(object_map(ma),object_map(mb))
    kept = {id(solution) for solution in manager.latest_fa_solutions}
    scores = [s.objective_score for s in captured if s.objective_score is not None]
    maximum = max(scores) if scores else None
    candidates = []
    for index,solution in enumerate(captured):
        row = dict(candidate_index=index,success=bool(solution.success),retained=id(solution) in kept,
                   associated_objects=solution.num_objs_associated,score=None if solution.objective_score is None else float(solution.objective_score),
                   score_fraction_of_best=None,passes_manager_geometry=False,proper_rotation=False,
                   error=None,correct=False,associations=[],manual_categories={})
        if solution.success:
            T = np.asarray(solution.transform)
            row['transform'] = T.tolist()
            row['proper_rotation'] = bool(np.linalg.det(T[:2,:2])>0)
            row['passes_manager_geometry'] = bool(np.allclose(T[0,0],T[1,1]))
            row['score_fraction_of_best'] = None if maximum==0 else float(solution.objective_score/maximum)
            expected = row['passes_manager_geometry'] and solution.objective_score>manager.max_opt_fraction*maximum
            assert row['retained'] == bool(expected)
            if row['proper_rotation']:
                row['error'] = pose_error(transform_2_xypsi(T),frame['truth'])
                row['correct'] = row['error'][0]<=1. and row['error'][1]<=5.
            associations = np.asarray(solution.associated_objs).reshape(-1,2).astype(int)
            assert len(associations) == solution.num_objs_associated
            manual = Counter()
            truth3 = np.asarray(frame['truth_se3'])
            for left,right in associations:
                edge = (int(left),int(right))
                task = task_by_edge.get(edge)
                residual = ma[left,:3]-(mb[right,:3]@truth3[:3,:3].T+truth3[:3,3])
                associated = dict(left_map_index=int(left),right_map_index=int(right),
                                  GT_xyz_distance_m=float(np.linalg.norm(residual)),GT_xy_distance_m=float(np.linalg.norm(residual[:2])),
                                  task_id=None if task is None else task['task_id'])
                if task is not None:
                    associated['recorded_label'] = labels[task['task_id']]['label']
                    associated['recorded_confidence'] = labels[task['task_id']]['confidence']
                    associated['left_track_id'] = task['left']['track_id']
                    associated['right_track_id'] = task['right']['track_id']
                    manual[associated['recorded_label']] += 1
                else:
                    manual['outside_annotated_pool'] += 1
                row['associations'].append(associated)
            row['manual_categories'] = dict(manual)
        candidates.append(row)
    good_pre = [c for c in candidates if c['correct'] and c['passes_manager_geometry']]
    good_post = [c for c in good_pre if c['retained']]
    assert len(measurements) == sum(c['retained'] for c in candidates)
    return dict(episode_id=episode['episode_id'],pair=episode['pair'],anchor=episode['anchor'],alpha=alpha,repeat=repeat,
                params=kwargs,candidate_pairs=len(manager.get_putative_assoc(object_map(ma),object_map(mb))),
                candidates=candidates,
                metrics=dict(good_pre_quality=float(bool(good_pre)),good_post_quality=float(bool(good_post)),
                             good_lost_entirely_at_quality=float(bool(good_pre) and not good_post),
                             good_candidates_pruned_at_quality=sum(not c['retained'] for c in good_pre),
                             raw_successful=float(sum(c['success'] for c in candidates)),
                             wrong_retained=float(sum(c['retained'] and not c['correct'] for c in candidates))))


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['episode_id']].append(row['metrics'])
    episodes = [dict(episode_id=eid,metrics={k:float(np.mean([v[k] for v in values])) for k in values[0]})
                for eid,values in sorted(grouped.items())]
    return dict(episodes=episodes,metrics={k:float(np.mean([e['metrics'][k] for e in episodes])) for k in episodes[0]['metrics']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-results',default='results/size_gate_study_20261009')
    parser.add_argument('--output',default='results/size_gate_candidate_stage_audit_20261009')
    args = parser.parse_args()
    main_results,out = ROOT/args.main_results,ROOT/args.output
    if out.exists() or not out.resolve().is_relative_to(ROOT/'results'):
        raise ValueError('Use a new separate results directory')
    primary = read(main_results/'summary.json')
    assert read(main_results/'final_checks.json')['all_checks_passed']
    manifest = read(ROOT/'data/size_gate_study/analysis_private/manifest.json')
    labels = validate_labels(manifest,read(ROOT/'data/size_gate_study/labels.json'))
    check_pins(manifest['protocol'])
    episodes = [e for e in manifest['episodes'] if e['split']=='holdout']
    sources = {pair:read(ROOT/'data/tcaff_baseline_release'/f'{pair}.json') for pair in {e['pair'] for e in episodes}}
    quality_fractions = {data['params']['max_opt_fraction'] for data in sources.values()}
    assert len(quality_fractions)==1
    quality_fraction = next(iter(quality_fractions))
    rows = defaultdict(list)
    out.mkdir(parents=True)
    (out/'runs').mkdir()
    for episode in episodes:
        task_by_edge = {(t['left']['map_index'],t['right']['map_index']):t for t in manifest['tasks'] if t['episode_id']==episode['episode_id']}
        for repeat in range(1,6):
            for a in [1.,primary['alpha']]:
                run = inspect(sources[episode['pair']],episode,a,repeat,task_by_edge,labels)
                (out/'runs'/f'{episode["episode_id"]}_a{a}_r{repeat}.json').write_text(json.dumps(run,ensure_ascii=False,indent=2)+'\n')
                rows[str(a)].append(run)
        print('completed '+episode['episode_id'],flush=True)
    summaries = {a:aggregate(values) for a,values in rows.items()}
    check_pins(manifest['protocol'])
    summary = dict(alpha=primary['alpha'],summaries=summaries,run_count=sum(map(len,rows.values())),
                   primary_summary_sha256=digest(main_results/'summary.json'),labels_sha256=primary['labels_sha256'],
                   script_sha256=digest(Path(__file__)),command=sys.argv,upstream_unchanged=True,
                   scope='New exploratory stochastic anchor solves, before and after existing manager quality filter; no temporal TCAFF or parameter tuning',
                   attribution='Within each call recorded raw solutions and actually retained object identities; manager return values unchanged',
                   limitation='New random solutions are not the selected associations of the earlier 420 runs; manual anchor pool is incomplete for all solver associations')
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    b,r = summaries['1.0']['metrics'],summaries[str(primary['alpha'])]['metrics']
    lines = ['# Аудит кандидатов до и после штатных проверок менеджера', '',
             'После основного сравнения выполнено 120 новых стохастических вызовов на всех 12 отложенных якорях: '
             'B0 и выбранное ослабление, по пять повторов. Временной фильтр не запускался. '
             'Обёртка сохраняет результаты `FrameAligner.align_objects`, возвращая те же объекты без изменения порядка и значений. '
             'Фактически сохранённые `latest_fa_solutions` сверены с исходными условиями менеджера.', '',
             '| Диагностика на якорях | B0 | Ослабление |', '|---|---:|---:|']
    for key,title in [('good_pre_quality','Есть правильное решение до относительного порога качества'),
                      ('good_post_quality','Есть правильное решение после порога'),
                      ('good_lost_entirely_at_quality','Все правильные решения потеряны на пороге качества')]:
        lines.append(f'| {title} | {100*b[key]:.2f}% | {100*r[key]:.2f}% |')
    lines += ['', f"Среднее число правильных кандидатов, исключённых порогом: {b['good_candidates_pruned_at_quality']:.4f} → {r['good_candidates_pruned_at_quality']:.4f}.", '',
              f'Проверка относительного качества: `objective_score > {quality_fraction:g} * max_opt`; исходный порог не менялся. '
              'Правильными здесь считаются только решения с корректной ориентацией, проходящие геометрическое условие менеджера, '
              'и ошибкой ≤1 м / ≤5°. Отражения не интерпретировались как SE(2)-позы.', '',
              'Сырые матрицы, оценки качества, число объектов, выбранные соответствия, GT-расстояния центров и доступные ручные метки '
              'сохранены по каждому вызову. Пары вне 798 размеченных не получили выдуманных семантических меток. '
              'Близость центров не считается доказательством идентичности.', '',
              'Это новые решения C++ со случайной инициализацией. Они не восстанавливают точные соответствия исходных 420 прогонов '
              'и не являются новым отложенным сравнением эффективности. Метрики усреднены по повторам внутри якоря, затем по 12 зависимым якорям одной записи. '
              'Исходный размерный вариант, стоимость, число попыток MNO и upstream неизменны.', '',
              '```bash',
              'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/tcaff-mpl \\',
              '  .venv/bin/python scripts/size_gate_candidate_stage_audit.py', '```', '']
    (out/'report.md').write_text('\n'.join(lines))
    print(json.dumps(dict(report=str(out/'report.md'),summaries=summaries),indent=2))


if __name__ == '__main__':
    main()
