"""Descriptive correctness-threshold sensitivity on completed saved runs.

No algorithm reruns, parameter selection or acceptance-rule changes. Every
threshold is reported, including the unchanged primary <=1m and <=5deg.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(full, cold, distance, angle):
    def good(frame):
        e = frame['error']
        return e is not None and e[0] <= distance and e[1] <= angle
    first = next((f for f in cold if f['estimate'] is not None),None)
    success = next((f for f in cold if good(f)),None)
    horizon = cold[-1]['time']-cold[0]['time']
    return dict(correct_update_availability=sum(good(f) for f in full)/len(full),
                wrong_first_rate=float(first is not None and not good(first)),
                correct_by_horizon_rate=float(success is not None),
                capped_time_to_correct_s=horizon if success is None else success['time']-cold[0]['time'])


def average(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['episode_id']].append(row['metrics'])
    episodes = [{key:float(np.mean([r[key] for r in repeats])) for key in repeats[0]}
                for repeats in grouped.values()]
    return {key:float(np.mean([e[key] for e in episodes])) for key in episodes[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',default='data/size_gate_runs_20261009')
    parser.add_argument('--main-results',default='results/size_gate_study_20261009')
    parser.add_argument('--output',default='results/size_gate_threshold_diagnosis_20261009')
    args = parser.parse_args()
    runs, main_results, out = ROOT/args.runs, ROOT/args.main_results, ROOT/args.output
    if out.exists() or not out.resolve().is_relative_to(ROOT/'results'):
        raise ValueError('Use a new separate results directory')
    primary = read(main_results/'summary.json')
    assert read(main_results/'final_checks.json')['all_checks_passed']
    manifest = read(ROOT/'data/size_gate_study/analysis_private/manifest.json')
    episodes = [e for e in manifest['episodes'] if e['split']=='holdout']
    alpha = primary['alpha']
    paths, cache, inputs = {},{},defaultdict(list)
    for episode in episodes:
        for a in [1.,alpha]:
            for repeat in range(1,6):
                full_path = runs/'holdout'/f'full_{episode["pair"]}_a{a}_r{repeat}.json'
                if full_path not in cache:
                    cache[full_path] = read(full_path)
                    paths[str(full_path.relative_to(ROOT))] = digest(full_path)
                cold_path = runs/'holdout'/f'{episode["episode_id"]}_a{a}_r{repeat}.json'
                cold = read(cold_path)['frames']
                paths[str(cold_path.relative_to(ROOT))] = digest(cold_path)
                full = cache[full_path]['frames'][episode['start']:episode['end']+1]
                assert [f['time'] for f in full] == [f['time'] for f in cold]
                assert all(f['truth'] is not None for f in full+cold)
                inputs[a].append((episode['episode_id'],full,cold))
    distances, angles = [.25,.5,1.,2.],[2.,5.,10.]
    grid = []
    for distance in distances:
        for angle in angles:
            row = dict(translation_limit_m=distance,yaw_limit_deg=angle,variants={})
            for a,streams in inputs.items():
                values = [dict(episode_id=eid,metrics=evaluate(full,cold,distance,angle)) for eid,full,cold in streams]
                row['variants'][str(a)] = average(values)
            grid.append(row)
    standard = next(row for row in grid if row['translation_limit_m']==1. and row['yaw_limit_deg']==5.)
    for a,key in [(1.,'rerun_B0'),(alpha,'selected_variant')]:
        metrics = standard['variants'][str(a)]
        np.testing.assert_allclose(metrics['correct_update_availability'],primary[key]['correct_update_availability'],atol=1e-12,rtol=0)
        for name in ['wrong_first_rate','correct_by_horizon_rate','capped_time_to_correct_s']:
            np.testing.assert_allclose(metrics[name],primary['cold_start'][str(a)][name],atol=1e-12,rtol=0)
    summary = dict(alpha=alpha,grid=grid,source_hashes=paths,
                   primary_summary_sha256=digest(main_results/'summary.json'),script_sha256=digest(Path(__file__)),
                   scope='Descriptive post-hoc threshold sensitivity, all 12 combinations reported; no retuning',
                   command=__import__('sys').argv,
                   correctness_unchanged_for_primary='Translation <=1m, yaw <=5deg',
                   inference='One recording; averaging repeats within each dependent episode; no significance claim')
    out.mkdir(parents=True)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    fig, axes = plt.subplots(1,2,figsize=(10,5))
    for ax,a,title in zip(axes,[1.,alpha],['B0',f'Ослабление α={alpha}']):
        matrix = np.array([[next(row for row in grid if row['translation_limit_m']==d and row['yaw_limit_deg']==t)
                            ['variants'][str(a)]['correct_update_availability']*100 for t in angles] for d in distances])
        im = ax.imshow(matrix,vmin=0,vmax=100,cmap='Blues',aspect='auto')
        for i in range(len(distances)):
            for j in range(len(angles)):
                ax.text(j,i,f'{matrix[i,j]:.1f}%',ha='center',va='center',color='white' if matrix[i,j]>60 else '#16304a')
        ax.set_xticks(range(len(angles)),[str(t) for t in angles])
        ax.set_yticks(range(len(distances)),[str(d) for d in distances])
        ax.add_patch(Rectangle((.5,1.5),1,1,fill=False,edgecolor='#dd7a00',linewidth=3))
        ax.set(xlabel='Предел ошибки угла, °',ylabel='Предел ошибки переноса, м',title=title)
    fig.colorbar(im,ax=axes,fraction=.04,pad=.04,label='Правильные оценки / все обновления, %')
    fig.suptitle('Чувствительность определения правильности на отложенных окнах',fontsize=13)
    fig.text(.5,.03,'Оранжевый контур: исходный критерий 1 м / 5°. Одна запись, зависимые окна.',ha='center',fontsize=10)
    fig.subplots_adjust(top=.85,bottom=.18,left=.09,right=.86,wspace=.35)
    fig.savefig(out/'correctness_thresholds.png',dpi=160)
    plt.close(fig)
    lines = ['# Чувствительность к порогам правильности', '',
             'Проверка research_plan.md §8 выполнена на сохранённых результатах: меняется только постфактум определение '
             'правильности, не вход алгоритма, порог принятия или выбранное ослабление. Основной результат остаётся 1 м / 5°.', '',
             '| Перенос, м / угол, ° | Правильные обновления B0 → вариант, % | Неверное первое решение B0 → вариант, % | Правильная инициализация к горизонту B0 → вариант, % |',
             '|---|---:|---:|---:|']
    for row in grid:
        b,r = row['variants']['1.0'],row['variants'][str(alpha)]
        entries = [f"{row['translation_limit_m']:g} / {row['yaw_limit_deg']:g}"]
        for key in ['correct_update_availability','wrong_first_rate','correct_by_horizon_rate']:
            entries.append(f'{100*b[key]:.2f} → {100*r[key]:.2f}')
        lines.append('| '+' | '.join(entries)+' |')
    lines += ['', 'Правильные обновления оцениваются с полной временной историей; две метрики инициализации — '
              'на отдельных холодных стартах. Знаменатель включает отсутствие результата. '
              'Усреднение: пять повторов внутри эпизода, затем 12 зависимых эпизодов одной записи.', '',
              '![Пороги правильности](correctness_thresholds.png)', '',
              'Вся сетка приведена без выбора удобного порога. Протокол основного сравнения не пересматривался. '
              'Совпадение метрик при 1 м / 5° с основным отчётом проверено до сохранения. '
              'Сырые ошибки, SHA входов и время до правильной оценки для каждого порога доступны в JSON. '
              'Независимость эпизодов и статистическая значимость не заявляются.', '',
              '```bash',
              'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/tcaff-mpl \\',
              '  .venv/bin/python scripts/size_gate_threshold_diagnosis.py', '```', '']
    (out/'report.md').write_text('\n'.join(lines))
    print(json.dumps(dict(report=str(out/'report.md'),primary_threshold_reproduced=True,grid=grid),indent=2))


if __name__ == '__main__':
    main()
