"""Post-hoc local WLS contribution of a recorded different-object pair.

Fixed exploratory example P0735, newly captured E14 alpha=2 repeat=1 solution.
No CLIPPER rerun, map merging, cost correction or temporal-filter comparison.
"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from size_gate_candidate_stage_audit import read,digest,check_pins
from tcaff_baseline_audit import pose_error
from tcaff.realign.wls import wls
from tcaff.utils.transform import transform_2_xypsi,T2d_2_T3d


def main():
    out = ROOT/'results/size_gate_semantic_case_20261009'
    if out.exists():
        raise ValueError('Refuse overwrite')
    manifest = read(ROOT/'data/size_gate_study/analysis_private/manifest.json')
    check_pins(manifest['protocol'])
    path = ROOT/'results/size_gate_candidate_stage_audit_20261009/runs/E14_a2.0_r1.json'
    run = read(path)
    candidate = next(c for c in run['candidates'] if c['retained'] and c['correct'] and
                     any(a['task_id']=='P0735' for a in c['associations']))
    source_path = ROOT/'data/tcaff_baseline_release'/f'{run["pair"]}.json'
    frame = read(source_path)['frames'][run['anchor']]
    ma,mb = np.asarray(frame['map_a']).reshape(-1,6),np.asarray(frame['map_b']).reshape(-1,6)
    pairs = np.array([[a['left_map_index'],a['right_map_index']] for a in candidate['associations']])
    left,right = pairs[:,0],pairs[:,1]
    weights = 1/(.01+ma[left,5]*mb[right,5])
    original = wls(ma[left,:2],mb[right,:2],weights)
    np.testing.assert_allclose(T2d_2_T3d(original),candidate['transform'],atol=1e-10,rtol=0)
    target = next(i for i,a in enumerate(candidate['associations']) if a['task_id']=='P0735')
    keep = np.arange(len(pairs))!=target
    removed = wls(ma[left[keep],:2],mb[right[keep],:2],weights[keep])
    task = next(t for t in manifest['tasks'] if t['task_id']=='P0735')
    result = dict(task_id='P0735',candidate_index=candidate['candidate_index'],
                  recorded_label='different_objects',recorded_confidence='high',
                  original_error=pose_error(transform_2_xypsi(original),frame['truth']),
                  pair_removed_error=pose_error(transform_2_xypsi(removed),frame['truth']),
                  original_objects=len(pairs),remaining_objects=int(keep.sum()),
                  original_transform_reproduced=True,target_normalized_age_weight=float(weights[target]/weights.sum()),
                  source_observation_ages_s=[task[side]['stale_s'] for side in ['left','right']],
                  selected_pair=candidate['associations'][target],
                  source_hashes={str(p.relative_to(ROOT)):digest(p) for p in [path,source_path]},
                  script_sha256=digest(Path(__file__)),command=sys.argv,
                  scope='Exploratory local fit with fixed other associations; not rerunning CLIPPER or the full pipeline',
                  identity_caveat='No mask and a blurred source image; saved map GT distances and pose correctness are not independent physical identity proof')
    check_pins(manifest['protocol'])
    out.mkdir()
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    before,after = result['original_error'],result['pair_removed_error']
    lines = ['# Локальная проверка пары P0735', '',
             'В новых кандидатах E14, α=2.0, повтор 1 найдено правильное по критерию ≤1 м / ≤5° решение, '
             'содержащее P0735: ручная метка «разные объекты», высокая уверенность. '
             'Трансформация из всех семи исходных соответствий точно воспроизведена авторской WLS-функцией. '
             'Затем только для диагностики повторена подгонка тех же остальных соответствий без P0735.', '',
             f'Исходная ошибка: {before[0]:.6f} м / {before[1]:.6f}°. Без пары: {after[0]:.6f} м / {after[1]:.6f}°. '
             f"Нормированная возрастная масса пары: {100*result['target_normalized_age_weight']:.4f}%. После удаления остаётся шесть соответствий.", '',
             'Это локальный вклад при фиксированных остальных парах, не результат нового поиска CLIPPER или временного TCAFF. '
             'Нельзя по правильной общей позе объявлять каждую пару полезной или менять физическую метку. '
             'Источники пары наблюдались в разные моменты; близость сохранённых центров не доказывает совпадение предметов. '
             'Масок нет, один исходный кадр размыт. Причина вертикального расхождения и статичность предметов не установлены.', '',
             '```bash',
             'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/tcaff-mpl \\',
             '  .venv/bin/python scripts/size_gate_semantic_case.py', '```', '']
    (out/'report.md').write_text('\n'.join(lines))
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
