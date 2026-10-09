"""Compare 3D annotation-pool support with planar algorithm support.

GT proximity defines a diagnostic candidate pool, never physical identity or a
new online filter. Runs only pairwise scoring and exact five-clique search.
"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from size_gate_candidate_stage_audit import read,digest,check_pins,object_map,TCAFFManager
from size_gate_label_audit import support


def main():
    out = ROOT/'results/size_gate_dimension_audit_20261009'
    if out.exists():
        raise ValueError('Refuse overwrite')
    manifest = read(ROOT/'data/size_gate_study/analysis_private/manifest.json')
    primary = read(ROOT/'results/size_gate_study_20261009/summary.json')
    check_pins(manifest['protocol'])
    sources = {pair:read(ROOT/'data/tcaff_baseline_release'/f'{pair}.json') for pair in {e['pair'] for e in manifest['episodes']}}
    rows = []
    for episode in manifest['episodes']:
        data = sources[episode['pair']]
        frame = data['frames'][episode['anchor']]
        ma,mb = np.asarray(frame['map_a']).reshape(-1,6),np.asarray(frame['map_b']).reshape(-1,6)
        truth = np.asarray(frame['truth_se3'])
        distance3 = np.linalg.norm(ma[:,None,:3]-(mb[:,:3]@truth[:3,:3].T+truth[:3,3])[None,:,:],axis=2)
        x,y,angle = frame['truth']
        rotation = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        distance2 = np.linalg.norm(ma[:,None,:2]-(mb[:,:2]@rotation.T+[x,y])[None,:,:],axis=2)
        kwargs = {k:v for k,v in data['params'].items() if k!='ts'}
        gates = {'before':np.ones((len(ma),len(mb)),dtype=bool)}
        for name,alpha in [('B0',1.),('selected',primary['alpha'])]:
            manager = TCAFFManager(**dict(kwargs,wh_scale_diff=1+alpha*(kwargs['wh_scale_diff']-1),h_diff=alpha*kwargs['h_diff']))
            indices = manager.get_putative_assoc(object_map(ma),object_map(mb))
            gate = np.zeros((len(ma),len(mb)),dtype=bool)
            gate[indices[:,0],indices[:,1]] = True
            gates[name] = gate
        record = dict(episode_id=episode['episode_id'],split=episode['split'],support={})
        for dimension,distances in [('XYZ',distance3),('XY_SE2',distance2)]:
            record['support'][dimension] = {}
            for name,gate in gates.items():
                edges = np.argwhere((distances<.6)&gate)
                tasks = [dict(task_id=f'{i}:{j}',left=dict(map_index=int(i)),right=dict(map_index=int(j))) for i,j in edges]
                record['support'][dimension][name] = support(tasks,ma,mb,data['params'])
        assert record['support']['XYZ']['before']['one_to_one_support']>=5
        assert record['support']['XYZ']['B0']['one_to_one_support']<5
        rows.append(record)
    counts = {}
    for split in ['all','development','holdout']:
        selected = [r for r in rows if split=='all' or r['split']==split]
        counts[split] = {dimension:{name:dict(episodes=len(selected),
            one_to_one_at_least_5=sum(r['support'][dimension][name]['one_to_one_support']>=5 for r in selected),
            clique_at_least_5=sum(r['support'][dimension][name]['clique_at_least_5'] for r in selected))
            for name in gates} for dimension in ['XYZ','XY_SE2']}
    check_pins(manifest['protocol'])
    out.mkdir()
    (out/'summary.json').write_text(json.dumps(dict(counts=counts,per_episode=rows,
        proximity_limit_m=.6,primary_summary_sha256=digest(ROOT/'results/size_gate_study_20261009/summary.json'),
        script_sha256=digest(Path(__file__)),command=sys.argv,
        XYZ_reference='Full saved GT SE3, original annotation pool',XY_reference='Saved planar GT SE2, matching coordinates used by algorithm',
        scope='Passive support audit, no semantic labels, solving, temporal updates or tuning',
        caveat='GT-close graph can contain different physical objects. Existence of five compatible edges does not ensure selected or correct pose.'),ensure_ascii=False,indent=2)+'\n')
    lines = ['# 3D-пул разметки и поддержка плоского совмещения', '',
             'Исходные 24 эпизода выбраны по 3D-близости центров <0.6 м после GT SE(3). '
             'Алгоритм сопоставляет только XY-центры. Проверен отдельный пул с расстоянием <0.6 м '
             'после того же планарного GT SE(2), по которому оценивается ошибка позы. '
             'Размерные параметры, исходная выборка и разметка не менялись.', '',
             '| Часть / координаты | ≥5 взаимно однозначных до / B0 / ослабление | Есть совместимая пятёрка до / B0 / ослабление |',
             '|---|---|---|']
    for split,dimensions in counts.items():
        for dimension,variants in dimensions.items():
            matching=' / '.join(str(variants[v]['one_to_one_at_least_5']) for v in ['before','B0','selected'])
            cliques=' / '.join(str(variants[v]['clique_at_least_5']) for v in ['before','B0','selected'])
            lines.append(f'| {split} ({next(iter(variants.values()))["episodes"]}) / {dimension} | {matching} | {cliques} |')
    lines += ['', 'Графы совместимости проверены по исходной C++-матрице ограничений CLIPPER. '
              'Это геометрический диагностический пул; в нём нет утверждения о физической идентичности. '
              'Все условия исходного 3D-отбора подтверждены. Если в XY после B0 остаётся пятёрка, '
              'потеря пяти 3D-близких пар не означает отсутствия всякой поддержки у плоского алгоритма.', '',
              'Даже наличие XY-клики не гарантирует правильную ориентацию, достаточное пространственное покрытие, '
              'выбор решателем или ошибку конечной позы ≤1 м / ≤5°. '
              'Семантическая разметка 798 пар остаётся отдельной проверкой, её нельзя автоматически перенести на новые XY-пары.', '',
              '```bash',
              'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/tcaff-mpl \\',
              '  .venv/bin/python scripts/size_gate_dimension_audit.py', '```', '']
    (out/'report.md').write_text('\n'.join(lines))
    print(json.dumps(counts,indent=2))


if __name__ == '__main__':
    main()
