"""Passive size-descriptor stability audit of original source-chain histories.

Track identity is not physical-object truth. No new masks, tracking, alignment,
parameter selection or changes to labels and baseline maps.
"""
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from size_gate_candidate_stage_audit import read,digest,check_pins
from size_gate_prepare import gate_details
from size_gate_evaluate import validate_labels


def history_key(side):
    last=side['source_chain'][-1]
    return f'{side["track_id"][0]}:{side["track_id"][1]}:{last[0]}:{last[1]}'


def main():
    out=ROOT/'results/size_gate_source_size_stability_20261009'
    if out.exists():raise ValueError('Refuse overwrite')
    manifest=read(ROOT/'data/size_gate_study/analysis_private/manifest.json')
    label_path=ROOT/'data/size_gate_study/labels.json'
    labels=validate_labels(manifest,read(label_path))
    primary=read(ROOT/'results/size_gate_study_20261009/summary.json')
    assert digest(label_path)==primary['labels_sha256']
    check_pins(manifest['protocol'])
    histories={}
    for task in manifest['tasks']:
        for name in ['left','right']:
            side=task[name];key=history_key(side)
            if key in histories:
                assert histories[key]['source_chain']==side['source_chain']
            else:histories[key]=side
    stats,source_hashes={},{}
    for robot in sorted({side['track_id'][0] for side in histories.values()}):
        path=ROOT/'data/tcaff_mot_data/fastsam_data'/f'{robot}.json'
        raw=read(path);source_hashes[str(path.relative_to(ROOT))]=digest(path)
        for key,side in histories.items():
            if side['track_id'][0]!=robot:continue
            values=[]
            for frame_index,detection_index,time in side['source_chain']:
                frame=raw[frame_index]
                assert abs(frame['time']-time)<1e-9
                values.append([frame['widths'][detection_index],frame['heights'][detection_index]])
            values=np.asarray(values,dtype=float)
            assert np.all(np.isfinite(values)) and np.all(values>=0)
            np.testing.assert_allclose(values[0],[side['first_observation']['observed_width_m'],side['first_observation']['observed_height_m']],rtol=0,atol=1e-12)
            np.testing.assert_allclose(values[-1],[side['last_observation']['observed_width_m'],side['last_observation']['observed_height_m']],rtol=0,atol=1e-12)
            percentiles=np.percentile(values,[10,50,90],axis=0)
            dummy=lambda v:[0.,0.,0.,float(v[0]),float(v[1]),1.]
            first_last=gate_details(dummy(values[0]),dummy(values[-1]))
            robust=gate_details(dummy(percentiles[0]),dummy(percentiles[2]))
            stats[key]=dict(track_id=side['track_id'],observations=len(values),
                            duration_s=side['source_chain'][-1][2]-side['source_chain'][0][2],
                            first_last_rejected=not first_last['admitted'],first_last_failed_checks=first_last['failed_checks'],
                            robust_p10_p90_rejected=not robust['admitted'],robust_failed_checks=robust['failed_checks'],
                            percentiles_p10_p50_p90_m=percentiles.tolist(),
                            absolute_robust_spread_m=(percentiles[2]-percentiles[0]).tolist(),
                            distinct_source_frames=len({ref[0] for ref in side['source_chain']}),
                            source_frames_with_multiple_detections=len(values)-len({ref[0] for ref in side['source_chain']}))
        del raw
    eligible=[s for s in stats.values() if s['observations']>=5]
    paired=[]
    for task in manifest['tasks']:
        answer=labels[task['task_id']]
        if answer['label']!='same_object' or answer['confidence']!='high':continue
        sides=[stats[history_key(task[name])] for name in ['left','right']]
        available=[s for s in sides if s['observations']>=5]
        paired.append(dict(task_id=task['task_id'],B0_rejected=not task['gate']['admitted'],
                           sides_with_at_least_5_observations=len(available),
                           either_side_robust_unstable=any(s['robust_p10_p90_rejected'] for s in available)))
    pair_groups={}
    for rejected in [False,True]:
        group=[p for p in paired if p['B0_rejected']==rejected]
        pair_groups['B0_rejected' if rejected else 'B0_admitted']=dict(pairs=len(group),
            both_sides_observed_at_least_5=sum(p['sides_with_at_least_5_observations']==2 for p in group),
            either_side_robust_unstable=sum(p['either_side_robust_unstable'] for p in group))
    totals=dict(unique_track_history_snapshots=len(stats),unique_track_ids=len({tuple(s['track_id']) for s in stats.values()}),
                histories_at_least_5_observations=len(eligible),first_last_rejected=sum(s['first_last_rejected'] for s in eligible),
                robust_p10_p90_rejected=sum(s['robust_p10_p90_rejected'] for s in eligible),
                histories_with_multiple_detections_in_source_frame=sum(s['observations']!=s['distinct_source_frames'] for s in stats.values()))
    check_pins(manifest['protocol']);assert digest(label_path)==primary['labels_sha256']
    out.mkdir();(out/'analysis_private').mkdir()
    summary=dict(totals=totals,high_confidence_same_object_pair_groups=pair_groups,source_hashes=source_hashes,
                 labels_sha256=primary['labels_sha256'],script_sha256=digest(Path(__file__)),command=sys.argv,
                 scope='Source descriptor histories through original anchors; no physical identity or causal attribution',
                 robust_criterion='Marginal p10-p90 widths/heights cross original ratio 1.35 or absolute difference 0.1m; percentiles need not co-occur')
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (out/'analysis_private/histories.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n')
    (out/'analysis_private/high_confidence_pairs.json').write_text(json.dumps(paired,indent=2)+'\n')
    n=len(eligible)
    lines=['# Стабильность исходных размерных наблюдений внутри треков','',
           f"Изучены {len(stats)} различных снимков истории и {totals['unique_track_ids']} ID треков, представленных в ручной выборке. "
           f'У {n} историй есть минимум пять исходных наблюдений. Каждый индекс и размер первого/последнего наблюдения сверены с оригинальным FastSAM JSON.','',
           f"При диагностическом сравнении размеров начала и конца {totals['first_last_rejected']}/{n} историй нарушают исходный размерный допуск. "
           f"Размах p10–p90 по ширине или высоте нарушает его в {totals['robust_p10_p90_rejected']}/{n} историях.",'',
           '| Высокая уверенность, «один объект» | Пар | С обеими историями ≥5 наблюдений | Вариативен хотя бы один трек |','|---|---:|---:|---:|']
    for name,g in pair_groups.items():lines.append(f"| {name} | {g['pairs']} | {g['both_sides_observed_at_least_5']} | {g['either_side_robust_unstable']} |")
    lines+=['','Ширина/высота здесь — исходные мгновенные измерения, не сглаженные размеры карты. '
            'Сравнение с самим треком через время диагностическое; межроботный алгоритм не заменялся таким сравнением. '
            'Процентили считаются отдельно по измерениям ширины и высоты и могут относиться к разным кадрам. '
            'Истории и повторные снимки зависимы, ID трека не доказывает постоянство физического предмета. '
            'Вариативность может отражать ракурс, сегментацию, динамику или ошибки связи; эти причины здесь не разделены. '
            'Ни метки, ни карты, ни функции стоимости не менялись.','',
            '```bash','OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/tcaff-mpl \\',
            '  .venv/bin/python scripts/size_gate_source_size_stability.py','```','']
    (out/'report.md').write_text('\n'.join(lines))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
