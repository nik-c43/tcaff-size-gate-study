"""Audit own labels and reaggregate saved error traces; standard library only.

This does not rerun CLIPPER, the temporal filter or ground-truth composition.
"""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]

def read(path):
    path = Path(path)
    if path.suffix == '.gz':
        with gzip.open(path, 'rt') as f: return json.load(f)
    return json.loads(path.read_text())

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def near(a, b):
    assert abs(a-b) < 1e-12, (a, b)

def frame_metrics(frames):
    # Compact fields: step, time, accepted, [translation m, yaw deg], correct,
    # runtime seconds, GT available. Errors are saved, not recalculated poses.
    valid = [f for f in frames if f[3] is not None]
    first = next((f for f in frames if f[2]), None)
    correct = next((f for f in frames if f[4]), None)
    return dict(availability=sum(f[2] for f in frames)/len(frames),
                correct_update_availability=sum(f[4] for f in frames)/len(frames),
                mean_translation_m=mean(f[3][0] for f in valid) if valid else None,
                mean_yaw_deg=mean(f[3][1] for f in valid) if valid else None,
                runtime_s=sum(f[5] for f in frames),
                wrong_first_rate=float(first is not None and not first[4]),
                no_first_rate=float(first is None),
                correct_by_horizon_rate=float(correct is not None),
                capped_time_to_correct_s=(correct[1] if correct else frames[-1][1])-frames[0][1])

def aggregate(records):
    episodes = defaultdict(list)
    for eid, frames in records: episodes[eid].append(frame_metrics(frames))
    per_episode = {}
    for eid, rows in episodes.items():
        per_episode[eid] = {k:mean(r[k] for r in rows if r[k] is not None)
                           if any(r[k] is not None for r in rows) else None for k in rows[0]}
    return {k:mean(r[k] for r in per_episode.values() if r[k] is not None)
            if any(r[k] is not None for r in per_episode.values()) else None
            for k in next(iter(per_episode.values()))}

def check_annotations():
    manifest = read(ROOT/'annotations/manifest.json')
    for record in manifest['files']:
        p = ROOT/record['path'];assert digest(p) == record['source_sha256']
        assert p.stat().st_size == record['source_bytes']
        x = read(p);assert len(x['labels']) == record['records']
        assert sum(bool(r['label']) for r in x['labels']) == record['assigned']
        assert x['fingerprint'] == manifest['fingerprint']
    for key in ['export', 'metadata', 'observation_chains']:
        record=manifest[key];assert digest(ROOT/record['path']) == record['sha256']
    for record in manifest['derived_files']:
        assert digest(ROOT/record['path']) == record['sha256']
    for line in (ROOT/'annotations/checksums.sha256').read_text().splitlines():
        expected, filename=line.split('  ',1)
        assert digest(ROOT/filename)==expected,filename
    final = read(ROOT/manifest['canonical'])
    assert final['annotator'].strip()
    rows=final['labels'];assert len(rows)==798 and len({r['task_id'] for r in rows})==798
    with (ROOT/manifest['export']['path']).open(newline='') as f:
        assert list(csv.DictReader(f)) == rows
    metadata = read(ROOT/manifest['metadata']['path'])
    assert metadata['fingerprint']==manifest['fingerprint']
    tasks = {r['task_id']:r for r in metadata['tasks']}
    chains=read(ROOT/manifest['observation_chains']['path'])
    assert tasks.keys() == {r['task_id'] for r in rows}
    counts=Counter();high=Counter();relaxed=Counter()
    for row in rows:
        assert row['label'] in ['same_object','different_objects','part_whole_or_ambiguous','cannot_tell']
        assert row['confidence'] in ['low','medium','high']
        task=tasks[row['task_id']]
        a,b=task['sides']
        for side in [a,b]:
            assert side['source_chain_id'] in chains
            assert side['track_id'][0]==side['robot']
            observations=chains[side['source_chain_id']]
            assert observations and all(c[0]>=0 and c[1]>=0 for c in observations)
            for view in side['views']:
                assert [view['frame_index'],view['detection_index'],view['time']] in observations
        def admitted(ratio, absolute):
            return all(max(a[k],b[k]) <= min(a[k],b[k])*ratio and abs(a[k]-b[k]) <= absolute
                       for k in ['map_width_m','map_height_m'])
        assert admitted(1.35,.1)==task['gate']['admitted']
        status='admitted' if task['gate']['admitted'] else 'rejected'
        counts[row['label']+'_'+status]+=1
        if row['confidence']=='high':high[row['label']+'_'+status]+=1
        relaxed[row['label']+('_admitted' if admitted(1.7,.2) else '_rejected')]+=1
    assert counts['same_object_rejected']==260 and counts['same_object_admitted']==54
    assert counts['different_objects_rejected']==304 and counts['different_objects_admitted']==37
    assert high['same_object_rejected']==178 and high['same_object_admitted']==42
    assert relaxed['same_object_admitted']==124 and relaxed['different_objects_admitted']==101
    return dict(records=len(rows),labels=dict(Counter(r['label'] for r in rows)),B0=dict(counts),high_confidence_B0=dict(high),alpha2=dict(relaxed),original_files=len(manifest['files']))

def check_results():
    traces=read(ROOT/'results/baseline/error_traces.json.gz')
    audit=read(ROOT/'results/baseline/audit.json')
    baseline={}
    for variant in ['release_tau9','paper_tau8_control']:
        pair_mean=[];accepted=0;updates=0
        for stream in traces:
            frames=stream['frames'];valid=[f[variant] for f in frames if f[variant]['translation_m'] is not None]
            assert len(frames)==337
            pair_mean.append([mean(r['translation_m'] for r in valid),mean(r['yaw_deg'] for r in valid)])
            accepted+=sum(f[variant]['accepted'] for f in frames);updates+=len(frames)
        value=dict(translation_m=mean(r[0] for r in pair_mean),yaw_deg=mean(r[1] for r in pair_mean),accepted=accepted,updates=updates,availability=accepted/updates)
        original=next(s for s in audit['summaries'] if s['variant']==variant)
        near(value['translation_m'],original['author_mean_of_pair_means_translation_m'])
        near(value['yaw_deg'],original['author_mean_of_pair_means_yaw_deg'])
        assert accepted==3598 and updates==4044
        baseline[variant]=value
    runs=read(ROOT/'results/size_gate/run_metrics.json.gz')['runs'];assert len(runs)==420
    episodes=read(ROOT/'annotations/episodes.json')['episodes']
    hold=[e for e in episodes if e['split']=='holdout'];assert len(hold)==12
    summary=read(ROOT/'results/size_gate/summary.json')
    result={}
    for alpha, key in [(1.,'rerun_B0'),(2.,'selected_variant')]:
        full=[r for r in runs if r['alpha']==alpha and '/full_' in r['source_path']]
        assert len(full)==30
        records=[]
        for e in hold:
            for r in full:
                if r['pair']==e['pair']:
                    frames=[f for f in r['frames'] if e['start']<=f[0]<=e['end']]
                    assert len(frames)==29
                    records.append((e['episode_id'],frames))
        aggregate_full=aggregate(records)
        for metric in ['availability','correct_update_availability','mean_translation_m','mean_yaw_deg','runtime_s']:
            near(aggregate_full[metric],summary[key][metric])
        correct=sum(f[4] for _,frames in records for f in frames)
        cold=[r for r in runs if r['stage']=='holdout' and '/full_' not in r['source_path'] and r['alpha']==alpha]
        assert len(cold)==60
        aggregate_cold=aggregate([(r['episode_id'],r['frames']) for r in cold])
        for metric in aggregate_cold:near(aggregate_cold[metric],summary['cold_start'][str(alpha)][metric])
        first=Counter(frame_metrics(r['frames'])['wrong_first_rate'] for r in cold)
        result[str(alpha)]=dict(full_history=aggregate_full,correct_updates=correct,total_updates=1740,cold_start=aggregate_cold,cold_wrong_first=int(first[1.]),cold_runs=60)
    assert result['1.0']['cold_wrong_first']==4 and result['2.0']['cold_wrong_first']==10
    cases=read(ROOT/'results/occlusion/cases.json');assert len(cases)==80
    assert sum(r['requested_bbox_occlusion_fraction']==0 for r in cases)==8
    return dict(baseline=baseline,size_gate=result,runs=len(runs),occlusion_cases=len(cases),runtime_ratio=result['2.0']['full_history']['runtime_s']/result['1.0']['full_history']['runtime_s'])

def verify_transfer():
    files=read(ROOT/'docs/source_inventory.json')
    for r in files:assert digest(ROOT/r['path'])==r['sha256'],r['path']
    return len(files)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=dict(annotations=check_annotations(),results=check_results(),transferred_files_checked=verify_transfer(),scope='Saved own labels/metrics integrity and reaggregation; no new experiment; saved replay/GT checks are not rerun here',all_checks_passed=True)
    if args.output:
        if args.output.exists():raise ValueError('Choose a new output file')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(all_checks_passed=True,annotations=result['annotations']['records'],runs=result['results']['runs'],baseline=result['results']['baseline']['release_tau9'],late_correct_updates={a:r['correct_updates'] for a,r in result['results']['size_gate'].items()},runtime_ratio=result['results']['runtime_ratio']),ensure_ascii=False))

if __name__=='__main__':main()
