"""Post-annotation counts and development/holdout comparison of one global gate.

Never runs an experiment without complete manual annotations. B0 data and upstream
remain untouched. Only the two size-gate parameters change in the relaxed factory.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'),str(ROOT/'vendor/tcaff')]
from size_gate_prepare import dump, check_pins
from tcaff_baseline_audit import digest, pose_error
from h1_map_audit import object_map
from tcaff.tcaff.tcaff_manager import TCAFFManager

LABELS = {'same_object','different_objects','part_whole_or_ambiguous','cannot_tell'}


def validate_labels(manifest, annotation):
    if annotation.get('fingerprint') != manifest['fingerprint']:
        raise ValueError('Annotation fingerprint does not match the frozen study')
    if not str(annotation.get('annotator','')).strip():
        raise ValueError('A nonempty human annotator name is required')
    expected = {t['task_id'] for t in manifest['tasks']}
    entries = annotation.get('labels',[])
    ids = [x.get('task_id') for x in entries]
    if len(ids)!=len(set(ids)) or set(ids)!=expected:
        raise ValueError('Every task must appear exactly once; missing/extra/duplicate task IDs')
    for x in entries:
        if x.get('label') not in LABELS or x.get('confidence') not in {'high','medium','low'}:
            raise ValueError(f'Incomplete/invalid annotation: {x.get("task_id")}')
        if not isinstance(x.get('notes',''),str):
            raise ValueError('Notes must be text')
    return {x['task_id']:x for x in entries}


def annotation_counts(manifest, labels):
    result=Counter(); by_episode=defaultdict(Counter); by_confidence=defaultdict(Counter); unique=defaultdict(set)
    for t in manifest['tasks']:
        label=labels[t['task_id']]['label']; decision='admitted' if t['gate']['admitted'] else 'rejected'
        key=label+'_'+decision
        result[key]+=1; by_episode[t['episode_id']][key]+=1
        by_confidence[labels[t['task_id']]['confidence']][key]+=1
        track_pair=(tuple(t['left']['track_id']),tuple(t['right']['track_id']))
        unique[track_pair].add(key)
    dedup=Counter(); varying=[]
    for pair,keys in unique.items():
        if len(keys)==1:dedup[next(iter(keys))]+=1
        else:varying.append(dict(track_pair=pair,observed_label_gate_combinations=sorted(keys)))
    rejected=result['same_object_rejected']+result['different_objects_rejected']
    correct=result['same_object_rejected']+result['same_object_admitted']
    incorrect=result['different_objects_rejected']+result['different_objects_admitted']
    return dict(scope='All near edges of the 24 selected anchors; no general-dataset or physical-object-level prevalence claim',
                pair_observation_counts=dict(result), confirmed_correct_excluded=result['same_object_rejected'],
                confirmed_incorrect_blocked=result['different_objects_rejected'],
                excluded_fraction_of_confirmed_correct=None if not correct else result['same_object_rejected']/correct,
                blocked_fraction_of_confirmed_incorrect=None if not incorrect else result['different_objects_rejected']/incorrect,
                correct_fraction_among_resolved_rejected=None if not rejected else result['same_object_rejected']/rejected,
                by_episode={k:dict(v) for k,v in by_episode.items()},
                by_confidence={k:dict(v) for k,v in by_confidence.items()},
                unique_map_track_pairs=len(unique), unchanging_unique_track_pair_counts=dict(dedup),
                varying_or_disagreeing_track_pairs=varying,
                caveat='Track-pair deduplication is not physical-identity deduplication. Multiple track IDs can represent one object; ambiguous/unresolved labels never counted as confirmed true or false.')


def metrics(frames):
    valid=[f for f in frames if f['error'] is not None]
    accepted=[f for f in frames if f['estimate'] is not None]
    first=next(iter(accepted),None)
    first_correct=next((f for f in valid if f['correct']),None)
    horizon=frames[-1]['time']-frames[0]['time']
    return dict(updates=len(frames),gt_valid_updates=sum(f['truth'] is not None for f in frames),
                accepted_updates=len(accepted),availability=len(accepted)/len(frames),
                correct_updates=sum(f['correct'] for f in frames),correct_update_availability=sum(f['correct'] for f in frames)/len(frames),
                mean_translation_m=None if not valid else float(np.mean([f['error'][0] for f in valid])),
                median_translation_m=None if not valid else float(np.median([f['error'][0] for f in valid])),
                mean_yaw_deg=None if not valid else float(np.mean([f['error'][1] for f in valid])),
                median_yaw_deg=None if not valid else float(np.median([f['error'][1] for f in valid])),
                first_solution='none' if first is None else ('correct' if first['correct'] else 'wrong'),
                first_solution_delay_s=None if first is None else first['time']-frames[0]['time'],
                correctly_initialized_by_horizon=first_correct is not None,
                time_to_correct_s=None if first_correct is None else first_correct['time']-frames[0]['time'],
                right_censored=first_correct is None,censor_horizon_s=horizon,
                capped_time_to_correct_s=horizon if first_correct is None else first_correct['time']-frames[0]['time'],
                runtime_s=sum(f['runtime_s'] for f in frames))


def run_episode(data, episode, alpha, repetition):
    kwargs={k:v for k,v in data['params'].items() if k!='ts'}
    kwargs['wh_scale_diff']=1+alpha*(kwargs['wh_scale_diff']-1)
    kwargs['h_diff']*=alpha
    manager=TCAFFManager(**kwargs)
    verifier=TCAFFManager(**kwargs).faf
    frames=[]
    for source in data['frames'][episode['start']:episode['end']+1]:
        ma,mb=object_map(source['map_a']),object_map(source['map_b'])
        count=len(manager.get_putative_assoc(ma,mb))
        began=time.perf_counter();manager.update(ma,mb);elapsed=time.perf_counter()-began
        estimate=None if manager.faf.main_tree is None else manager.faf.main_tree.optimal.xhat.ravel().tolist()
        zs=[z.copy() for z in manager.latest_zs]
        verifier.update(zs,[manager.R.copy() for _ in zs])
        replay=None if verifier.main_tree is None else verifier.main_tree.optimal.xhat.ravel()
        assert (replay is None)==(estimate is None)
        if replay is not None:np.testing.assert_allclose(replay,estimate,rtol=0,atol=1e-10)
        error=None if estimate is None or source['truth'] is None else pose_error(estimate,source['truth'])
        limits=(1.,5.)
        frames.append(dict(step=source['step'],time=source['time'],truth=source['truth'],estimate=estimate,error=error,
                           correct=error is not None and error[0]<=limits[0] and error[1]<=limits[1],
                           putative_size_admitted_pairs=count,zs=[z.ravel().tolist() for z in zs],R=manager.R.tolist(),
                           runtime_s=elapsed))
    return dict(episode_id=episode['episode_id'],pair=episode['pair'],start=episode['start'],end=episode['end'],
                temporal_block=episode['temporal_block'],alpha=alpha,repetition=repetition,params=kwargs,
                state=episode.get('state','Original factory cold start on unchanged saved full-history maps'),
                temporal_replay_verified=True,frames=frames,metrics=metrics(frames))


def aggregate(runs):
    # Average repetitions inside each episode before averaging episodes.
    grouped=defaultdict(list)
    for r in runs:grouped[r['episode_id']].append(r['metrics'])
    episode=[]
    for eid,rows in grouped.items():
        item=dict(episode_id=eid,repetitions=len(rows))
        for key in ['availability','correct_update_availability','mean_translation_m','median_translation_m',
                    'mean_yaw_deg','median_yaw_deg','capped_time_to_correct_s','runtime_s']:
            values=[r[key] for r in rows if r[key] is not None]
            item[key]=None if not values else float(np.mean(values))
            item[key+'_repeat_range']=None if not values else [min(values),max(values)]
        item['wrong_first_rate']=sum(r['first_solution']=='wrong' for r in rows)/len(rows)
        item['no_first_rate']=sum(r['first_solution']=='none' for r in rows)/len(rows)
        item['correct_by_horizon_rate']=sum(r['correctly_initialized_by_horizon'] for r in rows)/len(rows)
        episode.append(item)
    summary=dict(episodes=len(episode),solver_runs=len(runs),per_episode=episode)
    for key in ['availability','correct_update_availability','mean_translation_m','mean_yaw_deg',
                'capped_time_to_correct_s','runtime_s','wrong_first_rate','no_first_rate','correct_by_horizon_rate']:
        values=[e[key] for e in episode if e[key] is not None]
        summary[key]=None if not values else float(np.mean(values))
    summary['mean_wrong_first_count']=sum(e['wrong_first_rate'] for e in episode)
    summary['mean_correct_by_horizon_count']=sum(e['correct_by_horizon_rate'] for e in episode)
    return summary


def aggregate_context(runs, recorded_runtime=True):
    result=aggregate(runs)
    # A holdout window entered by an already running filter has no new first start.
    removed={'wrong_first_rate','no_first_rate','correct_by_horizon_rate','mean_wrong_first_count',
             'mean_correct_by_horizon_count','capped_time_to_correct_s','capped_time_to_correct_s_repeat_range'}
    if not recorded_runtime:removed.update({'runtime_s','runtime_s_repeat_range'})
    for row in [result,*result['per_episode']]:
        for key in removed:row.pop(key,None)
    return result


def choose_alpha(summaries):
    baseline=summaries[1.]
    feasible=[a for a,s in summaries.items() if a!=1. and s['mean_wrong_first_count']<=baseline['mean_wrong_first_count']]
    if not feasible:return min(a for a in summaries if a!=1.),False
    best=min(feasible,key=lambda a:(-summaries[a]['mean_correct_by_horizon_count'],
                                   summaries[a]['mean_wrong_first_count'],
                                   -summaries[a]['correct_update_availability'],a))
    return best,True


def evaluate(manifest, labels_path, stage, out, selection_path):
    config=manifest['protocol']; check_pins(config)
    config_path=ROOT/manifest['provenance']['configuration_path']
    if out.exists():raise ValueError('Refuse to overwrite experiment directory')
    if stage=='holdout':
        if not selection_path:raise ValueError('--selection from completed development is required')
        chosen=json.loads(selection_path.read_text())
        assert chosen['fingerprint']==manifest['fingerprint']
        assert chosen['labels_sha256']==digest(labels_path), 'Manual labels changed after selection'
        assert chosen['protocol_sha256']==digest(config_path)
        assert chosen['evaluator_sha256']==digest(Path(__file__))
        assert chosen['development_summary_sha256']==digest(selection_path.parent/'summary.json')
        alphas=[1.,chosen['alpha']]
    else:alphas=[1.,*config['relaxation']['alpha_grid_development_only']]
    episodes=[e for e in manifest['episodes'] if e['split']==stage]
    assert len(episodes)==12
    out.mkdir(parents=True)
    source={}
    for pair in {e['pair'] for e in episodes}:
        p=ROOT/config['baseline']/f'{pair}.json'
        assert digest(p)==manifest['provenance']['source_hashes'][pair]['sha256']
        source[pair]=json.loads(p.read_text())
    # No use of holdout maps, outcomes or labels in the development ranking.
    runs=defaultdict(list)
    for e in sorted(episodes,key=lambda x:(x['start'],x['pair'])):
        for repeat in range(config['relaxation']['repetitions']):
            # Rotate execution order to reduce systematic time/order effects.
            order=alphas[repeat%len(alphas):]+alphas[:repeat%len(alphas)]
            for alpha in order:
                print(f'{stage} {e["episode_id"]} alpha={alpha} repeat={repeat+1}',flush=True)
                run=run_episode(source[e['pair']],e,alpha,repeat+1)
                dump(out/f'{e["episode_id"]}_a{alpha}_r{repeat+1}.json',run)
                runs[alpha].append(run)
    summaries={a:aggregate(rows) for a,rows in runs.items()}
    summary=dict(fingerprint=manifest['fingerprint'],stage=stage,labels_sha256=digest(labels_path),
                 protocol_sha256=digest(config_path),evaluator_sha256=digest(Path(__file__)),
                 summaries=summaries,command=sys.argv,
                 units='One recording; episode means over stochastic solver repeats, correlated canonical pairs/windows',
                 original_frozen_B0_overwritten=False,scope='Size gate -> MNO-CLIPPER -> temporal TCAFF on fixed author maps, diagnostic cold starts',
                 caveat='Capped delay must be read with correct-by-horizon rate and individual censor indicators. Errors are conditional on acceptance.')
    dump(out/'summary.json',summary)
    if stage=='development':
        alpha,feasible=choose_alpha(summaries)
        dump(out/'selection.json',dict(fingerprint=manifest['fingerprint'],alpha=alpha,wrong_first_constraint_satisfied=feasible,
                                       labels_sha256=digest(labels_path),protocol_sha256=summary['protocol_sha256'],
                                       evaluator_sha256=summary['evaluator_sha256'],development_summary_sha256=digest(out/'summary.json'),
                                       selection_rule=config['relaxation']['selection'],holdout_evaluated=False))
    else:
        # Primary contextual check: preserve temporal history from the original recording
        # start. Window cold starts above are the supplemental initialization diagnosis.
        full_runs=defaultdict(list); held_context=defaultdict(list); frozen_context=[]
        for pair,data in sorted(source.items()):
            related=[e for e in episodes if e['pair']==pair]
            for e in related:
                frozen=[]
                for f in data['frames'][e['start']:e['end']+1]:
                    error=None if f['baseline_estimate'] is None or f['truth'] is None else pose_error(f['baseline_estimate'],f['truth'])
                    frozen.append(dict(time=f['time'],estimate=f['baseline_estimate'],truth=f['truth'],error=error,
                                       correct=error is not None and error[0]<=1. and error[1]<=5.,runtime_s=0.))
                frozen_context.append(dict(episode_id=e['episode_id'],metrics=metrics(frozen)))
            for repeat in range(config['relaxation']['repetitions']):
                for alpha in alphas:
                    print(f'full-history {pair} alpha={alpha} repeat={repeat+1}',flush=True)
                    descriptor=dict(episode_id=pair,pair=pair,start=0,end=len(data['frames'])-1,temporal_block='whole_record',
                                    state='Original factory from recording start; no temporal resets')
                    run=run_episode(data,descriptor,alpha,repeat+1)
                    dump(out/f'full_{pair}_a{alpha}_r{repeat+1}.json',run)
                    full_runs[alpha].append(run)
                    for e in related:
                        held_context[alpha].append(dict(episode_id=e['episode_id'],metrics=metrics(run['frames'][e['start']:e['end']+1])))
        dump(out/'full_history_summary.json',dict(fingerprint=manifest['fingerprint'],alpha=chosen['alpha'],
                                                  frozen_B0_holdout_context=aggregate_context(frozen_context,recorded_runtime=False),
                                                  rerun_holdout_context={a:aggregate_context(rows) for a,rows in held_context.items()},
                                                  whole_record_descriptive={a:aggregate(rows) for a,rows in full_runs.items()},
                                                  interpretation='Primary pose/availability comparison preserves full temporal history. Global first initialization is descriptive and occurs in the development prefix; cold-start holdout initialization metrics are separately in summary.json. A first accepted update of a full-history holdout window is NOT a new initialization.',
                                                  units='Single recording, six canonical streams; directions and windows not independent'))
    check_pins(config)
    for pair,record in manifest['provenance']['source_hashes'].items():assert digest(ROOT/record['path'])==record['sha256']


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['count','development','holdout'])
    parser.add_argument('--study',default='data/size_gate_study')
    parser.add_argument('--labels',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--selection')
    args=parser.parse_args()
    study=ROOT/args.study
    frozen=json.loads((study/'bundle_frozen.json').read_text())
    assert digest(study/'analysis_private/manifest.json')==frozen['manifest_sha256'], 'Study manifest changed after freeze'
    assert digest(study/'annotation_blind/tasks.json')==frozen['blind_tasks_sha256'], 'Blind task list changed after freeze'
    manifest=json.loads((study/'analysis_private/manifest.json').read_text())
    assert manifest['fingerprint']==frozen['fingerprint']
    assert digest(ROOT/manifest['provenance']['configuration_path'])==manifest['provenance']['protocol_sha256'], 'Protocol changed after freeze'
    for f,expected in manifest['provenance']['code_hashes'].items():
        assert digest(ROOT/f)==expected, f'Study code changed after freeze: {f}'
    labels_path=ROOT/args.labels
    labels=validate_labels(manifest,json.loads(labels_path.read_text()))
    out=ROOT/args.output
    if args.stage=='count':
        if out.exists():raise ValueError('Refuse to overwrite counts')
        dump(out,dict(fingerprint=manifest['fingerprint'],labels_sha256=digest(labels_path),
                      annotator=json.loads(labels_path.read_text())['annotator'],**annotation_counts(manifest,labels)))
    else:
        count_path=ROOT/args.study/'analysis_private/label_counts.json'
        if not count_path.exists() or json.loads(count_path.read_text()).get('labels_sha256')!=digest(labels_path):
            raise ValueError('Run count first with --output data/size_gate_study/analysis_private/label_counts.json')
        evaluate(manifest,labels_path,args.stage,out,None if args.selection is None else ROOT/args.selection)


if __name__=='__main__':main()
