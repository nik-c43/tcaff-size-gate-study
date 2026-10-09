"""Freeze final protocol, code hashes, blind package and RGB synchronization.

Only before delivery/labels; refuses an already frozen or annotated bundle.
Normally called by size_gate_prepare.py. Also finalizes a newly prepared bundle.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader

from size_gate_images import synchronized_anchor_indices, opaque_key, build_private_table
from size_gate_prepare import ROOT, dump, check_pins
from tcaff_baseline_audit import digest


def finalize(out):
    if (out/'bundle_frozen.json').exists() or (out/'labels.json').exists():
        raise ValueError('Refuse to modify an already frozen or annotated bundle')
    private=json.loads((out/'analysis_private/manifest.json').read_text())
    config_path=ROOT/private['provenance'].get('configuration_path','configs/size_gate_study.json')
    private['provenance']['configuration_path']=str(config_path.relative_to(ROOT))
    private['protocol']=json.loads(config_path.read_text())
    check_pins(private['protocol'])
    raw={r:json.loads((ROOT/f'data/tcaff_mot_data/fastsam_data/{r}.json').read_text()) for r in ['RR01','RR04','RR06','RR08']}
    times={r:np.asarray([f['time'] for f in frames]) for r,frames in raw.items()}
    provenance=json.loads((out/'analysis_private/rgb_provenance.json').read_text())
    images={(i['robot'],i['frame_index']):i for i in provenance['images']}
    requested={}
    for e in private['episodes']:
        e['anchor_rgb']={}
        for r,i in synchronized_anchor_indices(times,e['pair'],e['time']).items():
            e['anchor_rgb'][r]=dict(frame_index=i,time=float(times[r][i]))
            if (r,i) not in images:requested[r,i]=raw[r][i]
    for r in sorted({r for r,i in requested}):
        bags=sorted((ROOT/'data/tcaff_mot_data/data').glob(f'{r}_compressed*.bag'))
        with AnyReader(bags) as reader:
            conn=next(c for c in reader.connections if c.topic==f'/{r}/l515/color/image_raw/compressed')
            info=next(c for c in reader.connections if c.topic==f'/{r}/l515/color/camera_info')
            _,_,message=next(reader.messages(connections=[info]))
            K=np.asarray(reader.deserialize(message,info.msgtype).K).reshape(3,3)
            for (rr,i),target in sorted(requested.items()):
                if rr!=r:continue
                t=target['time'];candidates=[]
                for _,_,message in reader.messages(connections=[conn],start=int((t-.2)*1e9),stop=int((t+.2)*1e9)):
                    msg=reader.deserialize(message,conn.msgtype)
                    stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
                    candidates.append((abs(stamp-t),stamp,msg))
                dt,stamp,msg=min(candidates,key=lambda x:x[0]);assert dt<=.05
                jpeg=bytes(msg.data);name=f'{opaque_key(r,i)}_rgb.jpg'
                (out/'annotation_blind/images'/name).write_bytes(jpeg)
                xyz=np.asarray(target['detections']).reshape(-1,3);uv=xyz@K.T;uv=uv[:,:2]/uv[:,2:]
                err=float(np.max(np.linalg.norm(uv-np.asarray(target['pts_img']).reshape(-1,2),axis=1),initial=0.))
                assert err<1e-6
                images[r,i]=dict(robot=r,frame_index=i,observation_time=t,rgb_header_time=stamp,sync_error_s=dt,
                                 projection_error_px=err,raw_image=f'images/{name}',original_jpeg_sha256=hashlib.sha256(jpeg).hexdigest())
    private['provenance']['protocol_sha256']=digest(config_path)
    private['provenance']['code_hashes']={f:digest(ROOT/f) for f in ['scripts/size_gate_prepare.py','scripts/size_gate_images.py',
                                                                 'scripts/size_gate_finalize.py','scripts/size_gate_evaluate.py','scripts/size_gate_review.html']}
    original=json.loads((ROOT/'results/tcaff_baseline_release/audit.json').read_text())
    private['provenance']['frozen_B0_inputs']=original['input_hashes']
    private['provenance']['clipper_binary']=original['environment']['clipperpy_files']
    private['provenance']['clipper_untracked_build_artifacts']='test/CMakeCache.txt and test/CMakeFiles/; no tracked source differences'
    private.pop('fingerprint',None)
    fingerprint=hashlib.sha256(json.dumps(private,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    private['fingerprint']=fingerprint
    blind=out/'annotation_blind'
    payload=json.loads((blind/'tasks.json').read_text());payload['fingerprint']=fingerprint
    episodes={e['episode_id']:e for e in private['episodes']}
    for task in payload['tasks']:
        e=episodes[task['episode_id']]
        for side in task['sides']:
            r=side['robot'];image=images[r,e['anchor_rgb'][r]['frame_index']]
            side['anchor_image']=image['raw_image'];side['anchor_image_time']=image['rgb_header_time']
            side['anchor_offset_from_map_s']=image['rgb_header_time']-e['time']
            side['source_json']=f'data/tcaff_mot_data/fastsam_data/{r}.json'
        task['anchor_pair_skew_s']=abs(task['sides'][0]['anchor_image_time']-task['sides'][1]['anchor_image_time'])
    dump(blind/'tasks.json',payload)
    template=json.loads((blind/'labels_template.json').read_text());template['fingerprint']=fingerprint
    dump(blind/'labels_template.json',template)
    encoded=json.dumps(payload,ensure_ascii=False).replace('</','<\\/')
    (blind/'index.html').write_text((ROOT/'scripts/size_gate_review.html').read_text().replace('__DATA__',encoded))
    dump(out/'protocol.json',private['protocol'])
    dump(out/'analysis_private/manifest.json',private)
    provenance['images']=list(images.values());dump(out/'analysis_private/rgb_provenance.json',provenance)
    build_private_table(private,out)
    summary=json.loads((out/'preparation_summary.json').read_text())
    summary.update(fingerprint=fingerprint,unique_rgb_frames=len(images),
                   max_anchor_pair_skew_s=max(t['anchor_pair_skew_s'] for t in payload['tasks']),
                   max_anchor_offset_from_map_s=max(abs(s['anchor_offset_from_map_s']) for t in payload['tasks'] for s in t['sides']))
    dump(out/'preparation_summary.json',summary)
    dump(out/'bundle_frozen.json',dict(fingerprint=fingerprint,manual_labels_received=False,relaxation_run=False,
                                      manifest_sha256=digest(out/'analysis_private/manifest.json'),
                                      blind_tasks_sha256=digest(blind/'tasks.json'),protocol_sha256=digest(out/'protocol.json')))
    print('Bundle frozen before annotation; fingerprint '+fingerprint,flush=True)


if __name__=='__main__':finalize(ROOT/'data/size_gate_study')
