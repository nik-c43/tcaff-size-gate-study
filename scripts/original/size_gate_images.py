"""Actual author RGB and explicitly non-mask covariance overlays for review."""
from collections import defaultdict
import hashlib
import html
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from rosbags.highlevel import AnyReader

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def opaque_key(robot, frame_index, detection_index=None):
    return hashlib.sha256(f'{robot}:{frame_index}:{detection_index}'.encode()).hexdigest()[:18]


def synchronized_anchor_indices(times, pair, target):
    a,b=pair.split('_')
    ia=np.flatnonzero(abs(times[a]-target)<=.1)
    ib=np.flatnonzero(abs(times[b]-target)<=.1)
    assert len(ia) and len(ib)
    # Joint choice avoids opposite rounding to two author frames almost 100ms apart.
    skew,offset,i,j=min((abs(times[a][i]-times[b][j]),max(abs(times[a][i]-target),abs(times[b][j]-target)),int(i),int(j))
                        for i in ia for j in ib)
    return {a:i,b:j}


def marked_image(rgb, point, cov, index):
    result = rgb.copy()
    center = tuple(np.rint(point).astype(int))
    eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(cov))
    if np.all(np.isfinite(eigenvalues)) and np.all(eigenvalues >= 0):
        axes = tuple(np.maximum(1, np.rint(2*np.sqrt(eigenvalues[::-1]))).astype(int))
        angle = float(np.degrees(np.arctan2(eigenvectors[1,-1], eigenvectors[0,-1])))
        cv2.ellipse(result, center, axes, angle, 0, 360, (255, 0, 255), 3)
    cv2.drawMarker(result, center, (255,0,255), cv2.MARKER_CROSS, 20, 2)
    cv2.putText(result, f'det {index} (ellipse != mask)', (max(0, center[0]-90), max(30,center[1]-20)),
                cv2.FONT_HERSHEY_SIMPLEX, .6, (255,0,255), 2, cv2.LINE_AA)
    return result


def context_crop(marked, point, covariance):
    # This region is only context; it is never presented as the detection boundary.
    extent = max(120., 3.*np.sqrt(max(np.linalg.eigvalsh(covariance).max(), 0.)))
    extent = min(extent, 450.)
    h,w = marked.shape[:2]; u,v=point
    x0,x1 = max(0,int(u-extent)), min(w,int(u+extent))
    y0,y1 = max(0,int(v-extent)), min(h,int(v+extent))
    assert x1>x0 and y1>y0, 'Detection center is outside RGB'
    return marked[y0:y1,x0:x1], [x0,y0,x1,y1]


def build_images_and_review(private, out):
    blind = out / 'annotation_blind'; image_dir = blind / 'images'
    image_dir.mkdir(parents=True)
    raw = {r:json.loads((ROOT / f'data/tcaff_mot_data/fastsam_data/{r}.json').read_text())
           for r in ['RR01','RR04','RR06','RR08']}
    times = {r:np.asarray([f['time'] for f in frames]) for r,frames in raw.items()}
    needed = defaultdict(set)
    for e in private['episodes']:
        e['anchor_rgb'] = {}
        for r,i in synchronized_anchor_indices(times,e['pair'],e['time']).items():
            needed[r].add(i)
            e['anchor_rgb'][r] = dict(frame_index=i, time=float(times[r][i]))
    observations = {}
    for task in private['tasks']:
        for side in ['left','right']:
            obj = task[side]
            for kind in ['last_observation','first_observation']:
                obs = obj[kind]; r=obs['robot']; i=obs['frame_index']; j=obs['detection_index']
                needed[r].add(i)
                observations[r,i,j] = obs
    images, provenance = {}, []
    for r in sorted(needed):
        bags = sorted((ROOT / 'data/tcaff_mot_data/data').glob(f'{r}_compressed*.bag'))
        assert len(bags)==1
        stat = bags[0].stat()
        provenance.append(dict(path=str(bags[0].relative_to(ROOT)), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns))
        with AnyReader(bags) as reader:
            rgb_conn = next(c for c in reader.connections if c.topic == f'/{r}/l515/color/image_raw/compressed')
            info_conn = next(c for c in reader.connections if c.topic == f'/{r}/l515/color/camera_info')
            _,_,msgraw=next(reader.messages(connections=[info_conn]))
            K = np.asarray(reader.deserialize(msgraw,info_conn.msgtype).K).reshape(3,3)
            for n,i in enumerate(sorted(needed[r])):
                target=raw[r][i]; t=target['time']; candidates=[]
                for _,_,msgraw in reader.messages(connections=[rgb_conn],start=int((t-.2)*1e9),stop=int((t+.2)*1e9)):
                    msg=reader.deserialize(msgraw,rgb_conn.msgtype)
                    stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
                    candidates.append((abs(stamp-t),stamp,msg))
                assert candidates, (r,i,t)
                dt,stamp,msg=min(candidates,key=lambda x:x[0])
                assert dt <= .05, (r,i,dt)
                jpeg=bytes(msg.data); rgb=cv2.imdecode(np.frombuffer(jpeg,dtype=np.uint8),cv2.IMREAD_COLOR)
                assert rgb is not None
                points=np.asarray(target['detections']).reshape(-1,3)
                projected=points @ K.T
                uv=projected[:,:2]/projected[:,2:]
                err=float(np.max(np.linalg.norm(uv-np.asarray(target['pts_img']).reshape(-1,2),axis=1),initial=0.))
                assert err < 1e-6
                key=opaque_key(r,i)
                path=image_dir/f'{key}_rgb.jpg';path.write_bytes(jpeg)
                record=dict(robot=r,frame_index=i,observation_time=t,rgb_header_time=stamp,sync_error_s=dt,
                            projection_error_px=err,raw_image=f'images/{path.name}',
                            original_jpeg_sha256=hashlib.sha256(jpeg).hexdigest())
                images[r,i]=record
                for (rr,ii,j),obs in observations.items():
                    if rr!=r or ii!=i:continue
                    key=opaque_key(r,i,j)
                    marked=marked_image(rgb,obs['uv'],obs['image_covariance'],j)
                    crop,bounds=context_crop(marked,obs['uv'],obs['image_covariance'])
                    assert cv2.imwrite(str(image_dir/f'{key}_marked.jpg'),marked,[cv2.IMWRITE_JPEG_QUALITY,90])
                    assert cv2.imwrite(str(image_dir/f'{key}_context.jpg'),crop,[cv2.IMWRITE_JPEG_QUALITY,90])
                    obs.update(marked_image=f'images/{key}_marked.jpg',context_crop=f'images/{key}_context.jpg',
                               crop_context_bounds_px=bounds,**record)
                if n%25==0:print(f'RGB export {r}: {n+1}/{len(needed[r])}',flush=True)
        print(f'RGB export {r}: done {len(needed[r])} frames',flush=True)
    blind_tasks=[]
    episode_lookup={e['episode_id']:e for e in private['episodes']}
    for task in private['tasks']:
        e=episode_lookup[task['episode_id']]
        sides=[]
        for side in ['left','right']:
            obj=task[side];views=[]
            seen=set()
            for kind,label in [('last_observation','Последнее наблюдение'),('first_observation','Первое наблюдение трека')]:
                source=obj[kind];r,i,j=source['robot'],source['frame_index'],source['detection_index']
                if (i,j) in seen:continue
                seen.add((i,j)); obs=observations[r,i,j]
                views.append(dict(label=label,frame_index=i,detection_index=j,time=obs['time'],
                                  age_at_anchor_s=task['time']-obs['time'],
                                  crop=obs['context_crop'],full=obs['marked_image'],raw=obs['raw_image'],
                                  uv=obs['uv'],sync_error_s=obs['sync_error_s']))
            r=obj['track_id'][0];i=e['anchor_rgb'][r]['frame_index']
            anchor=images[r,i]
            sides.append(dict(robot=r,track_id=obj['track_id'],map_index=obj['map_index'],views=views,
                              anchor_image=anchor['raw_image'],anchor_image_time=anchor['rgb_header_time']))
        blind_tasks.append(dict(task_id=task['task_id'],episode_id=task['episode_id'],sides=sides,
                               anchor_pair_skew_s=abs(sides[0]['anchor_image_time']-sides[1]['anchor_image_time'])))
    payload=dict(fingerprint=private['fingerprint'],tasks=blind_tasks)
    write_json(blind/'tasks.json',payload)
    # Export contains no private decisions, numerical dimensions or outcome metadata.
    template=dict(fingerprint=private['fingerprint'],annotator='',labels=[dict(task_id=t['task_id'],label=None,confidence=None,notes='') for t in blind_tasks])
    write_json(blind/'labels_template.json',template)
    script = (ROOT/'scripts/size_gate_review.html').read_text()
    encoded=json.dumps(payload,ensure_ascii=False).replace('</','<\\/')
    (blind/'index.html').write_text(script.replace('__DATA__',encoded))
    (blind/'README.md').write_text('''# Ручная разметка объектных соответствий

Откройте index.html в браузере. Укажите имя разметчика. Для каждой пары сравните исходные RGB двух роботов: фрагменты с выделенной детекцией, полные кадры и синхронизированные кадры эпизода. Можно открывать первое наблюдение трека для проверки устойчивости его идентичности.

Розовые центры и эллипсы — сохранённые координаты и 2σ ковариация, **не маски и не границы сегментации**. Маски и точные границы отсутствуют. Фрагмент изображения показывает контекст; его рамка не является рамкой детекции. ID трека — ID построителя карты, а не доказанная физическая идентичность. Первое и последнее наблюдения могут относиться к разным предметам при ошибке трекера: тогда отметьте неоднозначность и поясните её.

Разметьте «Тот же предмет» только при убедительной визуальной идентичности; «Разные предметы» при различимых физических предметах; «Часть/целое или неоднозначность» для несовпадающих сегментов/объединений или сомнительной связи трека; «Не определить» при недостаточных изображениях. Не выводите идентичность из положения центров. Укажите уверенность и комментарий. Не открывайте соседний analysis_private до экспорта первичной разметки.

Все близкие пары выбранных эпизодов включены без отбора по решению фильтра; есть контрольные допустимые пары и альтернативы для проверки ложных соответствий. Их роли, размеры, решения фильтра и результат совмещения скрыты. Сначала разметьте визуальную идентичность, затем экспортируйте labels.json. Автосохранение в localStorage удобно, но экспорт JSON обязателен; при смене браузера импортируйте сохранённый JSON.

798 пар — полное множество близких рёбер для 24 выбранных эпизодов. Неопределимые пары тоже требуют явной отметки. Ориентир времени: 4–8 часов, больше при проверке историй. Метки не заполнены автоматически.
''')
    write_json(out/'analysis_private/rgb_provenance.json',dict(bags=provenance,images=list(images.values())))
    write_json(out/'analysis_private/manifest.json',private)
    build_private_table(private,out)
    return dict(annotation_tasks=len(blind_tasks),unique_rgb_frames=len(images),
                max_rgb_observation_dt_s=max(i['sync_error_s'] for i in images.values()),
                max_anchor_pair_skew_s=max(t['anchor_pair_skew_s'] for t in blind_tasks),
                max_projection_error_px=max(i['projection_error_px'] for i in images.values()),
                mask_warning='No masks or exact detection boundaries; covariance ellipses and contextual crops only')


def build_private_table(private,out):
    rows=[]
    for t in private['tasks']:
        a,b=t['left'],t['right'];g=t['gate']
        rows.append('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in
                    [t['task_id'],t['episode_id'],t['pair'],t['step'],a['track_id'],b['track_id'],
                     [round(v,4) for v in a['map_row'][3:5]],[round(v,4) for v in b['map_row'][3:5]],
                     g['admitted'],g['first_exclusion_reason'],g['failed_checks'],round(t['distance_m'],3)])+'</tr>')
    header=['Пара','Эпизод','Роботы','Шаг','ID A','ID B','w/h A, м','w/h B, м','Допущена','Первая причина','Все проверки','Расстояние, м']
    page='<meta charset="utf-8"><title>Закрытое приложение: только после первичной разметки</title><style>body{font:14px sans-serif}td,th{border:1px solid #ccc;padding:5px}table{border-collapse:collapse}</style><h1>Открывать после экспорта слепой разметки</h1><p>Исходные ограничения: отношение размеров ≤1.35 и разность ≤0.1 м отдельно для ширины и высоты. Причина — первое сработавшее условие upstream; также показаны все нарушенные ограничения. Допуск означает кандидат в get_putative_assoc, а не принятое MNO-CLIPPER соответствие.</p><table><tr>'+''.join('<th>'+x+'</th>' for x in header)+'</tr>'+''.join(rows)+'</table>'
    (out/'analysis_private/gate_details.html').write_text(page)
