"""Synthetic camera-view diagnostic using the pinned author's size functions.

Ideal cuboid ray intersections provide exact visible masks and noiseless depth.
No external occluders, segmentation errors, merging or cost modifications.
This tests a mechanism, not its frequency in the real recording.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from size_gate_prepare import check_pins, gate_details
from h1_map_audit import object_map
from tcaff.tcaff.tcaff_manager import TCAFFManager
from tcaff.fastsam3D.depth_mask_2_centroid import depth_mask_2_centroid, mask_depth_2_width_height


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ideal_view(dimensions, degrees, K, shape, distance):
    """Ray/OBB front-surface intersections, in camera RDF coordinates."""
    y, x = np.indices(shape)
    rays = np.stack(((x-K[0,2])/K[0,0], (y-K[1,2])/K[1,1], np.ones(shape)), axis=-1)
    angle = np.radians(degrees)
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c,0,s],[0,1,0],[-s,0,c]])
    center = np.array([0.,0.,distance])
    # dimensions are horizontal side, depth side, vertical height.
    half = np.array([dimensions[0],dimensions[2],dimensions[1]])/2
    origin = -rotation.T@center
    local_rays = rays@rotation
    with np.errstate(divide='ignore',invalid='ignore'):
        t0 = (-half-origin)/local_rays
        t1 = (half-origin)/local_rays
    near = np.max(np.minimum(t0,t1),axis=-1)
    far = np.min(np.maximum(t0,t1),axis=-1)
    valid = (near>0)&np.isfinite(near)&(near<=far)
    mask = valid.astype(np.uint8)
    depth = np.where(valid,near*1000.,0.)
    assert np.any(valid) and not np.any(valid[[0,-1],:]) and not np.any(valid[:,[0,-1]])
    ys, xs = np.nonzero(mask)
    center_pixel = (float(np.mean(xs)),float(np.mean(ys)))
    centroid = depth_mask_2_centroid(depth,mask,center_pixel,K)
    width, height = mask_depth_2_width_height(centroid[2],mask,K)
    np.testing.assert_allclose([width,height],[(xs.max()-xs.min())*centroid[2]/K[0,0],
                                               (ys.max()-ys.min())*centroid[2]/K[1,1]],atol=1e-14,rtol=0.)
    return dict(angle_deg=degrees, width_m=float(width),height_m=float(height),
                median_depth_m=float(centroid[2]), mask_pixels=int(valid.sum()),
                map_row=[*centroid.tolist(),float(width),float(height),1.]),mask,depth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-results',default='results/size_gate_study_20261009')
    parser.add_argument('--output',default='results/size_gate_viewpoint_diagnosis_20261009')
    args = parser.parse_args()
    primary_dir, out = ROOT/args.main_results, ROOT/args.output
    if out.exists() or not out.resolve().is_relative_to(ROOT/'results'):
        raise ValueError('Use a new separate results directory')
    primary = read(primary_dir/'summary.json')
    assert read(primary_dir/'final_checks.json')['all_checks_passed']
    manifest = read(ROOT/'data/size_gate_study/analysis_private/manifest.json')
    check_pins(manifest['protocol'])
    source = read(ROOT/'data/tcaff_baseline_release/RR01_RR04.json')
    base_kwargs = {k:v for k,v in source['params'].items() if k!='ts'}
    alpha = primary['alpha']
    managers = {1.:TCAFFManager(**base_kwargs),alpha:TCAFFManager(**dict(base_kwargs,
                wh_scale_diff=primary['selected_wh_scale_diff'],h_diff=primary['selected_h_diff_m']))}
    K = np.array([[600.,0.,320.],[0.,600.,240.],[0.,0.,1.]])
    shape, distance = (480,640),3.
    dimensions = {'cube':[.4,.4,.4],'rectangular_box':[.5,.3,.4]}
    views, pairs, display = {},[],{}
    for name, size in dimensions.items():
        views[name] = []
        for angle in range(0,91,5):
            row, mask, depth = ideal_view(size,angle,K,shape,distance)
            views[name].append(row)
            if angle in [0,45,90]:
                display[name,angle] = (mask,depth)
        for left, right in ((a,b) for i,a in enumerate(views[name]) for b in views[name][i+1:]):
            decisions = {}
            for a,manager in managers.items():
                decision = len(manager.get_putative_assoc(object_map([left['map_row']]),object_map([right['map_row']])))==1
                reference = gate_details(left['map_row'],right['map_row'],manager.wh_scale_diff,manager.h_diff)
                assert decision == reference['admitted']
                decisions[str(a)] = reference
            pairs.append(dict(object=name,left_angle_deg=left['angle_deg'],right_angle_deg=right['angle_deg'],decisions=decisions))
    # Reporting is deliberately separate from algorithm execution.
    summaries = {name:{str(a):dict(view_pairs=171,rejected=sum(not row['decisions'][str(a)]['admitted'] for row in pairs if row['object']==name))
                     for a in managers} for name in dimensions}
    summary = dict(alpha=alpha, dimensions_m=dimensions, camera_K=K.tolist(),image_shape=list(shape),
                   object_center_depth_m=distance, relative_view_angles_deg=list(range(0,91,5)),
                   summaries=summaries,views=views,
                   upstream_commit=manifest['protocol']['tcaff_commit'],
                   input_main_summary_sha256=digest(primary_dir/'summary.json'),script_sha256=digest(Path(__file__)),command=sys.argv,
                   scope='Mechanism demonstration for two ideal cuboids; not real-dataset frequency or pose evaluation',
                   assumptions='Ideal silhouette, continuous noiseless front-surface depth; no external occlusion. Relative box orientation represents a camera orbit around a stationary box.',
                   algorithm_run='Pinned depth_mask_2_centroid and mask_depth_2_width_height; only size gate, no MNO-CLIPPER solving or temporal TCAFF')
    check_pins(manifest['protocol'])
    out.mkdir(parents=True)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (out/'view_pair_decisions.json').write_text(json.dumps(pairs,indent=2)+'\n')
    fig, axes = plt.subplots(2,3,figsize=(11,6))
    for i,name in enumerate(dimensions):
        for j,angle in enumerate([0,45,90]):
            mask,depth = display[name,angle]
            ax = axes[i,j]
            # Same pixel canvas and magnification in all six panels.
            cropped = np.where(mask,depth/1000.,np.nan)[170:311,230:411]
            ax.imshow(cropped,cmap='viridis',vmin=2.65,vmax=3.2)
            row = next(r for r in views[name] if r['angle_deg']==angle)
            title = 'Куб' if name=='cube' else 'Прямоугольная коробка'
            ax.set_title(f'{title}, ракурс {angle}°\nширина {row["width_m"]:.3f} м, высота {row["height_m"]:.3f} м',fontsize=10)
            ax.set_axis_off()
    fig.suptitle('Синтетическая диагностика: идеальные видимые маски и глубина',fontsize=14)
    fig.text(.5,.02,'Один физический объект в строке; внешнего перекрытия и ошибки сегментации нет.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.055,1,.94])
    fig.savefig(out/'ideal_cuboid_views.png',dpi=160)
    plt.close(fig)
    examples = []
    for name in dimensions:
        baseline_rejected = [row for row in pairs if row['object']==name and not row['decisions']['1.0']['admitted']]
        if baseline_rejected:
            # Fixed diagnostic examples, never used for selecting the relaxation.
            fixed = next((row for row in baseline_rejected if row['left_angle_deg']==0 and row['right_angle_deg']==45),None)
            example = fixed if fixed is not None else baseline_rejected[0]
            examples.append(example)
    lines = ['# Размерный признак и ракурс: контролируемая диагностика', '',
             'Проверена отдельная гипотеза: исходный размерный допуск может исключить наблюдения одного жёсткого '
             'объекта только из-за изменения ракурса, без внешнего перекрытия и без ошибки сегментации. '
             'Диагностика выполнена после основного сравнения; выбранный на реальных development-эпизодах α не менялся.', '',
             'В камере построены точные пересечения лучей с двумя параллелепипедами. '
             'Видимая маска идеальна, глубина поверхности непрерывная и без шума. Относительный поворот вокруг вертикальной '
             'оси соответствует обходу камерой неподвижной коробки. Это синтетическая модель, а не реальные RGB-D кадры.', '',
             '| Объект | Габариты: две стороны × высота, м | Исключено B0 из 171 пар ракурсов | Исключено выбранным вариантом |',
             '|---|---|---:|---:|']
    for name,ds in dimensions.items():
        lines.append(f"| {name} | {' × '.join(str(v) for v in ds)} | {summaries[name]['1.0']['rejected']} | {summaries[name][str(alpha)]['rejected']} |")
    lines += ['', 'Перебор ракурсов: 0–90° с шагом 5°; 19 ракурсов дают 171 неупорядоченную пару. '
              'Эти пары зависимы. Их доля исключения описывает выбранную модель и сетку, не частоту отказов на реальных данных.', '',
              '![Идеальные наблюдения](ideal_cuboid_views.png)', '', '## Примеры', '']
    for row in examples:
        by_angle = {v['angle_deg']:v for v in views[row['object']]}
        left,right = by_angle[row['left_angle_deg']],by_angle[row['right_angle_deg']]
        lines.append(f"- {row['object']}, {row['left_angle_deg']}° → {row['right_angle_deg']}°: ширина "
                     f"{left['width_m']:.4f} → {right['width_m']:.4f} м; высота {left['height_m']:.4f} → {right['height_m']:.4f} м. "
                     f"Нарушения B0: {', '.join(row['decisions']['1.0']['failed_checks'])}. "
                     f"Выбранный вариант: {'допускает' if row['decisions'][str(alpha)]['admitted'] else 'исключает'}.")
    lines += ['', '## Что установлено', '',
              '`mask_depth_2_width_height` берёт крайние координаты пикселей маски, проектирует их с одной медианной глубиной '
              'и возвращает разности координат: `w=(xmax−xmin)·d/fx`, `h=(ymax−ymin)·d/fy`. '
              'Функция не оценивает инвариантные размеры 3D-тела. Идеальный силуэт может менять ширину при ракурсе. '
              'Решения сверены с исходным `TCAFFManager.get_putative_assoc`.', '',
              'Если B0 отвергает хотя бы одну пару идеальных ракурсов одного объекта, это подтверждает возможность '
              'механизма без внешнего перекрытия. Это не устанавливает причину конкретных отказов в авторской записи: '
              'там нет масок, а треки могут представлять части, группы или разные сущности. '
              'Улучшение позы, MOTA и универсальность ослабления этим опытом не проверяются. '
              'Временной фильтр и решатель MNO-CLIPPER не запускались.', '',
              f"Коммит TCAFF: `{manifest['protocol']['tcaff_commit']}`. Функции: "
              ' `depth_mask_2_centroid`, `mask_depth_2_width_height` '
              '(`vendor/tcaff/tcaff/fastsam3D/depth_mask_2_centroid.py`) и '
              '`TCAFFManager.get_putative_assoc` (`vendor/tcaff/tcaff/tcaff/tcaff_manager.py`).', '',
              'Все размеры, параметры камеры, масочные площади и решения каждой пары сохранены в JSON. '
              'Новых параметров на отложенных данных не подбирали; стоимость и upstream не изменяли.', '',
              '```bash',
              'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/tcaff-mpl \\',
              '  .venv/bin/python scripts/size_gate_viewpoint_diagnosis.py', '```', '']
    (out/'report.md').write_text('\n'.join(lines))
    print(json.dumps(dict(report=str(out/'report.md'),summaries=summaries),indent=2))


if __name__ == '__main__':
    main()
