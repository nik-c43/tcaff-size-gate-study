"""Controlled visible-mask occlusion: descriptor and centroid changes only.

The physical cuboid and camera stay fixed. An exact foreground screen removes
one side of its ideal visible mask; remaining object depth is unchanged. This
is not a FastSAM segmentation run or an alternative TCAFF pose evaluation.
"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from size_gate_viewpoint_diagnosis import ideal_view, read, digest
from size_gate_prepare import check_pins, gate_details
from h1_map_audit import object_map
from tcaff.fastsam3D.depth_mask_2_centroid import depth_mask_2_centroid, mask_depth_2_width_height
from tcaff.tcaff.tcaff_manager import TCAFFManager


def hidden_side(mask, fraction, axis):
    """Hide the left/top fraction of the silhouette bounding-box extent."""
    assert axis in ['left', 'top'] and 0 <= fraction < 1
    ys, xs = np.nonzero(mask)
    coordinates = xs if axis == 'left' else ys
    low, high = int(coordinates.min()), int(coordinates.max())
    removed = int(round(fraction * (high-low+1)))
    visible = mask.copy()
    if axis == 'left':
        visible[:, :low+removed] = 0
    else:
        visible[:low+removed, :] = 0
    assert np.any(visible) and np.all(visible <= mask)
    return visible


def describe(mask, depth, K):
    ys, xs = np.nonzero(mask)
    mean = [float(xs.mean()), float(ys.mean())]
    center = depth_mask_2_centroid(depth, mask, mean, K)
    width, height = mask_depth_2_width_height(center[2], mask, K)
    assert np.all(np.isfinite(center)) and width > 0 and height > 0
    return dict(visible_pixels=int(mask.sum()), pixel_mean=mean, centroid_camera_RDF_m=center.tolist(),
                width_m=float(width), height_m=float(height), map_row=[*center.tolist(), float(width), float(height), 1.])


def main():
    out = ROOT / 'results/size_gate_occlusion_diagnosis_20261009'
    if out.exists():
        raise ValueError('Refuse overwrite')
    manifest = read(ROOT / 'data/size_gate_study/analysis_private/manifest.json')
    primary_path = ROOT / 'results/size_gate_study_20261009/summary.json'
    primary = read(primary_path)
    check_pins(manifest['protocol'])
    source = read(ROOT / 'data/tcaff_baseline_release/RR01_RR04.json')
    kwargs = {k: v for k, v in source['params'].items() if k != 'ts'}
    managers = {1.: TCAFFManager(**kwargs), primary['alpha']: TCAFFManager(**dict(kwargs,
                wh_scale_diff=primary['selected_wh_scale_diff'], h_diff=primary['selected_h_diff_m']))}
    K = np.array([[600., 0., 320.], [0., 600., 240.], [0., 0., 1.]])
    shape, distance = (480, 640), 3.
    dimensions = {'cube': [.4, .4, .4], 'wide_box': [1.2, .4, .8]}
    rows, display = [], {}
    for name, sizes in dimensions.items():
        for angle in [0, 45]:
            _, full_mask, depth = ideal_view(sizes, angle, K, shape, distance)
            full = describe(full_mask, depth, K)
            for axis in ['left', 'top']:
                for percentage in range(0, 91, 10):
                    fraction = percentage / 100.
                    visible = hidden_side(full_mask, fraction, axis)
                    current = describe(visible, depth, K)
                    shift = np.asarray(current['centroid_camera_RDF_m']) - full['centroid_camera_RDF_m']
                    gates = {}
                    for alpha, manager in managers.items():
                        actual = len(manager.get_putative_assoc(object_map([full['map_row']]), object_map([current['map_row']]))) == 1
                        detail = gate_details(full['map_row'], current['map_row'], manager.wh_scale_diff, manager.h_diff)
                        assert actual == detail['admitted']
                        gates[str(alpha)] = detail
                    if percentage == 0:
                        assert np.all(shift == 0) and all(g['admitted'] for g in gates.values())
                    rows.append(dict(object=name, dimensions_horizontal_depth_height_m=sizes,
                                     angle_deg=angle, hidden_side=axis, requested_bbox_occlusion_fraction=fraction,
                                     actual_mask_area_removed_fraction=1-current['visible_pixels']/full['visible_pixels'],
                                     reference=full, occluded=current, centroid_shift_RDF_m=shift.tolist(),
                                     centroid_shift_norm_m=float(np.linalg.norm(shift)),
                                     level_camera_horizontal_plane_shift_m=float(np.linalg.norm(shift[[0, 2]])),
                                     gates=gates))
                    if name == 'cube' and angle == 0 and axis == 'left' and percentage in [0, 30, 60]:
                        display[percentage] = visible
    assert len(rows) == 80
    summary = dict(cases=len(rows), dimensions_m=dimensions, angles_deg=[0, 45], screen_directions=['left', 'top'],
                   bbox_occlusion_fractions=[i/100 for i in range(0, 91, 10)], camera_K=K.tolist(), image_shape=list(shape),
                   object_center_depth_m=distance, selected_alpha=primary['alpha'],
                   primary_summary_sha256=digest(primary_path), model_script_sha256=digest(ROOT / 'scripts/size_gate_viewpoint_diagnosis.py'),
                   script_sha256=digest(Path(__file__)), command=sys.argv,
                   scope='Synthetic visible-mask mechanism; no segmentation network, map smoothing, MNO solving or temporal TCAFF',
                   centroid_reference='Shift relative to the original full visible-surface measurement, not error relative to physical object center.',
                   occlusion_definition='Left/top foreground screen removes a fraction of the full silhouette bounding-box pixel extent; mask area fraction recorded separately.',
                   assumptions='Exact remaining silhouette and noiseless front-surface depth; fixed object and camera; masked-out occluder excluded exactly.',
                   counts={str(alpha): dict(nonzero_cases=72, rejected=sum(not r['gates'][str(alpha)]['admitted'] for r in rows if r['requested_bbox_occlusion_fraction'] > 0),
                            admitted_shift_over_0_1m=sum(r['gates'][str(alpha)]['admitted'] and r['centroid_shift_norm_m'] > .1 for r in rows))
                           for alpha in managers})
    check_pins(manifest['protocol'])
    out.mkdir()
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    (out / 'cases.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2)+'\n')
    np.savez_compressed(out / 'display_masks.npz', **{f'cube_left_{percentage}': mask for percentage, mask in display.items()}, camera_K=K)
    fig = plt.figure(figsize=(12, 7.5))
    grid = fig.add_gridspec(2, 6, height_ratios=[1, 1.5], hspace=.43, wspace=.4)
    for i, percentage in enumerate([0, 30, 60]):
        ax = fig.add_subplot(grid[0, 2*i:2*i+2])
        row = next(r for r in rows if r['object']=='cube' and r['angle_deg']==0 and r['hidden_side']=='left' and r['requested_bbox_occlusion_fraction']==percentage/100)
        ax.imshow(display[percentage][170:311, 230:411], cmap='Greys', vmin=0, vmax=1)
        ax.set_title(f"Куб, скрыто {percentage}% ширины\nw={row['occluded']['width_m']:.3f} м; Δc={row['centroid_shift_norm_m']:.3f} м", fontsize=10)
        ax.set_axis_off()
    for j, name in enumerate(dimensions):
        ax = fig.add_subplot(grid[1, j*3:j*3+3])
        for angle in [0, 45]:
            selected = [r for r in rows if r['object']==name and r['angle_deg']==angle and r['hidden_side']=='left']
            x = [100*r['requested_bbox_occlusion_fraction'] for r in selected]
            y = [r['centroid_shift_norm_m'] for r in selected]
            line, = ax.plot(x, y, label=f'Ракурс {angle}°')
            for alpha, marker in [(1., 'o'), (primary['alpha'], 's')]:
                accepted = [r for r in selected if r['gates'][str(alpha)]['admitted']]
                ax.scatter([100*r['requested_bbox_occlusion_fraction'] for r in accepted],
                           [r['centroid_shift_norm_m'] for r in accepted], marker=marker, s=65 if alpha==1. else 105,
                           facecolors='none', edgecolors=line.get_color(), linewidths=1.2)
        ax.set_title('Куб 0.4 × 0.4 × 0.4 м' if name=='cube' else 'Коробка 1.2 × 0.4 × 0.8 м')
        ax.set_xlabel('Скрытая доля ширины силуэта, %')
        ax.set_ylabel('Сдвиг центра относительно полной маски, м')
        ax.grid(alpha=.25)
        ax.legend(fontsize=9)
    fig.suptitle('Идеальное частичное перекрытие меняет размер и центр одного объекта', fontsize=13, y=.99)
    fig.text(.5, .02, 'Кружок: пара проходит B0; квадрат: проходит выбранный α=2. Камера и физический объект неподвижны.', ha='center', fontsize=10)
    fig.subplots_adjust(bottom=.13, top=.90)
    fig.savefig(out / 'occlusion_centroid.png', dpi=160)
    plt.close(fig)
    lines = ['# Частичное перекрытие: размер и измеренный центр', '',
             'Контролируемая синтетическая проверка выполняется после основного эксперимента. '
             'Неподвижные камера и параллелепипед дают идеальную видимую маску и глубину поверхности. '
             'Передний экран скрывает левую или верхнюю часть объекта; оставшаяся маска объекта точна, пиксели экрана исключены. '
             'Используются исходные функции расчёта центра/размера и исходный размерный допуск TCAFF.', '',
             'Сдвиг считается относительно измерения по полной видимой маске в той же камере. '
             'Он не является ошибкой относительно истинного объёмного центра. Положение физического предмета не меняется.', '',
             '![Размер и центр при перекрытии](occlusion_centroid.png)', '',
             '| Фронтальная маска куба, экран слева | Ширина, м | Сдвиг центра, м | B0 | α=2 |', '|---|---:|---:|---|---|']
    for row in rows:
        if row['object']=='cube' and row['angle_deg']==0 and row['hidden_side']=='left':
            decisions = ['допускает' if row['gates'][str(alpha)]['admitted'] else 'исключает' for alpha in [1., primary['alpha']]]
            lines.append(f"| {100*row['requested_bbox_occlusion_fraction']:.0f}% | {row['occluded']['width_m']:.4f} | {row['centroid_shift_norm_m']:.4f} | {decisions[0]} | {decisions[1]} |")
    lines += ['', 'Проверены две коробки, ракурсы 0°/45°, два направления экрана, скрытые доли 0–90% с шагом 10%: '
              '80 случаев, включая 8 повторяющихся контролей без перекрытия. Это зависимые случаи модели, не частота реальных отказов. '
              'Процент относится к протяжённости bounding box маски; доля реально удалённой площади записана отдельно.', '',
              'Доказана возможность одновременного изменения размера и центра одного неподвижного предмета при идеальном частичном перекрытии. '
              'Поэтому физически правильная пара, возвращённая ослаблением размерного допуска, не обязана давать одинаковые геометрические центры. '
              'Ни вклад этого механизма в реальные отказы, ни улучшение позы здесь не установлены. '
              'Масок авторского датасета нет; его отказ нельзя автоматически приписать перекрытию.', '',
              'FastSAM, сглаживание карты, MNO и временной TCAFF не запускались. Стоимость и upstream не менялись; α не подбирался. '
              'В JSON сохранены камера, физические размеры, доли перекрытия, центры и все размерные решения; показанные маски сохранены в NPZ.', '',
              '```bash', 'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/tcaff-mpl \\',
              '  .venv/bin/python scripts/size_gate_occlusion_diagnosis.py', '```', '']
    (out / 'report.md').write_text('\n'.join(lines))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
