"""Render completed occlusion cases with readable scientific figure labels."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from size_gate_candidate_stage_audit import read, digest


def main():
    out = ROOT / 'results/size_gate_occlusion_diagnosis_20261009'
    target = out / 'occlusion_centroid_readable.png'
    if target.exists():
        raise ValueError('Refuse overwrite')
    rows = read(out / 'cases.json')
    summary = read(out / 'summary.json')
    alpha = summary['selected_alpha']
    masks = np.load(out / 'display_masks.npz')
    fig = plt.figure(figsize=(12, 7.5))
    grid = fig.add_gridspec(2, 6, height_ratios=[1, 1.5], hspace=.45, wspace=1.0)
    for i, percentage in enumerate([0, 30, 60]):
        ax = fig.add_subplot(grid[0, 2*i:2*i+2])
        row = next(r for r in rows if r['object']=='cube' and r['angle_deg']==0 and r['hidden_side']=='left' and r['requested_bbox_occlusion_fraction']==percentage/100)
        ax.imshow(masks[f'cube_left_{percentage}'][170:311, 230:411], cmap='Greys', vmin=0, vmax=1)
        ax.set_title(f"Куб, скрыто {percentage}% ширины\nw={row['occluded']['width_m']:.3f} м; Δc={row['centroid_shift_norm_m']:.3f} м", fontsize=10)
        ax.set_axis_off()
    for j, name in enumerate(summary['dimensions_m']):
        ax = fig.add_subplot(grid[1, j*3:j*3+3])
        for angle in [0, 45]:
            selected = [r for r in rows if r['object']==name and r['angle_deg']==angle and r['hidden_side']=='left']
            line, = ax.plot([100*r['requested_bbox_occlusion_fraction'] for r in selected],
                            [r['centroid_shift_norm_m'] for r in selected], label=f'Ракурс {angle}°')
            for a, marker in [(1., 'o'), (alpha, 's')]:
                accepted = [r for r in selected if r['gates'][str(a)]['admitted']]
                ax.scatter([100*r['requested_bbox_occlusion_fraction'] for r in accepted],
                           [r['centroid_shift_norm_m'] for r in accepted], marker=marker, s=65 if a==1. else 105,
                           facecolors='none', edgecolors=line.get_color(), linewidths=1.2)
        ax.set_title('Куб 0.4 × 0.4 × 0.4 м' if name=='cube' else 'Коробка 1.2 × 0.4 × 0.8 м', fontsize=11)
        ax.set_xlabel('Скрытая доля ширины силуэта, %', fontsize=10)
        ax.set_ylabel('Сдвиг измеренного центра, м', fontsize=10)
        ax.grid(alpha=.25)
        ax.legend(fontsize=9)
    fig.suptitle('Идеальное частичное перекрытие меняет размер и центр одного объекта', fontsize=13, y=.98)
    fig.text(.5, .045, f'Кружок: пара проходит B0; квадрат: проходит выбранный α={alpha:g}. Камера и предмет неподвижны.', ha='center', fontsize=10)
    fig.text(.5, .017, 'Сдвиг относительно измерения по полной видимой маске; это не ошибка относительно объёмного центра.', ha='center', fontsize=9)
    fig.subplots_adjust(bottom=.155, top=.89, left=.085, right=.97)
    fig.savefig(target, dpi=160)
    plt.close(fig)
    original = out / 'report_original.md'
    assert not original.exists()
    report = out / 'report.md'
    original.write_bytes(report.read_bytes())
    report.write_text(report.read_text().replace('(occlusion_centroid.png)', '(occlusion_centroid_readable.png)'))
    provenance = dict(command=sys.argv, script_sha256=digest(Path(__file__)),
                      summary_sha256=digest(out / 'summary.json'), cases_sha256=digest(out / 'cases.json'),
                      masks_sha256=digest(out / 'display_masks.npz'), figure_sha256=digest(target),
                      original_report_sha256=digest(original), current_report_sha256=digest(report),
                      scope='Figure layout only; original numerical outputs and original figure retained')
    (out / 'plot_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps(dict(figure=str(target), numerical_outputs_unchanged=True)))


if __name__ == '__main__':
    main()
