"""Build report figures from saved own metrics and ideal masks; no experiments."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from inspect_saved import ROOT, read, digest

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    out=args.output
    if out.exists():raise ValueError('Choose a new output directory')
    out.mkdir(parents=True)
    summary=read(ROOT/'results/size_gate/summary.json')
    fig,axes=plt.subplots(1,5,figsize=(15,4.5))
    colors=['#35629e','#c96729']
    metrics=[('correct_update_availability','Правильные обновления','%',100),('mean_translation_m','Ошибка переноса','м',1),('mean_yaw_deg','Ошибка угла','°',1),('availability','Доступность','%',100),('runtime_s','Время окна','с',1)]
    for ax,(key,title,units,factor) in zip(axes,metrics):
        values=[summary[k][key]*factor for k in ['rerun_B0','selected_variant']]
        ax.bar(['B0','α=2'],values,color=colors,width=.6)
        ax.set_title(title,fontsize=11);ax.set_ylabel(units)
        ax.set_ylim(0,max(values)*1.25)
        for x,value in enumerate(values):ax.text(x,value+max(values)*.04,f'{value:.2f}',ha='center',fontsize=10)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Поздние окна с полной историей: 12 эпизодов × 29 обновлений × 5 повторов',fontsize=12)
    fig.text(.5,.02,'Одна запись; среднее повторов внутри эпизода, затем среднее эпизодов. Правильно: ≤1 м и ≤5°.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.06,1,.91))
    for ext in ['png','svg']:fig.savefig(out/f'late_windows.{ext}',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,4.5))
    for ax,key,title in zip(axes,['wrong_first_rate','correct_by_horizon_rate'],['Неверное первое решение','Правильное решение до конца окна']):
        values=[summary['cold_start'][str(a)][key]*100 for a in [1.,2.]]
        ax.bar(['B0','α=2'],values,color=colors,width=.6);ax.set_title(title,fontsize=11);ax.set_ylabel('% стартов');ax.set_ylim(0,65)
        for i,v in enumerate(values):ax.text(i,v+2,f'{v:.2f}%',ha='center')
    fig.suptitle('Отдельная диагностика холодного старта: 12 окон × 5 повторов',fontsize=12)
    fig.text(.5,.015,'Отсутствие первого решения: 40/60 → 25/60. Повторы и окна одной записи зависимы.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.05,1,.92));fig.savefig(out/'cold_start.png',dpi=160);plt.close(fig)
    cases=read(ROOT/'results/occlusion/cases.json');masks=np.load(ROOT/'results/occlusion/display_masks.npz')
    fig=plt.figure(figsize=(12,7.5));grid=fig.add_gridspec(2,6,height_ratios=[1,1.5],hspace=.45,wspace=1)
    for i,percentage in enumerate([0,30,60]):
        row=next(r for r in cases if r['object']=='cube' and r['angle_deg']==0 and r['hidden_side']=='left' and r['requested_bbox_occlusion_fraction']==percentage/100)
        ax=fig.add_subplot(grid[0,2*i:2*i+2]);ax.imshow(masks[f'cube_left_{percentage}'][170:311,230:411],cmap='Greys',vmin=0,vmax=1)
        ax.set_title(f"Скрыто {percentage}% ширины\nw={row['occluded']['width_m']:.3f} м; Δc={row['centroid_shift_norm_m']:.3f} м",fontsize=10);ax.set_axis_off()
    for j,name in enumerate(['cube','wide_box']):
        ax=fig.add_subplot(grid[1,j*3:j*3+3])
        for angle in [0,45]:
            rows=[r for r in cases if r['object']==name and r['angle_deg']==angle and r['hidden_side']=='left']
            line,=ax.plot([100*r['requested_bbox_occlusion_fraction'] for r in rows],[r['centroid_shift_norm_m'] for r in rows],label=f'Ракурс {angle}°')
            for alpha,marker in [(1.,'o'),(2.,'s')]:
                admitted=[r for r in rows if r['gates'][str(alpha)]['admitted']]
                ax.scatter([100*r['requested_bbox_occlusion_fraction'] for r in admitted],[r['centroid_shift_norm_m'] for r in admitted],marker=marker,s=65 if alpha==1. else 105,facecolors='none',edgecolors=line.get_color())
        ax.set_title('Куб 0.4 × 0.4 × 0.4 м' if name=='cube' else 'Коробка 1.2 × 0.4 × 0.8 м',fontsize=11)
        ax.set_xlabel('Скрытая доля ширины силуэта, %');ax.set_ylabel('Сдвиг измеренного центра, м');ax.grid(alpha=.25);ax.legend()
    fig.suptitle('Идеальное перекрытие: камера и предмет неподвижны',fontsize=13)
    fig.text(.5,.045,'Кружок: проходит B0; квадрат: проходит α=2. Идеальные оставшиеся маски и глубина.',ha='center',fontsize=10)
    fig.text(.5,.017,'Сдвиг относительно измерения полной видимой поверхности, а не объёмного центра.',ha='center',fontsize=9)
    fig.subplots_adjust(bottom=.155,top=.89,left=.085,right=.97);fig.savefig(out/'occlusion.png',dpi=160);plt.close(fig)
    inputs=['results/size_gate/summary.json','results/occlusion/cases.json','results/occlusion/display_masks.npz']
    provenance=dict(scope='Layout only; no new experiment or changed numerical input',command='python3 scripts/render_saved.py --output <new-directory>',script_sha256=digest(Path(__file__)),inputs={f:digest(ROOT/f) for f in inputs},outputs={p.name:digest(p) for p in out.iterdir() if p.is_file()})
    (out/'plot_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('Rendered:',len(provenance['outputs']),'figures')

if __name__=='__main__':main()
