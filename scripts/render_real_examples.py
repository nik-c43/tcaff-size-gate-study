"""Render selected original RGB contexts locally; never download or bundle images."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from inspect_saved import ROOT,read,digest

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pool',type=Path,required=True,help='External annotation_blind directory containing images/')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--blind',action='store_true');args=p.parse_args()
    if args.output.exists():raise ValueError('Choose a new output file')
    if args.output.resolve().is_relative_to(ROOT):raise ValueError('Image publication permission is not established; write outside the repository')
    tasks={r['task_id']:r for r in read(ROOT/'annotations/task_metadata.json')['tasks']}
    labels={r['task_id']:r for r in read(ROOT/'annotations/originals/labels-final-canonical.json')['labels']}
    examples=read(ROOT/'annotations/examples.json')['tasks']
    fig,axes=plt.subplots(len(examples),2,figsize=(13,4*len(examples)))
    images={}
    for i,e in enumerate(examples):
        task=tasks[e['task_id']];label=labels[e['task_id']]
        for j,side in enumerate(task['sides']):
            path=args.pool/side['views'][0]['full']
            if not path.is_file():raise FileNotFoundError(path)
            with Image.open(path) as im:axes[i,j].imshow(im)
            images[side['views'][0]['full']]=digest(path)
            title=f"{task['task_id']} · {side['robot']} · track {side['track_id'][1]} · det {side['views'][0]['detection_index']}"
            if not args.blind:title+=f"\n{label['label']} ({label['confidence']}); B0 {'допускает' if task['gate']['admitted'] else 'исключает'}; w={side['map_width_m']:.3f}, h={side['map_height_m']:.3f} м"
            axes[i,j].set_title(title,fontsize=9);axes[i,j].set_axis_off()
    fig.suptitle('Выбранные реальные наблюдения. Эллипс ковариации ≠ маска. Наблюдения треков могут быть разновременными.',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.97));fig.savefig(args.output,dpi=130);plt.close(fig)
    meta=dict(scope='Local-only original author RGB contexts; selected illustrations, not a representative sample',blind=args.blind,task_ids=[e['task_id'] for e in examples],image_hashes=images,script_sha256=digest(Path(__file__)),figure_sha256=digest(args.output))
    args.output.with_suffix('.provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
    print('Rendered local examples:',len(examples))

if __name__=='__main__':main()
