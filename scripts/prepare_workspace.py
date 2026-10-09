"""Materialize unchanged archived runners outside the publishable repository.

Only external-data pointers and a binary audit path may be relocated. No gate,
cost, experimental configuration or archived code is edited.
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path
from inspect_saved import ROOT, read, digest

def link(target, source):
    target.parent.mkdir(parents=True,exist_ok=True)
    target.symlink_to(source.resolve(),target_is_directory=source.is_dir())

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--venv',type=Path,required=True)
    p.add_argument('--frozen-project',type=Path,help='External original snapshot for the exact frozen size-gate experiment')
    args=p.parse_args()
    work=args.workspace.resolve();dataset=args.dataset.resolve();venv=args.venv.resolve()
    if work==ROOT or work.is_relative_to(ROOT):raise ValueError('Workspace must be outside the repository')
    if work.exists():raise ValueError('Choose a new workspace')
    if not (venv/'bin/python').is_file():raise ValueError('Missing external virtual environment')
    for robot in ['RR01','RR04','RR06','RR08']:
        assert (dataset/f'fastsam_data/{robot}.json').is_file()
        assert (dataset/f'data/kimera_odom/{robot}.bag').is_file()
    pins={'tcaff':'ff4ceab04b03fecabc4de9c9cbe7dbba4ab50df9','clipper':'e514dc29c273837ffdfeebbefbdcb2a93d970969'}
    for name,expected in pins.items():
        actual=subprocess.check_output(['git','-C',str(ROOT/'vendor'/name),'rev-parse','HEAD'],text=True).strip()
        assert actual==expected,(name,actual)
    relocated=[]
    if args.frozen_project:
        frozen=args.frozen_project.resolve()
        audit=read(frozen/'results/tcaff_baseline_release/audit.json')
        binaries=json.loads(subprocess.check_output([str(venv/'bin/python'),'-c','import clipperpy,json;from pathlib import Path;print(json.dumps([str(p) for p in Path(clipperpy.__file__).parent.glob("*.so")]))'],text=True))
        available={digest(Path(path)):path for path in binaries}
        for binary in audit['environment']['clipperpy_files']:
            if binary['sha256'] not in available:
                raise ValueError('Frozen binary SHA does not match this environment. Exact snapshot replay requires its original binary; a rebuilt binary is a distinct reproduction, not a silent replacement.')
            binary['path']=available[binary['sha256']];relocated.append(binary['sha256'])
        bundle=read(frozen/'data/size_gate_study/analysis_private/manifest.json')
        for file,expected in bundle['provenance']['code_hashes'].items():
            archived=ROOT/'scripts/original'/Path(file).name
            assert digest(archived)==expected,file
        for rec in audit['input_hashes']:
            assert digest(frozen/rec['path'])==rec['sha256'],rec['path']
    work.mkdir()
    shutil.copytree(ROOT/'scripts/original',work/'scripts')
    shutil.copytree(ROOT/'src',work/'src')
    shutil.copytree(ROOT/'configs',work/'configs')
    link(work/'vendor/tcaff',ROOT/'vendor/tcaff');link(work/'clipper',ROOT/'vendor/clipper')
    link(work/'.venv',venv);link(work/'data/tcaff_mot_data',dataset)
    if args.frozen_project:
        link(work/'data/tcaff_baseline_release',frozen/'data/tcaff_baseline_release')
        # A private working copy avoids any possible write to the original bundle.
        shutil.copytree(frozen/'data/size_gate_study',work/'data/size_gate_study',ignore=shutil.ignore_patterns('images','*.html'))
        target=work/'results/tcaff_baseline_release';target.mkdir(parents=True)
        (target/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
        shutil.copyfile(ROOT/'annotations/originals/labels-final-canonical.json',work/'data/size_gate_study/labels.json')
    info=dict(pins=pins,mode='exact frozen size study' if args.frozen_project else 'fresh released baseline',archived_code_unchanged=True,relocated_binary_hashes=relocated,warning='No experiment started. Run the commands in docs/reproduction.md inside this external workspace.')
    (work/'workspace_manifest.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info))

if __name__=='__main__':main()
