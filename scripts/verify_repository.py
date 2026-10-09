"""Check publication contents, local navigation, pins and unchanged adapted B0."""
import argparse
import ast
import csv
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote
from inspect_saved import ROOT,read,digest,check_annotations,check_results,verify_transfer

def slug(text):
    text=re.sub(r'[`*_]','',text).lower().strip()
    text=re.sub(r'[^\w\- ]','',text)
    return text.replace(' ','-')

def ast_item(path, name, method=None):
    node=next(n for n in ast.parse(path.read_text()).body if getattr(n,'name',None)==name)
    if method:node=next(n for n in node.body if getattr(n,'name',None)==method)
    return ast.dump(node,include_attributes=False)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);args=p.parse_args()
    tracked=subprocess.check_output(['git','-C',str(ROOT),'ls-files','--stage'],text=True).splitlines()
    own=[];submodules=[]
    for line in tracked:
        mode,obj,stage,path=line.split(maxsplit=3)
        (submodules if mode=='160000' else own).append(path)
    assert set(submodules)=={'vendor/tcaff','vendor/clipper'},submodules
    assert not any('roman' in f.lower() for f in own), 'ROMAN filename'
    pins={'tcaff':'ff4ceab04b03fecabc4de9c9cbe7dbba4ab50df9','clipper':'e514dc29c273837ffdfeebbefbdcb2a93d970969'}
    for name,pin in pins.items():
        assert subprocess.check_output(['git','-C',str(ROOT/'vendor'/name),'rev-parse','HEAD'],text=True).strip()==pin
        assert not subprocess.check_output(['git','-C',str(ROOT/'vendor'/name),'status','--porcelain','--untracked-files=no'],text=True).strip()
    assert 'roman' not in (ROOT/'.gitmodules').read_text().lower()
    for f in own:
        path=ROOT/f
        assert not path.is_symlink(),f
        assert path.stat().st_size<10_000_000,(f,path.stat().st_size)
        assert path.suffix not in ['.bag','.pt','.pth','.onnx','.so','.o','.jpg','.jpeg','.tar','.zip'],f
        assert not any(part.startswith('.venv') or part in ['__pycache__','fastsam_data'] for part in path.parts),f
        if path.suffix in ['.md','.py','.sh','.txt','.json','.csv','.patch','.html']:
            text=path.read_text()
            assert not re.search(r'/(?:home)/[^/\s]+|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN (?:OPENSSH|RSA|EC) PRIVATE KEY',text),f
        if path.suffix=='.py':
            for n in ast.walk(ast.parse(path.read_text())):
                if isinstance(n,ast.Import):assert all(a.name.split('.')[0]!='roman' for a in n.names),f
                if isinstance(n,ast.ImportFrom):assert (n.module or '').split('.')[0]!='roman',f
    for filename in ['requirements.txt','requirements-light.txt']:
        assert 'roman' not in (ROOT/filename).read_text().lower()
    links=0
    for f in own:
        path=ROOT/f
        if path.suffix!='.md':continue
        text=path.read_text()
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
            if target.startswith(('https://','http://','mailto:')):continue
            target=unquote(target.strip('<>'));file,_,anchor=target.partition('#')
            dest=(path.parent/file).resolve() if file else path
            assert dest.exists(),(f,target)
            if anchor and dest.suffix=='.md':
                headings=[slug(x) for x in re.findall(r'^#+\s+(.+)$',dest.read_text(),re.M)]
                assert anchor in headings,(f,target,headings)
            links+=1
    manifest=read(ROOT/'annotations/manifest.json')
    for rec in manifest['files']:assert rec['path'] in own,rec['path']
    assert 'annotations/labels.csv' in own
    upstream=ROOT/'vendor/tcaff/demo';local=ROOT/'src/demo_cpu'
    for name in ['robot.py','tcaff_data_processory.py']:assert digest(upstream/name)==digest(local/name)
    assert ast_item(upstream/'detections.py','DetectionData')==ast_item(local/'detections.py','DetectionData')
    assert ast_item(upstream/'alignment_results.py','AlignmentResults','get_Tij_gt')==ast_item(local/'alignment_results.py','AlignmentResults','get_Tij_gt')
    runs=read(ROOT/'results/size_gate/run_metrics.json.gz')['runs'];params={k:v for k,v in read(ROOT/'configs/manager.json').items() if k!='ts'}
    for run in runs:
        expected=dict(params,wh_scale_diff=1+run['alpha']*(params['wh_scale_diff']-1),h_diff=run['alpha']*params['h_diff'])
        assert run['params']==expected,run['source_path']
        assert run['temporal_replay_verified']
    evidence=read(ROOT/'results/evidence.json')
    with (ROOT/'results/evidence.csv').open(newline='') as f:
        exported=list(csv.DictReader(f))
    assert len(exported)==len(evidence)==35
    for row in evidence:
        assert all(row.values()),row['experiment_id']
        for file in row['source_file'].split('; '):assert (ROOT/file).is_file(),file
    annotations=check_annotations();results=check_results();transferred=verify_transfer()
    result=dict(all_checks_passed=True,own_indexed_files=len(own),submodules=pins,local_links_checked=links,canonical_answers=annotations['records'],original_label_files=annotations['original_files'],source_files_checked=transferred,primary_runs_with_unchanged_non_gate_parameters=len(runs),no_author_dataset_in_parent_index=True,no_ROMAN_dependency_or_import=True,adapted_mapping_and_GT_unchanged=True,evidence_rows=len(evidence),script_sha256=digest(Path(__file__)),scope='Local packaging and saved metrics; no clean installation or broad rerun')
    if args.output:
        if args.output.exists():raise ValueError('Choose a new check output')
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
