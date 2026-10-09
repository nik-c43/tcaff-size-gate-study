"""Recheck the existing 80-case ideal model with pinned upstream descriptor/gate.

No author data, network inference, CLIPPER solve or temporal-filter evaluation.
"""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from inspect_saved import ROOT, read, digest
sys.path[:0]=[str(ROOT/'vendor/tcaff'),str(ROOT/'src')]
from model_geometry import ideal_view,hidden_side,describe,object_map,gate_details
from tcaff.tcaff.tcaff_manager import TCAFFManager

def compare(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        return max((compare(a[k],b[k]) for k in a),default=0.)
    if isinstance(a,list):
        assert len(a)==len(b)
        return max((compare(x,y) for x,y in zip(a,b)),default=0.)
    if isinstance(a,(int,float)) and not isinstance(a,bool):
        assert abs(a-b)<1e-12,(a,b)
        return float(abs(a-b))
    assert a==b,(a,b)
    return 0.

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);args=p.parse_args()
    if args.output and args.output.exists():raise ValueError('Choose a new output file')
    for name,pin in [('tcaff','ff4ceab04b03fecabc4de9c9cbe7dbba4ab50df9'),('clipper','e514dc29c273837ffdfeebbefbdcb2a93d970969')]:
        assert subprocess.check_output(['git','-C',str(ROOT/'vendor'/name),'rev-parse','HEAD'],text=True).strip()==pin
    functions={node.name:ast.dump(node,include_attributes=False) for node in ast.parse((ROOT/'src/model_geometry.py').read_text()).body if isinstance(node,ast.FunctionDef)}
    origins={'size_gate_viewpoint_diagnosis.py':['ideal_view'],'size_gate_occlusion_diagnosis.py':['hidden_side','describe'],'h1_map_audit.py':['object_map'],'size_gate_prepare.py':['gate_details']}
    for file,names in origins.items():
        original={node.name:ast.dump(node,include_attributes=False) for node in ast.parse((ROOT/'scripts/original'/file).read_text()).body if isinstance(node,ast.FunctionDef)}
        for name in names:assert functions[name]==original[name],name
    kwargs={k:v for k,v in read(ROOT/'configs/manager.json').items() if k!='ts'}
    managers={a:TCAFFManager(**dict(kwargs,wh_scale_diff=1+a*(kwargs['wh_scale_diff']-1),h_diff=a*kwargs['h_diff'])) for a in [1.,2.]}
    K=np.array([[600.,0.,320.],[0.,600.,240.],[0.,0.,1.]])
    rows=[]
    for name,sizes in {'cube':[.4,.4,.4],'wide_box':[1.2,.4,.8]}.items():
        for angle in [0,45]:
            _,mask,depth=ideal_view(sizes,angle,K,(480,640),3.)
            full=describe(mask,depth,K)
            for axis in ['left','top']:
                for percentage in range(0,91,10):
                    fraction=percentage/100.;visible=hidden_side(mask,fraction,axis);current=describe(visible,depth,K)
                    shift=np.asarray(current['centroid_camera_RDF_m'])-full['centroid_camera_RDF_m'];gates={}
                    for alpha,manager in managers.items():
                        actual=len(manager.get_putative_assoc(object_map([full['map_row']]),object_map([current['map_row']])))==1
                        detail=gate_details(full['map_row'],current['map_row'],manager.wh_scale_diff,manager.h_diff)
                        assert actual==detail['admitted'];gates[str(alpha)]=detail
                    rows.append(dict(object=name,dimensions_horizontal_depth_height_m=sizes,angle_deg=angle,hidden_side=axis,requested_bbox_occlusion_fraction=fraction,actual_mask_area_removed_fraction=1-current['visible_pixels']/full['visible_pixels'],reference=full,occluded=current,centroid_shift_RDF_m=shift.tolist(),centroid_shift_norm_m=float(np.linalg.norm(shift)),level_camera_horizontal_plane_shift_m=float(np.linalg.norm(shift[[0,2]])),gates=gates))
    error=compare(rows,read(ROOT/'results/occlusion/cases.json'))
    result=dict(cases=len(rows),controls=sum(r['requested_bbox_occlusion_fraction']==0 for r in rows),max_numeric_difference=error,geometry_function_AST_unchanged=True,scope='Existing ideal descriptor/gate model recheck, not a new research series',script_sha256=digest(Path(__file__)),saved_cases_sha256=digest(ROOT/'results/occlusion/cases.json'),all_checks_passed=True)
    if args.output:args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
