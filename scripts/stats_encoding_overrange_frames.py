"""Exact frame-level encoding exceedance on the policy train split and one rollout.

Both tactile sides are ORed per timestamp. Sparse queries can certify an
exceedance; all remaining candidate frames use the complete 224x224 field.
Training visibility is inferred from duplicate padding, not reconstructed UV.
"""
from pathlib import Path
from functools import lru_cache
import json
import csv
import time
import sys

import cv2
import h5py
import numpy as np
from scipy.spatial import Delaunay, cKDTree

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'encoder'))
from depth_marker_to_deform import _interpolation_map

RUN=ROOT/'policy_results/20261008-114326-417910781_starvla_groot_multi_task_all_encode_depth_deform_VAE_S_full_encoder'
OUT=ROOT/'.cache/encoding-overrange-frames'
SIZE=224
COUNTERS={'sparse_certified':0,'full_field_checked':0}


def weights_at(reference, query):
    tri=Delaunay(reference)
    simplex=tri.find_simplex(query);inside=simplex>=0
    width=min(8,len(reference))
    vertices=np.zeros((len(query),max(3,width)),np.int32)
    weights=np.zeros_like(vertices,dtype=np.float32)
    vertices[inside,:3]=tri.simplices[simplex[inside]]
    affine=tri.transform[simplex[inside]]
    xy=np.einsum('nij,nj->ni',affine[:,:2],query[inside]-affine[:,2,:])
    weights[inside,:2]=xy;weights[inside,2]=1-xy.sum(1)
    if (~inside).any():
        dist,idx=cKDTree(reference).query(query[~inside],k=width)
        inverse=1/np.maximum(dist,1e-6)**2
        vertices[~inside,:width]=idx
        weights[~inside,:width]=inverse/inverse.sum(1,keepdims=True)
    return vertices,weights


@lru_cache(maxsize=128)
def sparse_map(key,n,h,w):
    p=np.frombuffer(key,np.float32).reshape(n,2)
    ix=np.clip(np.rint(p[:,0]*SIZE/w-.5),0,SIZE-1).astype(int)
    iy=np.clip(np.rint(p[:,1]*SIZE/h-.5),0,SIZE-1).astype(int)
    x=(np.arange(SIZE,dtype=np.float32)+.5)*w/SIZE
    y=(np.arange(SIZE,dtype=np.float32)+.5)*h/SIZE
    return weights_at(p,np.column_stack((x[ix],y[iy])))


def deform_exceeds(marker,h,w,limit):
    p,q=np.asarray(marker,np.float32)
    delta=q-p
    # Convex linear/IDW combinations cannot exceed component bounds.
    if np.abs(delta).max() <= limit-1e-4:
        return False
    if len(np.unique(p,axis=0))<3:
        raise ValueError('Insufficient independent markers for the production interpolator')
    key=p.tobytes()
    vertices,weights=sparse_map(key,len(p),h,w)
    sparse=np.einsum('nvc,nv->nc',delta[vertices],weights)
    if np.any(np.abs(sparse)>limit+1e-4):
        COUNTERS['sparse_certified']+=1
        return True
    vertices,weights=_interpolation_map(key,len(p),h,w,SIZE)
    field=np.einsum('nvc,nv->nc',delta[vertices],weights)
    COUNTERS['full_field_checked']+=1
    return bool(np.any(np.abs(field)>limit))


def scan_sensor(g,indices=None,online=False):
    markers=g['marker'][:] if indices is None else g['marker'][indices]
    full=(g['marker_full'][:] if indices is None else g['marker_full'][indices]) if online else markers
    n=len(markers);h,w=g['depth'].shape[-2:]
    flags={k:np.zeros(n,bool) for k in ('outside','deficient','old_depth','new_depth','old_deform','new_deform')}
    if online:
        visible=g['marker_in_view'][:] if indices is None else g['marker_in_view'][indices]
        flags['outside']=(~visible).any(axis=1)
    else:
        p=markers[:,0]
        order=np.lexsort((p[:,:,1],p[:,:,0]),axis=1)
        sorted_p=np.take_along_axis(p,order[:,:,None],axis=1)
        flags['deficient']=np.any(np.all(sorted_p[:,1:]==sorted_p[:,:-1],axis=2),axis=1)
        current=markers[:,1]
        flags['outside']=((current[:,:,0]<=0)|(current[:,:,0]>=w)|(current[:,:,1]<=0)|(current[:,:,1]>=h)).any(axis=1)
    for start in range(0,n,32):
        selection=slice(start,min(start+32,n))
        press=g['press_depth'][selection] if indices is None else g['press_depth'][indices[selection]]
        peaks=press.max(axis=(1,2))
        for local in np.flatnonzero(peaks>3):
            resized=cv2.resize(press[local],(SIZE,SIZE),interpolation=cv2.INTER_AREA)
            flags['old_depth'][start+local]=bool(np.any(resized>4))
            flags['new_depth'][start+local]=bool(np.any(resized>3))
    for i in range(n):
        flags['old_deform'][i]=deform_exceeds(markers[i],h,w,20.)
        flags['new_deform'][i]=deform_exceeds(full[i],h,w,100.)
    flags['old_any']=flags['old_depth']|flags['old_deform']
    flags['new_any']=flags['new_depth']|flags['new_deform']
    return flags


def result(label,flags):
    total=len(next(iter(flags.values())))
    return dict(dataset=label,frames=total,
                counts={k:int(v.sum()) for k,v in flags.items()},
                percent={k:float(100*v.mean()) for k,v in flags.items()})


def finalize_training_inputs():
    """Exclude each trajectory's terminal frame, as UniVTACDataset does."""
    summary=json.loads((OUT/'summary.json').read_text())
    split=json.loads((RUN/'episodes.json').read_text())['train']
    all_valid=[];tasks=[];rows=[]
    for task,names in split.items():
        pieces=[]
        for name in names:
            with h5py.File(ROOT/'resources/data/isaac51'/task/'hdf5'/name) as f:
                valid=max(0,len(f['embodiment/joint'])-1)
            with np.load(OUT/f'{task}_{name}.npz') as f:
                flags={k:f[k][:valid] for k in f.files}
                assert len(f['outside'])==valid+1
            all_valid.append(flags);pieces.append(flags)
            r=result(task+'/'+name,flags)
            rows.append({'task':task,'episode':name,'frames':r['frames'],**r['counts']})
        combined={k:np.concatenate([x[k] for x in pieces]) for k in pieces[0]}
        tasks.append(result(task,combined))
    combined={k:np.concatenate([x[k] for x in all_valid]) for k in all_valid[0]}
    summary['policy_train_recorded_frames']=summary['policy_train']
    summary['per_task_recorded_frames']=summary['per_task']
    summary['policy_train']=result('policy_train_actual_input_frames',combined)
    summary['per_task']=tasks
    summary['notes'].append('Policy training results exclude the final frame of each episode, matching UniVTACDataset range(length - 1).')
    with (OUT/'actual_training_episodes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    columns=['数据','统计帧数','marker出界或缺点代理比例','超旧量程比例（depth4mm_deform20px）','超新量程比例（depth3mm_deform100px）','说明']
    table=[['encoder预训练数据','未提供','0%','0%','0%','沿用用户已有统计，本次未复核']]
    for key,label in [('policy_train','policy训练数据'),('rollout_action_rows','Pull out key rollout（300条记录）'),('rollout_unique','Pull out key rollout（19个独立输入）')]:
        r=summary[key];proxy='deficient' if key=='policy_train' else 'outside'
        table.append([label,r['frames'],f"{r['percent'][proxy]:.6f}%",f"{r['percent']['old_any']:.6f}%",f"{r['percent']['new_any']:.6f}%",'缺点/重复填充代理，实际出界坐标不可恢复' if key=='policy_train' else '完整marker；300条记录重复了19个独立输入'])
    with (OUT/'comparison_table.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(columns);w.writerows(table)
    with (OUT/'per_task.csv').open('w',newline='') as f:
        task_rows=[{'task':r['dataset'],'frames':r['frames'],**{k+'_count':v for k,v in r['counts'].items()},**{k+'_percent':v for k,v in r['percent'].items()}} for r in tasks]
        w=csv.DictWriter(f,fieldnames=list(task_rows[0]));w.writeheader();w.writerows(task_rows)
    return summary


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    split=json.loads((RUN/'episodes.json').read_text())['train']
    all_flags=[];rows=[];task_results=[];started=time.time();done=0
    total=sum(map(len,split.values()))
    for task,names in split.items():
        task_flags=[]
        for j,name in enumerate(names):
            with h5py.File(ROOT/'resources/data/isaac51'/task/'hdf5'/name) as f:
                a=scan_sensor(f['tactile/left_tactile'])
                b=scan_sensor(f['tactile/right_tactile'])
            if len(a['outside'])!=len(b['outside']):raise ValueError('Sides not aligned')
            flags={k:a[k]|b[k] for k in a}
            all_flags.append(flags);task_flags.append(flags)
            np.savez_compressed(OUT/f'{task}_{name}.npz',**flags)
            r=result(task+'/'+name,flags);rows.append({'task':task,'episode':name,'frames':r['frames'],**r['counts']})
            done+=1
            if (j+1)%10==0 or j+1==len(names):
                progress=dict(episodes_done=done,episodes_total=total,task=task,
                    task_episodes_done=j+1,frames_done=sum(len(x['outside']) for x in all_flags),
                    elapsed_seconds=round(time.time()-started,1),counters=COUNTERS)
                (OUT/'progress.json').write_text(json.dumps(progress,indent=2))
                print(json.dumps(progress),flush=True)
        combined={k:np.concatenate([x[k] for x in task_flags]) for k in task_flags[0]}
        task_results.append(result(task,combined))
    combined={k:np.concatenate([x[k] for x in all_flags]) for k in all_flags[0]}
    summary=dict(policy_train=result('policy_train',combined),per_task=task_results,
        encoder_pretraining={'outside_percent':0,'old_any_percent':0,'new_any_percent':0,'provenance':'Existing statistics supplied by user; not rescanned locally.'},
        thresholds={'old':{'indentation_mm':4,'deform_px':20},'new':{'indentation_mm':3,'deform_px':100}},
        notes=['Both sides ORed at each timestamp; any dense pixel/component exceeds means affected frame.',
               'Strict > thresholds; touching a range endpoint is not exceeding it.',
               'Training deficient is a duplicate-padding proxy for missing/out-of-view markers, not measured full geometry.',
               'Online legacy fields for old encoding, full recorded markers for new encoding.',
               '19 unique online inputs are repeated across 300 saved action rows; these are not 300 independent sensor observations.'])
    source=ROOT/'.cache/key-online-deform/full_marker_capture/policy_inputs/1000000_failed.hdf5'
    with h5py.File(source) as f:
        sources=f['source_action_index'][:]
        _,indices=np.unique(sources,return_index=True);indices.sort()
        a=scan_sensor(f['tactile/left_tactile'],indices,True)
        b=scan_sensor(f['tactile/right_tactile'],indices,True)
        flags={k:a[k]|b[k] for k in a}
        summary['rollout_unique']=result('rollout_unique',flags)
        lookup={int(sources[i]):j for j,i in enumerate(indices)}
        mapping=np.array([lookup[int(x)] for x in sources])
        expanded={k:v[mapping] for k,v in flags.items()}
        summary['rollout_action_rows']=result('rollout_action_rows',expanded)
        summary['rollout_source_actions']=sources[indices].tolist()
        summary['rollout_affected_source_actions']={k:sources[indices][v].tolist() for k,v in flags.items()}
        np.savez_compressed(OUT/'rollout_unique_flags.npz',source_action=sources[indices],**flags)
    with (OUT/'episodes.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    summary=finalize_training_inputs()
    print(json.dumps({k:v for k,v in summary.items() if k in ('policy_train','rollout_unique','rollout_action_rows')},indent=2),flush=True)


if __name__=='__main__':main()
