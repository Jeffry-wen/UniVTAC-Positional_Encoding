"""Compare +/-20, +/-80, +/-160, +/-240px frame exceedance, holding marker source fixed."""
import csv
import json
from pathlib import Path
import time

import h5py
import numpy as np
from scripts.stats_encoding_overrange_frames import ROOT,RUN,OUT as PREVIOUS,deform_exceeds

OUT=ROOT/'.cache/deform-threshold-frames'
LIMITS=(20,80,160,240)


def summarize(name,flags,outside,proxy=False):
    n=len(outside)
    return dict(dataset=name,frames=n,marker_outside_count=int(outside.sum()),
        marker_outside_percent=float(outside.mean()*100),marker_outside_is_proxy=proxy,
        deform_exceeded_counts={str(k):int(flags[k].sum()) for k in LIMITS},
        deform_exceeded_percent={str(k):float(flags[k].mean()*100) for k in LIMITS})


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    split=json.loads((RUN/'episodes.json').read_text())['train']
    totals={k:[] for k in LIMITS};outside_parts=[];tasks=[];done=0;started=time.time()
    for task,names in split.items():
        pieces={k:[] for k in LIMITS};missing=[]
        for name in names:
            with h5py.File(ROOT/'resources/data/isaac51'/task/'hdf5'/name) as f:
                n=len(f['embodiment/joint'])-1
                flags={k:np.zeros(n,bool) for k in LIMITS}
                with np.load(PREVIOUS/f'{task}_{name}.npz') as cached:
                    flags[20]=cached['old_deform'][:n].copy()
                    deficient=cached['deficient'][:n].copy()
                for side in ('left','right'):
                    g=f[f'tactile/{side}_tactile'];m=np.asarray(g['marker'][:n],np.float32)
                    h,w=g['depth'].shape[-2:]
                    peaks=np.abs(m[:,1]-m[:,0]).max(axis=(1,2))
                    for limit in LIMITS[1:]:
                        for i in np.flatnonzero(peaks>limit-1e-4):
                            if not flags[limit][i]:
                                flags[limit][i]=deform_exceeds(m[i],h,w,float(limit))
            assert all(np.all(~flags[hi]|flags[lo]) for lo,hi in zip(LIMITS,LIMITS[1:]))
            for k in LIMITS:pieces[k].append(flags[k]);totals[k].append(flags[k])
            missing.append(deficient);outside_parts.append(deficient)
            np.savez_compressed(OUT/f'{task}_{name}.npz',**{f'limit_{k}':v for k,v in flags.items()},marker_missing_proxy=deficient)
            done+=1
            if done%30==0:
                p=dict(episodes_done=done,episodes_total=720,task=task,elapsed_seconds=round(time.time()-started,1))
                (OUT/'progress.json').write_text(json.dumps(p,indent=2));print(json.dumps(p),flush=True)
        tasks.append(summarize(task,{k:np.concatenate(pieces[k]) for k in LIMITS},np.concatenate(missing),True))
    summary=dict(policy_train=summarize('policy_train',{k:np.concatenate(totals[k]) for k in LIMITS},np.concatenate(outside_parts),True),
                 per_task=tasks,limits_px=LIMITS,
                 encoder_pretraining={'marker_outside_percent':0,'deform_exceeded_percent':{'20':7.648,'80':None,'160':None,'240':None},'provenance':'20px and visibility supplied by user; remaining encoder thresholds not computed at user request.'})
    path=ROOT/'.cache/key-online-deform/full_marker_capture/policy_inputs/1000000_failed.hdf5'
    with h5py.File(path) as f:
        sources=f['source_action_index'][:];_,indices=np.unique(sources,return_index=True);indices.sort()
        n=len(indices);flags={k:np.zeros(n,bool) for k in LIMITS};outside=np.zeros(n,bool)
        for side in ('left','right'):
            g=f[f'tactile/{side}_tactile'];markers=g['marker_full'][indices]
            outside |= (~g['marker_in_view'][indices]).any(axis=1)
            h,w=g['depth'].shape[-2:]
            for i,m in enumerate(markers):
                for limit in LIMITS:flags[limit][i] |= deform_exceeds(m,h,w,float(limit))
        summary['rollout_unique']=summarize('rollout_unique',flags,outside)
        summary['rollout_source_actions']=sources[indices].tolist()
        summary['rollout_exceeded_source_actions']={str(k):sources[indices][flags[k]].tolist() for k in LIMITS}
        np.savez_compressed(OUT/'rollout_unique.npz',source_action=sources[indices],marker_outside=outside,**{f'limit_{k}':v for k,v in flags.items()})
    summary['notes']=['Both tactile sides ORed per input timestamp; any interpolated dx or dy component strictly exceeds the threshold.',
        'Policy training uses existing saved markers and excludes terminal frames, matching the training loader.',
        'Policy marker out-of-view statistic is the duplicate-padding/missing-marker proxy; actual removed positions unavailable.',
        'All four rollout ranges use the same full-marker field; 19 distinct inference observations, not 300 repeated action rows.',
        'Depth is not included in any deform exceedance column.',
        'Existing exact 20px policy interpolation flags reused; 80/160/240px evaluated with the same production interpolation.']
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    cols=['数据','帧数','marker出界','Deform超出±20px','Deform超出±80px','Deform超出±160px','Deform超出±240px']
    rows=[['encoder预训练数据','未提供','0%','7.648%','—','—','—']]
    for key,label in [('policy_train','policy训练数据'),('rollout_unique','仿真rollout（pull out key）')]:
        r=summary[key];rows.append([label,r['frames'],f"{r['marker_outside_percent']:.3f}%",*[f"{r['deform_exceeded_percent'][str(k)]:.3f}%" for k in LIMITS]])
    with (OUT/'comparison_table.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(cols);w.writerows(rows)
    with (OUT/'per_task.csv').open('w',newline='') as f:
        data=[{'task':r['dataset'],'frames':r['frames'],'marker_missing_proxy_count':r['marker_outside_count'],
               **{f'deform_exceeds_{k}px_count':r['deform_exceeded_counts'][str(k)] for k in LIMITS},
               **{f'deform_exceeds_{k}px_percent':r['deform_exceeded_percent'][str(k)] for k in LIMITS}} for r in tasks]
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    print(json.dumps({k:summary[k] for k in ('policy_train','rollout_unique')},indent=2),flush=True)


if __name__=='__main__':main()
