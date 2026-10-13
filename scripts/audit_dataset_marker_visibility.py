"""Audit stored marker coordinates and padding across the exact policy split.

Stored out-of-bounds points and missing observations are different quantities:
the current sensor filters out-of-frame points before padding. Exact repeated
reference/current pairs detect padding; its cause cannot be proven from HDF5.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import h5py
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN='20261008-114326-417910781_starvla_groot_multi_task_all_encode_depth_deform_VAE_S_full_encoder'


def write(path,rows):
    if not rows:
        return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def longest_run(mask):
    indices=np.flatnonzero(np.diff(np.r_[False,mask,False]))
    return int((indices[1::2]-indices[::2]).max()) if len(indices) else 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes',type=Path,default=ROOT/f'policy_results/{RUN}/episodes.json')
    parser.add_argument('--data',type=Path,default=ROOT/'resources/data/isaac51')
    parser.add_argument('--output',type=Path,default=ROOT/'.cache/marker-visibility-audit')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    splits=json.loads(args.episodes.read_text())
    sensors=[];affected=[];episodes=[]
    for split,tasks in splits.items():
        for task,names in tasks.items():
            for name in names:
                masks=[]
                with h5py.File(args.data/task/'hdf5'/name) as h5:
                    for side in ('left','right'):
                        g=h5[f'tactile/{side}_tactile'];m=g['marker'][:]
                        t,_,count,_=m.shape;h,w=g['depth'].shape[-2:]
                        assert m.shape[1]==2 and m.shape[-1]==2
                        uniq=[];pair_uniq=[];hist=Counter()
                        allzero=np.all(m==0,axis=(1,2,3))
                        finite=np.isfinite(m).all(axis=(1,2,3))
                        current=m[:,1]
                        out=(current[:,:,0]<=0)|(current[:,:,0]>=w)|(current[:,:,1]<=0)|(current[:,:,1]>=h)
                        edge=(current[:,:,0]==0)|(current[:,:,0]==w-1)|(current[:,:,1]==0)|(current[:,:,1]==h-1)
                        for frame in range(t):
                            n=0 if allzero[frame] else len(np.unique(m[frame,0],axis=0))
                            paired=0 if allzero[frame] else len(np.unique(np.column_stack((m[frame,0],m[frame,1])),axis=0))
                            uniq.append(n);pair_uniq.append(paired);hist[n]+=1
                            if n<count or out[frame].any() or not finite[frame]:
                                affected.append(dict(split=split,task=task,episode=name,side=side,frame=frame,
                                                     unique_reference_points=n,unique_pairs=paired,
                                                     returned_slots=count,padding_or_empty_slots=count-paired,
                                                     stored_outside_slots=int(out[frame].sum()),
                                                     all_zero=bool(allzero[frame]),finite=bool(finite[frame])))
                        uniq=np.array(uniq);pair_uniq=np.array(pair_uniq)
                        deficient=uniq<count;masks.append(deficient)
                        ref=m[:,0].reshape(-1,2)
                        ref=ref[np.isfinite(ref).all(axis=1)&(ref!=0).any(axis=1)]
                        union=len(np.unique(ref,axis=0))
                        sensors.append(dict(split=split,task=task,episode=name,side=side,
                                            sensor_frames=t,returned_slots=count,
                                            deficient_frames=int(deficient.sum()),min_unique=int(uniq.min()),
                                            padding_or_empty_slots=int((count-pair_uniq).sum()),
                                            reference_pair_unique_count_mismatch_frames=int((uniq!=pair_uniq).sum()),
                                            stored_outside_frames=int(out.any(axis=1).sum()),
                                            stored_outside_slots=int(out.sum()),
                                            exact_boundary_slots=int(edge.sum()),all_zero_frames=int(allzero.sum()),
                                            nonfinite_frames=int((~finite).sum()),
                                            distinct_reference_coordinates_across_episode=union,
                                            longest_deficient_run=longest_run(deficient),
                                            unique_count_histogram=json.dumps(dict(sorted(hist.items())))))
                    assert masks[0].shape==masks[1].shape
                    either=masks[0]|masks[1];both=masks[0]&masks[1]
                    episodes.append(dict(split=split,task=task,episode=name,frames=len(either),
                                         either_side_deficient_frames=int(either.sum()),
                                         both_sides_deficient_frames=int(both.sum())))
            print(split,task,len(names),'episodes',flush=True)
    groups=[]
    for task in [*sorted(set(r['task'] for r in sensors)),'ALL']:
        for split in ('train','val'):
            rows=[r for r in sensors if r['split']==split and (task=='ALL' or r['task']==task)]
            ep=[r for r in episodes if r['split']==split and (task=='ALL' or r['task']==task)]
            if not rows:continue
            sf=sum(r['sensor_frames'] for r in rows);ef=sum(r['frames'] for r in ep)
            deficient=sum(r['deficient_frames'] for r in rows)
            hist=Counter()
            for r in rows:hist.update({int(k):v for k,v in json.loads(r['unique_count_histogram']).items()})
            groups.append(dict(task=task,split=split,episodes=len(ep),
                               affected_episodes=sum(r['either_side_deficient_frames']>0 for r in ep),
                               trajectory_frames=ef,either_side_deficient_frames=sum(r['either_side_deficient_frames'] for r in ep),
                               either_side_deficient_percent=100*sum(r['either_side_deficient_frames'] for r in ep)/ef,
                               sensor_frames=sf,deficient_sensor_frames=deficient,deficient_sensor_percent=100*deficient/sf,
                               min_unique_markers=min(r['min_unique'] for r in rows),
                               padding_or_empty_slots=sum(r['padding_or_empty_slots'] for r in rows),
                               stored_outside_frames=sum(r['stored_outside_frames'] for r in rows),
                               stored_outside_slots=sum(r['stored_outside_slots'] for r in rows),
                               all_zero_frames=sum(r['all_zero_frames'] for r in rows),
                               nonfinite_frames=sum(r['nonfinite_frames'] for r in rows),
                               unique_count_histogram=dict(sorted(hist.items()))))
    write(args.output/'episode_sensors.csv',sensors)
    write(args.output/'episodes.csv',episodes)
    write(args.output/'affected_frames.csv',affected)
    summary=dict(episodes_file=str(args.episodes.resolve()),data_root=str(args.data.resolve()),
                 groups=groups,notes=['Full scan of every frame in the saved policy train/val split.',
                                     'Sensor-frame means one side at one trajectory timestep.',
                                     'Unique reference versus unique pair counts are cross-checked.',
                                     'Padding detects insufficient distinct returned points, not the exact cause of point loss.',
                                     'Filtered out-of-view coordinates cannot be recovered from padding.',
                                     'All-zero marker frames are treated as zero valid points, not one.'])
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
