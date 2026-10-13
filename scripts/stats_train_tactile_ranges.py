"""Full-frame per-task training indentation and deduplicated marker dx/dy ranges.

Depth quantiles describe each SENSOR FRAME's maximum positive indentation.
Displacement quantiles describe stored DISTINCT marker observations, not dense
pixels. Missing out-of-view marker coordinates in old HDF5 are not recoverable.
"""
import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN='20261008-114326-417910781_starvla_groot_multi_task_all_encode_depth_deform_VAE_S_full_encoder'
PERCENTILES=[50,90,95,99,99.9,100]


def write_csv(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def summarize(task,side,items):
    u=np.concatenate([r['displacement'] for r in items])
    maxima=np.concatenate([r['press_max'] for r in items])
    row=dict(task=task,side=side,episodes=len(set(r['episode'] for r in items)),
             sensor_frames=len(maxima),distinct_marker_observations=len(u),
             duplicate_padding_slots_removed=sum(r['removed'] for r in items),
             indentation_min_mm=min(r['press_min'] for r in items),indentation_max_mm=float(maxima.max()),
             frame_peak_indentation_mean_mm=float(maxima.mean()))
    for percentile,value in zip(PERCENTILES,np.percentile(maxima,PERCENTILES)):
        row[f'frame_peak_indentation_p{percentile:g}_mm']=float(value)
    for axis,index in [('dx',0),('dy',1)]:
        values=u[:,index]
        row[axis+'_min_px']=float(values.min());row[axis+'_max_px']=float(values.max())
        row[axis+'_mean_px']=float(values.mean())
        for percentile,value in zip(PERCENTILES,np.percentile(np.abs(values),PERCENTILES)):
            row[f'{axis}_abs_p{percentile:g}_px']=float(value)
        for limit in [20,40,80,160,240]:
            row[f'{axis}_outside_pm{limit}_percent']=float(100*np.mean(np.abs(values)>limit))
    for limit in [1,1.5,2,3,4,5]:
        row[f'frames_indentation_above_{limit:g}mm_percent']=float(100*np.mean(maxima>limit))
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes',type=Path,default=ROOT/f'policy_results/{RUN}/episodes.json')
    p.add_argument('--data',type=Path,default=ROOT/'resources/data/isaac51')
    p.add_argument('--output',type=Path,default=ROOT/'.cache/train-tactile-ranges')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    train=json.loads(args.episodes.read_text())['train']
    summaries=[];episode_rows=[];extrema=[];overall=[]
    for task,names in train.items():
        items=[]
        for name in names:
            with h5py.File(args.data/task/'hdf5'/name) as f:
                for side in ('left','right'):
                    g=f[f'tactile/{side}_tactile'];marker=g['marker'][:]
                    if not np.isfinite(marker).all():raise ValueError(f'Nonfinite marker: {task}/{name}/{side}')
                    displacement=marker[:,1]-marker[:,0]
                    order=np.lexsort((marker[:,0,:,1],marker[:,0,:,0]),axis=1)
                    reference=np.take_along_axis(marker[:,0],order[:,:,None],axis=1)
                    sorted_u=np.take_along_axis(displacement,order[:,:,None],axis=1)
                    keep=np.ones(reference.shape[:2],dtype=bool)
                    keep[:,1:]=(reference[:,1:]!=reference[:,:-1]).any(axis=2)
                    allzero=np.all(marker==0,axis=(1,2,3));keep[allzero]=False
                    if np.any((~keep[:,1:])&(np.abs(sorted_u[:,1:]-sorted_u[:,:-1])>1e-5).any(axis=2)):
                        raise ValueError('Duplicate reference coordinates have different displacements')
                    unique_u=sorted_u[keep]
                    count=len(marker);maximum=np.empty(count,np.float32);minimum=np.empty(count,np.float32)
                    for start in range(0,count,32):
                        chunk=g['press_depth'][start:start+32]
                        if not np.isfinite(chunk).all():raise ValueError(f'Nonfinite press depth: {task}/{name}/{side}')
                        maximum[start:start+len(chunk)]=chunk.max(axis=(1,2))
                        minimum[start:start+len(chunk)]=chunk.min(axis=(1,2))
                    if minimum.min()<0:raise ValueError('press_depth is not positive indentation')
                    removed=int(keep.size-keep.sum())
                    item=dict(side=side,episode=name,displacement=unique_u,press_max=maximum,
                              press_min=float(minimum.min()),removed=removed)
                    items.append(item)
                    episode_rows.append(dict(task=task,episode=name,side=side,sensor_frames=count,
                                             distinct_marker_observations=len(unique_u),padding_slots_removed=removed,
                                             indentation_min_mm=item['press_min'],indentation_max_mm=float(maximum.max()),
                                             dx_min_px=float(unique_u[:,0].min()),dx_max_px=float(unique_u[:,0].max()),
                                             dy_min_px=float(unique_u[:,1].min()),dy_max_px=float(unique_u[:,1].max())))
                    peak=int(maximum.argmax());press=g['press_depth'][peak];py,px=np.unravel_index(press.argmax(),press.shape)
                    extrema.append(dict(task=task,episode=name,side=side,quantity='indentation_max',
                                        value=float(maximum[peak]),unit='mm',frame=peak,image_xy=[int(px),int(py)]))
                    for axis,index in [('dx',0),('dy',1)]:
                        for label,func in [('min',np.argmin),('max',np.argmax)]:
                            frame,slot=np.unravel_index(func(displacement[:,:,index]),displacement[:,:,index].shape)
                            extrema.append(dict(task=task,episode=name,side=side,quantity=axis+'_'+label,
                                                value=float(displacement[frame,slot,index]),unit='px',frame=int(frame),
                                                legacy_slot=int(slot),reference_xy=marker[frame,0,slot].tolist(),
                                                current_xy=marker[frame,1,slot].tolist()))
        for side in ('left','right','both'):
            subset=items if side=='both' else [r for r in items if r['side']==side]
            summaries.append(summarize(task,side,subset))
        overall.extend(items)
        print('DONE',task,len(names),'episodes',sum(len(r['press_max']) for r in items),'sensor frames',flush=True)
    for side in ('left','right','both'):
        subset=overall if side=='both' else [r for r in overall if r['side']==side]
        row=summarize('ALL',side,subset)
        row['episodes']=sum(len(names) for names in train.values())
        summaries.append(row)
    write_csv(args.output/'task_side_summary.csv',summaries)
    combined=[r for r in summaries if r['side']=='both']
    write_csv(args.output/'task_summary.csv',combined)
    write_csv(args.output/'episode_summary.csv',episode_rows)
    peaks=[]
    for task in train:
        for quantity in ('indentation_max','dx_min','dx_max','dy_min','dy_max'):
            subset=[r for r in extrema if r['task']==task and r['quantity']==quantity]
            peaks.append((min if quantity.endswith('_min') else max)(subset,key=lambda r:r['value']))
    (args.output/'extrema.json').write_text(json.dumps(peaks,indent=2))
    summary=dict(episodes_file=str(args.episodes.resolve()),data_root=str(args.data.resolve()),split='train',
                 quantile_definitions=dict(depth='Quantiles over each sensor frame maximum of saved press_depth, mm.',
                                           dx_dy='Absolute-value quantiles over distinct stored marker displacements, px.'),
                 weighting='Frame-weighted depth and marker-observation-weighted displacement; all frames scanned.',
                 notes=['No validation or online evaluation observations included.',
                        'Only old HDF5 stored points; historically removed out-of-view coordinates cannot be recovered.',
                        'Exact reference-coordinate duplicate padding removed before displacement distribution statistics.',
                        'dx/dy are marker values, not dense-field pixel quantiles. Original linear/IDW weights are nonnegative, so marker component extrema bound the interpolated field, apart from floating-point roundoff.'],
                 groups=summaries)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    rows=[r for r in combined if r['task']!='ALL'];labels=[r['task'] for r in rows];x=np.arange(len(rows))
    fig,axes=plt.subplots(1,3,figsize=(18,5.5),layout='constrained')
    axes[0].bar(x-.18,[r['frame_peak_indentation_p99_mm'] for r in rows],.36,label='P99 frame peak')
    axes[0].bar(x+.18,[r['indentation_max_mm'] for r in rows],.36,label='Maximum')
    axes[0].set_ylabel('Positive indentation (mm)');axes[0].legend()
    for ax,axis in zip(axes[1:],('dx','dy')):
        ax.bar(x-.25,[r[axis+'_abs_p99_px'] for r in rows],.25,label='P99 |'+axis+'|')
        ax.bar(x,[r[axis+'_abs_p99.9_px'] for r in rows],.25,label='P99.9 |'+axis+'|')
        ax.bar(x+.25,[r[axis+'_abs_p100_px'] for r in rows],.25,label='Maximum |'+axis+'|')
        ax.set_ylabel('Marker displacement (px)');ax.legend()
    for ax in axes:
        ax.set_xticks(x,labels,rotation=35,ha='right');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Training only | full scan | both tactile sensors | depth frame peaks and distinct marker displacements')
    fig.savefig(args.output/'per_task_ranges.png',dpi=170,bbox_inches='tight');plt.close(fig)
    print('SAVED',args.output,flush=True)


if __name__=='__main__':main()
