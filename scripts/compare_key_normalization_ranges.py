"""Static range sweep on recorded out-of-view marker states, no rollout changes.

G/B = clip(displacement / (2*L) + 0.5, 0, 1), L in original-image pixels.
R is copied unchanged. Unclipped old/full displacement fields are held fixed.
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


def encode(r,field,limit):
    return np.concatenate((r[None],np.clip(field/(2*limit)+.5,0,1)),axis=0).transpose(1,2,0)


def stats(field,encoded,limit):
    decoded=(encoded[:,:,1:].transpose(2,0,1)-.5)*(2*limit)
    return dict(clipped_component_fraction=float((np.abs(field)>limit).mean()),
                any_component_clipped_pixel_fraction=float((np.abs(field)>limit).any(axis=0).mean()),
                clipping_mean_epe_px=float(np.linalg.norm(field-decoded,axis=0).mean()),
                clipping_max_epe_px=float(np.linalg.norm(field-decoded,axis=0).max()),
                exactly_black_pixel_fraction=float((encoded==0).all(axis=-1).mean()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hdf5',type=Path,default=ROOT/'exports/pull_out_key_marker2depth_seed1000000/episode.h5')
    p.add_argument('--output',type=Path,default=ROOT/'.cache/key-online-deform/normalization_sweep')
    p.add_argument('--half-ranges',type=float,nargs='+',default=[20,40,80,160,240])
    p.add_argument('--source-actions',type=int,nargs='+',default=[64,80,112,176])
    args=p.parse_args()
    if any(x<=0 for x in args.half_ranges):p.error('Range limits must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[];case_data=[]
    with h5py.File(args.hdf5) as f:
        sources=f['frames/source_action_index'][:]
        for source in args.source_actions:
            match=np.flatnonzero(sources==source)
            if len(match)!=1:raise ValueError(f'Source action {source} not recorded')
            i=int(match[0]);g=f['frames/tactile/right_tactile'];d=f['derived/right']
            marker=g['marker_full'][i];rgb=g['rgb_marker'][i]
            outside=int((~g['marker_in_view'][i]).sum())
            if not outside:raise ValueError(f'Source action {source} has no out-of-view markers')
            old=d['displacement_old_raw'][i];full=d['displacement_new_raw'][i]
            r=d['depth_deform_old'][i,0]
            np.testing.assert_array_equal(r,d['depth_deform_new'][i,0])
            images={}
            for limit in args.half_ranges:
                a,b=encode(r,old,limit),encode(r,full,limit)
                if limit==20:
                    np.testing.assert_allclose(a,d['depth_deform_old'][i].transpose(1,2,0),atol=1e-6,rtol=0)
                    np.testing.assert_allclose(b,d['depth_deform_new'][i].transpose(1,2,0),atol=1e-6,rtol=0)
                np.testing.assert_array_equal(a[:,:,0],r)
                np.testing.assert_array_equal(b[:,:,0],r)
                images[limit]=(a,b)
                for mode,field,image in [('old',old,a),('full',full,b)]:
                    rows.append(dict(source_action=source,side='right',outside_markers=outside,
                                     marker_count=marker.shape[1],representation=mode,half_range_px=limit,
                                     raw_max_abs_component_px=float(np.abs(field).max()),
                                     old_new_encoded_gb_mean_abs_difference=float(np.abs(a[:,:,1:]-b[:,:,1:]).mean()),
                                     **stats(field,image,limit)))
            case_data.append(dict(source=source,outside=outside,marker=marker,rgb=rgb,old=old,full=full,images=images))
    count=len(args.half_ranges)
    # Compact overview: fixed full-marker fields, only normalization changes.
    fig,axes=plt.subplots(len(case_data),count+1,figsize=(3*(count+1),3.3*len(case_data)),squeeze=False,layout='constrained')
    for row,c in enumerate(case_data):
        axes[row,0].imshow(c['rgb']);axes[row,0].set_title(f'Marker_RGB | action {c["source"]}\n{c["outside"]}/63 outside image',fontsize=11)
        for col,limit in enumerate(args.half_ranges,1):
            image=c['images'][limit][1];s=stats(c['full'],image,limit)
            axes[row,col].imshow(image)
            axes[row,col].set_title(f'Full-marker deform +/-{limit:g} px',fontsize=10)
            axes[row,col].set_xlabel(f'Clipped G/B components: {100*s["clipped_component_fraction"]:.1f}%\nClipping EPE: {s["clipping_mean_epe_px"]:.2f} px',fontsize=9)
    for ax in axes.ravel():ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle('Pull Out Key | right tactile | fixed full-marker interpolation, varying G/B range only\nG/B = clip(displacement / (2L) + 0.5). R unchanged. +/-20 px is the existing baseline.',fontsize=13)
    fig.savefig(args.output/'full_marker_ranges_overview.png',dpi=180,bbox_inches='tight');plt.close(fig)
    # Per-case old/full pairs isolate whether clipping hides added-marker effects.
    for c in case_data:
        fig,axes=plt.subplots(2,count+1,figsize=(3.2*(count+1),7.3),squeeze=False,layout='constrained')
        axes[0,0].imshow(c['rgb']);axes[0,0].set_title('Same-state Marker_RGB',fontsize=11)
        marker=c['marker'];u=marker[1]-marker[0]
        axes[1,0].scatter(marker[0,:,0],marker[0,:,1],s=7,c='black')
        q=axes[1,0].quiver(marker[0,:,0],marker[0,:,1],u[:,0],u[:,1],np.linalg.norm(u,axis=1),
                          angles='xy',scale_units='xy',scale=4,pivot='tail',cmap='viridis',width=.004)
        axes[1,0].set(xlim=(0,320),ylim=(240,0),aspect='equal')
        axes[1,0].set_title('All 63 measured displacement arrows\nDisplay length = 0.25 x displacement',fontsize=10)
        for col,limit in enumerate(args.half_ranges,1):
            for row,mode in enumerate(('old','full')):
                image=c['images'][limit][row];field=c['old'] if mode=='old' else c['full'];s=stats(field,image,limit)
                axes[row,col].imshow(image)
                axes[row,col].set_title(f'{mode.upper()} +/-{limit:g} px',fontsize=11)
                axes[row,col].set_xlabel(f'Clipped components: {100*s["clipped_component_fraction"]:.1f}%\nClipping EPE: {s["clipping_mean_epe_px"]:.2f} px',fontsize=9)
        for ax in axes.ravel():ax.set_xticks([]);ax.set_yticks([])
        fig.suptitle(f'Pull Out Key | action {c["source"]} | right | {c["outside"]}/63 markers outside image\n'
                     'Rows: old filtered/padded field versus full-marker field. Same R and normalization within each column.',fontsize=13)
        fig.savefig(args.output/f'action_{c["source"]:03d}_old_full_ranges.png',dpi=180,bbox_inches='tight');plt.close(fig)
        np.savez_compressed(args.output/f'action_{c["source"]:03d}_range_inputs.npz',
                            half_ranges_px=np.asarray(args.half_ranges),
                            old=np.stack([c['images'][r][0].transpose(2,0,1) for r in args.half_ranges]),
                            full=np.stack([c['images'][r][1].transpose(2,0,1) for r in args.half_ranges]),
                            old_raw=c['old'],full_raw=c['full'],marker_full=c['marker'])
    with (args.output/'metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=dict(source_hdf5=str(args.hdf5.resolve()),source_actions=args.source_actions,
                 half_ranges_px=args.half_ranges,formula='G/B = clip(u / (2*L) + 0.5, 0, 1)',
                 depth_channel='R copied unchanged from original representation',metrics=rows,
                 note='Clipping EPE compares the raw interpolated field to its normalized-and-decoded version. It is not interpolation ground-truth error. Static visualization only; policy/encoder unchanged. Broader ranges reduce response to small displacement. Archive unchanged.')
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print('Saved',args.output)
    for source in args.source_actions:
        print('Action',source)
        for r in rows:
            if r['source_action']==source and r['representation']=='full':
                print(f'  +/-{r["half_range_px"]:g}: clip {100*r["clipped_component_fraction"]:.2f}%, EPE {r["clipping_mean_epe_px"]:.2f}px')


if __name__=='__main__':main()
