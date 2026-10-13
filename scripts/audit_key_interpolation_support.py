"""Separate raw interpolation, duplicate padding, and extrapolation effects.

Uses online raw marker arrays. Removes all copies of each held-out marker in
LOO. Dense differences do not identify a physically correct dense field.
"""
import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import ConvexHull,Delaunay,cKDTree

from analyze_key_deform import ROOT,linear_weights,loo_maps,write_csv
from analyze_key_online_inputs import arrays,grid


def compute(h5,index):
    raw_marker=h5['tactile/right_tactile/marker'][index]
    full_p=raw_marker[0].astype(np.float32)
    full_u=raw_marker[1]-full_p
    p,u,current,decoded,actual,rgb,depth=arrays(h5,index,'right')
    p=p.astype(np.float32)
    xx,yy=grid();query=np.column_stack((xx.ravel(),yy.ravel()))
    inside=(Delaunay(p).find_simplex(query)>=0).reshape(224,224)
    dedup=(linear_weights(p,query)@u).reshape(224,224,2)
    tps=RBFInterpolator(p/320,u,kernel='thin_plate_spline',smoothing=0)(query/320).reshape(224,224,2)
    padding_difference=np.linalg.norm(current-dedup,axis=-1)
    assert padding_difference[inside].max()<1e-3, 'Deduplication unexpectedly changed interior interpolation'
    _,inverse=np.unique(full_p,axis=0,return_inverse=True)
    _,neighbors=cKDTree(full_p).query(query[~inside.ravel()],k=8)
    identities=np.sort(inverse[neighbors],axis=1)
    neighbor_count=1+(np.diff(identities,axis=1)!=0).sum(axis=1)
    # Production-style LOO: remove ALL duplicated copies of the held-out location.
    predictions=[]
    for point in p:
        keep=(full_p!=point).any(axis=1)
        predictions.append((linear_weights(full_p[keep],point[None])@full_u[keep])[0])
    predictions=np.asarray(predictions)
    loo,boundary=loo_maps(p.tobytes(),len(p))
    pred_dedup=loo['linear_idw8']@u
    pred_tps=loo['thin_plate']@u
    err=np.linalg.norm(predictions-u,axis=1)
    metrics=dict(source_action=int(h5['source_action_index'][index]),unique_markers=len(p),
                 marker_disp_max_px=float(np.linalg.norm(u,axis=1).max()),
                 extrapolated_pixel_fraction=float((~inside).mean()),
                 outside_fewer_than_8_unique_neighbors_fraction=float((neighbor_count<8).mean()),
                 outside_only_1_unique_neighbor_fraction=float((neighbor_count==1).mean()),
                 outside_mean_unique_neighbor_count=float(neighbor_count.mean()),
                 padding_effect_inside_max_px=float(padding_difference[inside].max()),
                 padding_effect_outside_mean_px=float(padding_difference[~inside].mean()),
                 padding_effect_outside_max_px=float(padding_difference[~inside].max()),
                 current_vs_tps_inside_mean_px=float(np.linalg.norm(current-tps,axis=-1)[inside].mean()),
                 current_vs_tps_outside_mean_px=float(np.linalg.norm(current-tps,axis=-1)[~inside].mean()),
                 current_loo_mean_epe_px=float(err.mean()),current_loo_max_epe_px=float(err.max()),
                 dedup_loo_mean_epe_px=float(np.linalg.norm(pred_dedup-u,axis=1).mean()),
                 tps_loo_mean_epe_px=float(np.linalg.norm(pred_tps-u,axis=1).mean()))
    for name,pred in [('current',predictions),('dedup',pred_dedup),('tps',pred_tps)]:
        error=np.linalg.norm(pred-u,axis=1)
        metrics[name+'_loo_interior_epe_px']=float(error[~boundary].mean())
        metrics[name+'_loo_boundary_epe_px']=float(error[boundary].mean())
        moving=np.linalg.norm(u,axis=1)>1
        metrics[name+'_loo_relative_epe_moving']=float(np.mean(error[moving]/np.linalg.norm(u[moving],axis=1))) if moving.any() else None
        valid=(np.linalg.norm(u,axis=1)>1)&(np.linalg.norm(pred,axis=1)>1)
        cosine=np.sum(pred[valid]*u[valid],axis=1)/(np.linalg.norm(pred[valid],axis=1)*np.linalg.norm(u[valid],axis=1))
        metrics[name+'_loo_mean_angle_deg']=float(np.rad2deg(np.arccos(np.clip(cosine,-1,1))).mean()) if valid.any() else None
    return metrics,(p,u,current,dedup,tps,padding_difference,predictions,err,inside,full_p)


def render(metrics,data,output):
    p,u,current,dedup,tps,difference,pred,err,inside,full_p=data
    xx,yy=grid();sl=(slice(None,None,6),slice(None,None,6))
    hull=p[ConvexHull(p).vertices];hull=np.vstack((hull,hull[:1]))
    vmax=max(40,np.linalg.norm(current,axis=-1).max(),np.linalg.norm(tps,axis=-1).max())
    fig,axes=plt.subplots(2,4,figsize=(19,10),layout='constrained')
    opts=dict(angles='xy',scale_units='xy',scale=4,pivot='tail',minlength=0)
    im=axes[0,0].quiver(p[:,0],p[:,1],u[:,0],u[:,1],np.linalg.norm(u,axis=1),cmap='viridis',norm=Normalize(0,vmax),width=.004,**opts)
    axes[0,0].scatter(p[:,0],p[:,1],s=5,color='black');axes[0,0].set_title('Observed marker displacements')
    for ax,field,title in zip(axes[0,1:],(current,dedup,tps),('Current raw field (padded points)','Same linear + IDW8, UNIQUE points','TPS, UNIQUE points')):
        im=ax.imshow(np.linalg.norm(field,axis=-1),extent=(0,320,240,0),vmin=0,vmax=vmax,cmap='viridis')
        ax.quiver(xx[sl],yy[sl],field[sl][...,0],field[sl][...,1],color='white',width=.0018,**opts)
        ax.plot(hull[:,0],hull[:,1],'w--',lw=1);ax.set_title(title)
    fig.colorbar(im,ax=axes[0].tolist(),location='bottom',shrink=.6,label='Raw displacement magnitude (px); same scale; NO clipping')
    _,idx,counts=np.unique(full_p,axis=0,return_index=True,return_counts=True)
    axes[1,0].scatter(full_p[idx,0],full_p[idx,1],c=counts,s=30+counts*5,cmap='Reds',vmin=1,vmax=40)
    for point,count in zip(full_p[idx],counts):
        if count>1:axes[1,0].annotate(f'{count} copies',(point[0],point[1]),xytext=(-70,15),textcoords='offset points')
    axes[1,0].plot(hull[:,0],hull[:,1],'k--');axes[1,0].set_title(f'Available support: {len(p)}/63 unique points\nOutside hull: {100*metrics["extrapolated_pixel_fraction"]:.1f}% of pixels')
    for ax,value,title in [(axes[1,1],difference,'Padding effect: current vs UNIQUE IDW'),
                           (axes[1,2],np.linalg.norm(current-tps,axis=-1),'Current vs TPS (disagreement, not GT error)')]:
        im=ax.imshow(value,extent=(0,320,240,0),cmap='inferno',vmin=0,vmax=max(5,float(value.max())))
        ax.plot(hull[:,0],hull[:,1],'w--');ax.set_title(title)
        fig.colorbar(im,ax=ax,shrink=.65,label='Vector difference (px)')
    axes[1,3].quiver(p[:,0],p[:,1],u[:,0],u[:,1],color='gray',width=.003,**opts)
    axes[1,3].quiver(p[:,0],p[:,1],pred[:,0],pred[:,1],color='magenta',width=.003,**opts)
    im=axes[1,3].scatter(p[:,0],p[:,1],c=err,cmap='inferno',s=15)
    axes[1,3].set_title('LOO: gray observed / magenta predicted\nAll copies of held-out point removed')
    fig.colorbar(im,ax=axes[1,3],shrink=.65,label='Held-out EPE (px)')
    for ax in axes.ravel():
        ax.set(xlim=(0,320),ylim=(240,0),aspect='equal');ax.title.set_fontsize(10)
    fig.suptitle(f'Pull Out Key ONLINE RERUN | right tactile | input action {metrics["source_action"]}\n'
                 'Raw interpolation audit: component clipping is excluded. Dense disagreement is not dense ground-truth error.',fontsize=14)
    path=output/f'raw_interpolation_action_{metrics["source_action"]:03d}.png'
    fig.savefig(path,dpi=160,bbox_inches='tight');plt.close(fig)
    return str(path.resolve())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('hdf5',type=Path)
    p.add_argument('--output',type=Path,default=ROOT/'.cache/key-online-deform/interpolation')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with h5py.File(args.hdf5) as h5:
        fresh=np.flatnonzero(h5['action_index'][:]==h5['source_action_index'][:])
        rows=[]
        for index in fresh:
            metrics,_=compute(h5,int(index));metrics['record']=int(index);rows.append(metrics)
        selected={max(rows,key=lambda r:r['current_loo_mean_epe_px'])['record'],
                  max(rows,key=lambda r:r['padding_effect_outside_mean_px'])['record'],
                  max(rows,key=lambda r:r['marker_disp_max_px'])['record']}
        figures=[]
        for index in sorted(selected):
            metrics,data=compute(h5,index);figures.append(render(metrics,data,args.output))
        write_csv(args.output/'metrics.csv',rows)
        summary=dict(source=str(args.hdf5.resolve()),metrics=rows,figures=figures,
                     note='Unique-point variant isolates padding from algorithm choice. LOO does not provide dense ground truth and changes boundary support. No clipping used in these comparisons.')
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
