"""Inspect recorded online Pull Out Key inputs, including out-of-frame markers."""
import argparse
import json
import sys
from pathlib import Path

import cv2
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'encoder'))
from depth_marker_to_deform import _interpolation_map, convert


def arrays(h5,index,side):
    g=h5[f'tactile/{side}_tactile']
    m=g['marker'][index]
    p,moved=m
    _,idx=np.unique(p,axis=0,return_index=True)
    u=moved-p
    depth=g['depth'][index]
    v,w=_interpolation_map(p.astype(np.float32).tobytes(),len(p),*depth.shape,224)
    raw=np.einsum('nvc,nv->nc',u[v],w).reshape(224,224,2)
    actual=h5[f'policy_input/observation.images.tac_{side}'][index].transpose(1,2,0)
    np.testing.assert_allclose(actual,convert(depth,m).transpose(1,2,0),atol=1e-6)
    decoded=(actual[:,:,1:]-.5)*40
    rgb=g['rgb_marker'][index]
    if rgb.dtype!=np.uint8:
        rgb=np.clip(rgb,0,255).astype(np.uint8)
    return p[idx],u[idx],raw,decoded,actual,rgb,depth


def grid():
    xx,yy=np.meshgrid((np.arange(224)+.5)*320/224,(np.arange(224)+.5)*240/224)
    return xx,yy


def draw_arrows(p,u,vmax):
    image=np.full((240,320,3),255,np.uint8)
    colors=plt.get_cmap('viridis')(np.clip(np.linalg.norm(u,axis=1)/vmax,0,1))[:,:3]*255
    for start,vector,color in zip(p,u,colors):
        a=tuple(np.rint(start).astype(int));b=tuple(np.rint(start+vector*.25).astype(int))
        cv2.arrowedLine(image,a,b,tuple(int(c) for c in color),1,cv2.LINE_AA,tipLength=.3)
    return image


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('hdf5',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'.cache/key-online-deform/analysis')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    xx,yy=grid();sl=(slice(None,None,4),slice(None,None,4))
    points=np.column_stack((xx[sl].ravel(),yy[sl].ravel()))
    with h5py.File(args.hdf5) as h5:
        source=h5['source_action_index'][:]
        action=h5['action_index'][:]
        fresh=np.flatnonzero(source==action)
        metrics=[]
        for index in fresh:
            for side in ('left','right'):
                p,u,raw,decoded,actual,rgb,depth=arrays(h5,int(index),side)
                metrics.append(dict(record=int(index),source_action=int(source[index]),side=side,
                                    unique_markers=len(p),marker_disp_max_px=float(np.linalg.norm(u,axis=1).max()),
                                    marker_disp_p95_px=float(np.percentile(np.linalg.norm(u,axis=1),95)),
                                    max_abs_component_px=float(np.abs(u).max()),
                                    marker_clip_fraction=float((np.abs(u)>20).mean()),
                                    marker_clipping_epe_px=float(np.linalg.norm(np.clip(u,-20,20)-u,axis=1).mean()),
                                    field_clip_fraction=float((np.abs(raw)>20).mean()),
                                    press_max_mm=float(h5[f'tactile/{side}_tactile/press_depth'][index].max()),
                                    raw_depth_min_mm=float(depth.min())))
        vmax=max(40,np.ceil(max(r['marker_disp_max_px'] for r in metrics)/10)*10)
        chosen={int(fresh[0]),int(fresh[-1])}
        for side in ('left','right'):
            rows=[r for r in metrics if r['side']==side]
            chosen.add(max(rows,key=lambda r:r['marker_disp_max_px'])['record'])
            chosen.add(min(rows,key=lambda r:(r['unique_markers'],-r['marker_disp_max_px']))['record'])
        figures=[]
        for index in sorted(chosen):
            fig,axes=plt.subplots(2,5,figsize=(23,10),layout='constrained')
            for row,side in enumerate(('left','right')):
                p,u,raw,decoded,actual,rgb,depth=arrays(h5,index,side)
                titles=['Saved sensor marker RGB','Observed marker displacement','Current raw deform (3,136 arrows)',
                        'Actual policy depth_deform','Decoded G/B arrows (after clipping)']
                axes[row,0].imshow(rgb,extent=(0,320,240,0))
                axes[row,3].imshow(actual,extent=(0,320,240,0))
                opts=dict(angles='xy',scale_units='xy',scale=4,pivot='tail',cmap='viridis',norm=Normalize(0,vmax),minlength=0)
                axes[row,1].quiver(p[:,0],p[:,1],u[:,0],u[:,1],np.linalg.norm(u,axis=1),width=.004,**opts)
                axes[row,1].scatter(p[:,0],p[:,1],s=5,color='black')
                for ax,field in ((axes[row,2],raw),(axes[row,4],decoded)):
                    vectors=field[sl].reshape(-1,2)
                    im=ax.quiver(points[:,0],points[:,1],vectors[:,0],vectors[:,1],np.linalg.norm(vectors,axis=1),width=.0016,**opts)
                    hull=p[ConvexHull(p).vertices];hull=np.vstack((hull,hull[:1]))
                    ax.plot(hull[:,0],hull[:,1],'--',color='gray',lw=.8)
                for ax,title in zip(axes[row],titles):
                    ax.set(xlim=(0,320),ylim=(240,0),aspect='equal',title=title)
                    ax.title.set_fontsize(10)
                axes[row,0].set_ylabel(side+f' | {len(p)} unique markers')
            fig.colorbar(im,ax=axes.ravel().tolist(),location='bottom',shrink=.5,aspect=60,label='Displacement magnitude (px), shared scale; arrows displayed at 0.25 x displacement')
            fig.suptitle(f'Pull Out Key ONLINE RERUN | seed 1000000 | actual inference input at action {source[index]}\n'
                         'Reference coordinates for arrows; raw field and actual encoded input shown separately. Not reconstructed from the old MP4.',fontsize=14)
            path=args.output/f'input_action_{source[index]:03d}.png'
            fig.savefig(path,dpi=170,bbox_inches='tight');plt.close(fig)
            figures.append(str(path.resolve()))
        # A diagnostic video shares the recorded model-input timestamps. Reuse
        # rendered fields within each queued chunk, just as the policy does.
        output=args.output/'marker_vs_actual_deform.mp4'
        writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*'mp4v'),10,(1600,528))
        if not writer.isOpened():
            raise RuntimeError('Cannot open diagnostic video writer')
        cached_source=None
        for index in range(len(action)):
            if source[index]!=cached_source:
                canvas=np.zeros((528,1600,3),np.uint8)
                for row,side in enumerate(('left','right')):
                    p,u,raw,decoded,actual,rgb,depth=arrays(h5,index,side)
                    panels=[rgb,draw_arrows(p,u,vmax),
                            draw_arrows(points,raw[sl].reshape(-1,2),vmax),
                            np.rint(np.clip(cv2.resize(actual,(320,240)),0,1)*255).astype(np.uint8),
                            draw_arrows(points,decoded[sl].reshape(-1,2),vmax)]
                    for col,(panel,label) in enumerate(zip(panels,('sensor marker RGB','marker displacement','raw deform arrows','actual depth_deform','decoded deform arrows'))):
                        canvas[row*264+24:(row+1)*264,col*320:(col+1)*320]=panel
                        cv2.putText(canvas,f'{side} {label}',(col*320+4,row*264+17),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1)
                cached_source=source[index]
            writer.write(cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR))
        writer.release()
        summary=dict(source_hdf5=str(args.hdf5.resolve()),checkpoint=h5.attrs['checkpoint'],
                     recorded_actions=len(action),fresh_inference_inputs=len(fresh),metrics=metrics,figures=figures,
                     diagnostic_video=str(output.resolve()),note='Current-environment rerun; old MP4 lacks depth and reference marker arrays. Indentation is not force. Unique marker loss can shrink interpolation support.')
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
        print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
