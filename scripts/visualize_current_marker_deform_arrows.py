"""Same-scale marker versus interpolated vectors at 3mm / +/-100px encoding."""
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np

from scripts.compare_train_online_depth_deform import sample, dense, TRAIN, ONLINE, ROOT
from encoder.linear_depth_deform import DEPTH_DEFORM_SPEC

OUT=ROOT/'.cache/key-online-deform/marker_dense_arrows_3mm_100px'
STRIDE=8
DISPLAY_SCALE=4


def build(path,step,online,label):
    c=sample(path,step,online)
    m=np.asarray(c['marker'],np.float32)
    field=dense(m,c['press_depth'].shape)
    # Decode the actual float model input, including componentwise clipping.
    encoded=c['new'][1:]
    decoded=(encoded*255.-128.)/127.*DEPTH_DEFORM_SPEC['deform_range_px']
    clipped=np.clip(field,-100,100)
    np.testing.assert_allclose(decoded,clipped,rtol=0,atol=3e-5)
    raw_displacement=m[1]-m[0]
    zero=np.zeros_like(field)
    zero_encoded=(128+127*np.clip(zero/100,-1,1))/255
    np.testing.assert_allclose((zero_encoded*255-128)/127*100,0,atol=1e-5)
    c.update(label=label,field=field,decoded=decoded,
             displacement=raw_displacement,clipmask=(np.abs(field)>100).any(axis=0))
    return c


def arrows(ax,x,y,u,norm,dense=False):
    return ax.quiver(x,y,u[:,0],u[:,1],np.linalg.norm(u,axis=1),
        angles='xy',scale_units='xy',scale=DISPLAY_SCALE,pivot='tail',
        cmap='viridis',norm=norm,width=.0017 if dense else .0045,
        headwidth=3.3,headlength=4,headaxislength=3.5,minlength=0,
        alpha=.8 if dense else 1)


def render(cases,path,zoom=False):
    fig,axes=plt.subplots(len(cases),4,figsize=(17,3.65*len(cases)),
                          squeeze=False,layout='constrained')
    vmax=np.ceil(max(np.linalg.norm(c['field'],axis=0).max() for c in cases)/10)*10
    norm=Normalize(0,vmax)
    titles=['Marker RGB','Marker Displacement','Interpolated Deform (Raw)','Deform (Normalized +/-100 px)']
    h,w=240,320
    coord_x=(np.arange(224,dtype=np.float32)+.5)*w/224
    coord_y=(np.arange(224,dtype=np.float32)+.5)*h/224
    xx,yy=np.meshgrid(coord_x,coord_y)
    sl=(slice(None,None,STRIDE),slice(None,None,STRIDE))
    x,y=xx[sl].ravel(),yy[sl].ravel()
    for row,c in enumerate(cases):
        p=c['marker'][0];u=c['displacement']
        roi=(0,w,0,h)
        if zoom:
            strongest=np.linalg.norm(u,axis=1)>=np.percentile(np.linalg.norm(u,axis=1),80)
            center=np.median(p[strongest],axis=0)
            x0=float(np.clip(center[0]-55,0,w-110));y0=float(np.clip(center[1]-55,0,h-110))
            roi=(x0,x0+110,y0,y0+110)
        axes[row,0].imshow(c['rgb'],extent=(0,w,h,0))
        axes[row,1].scatter(p[:,0],p[:,1],s=8,c='#303030',zorder=3)
        q=arrows(axes[row,1],p[:,0],p[:,1],u,norm)
        for col,key in ((2,'field'),(3,'decoded')):
            ax=axes[row,col]
            if col==3:
                overlay=np.zeros((224,224,4),np.float32)
                overlay[c['clipmask']]=[1,.2,.1,.13]
                ax.imshow(overlay,extent=(0,w,h,0))
            a=c[key].transpose(1,2,0)[sl].reshape(-1,2)
            arrows(ax,x,y,a,norm,dense=True)
        for col,ax in enumerate(axes[row]):
            if row==0:ax.set_title(titles[col],fontsize=15,pad=8)
            ax.set(xlim=roi[:2],ylim=(roi[3],roi[2]),aspect='equal')
            ax.set_xticks([]);ax.set_yticks([])
        if row==0:
            axes[row,1].quiverkey(q,.65,.94,40,'40 px',labelpos='E',coordinates='axes',fontproperties={'size':9})
        axes[row,0].set_ylabel(c['label']+f"\n{c['stats']['indentation_max_mm']:.3f} mm",fontsize=13)
        axes[row,3].set_xlabel(f"Clipped dx/dy components: {c['stats']['new_deform_clipped_component_percent']:.1f}%",fontsize=10)
    fig.colorbar(ScalarMappable(norm=norm,cmap='viridis'),ax=axes.ravel().tolist(),
                 location='bottom',shrink=.62,aspect=65,pad=.025,
                 label='Displacement magnitude (px) | same color and arrow scale in all panels | arrow length = displacement / 4')
    for ext in ('png','pdf'):
        fig.savefig(path.with_suffix('.'+ext),dpi=180,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    medium=json.loads((ROOT/'.cache/key-online-deform/medium_contact_selection.json').read_text())
    cases=[build(TRAIN.parent/medium['episode'],medium['frame'],False,'Medium Contact'),
           build(TRAIN,171,False,'Strong Contact (Train)'),
           build(ONLINE,112,True,'Extreme Contact (Rollout)')]
    render(cases,OUT/'marker_vs_dense_arrows')
    render(cases,OUT/'marker_vs_dense_arrows_zoom',zoom=True)
    for i,c in enumerate(cases):
        np.savez_compressed(OUT/f'case_{i}_fields.npz',marker=c['marker'],marker_ids=c['ids'],
            marker_rgb=c['rgb'],raw_field_px=c['field'],decoded_normalized_field_px=c['decoded'],
            encoded_depth_deform=c['new'],clipped_pixel_mask=c['clipmask'])
    summary=dict(encoding=DEPTH_DEFORM_SPEC,arrow_display_multiplier=1/DISPLAY_SCALE,
        dense_stride=STRIDE,dense_arrow_count=len(range(0,224,STRIDE))**2,
        marker_arrow_count=63,notes=[
            'All vectors start at reference coordinates, including full markers whose current positions are out of view.',
            'Sparse arrows are measured marker displacements; dense arrows are interpolated estimates, not independent measurements.',
            'Final column decodes actual normalized G/B with (255*value - 128)/127*100; length/direction may change when either component clips.',
            'Red tint marks pixels where at least one component exceeds +/-100px.',
            'Depth range is 0..3mm but depth does not enter the plotted tangential arrows.',
            'Both train cases use existing stored coordinates; extreme case uses recorded full coordinates and stable IDs.'],
        cases=[c['stats'] for c in cases])
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
