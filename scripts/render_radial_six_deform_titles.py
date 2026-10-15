"""Six encoding previews on shared TPS inputs; no VAE or production changes."""
from pathlib import Path
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'encoder'))
from nonlinear_depth_deform import encode_displacement
OUT=ROOT/'.cache/radial_six_mapping_comparison'
OUT.mkdir(parents=True,exist_ok=True)
NAMES=['Radial nonlinear','Original nonlinear','Linear20','Linear80','Linear160','Linear240']
COLORS=['#d64914','#222222','#a5791f','#5a9850','#8c64ab','#2878ad']
STYLES=['-','--','-', '-', '-', '-']


def amplitude(r):
    return np.where(r<=20,.015*r,.5-.2/(1+np.maximum(r-20,0)/(40/3)))


def encode(f,name):
    f=np.asarray(f,dtype=np.float64)
    if name.startswith('Linear'):
        cap=float(name[6:]);return (128+127*np.clip(f/cap,-1,1))/255
    if name=='Original nonlinear':return encode_displacement(f).astype(np.float64)
    r=np.linalg.norm(f,axis=0)
    gain=np.divide(amplitude(r),r,out=np.full_like(r,.015),where=r>0)
    return .5+f*gain


def response(f,name):
    e=encode(f,name)
    return (e-128/255)/(127/255) if name.startswith('Linear') else 2*(e-.5)


def angle_error(f,p):
    return np.degrees(np.abs(np.arctan2(f[0]*p[1]-f[1]*p[0],np.sum(f*p,axis=0))))


def save(fig,name):
    for ext in ('png','pdf'):fig.savefig(OUT/f'{name}.{ext}',dpi=180,facecolor='white')
    plt.close(fig)


train=np.load(ROOT/'.cache/policy_train_deform_preview/sample_arrays.npz')
meta=json.loads((ROOT/'.cache/policy_train_deform_preview/summary.json').read_text())
test=np.load(ROOT/'.cache/three_linear_tail04_preview/max_deform_arrays.npz')
cases=[('policy_train',train['fields'][2],f"POLICY TRAIN | {meta['sample']['task']} / episode {Path(meta['sample']['path']).stem} / {meta['sample']['side']} / frame {meta['sample']['step']}"),('test_max',test['raw_tps_field_px'],'Maximum-deform TEST | action 176 / right')]
metrics={};arrays={}
for key,raw,title in cases:
    raw=raw.astype(np.float64);mag=np.linalg.norm(raw,axis=0);valid=mag>5
    fig,axes=plt.subplots(2,3,figsize=(16,10),layout='constrained')
    ix=np.arange(8,224,16);xx,yy=np.meshgrid((ix+.5)*320/224,(ix+.5)*240/224)
    case_metrics={}
    for ax,name,color in zip(axes.ravel(),NAMES,COLORS):
        p=response(raw,name);e=encode(raw,name)
        assert np.isfinite(e).all() and e.min()>=0 and e.max()<=1
        err=angle_error(raw,p);sample=p[:,ix[:,None],ix[None,:]]
        q=ax.quiver(xx,yy,*sample,np.hypot(*sample),cmap='viridis',norm=Normalize(0,np.sqrt(2)),angles='xy',scale_units='xy',scale=.04,pivot='tail',width=.004,minlength=0)
        ax.set(xlim=(0,320),ylim=(240,0),aspect='equal',facecolor='#f5f6f8');ax.set_xticks([]);ax.set_yticks([])
        ax.set_title(name,fontsize=14,color=color)
        mean=float(err[valid].mean());p95=float(np.percentile(err[valid],95))
        clipping=float(100*(np.abs(raw)>float(name[6:])).mean()) if name.startswith('Linear') else 0.
        ax.set_xlabel(f'Direction change: mean {mean:.2f} deg | P95 {p95:.2f} deg\n'+(f'Clipped components: {clipping:.2f}%' if name.startswith('Linear') else 'No hard clipping'),fontsize=10)
        ax.quiverkey(q,.58,.04,.5,'0.5 response',labelpos='E',coordinates='axes')
        case_metrics[name]={'mean_direction_error_deg_raw_gt5px':mean,'p95_direction_error_deg_raw_gt5px':p95,'clipped_components_percent':clipping}
        arrays[f'{key}_{name.replace(" ","_")}']=p
    fig.colorbar(q,ax=axes.ravel().tolist(),shrink=.85,label='Centered encoded vector magnitude (dimensionless)')
    fig.suptitle('Training deformation' if key == 'policy_train' else 'Test Deformation', fontsize=18)
    save(fig,key+'_six_mapping_deform')
    metrics[key]={'raw_max_vector_px':float(mag.max()),'modes':case_metrics}
    arrays[key+'_raw_tps_px']=raw
