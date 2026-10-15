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


fig,axs=plt.subplots(2,3,figsize=(19,11),layout='constrained')
# Individual component: explicitly hold the other component at zero.
for ax,bound in zip(axs[0,:2],(300,25)):
    x=np.linspace(-bound,bound,4001);f=np.stack([x,np.zeros_like(x)])
    for name,color,style in zip(NAMES,COLORS,STYLES):
        ax.plot(x,encode(f,name)[0],color=color,ls=style,lw=2.5 if name=='Radial nonlinear' else 1.7,label=name)
    ax.set(xlim=(-bound,bound),ylim=(-.025,1.025),xlabel='dx (sensor px), dy = 0',ylabel='Encoded G',title='1D component curve'+(' | full' if bound==300 else ' | small range'))
    ax.legend(fontsize=8);ax.grid(alpha=.2)
# Coupled: show that G now depends on dy as well as dx.
x=np.linspace(-300,300,4001);f=np.stack([x,np.full_like(x,20)])
for name,color,style in zip(NAMES,COLORS,STYLES):
    axs[0,2].plot(x,encode(f,name)[0],color=color,ls=style,lw=2.5 if name=='Radial nonlinear' else 1.7,label=name)
axs[0,2].set(xlim=(-300,300),ylim=(-.025,1.025),xlabel='dx (sensor px), dy fixed at 20 px',ylabel='Encoded G',title='Coupled component curve | dy = 20 px')
axs[0,2].grid(alpha=.2);axs[0,2].legend(fontsize=8)
# Joint equal components: vector length on x axis, not per-component value.
for ax,bound in zip(axs[1,:2],(300,25)):
    r=np.linspace(0,bound,3001);f=np.stack([r/np.sqrt(2),r/np.sqrt(2)])
    for name,color,style in zip(NAMES,COLORS,STYLES):
        ax.plot(r,np.linalg.norm(response(f,name),axis=0),color=color,ls=style,lw=2.5 if name=='Radial nonlinear' else 1.7,label=name)
    ax.set(xlim=(0,bound),ylim=(0,1.45),xlabel='Raw vector magnitude r (sensor px)',ylabel='Centered encoded vector magnitude',title='Joint dx = dy (45 deg)'+(' | full' if bound==300 else ' | small range'))
    ax.grid(alpha=.2);ax.legend(fontsize=8)
deg=np.linspace(0,90,1801);r=240
f=np.stack([r*np.cos(np.radians(deg)),r*np.sin(np.radians(deg))])
for name,color,style in zip(NAMES,COLORS,STYLES):
    p=response(f,name)
    err=np.degrees(np.arctan2(p[1],p[0]))-deg
    axs[1,2].plot(deg,err,color=color,ls=style,lw=2.5 if name=='Radial nonlinear' else 1.7,label=name,zorder=10 if name=='Radial nonlinear' else 2)
axs[1,2].set(xlim=(0,90),xlabel='Raw vector direction (degrees)',ylabel='Encoded direction minus raw direction (degrees)',title='Joint direction curve | r = 240 px')
axs[1,2].grid(alpha=.2);axs[1,2].legend(fontsize=8)
fig.suptitle('Six mapping curves | 1D components AND joint dx/dy behavior\nRadial and original nonlinear overlap when dy = 0; joint plots expose their difference\nCentered response removes each encoding zero/gain; arrows use this same response',fontsize=15)
save(fig,'curves_six_mappings')

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
    fig.suptitle(f'{title}\nRaw max vector {mag.max():.2f} px | SAME TPS field | same arrow/color scales in training and test figures\nEncoded responses, not sensor-pixel arrows; direction statistics over raw magnitude > 5 px; no VAE',fontsize=14)
    save(fig,key+'_six_mapping_deform')
    metrics[key]={'raw_max_vector_px':float(mag.max()),'modes':case_metrics}
    arrays[key+'_raw_tps_px']=raw
# Check radial direction, identical core, and identical 1D curve to original.
rng=np.random.default_rng(42);probe=rng.normal(size=(2,1000))*150
assert angle_error(probe,response(probe,'Radial nonlinear')).max()<1e-9
small=probe[:,np.linalg.norm(probe,axis=0)<=20]
np.testing.assert_allclose(encode(small,'Radial nonlinear'),encode(small,'Original nonlinear'),atol=6e-8)
x=np.linspace(-300,300,6001);axis=np.stack([x,np.zeros_like(x)])
np.testing.assert_allclose(encode(axis,'Radial nonlinear'),encode(axis,'Original nonlinear'),atol=6e-8)
summary={'status':'Preview only, no production changes or VAE training/inference','order':NAMES,'policy_sample':meta['sample'],'train_membership_verified':meta['train_membership_verified'],'notes':['All modes share TPS fields to isolate normalization; actual production linear modes use triangle/IDW.','1D curves fix the other component to zero. Coupled plots include fixed dy=20, dx=dy and variable angle at radius240.','Linear encodings use center128/255 and gain127/255, nonlinear encodings center0.5. All arrow responses remove their own center/gain.','Radial nonlinear uses the original amplitude function on vector magnitude; direction is preserved, large-magnitude compression remains.'],'metrics':metrics}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
np.savez_compressed(OUT/'arrays.npz',**arrays)
print(json.dumps(summary,indent=2),flush=True)
