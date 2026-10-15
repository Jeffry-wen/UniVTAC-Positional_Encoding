"""Preview direction-preserving radial nonlinear; no registered mode changes."""
from pathlib import Path
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'encoder'))
from nonlinear_depth_deform import encode_displacement
OUT=ROOT/'.cache/radial_nonlinear_preview'
OUT.mkdir(parents=True,exist_ok=True)


def amplitude(r):
    r=np.asarray(r,dtype=np.float64)
    return np.where(r<=20,.015*r,.5-.2/(1+np.maximum(r-20,0)/(40/3)))


def radial(field):
    f=np.asarray(field,dtype=np.float64)
    r=np.linalg.norm(f,axis=0)
    scale=np.divide(amplitude(r),r,out=np.full_like(r,.015),where=r>0)
    return .5+f*scale


def response(field,mode):
    if mode=='Linear240':return np.clip(field/240,-1,1)
    if mode=='Original nonlinear':return 2*(encode_displacement(field.astype(np.float64)).astype(np.float64)-.5)
    return 2*(radial(field)-.5)


def direction_error(raw,encoded):
    # atan2(cross,dot) is more stable than arccos near zero angle.
    cross=raw[0]*encoded[1]-raw[1]*encoded[0]
    dot=np.sum(raw*encoded,axis=0)
    return np.degrees(np.abs(np.arctan2(cross,dot)))


def save(fig,name):
    for ext in ('png','pdf'):fig.savefig(OUT/f'{name}.{ext}',dpi=180,facecolor='white')
    plt.close(fig)


names=['Linear240','Original nonlinear','Radial nonlinear']
colors=['#2878ad','#777777','#d64914']
# Sanity checks cover the origin, linear core, bounded output, direction and inverse.
rng=np.random.default_rng(42)
probe=rng.normal(size=(2,10000))*180
p=radial(probe)
assert np.all((p>=0)&(p<=1))
assert np.max(direction_error(probe,p-.5))<1e-8
np.testing.assert_allclose(radial(np.zeros((2,3))),.5)
small=probe[:,np.linalg.norm(probe,axis=0)<=20]
np.testing.assert_allclose(radial(small),.5+.015*small,atol=1e-14)
a=np.linalg.norm(p-.5,axis=0)
rr=np.where(a<=.3,a/.015,20+(40/3)*(.2/np.maximum(.5-a,1e-12)-1))
recovered=(p-.5)*np.divide(rr,a,out=np.zeros_like(a),where=a>0)
np.testing.assert_allclose(recovered,probe,atol=1e-8)

fig,axes=plt.subplots(1,3,figsize=(18,5.3),layout='constrained')
for ax,bound in zip(axes[:2],(300,25)):
    r=np.linspace(0,bound,2001)
    ax.plot(r,np.minimum(r/240,1),label='Linear240 (axis)',color=colors[0],lw=2)
    ax.plot(r,2*amplitude(r),label='Radial nonlinear (any direction)',color=colors[2],lw=3)
    for deg,style in ((0,'--'),(45,':')):
        angle=np.radians(deg);f=np.stack([r*np.cos(angle),r*np.sin(angle)])
        ax.plot(r,np.linalg.norm(response(f,'Original nonlinear'),axis=0),style,color=colors[1],label=f'Original nonlinear ({deg} deg)',lw=1.8)
    ax.axvline(20,color='grey',ls=':',lw=.8)
    ax.set(xlabel='Raw vector magnitude r (sensor px)',ylabel='Centered encoded vector magnitude',xlim=(0,bound),ylim=(0,1.45 if bound==300 else .82))
    ax.grid(alpha=.2);ax.legend(fontsize=8)
    ax.set_title('Full magnitude curve' if bound==300 else 'Small displacement: r <= 25 px')
deg=np.linspace(0,90,901)
for r,c in ((10,'#81a68c'),(80,'#c6a260'),(240,'#777777')):
    f=np.stack([r*np.cos(np.radians(deg)),r*np.sin(np.radians(deg))])
    signed=np.degrees(np.arctan2(response(f,'Original nonlinear')[1],response(f,'Original nonlinear')[0]))-deg
    axes[2].plot(deg,signed,label=f'Original nonlinear, r={r} px',color=c)
axes[2].axhline(0,color=colors[2],lw=2,label='Radial nonlinear / Linear240')
axes[2].set(xlabel='Raw vector direction (degrees)',ylabel='Encoded direction minus raw direction (degrees)',title='Direction distortion',xlim=(0,90))
axes[2].grid(alpha=.2);axes[2].legend(fontsize=8)
fig.suptitle('Radial nonlinear PREVIEW | same original amplitude law, one shared scale for dx and dy\nFor r <= 20 px: identical to original nonlinear; outside: direction preserved, magnitude still compressed',fontsize=13)
save(fig,'curves')

train=np.load(ROOT/'.cache/policy_train_deform_preview/sample_arrays.npz')
meta=json.loads((ROOT/'.cache/policy_train_deform_preview/summary.json').read_text())
test=np.load(ROOT/'.cache/three_linear_tail04_preview/max_deform_arrays.npz')
cases=[('policy_train',train['fields'][2],f"POLICY TRAIN | {meta['sample']['task']} / episode {Path(meta['sample']['path']).stem} / {meta['sample']['side']} / frame {meta['sample']['step']}"),('test_max',test['raw_tps_field_px'],'Maximum-deform TEST | action 176 / right')]
metrics={};arrays={}
for key,raw,title in cases:
    raw=raw.astype(np.float64);mag=np.linalg.norm(raw,axis=0)
    valid=mag>5
    fig,axes=plt.subplots(1,3,figsize=(16,5.8),layout='constrained')
    ix=np.arange(8,224,16);xx,yy=np.meshgrid((ix+.5)*320/224,(ix+.5)*240/224)
    case_metrics={}
    for ax,name in zip(axes,names):
        resp=response(raw,name);error=direction_error(raw,resp)
        f=resp[:,ix[:,None],ix[None,:]]
        q=ax.quiver(xx,yy,*f,np.hypot(*f),cmap='viridis',norm=Normalize(0,np.sqrt(2)),angles='xy',scale_units='xy',scale=.04,pivot='tail',width=.004,minlength=0)
        ax.set(xlim=(0,320),ylim=(240,0),aspect='equal',facecolor='#f5f6f8');ax.set_xticks([]);ax.set_yticks([])
        ax.set_title(name,fontsize=14,color=colors[names.index(name)])
        mean=float(error[valid].mean());p95=float(np.percentile(error[valid],95))
        ax.set_xlabel(f'Direction change: mean {mean:.2f} deg, P95 {p95:.2f} deg\n(statistics over raw magnitude > 5 px)',fontsize=10)
        ax.quiverkey(q,.58,.04,.5,'0.5 response',labelpos='E',coordinates='axes')
        case_metrics[name]={'mean_direction_error_deg_raw_gt5px':mean,'p95_direction_error_deg_raw_gt5px':p95,'encoded_magnitude_max':float(np.linalg.norm(resp,axis=0).max())}
        arrays[f'{key}_{name.replace(" ","_")}']=resp
    fig.colorbar(q,ax=axes.tolist(),shrink=.8,label='Centered encoded vector magnitude (dimensionless)')
    fig.suptitle(f'{title}\nRaw max vector {mag.max():.2f} px | same TPS field, fixed arrow/color scales across BOTH sample figures\nCentered encoded components shown, not sensor-pixel displacement; no VAE',fontsize=13)
    save(fig,key+'_deform')
    metrics[key]={'raw_max_vector_px':float(mag.max()),'modes':case_metrics}
    arrays[key+'_raw_tps_px']=raw
summary={'status':'Preview only; no production changes, no VAE training or inference','amplitude':'A(r)=0.015r for r<=20; 0.5-0.2/(1+(r-20)/(40/3)) otherwise','encoding':'GB=0.5 + A(r)*[dx,dy]/r; r=0 -> GB=0.5','arrow_response':'Linear240: (GB-128/255)/(127/255); nonlinear and radial: 2*(GB-0.5)','notes':['All three comparisons share the same TPS field to isolate encoding. Production linear interpolation is triangle/IDW.','Small-range equivalence requires vector magnitude <=20 px, not merely each component <=20 px.','Radial nonlinear preserves direction but retains strong large-magnitude compression and inverse sensitivity.','At the same vector magnitude, radial nonlinear uses the same encoded magnitude for every direction; original componentwise nonlinear does not.'],'policy_sample':meta['sample'],'policy_train_membership_verified':meta['train_membership_verified'],'metrics':metrics}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
np.savez_compressed(OUT/'arrays.npz',**arrays)
print(json.dumps(summary,indent=2),flush=True)
