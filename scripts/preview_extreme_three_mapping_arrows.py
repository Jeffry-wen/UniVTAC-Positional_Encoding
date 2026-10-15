"""Extreme rollout: isolate mapping effects on the same dense TPS field."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'encoder'))
from nonlinear_depth_deform import encode_displacement
def linear(field,limit):
    return (128+127*np.clip(field/limit,-1,1))/255
OUT=Path(__file__).resolve().parents[1]/'.cache/three_linear_tail04_preview'
d=np.load(OUT/'max_deform_arrays.npz');field=d['raw_tps_field_px']
names=['Linear20','Linear240','Original nonlinear']
encoded=[linear(field,20),linear(field,240),encode_displacement(field)]
response=[(encoded[0]-128/255)/(127/255),(encoded[1]-128/255)/(127/255),2*(encoded[2]-.5)]
s=encoded[2].astype(np.float64)-.5;a=np.abs(s)
recovered=[response[0]*20,response[1]*240,np.sign(s)*np.where(a<=.3,a/.015,20+(40/3)*(.2/np.maximum(.5-a,1e-12)-1))]
ix=np.arange(8,224,16);xx,yy=np.meshgrid((ix+.5)*320/224,(ix+.5)*240/224)
fig,axes=plt.subplots(2,3,figsize=(16,9),layout='constrained')
metrics={}
rawmag=np.linalg.norm(field,axis=0)
for col,name in enumerate(names):
    r=response[col]
    rm=np.linalg.norm(r,axis=0)
    valid=(rawmag>5)&(rm>1e-8)
    cos=np.sum(field*r,axis=0)/np.maximum(rawmag*rm,1e-12)
    angle=np.degrees(np.arccos(np.clip(cos,-1,1)))
    metrics[name]={'encoded_mean_direction_error_degrees_where_raw_magnitude_gt_5px':float(angle[valid].mean()),'encoded_p95_direction_error_degrees_where_raw_magnitude_gt_5px':float(np.percentile(angle[valid],95)),'inverse_roundtrip_component_rmse_px':float(np.sqrt(np.mean((recovered[col]-field)**2))),'clipped_component_percent':float(100*(np.abs(field)>(20 if col==0 else 240)).mean()) if col<2 else 0}
for row,fields in enumerate((response,recovered)):
    for col,arrows in enumerate(fields):
        ax=axes[row,col];f=arrows[:,ix[:,None],ix[None,:]]
        q=ax.quiver(xx,yy,*f,np.hypot(*f),cmap='viridis',norm=Normalize(0,np.sqrt(2) if row==0 else 250),angles='xy',scale_units='xy',scale=.04 if row==0 else 10,pivot='tail',width=.004,minlength=0)
        ax.set(xlim=(0,320),ylim=(240,0),aspect='equal',facecolor='#f5f6f8');ax.set_xticks([]);ax.set_yticks([])
        ax.set_title(names[col],fontsize=14)
        if col==0:ax.set_ylabel('Centered encoded response' if row==0 else 'Inverse-mapped displacement',fontsize=12)
        if row==0:ax.set_xlabel(f"Mean direction change: {metrics[names[col]]['encoded_mean_direction_error_degrees_where_raw_magnitude_gt_5px']:.2f} deg (raw magnitude > 5 px)")
        else:ax.set_xlabel(f"Encoding round-trip RMSE: {metrics[names[col]]['inverse_roundtrip_component_rmse_px']:.4f} px")
        ax.quiverkey(q,.61,.04,.5 if row==0 else 100,'0.5 response' if row==0 else '100 px',labelpos='E',coordinates='axes')
    fig.colorbar(q,ax=axes[row].tolist(),shrink=.8,label='Encoded magnitude (dimensionless)' if row==0 else 'Displacement magnitude (sensor px)')
fig.suptitle('Extreme TEST deform | action 176, right | max vector 244.0 px, max component 231.3 px\nSame dense TPS field for all three mappings; shared scales within each row; no VAE\nTop shows componentwise encoding effects; bottom shows displacement recovered from encoding',fontsize=14)
for ext in ('png','pdf'):fig.savefig(OUT/f'extreme_linear20_linear240_nonlinear_arrows.{ext}',dpi=180,facecolor='white')
(OUT/'extreme_three_mapping_metrics.json').write_text(json.dumps({'metrics':metrics,'note':'Shared TPS isolates encoding. Production linear modes use triangle/IDW instead. Direction statistics compare centered encoded vectors against the raw TPS field, not independently measured physical ground truth. No VAE inference.'},indent=2)+'\n')
print(json.dumps(metrics,indent=2))
