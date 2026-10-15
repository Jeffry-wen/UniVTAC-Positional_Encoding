"""Render the requested three curve panels in one row with a shared title."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'encoder'))
from nonlinear_depth_deform import encode_displacement
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'.cache/radial_six_mapping_comparison'
NAMES=['Radial nonlinear','Original nonlinear','Linear20','Linear80','Linear160','Linear240']
COLORS=['#d64914','#222222','#a5791f','#5a9850','#8c64ab','#2878ad']


def encode(f,name):
    if name.startswith('Linear'):
        return (128+127*np.clip(f/float(name[6:]),-1,1))/255
    if name=='Original nonlinear':return encode_displacement(f).astype(np.float64)
    r=np.linalg.norm(f,axis=0)
    a=np.where(r<=20,.015*r,.5-.2/(1+np.maximum(r-20,0)/(40/3)))
    k=np.divide(a,r,out=np.full_like(r,.015),where=r>0)
    return .5+f*k


def response(f,name):
    e=encode(f,name)
    return (e-128/255)/(127/255) if name.startswith('Linear') else 2*(e-.5)


plt.rcParams.update({'font.size':17,'axes.labelsize':18,'xtick.labelsize':16,'ytick.labelsize':16,'pdf.fonttype':42})
fig,axes=plt.subplots(1,3,figsize=(24,5),layout='constrained')
x=np.linspace(-300,300,4001);one_d=np.stack([x,np.zeros_like(x)])
r=np.linspace(0,300,3001);joint=np.stack([r/np.sqrt(2),r/np.sqrt(2)])
deg=np.linspace(0,90,1801);direction=np.stack([240*np.cos(np.radians(deg)),240*np.sin(np.radians(deg))])
for name,color in zip(NAMES,COLORS):
    style='--' if name=='Original nonlinear' else '-'
    kw=dict(color=color,ls=style,lw=2.5 if name=='Radial nonlinear' else 1.7,label=name)
    axes[0].plot(x,encode(one_d,name)[0],**kw)
    axes[1].plot(r,np.linalg.norm(response(joint,name),axis=0),**kw)
    p=response(direction,name)
    axes[2].plot(deg,np.degrees(np.arctan2(p[1],p[0]))-deg,**kw,zorder=10 if name=='Radial nonlinear' else 2)
axes[0].set(xlim=(-300,300),ylim=(-.025,1.025),xlabel='dx (sensor px), dy = 0',ylabel='1D Encoded Deformation')
axes[1].set(xlim=(0,300),ylim=(0,1.45),xlabel='Raw vector magnitude r (sensor px), dx = dy',ylabel='2D Encoded Deformation')
axes[2].set(xlim=(0,90),xlabel='Raw vector direction (degrees), r = 240 px',ylabel='Direction Change')
for ax in axes:
    ax.grid(alpha=.2);ax.legend(fontsize=14)
fig.suptitle('Deformation Normalization Curves',fontsize=26)
for ext in ('png','pdf'):
    fig.savefig(OUT/f'curves_six_mappings.{ext}',dpi=180,facecolor='white')
print(OUT/'curves_six_mappings.png')
