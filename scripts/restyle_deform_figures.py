"""Restyle four cached deform figures without changing arrays or model inference."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
ROOT=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.size':17,'axes.titlesize':20,'axes.labelsize':17,'xtick.labelsize':16,'ytick.labelsize':16,'pdf.fonttype':42})
ix=np.arange(8,224,16);xx,yy=np.meshgrid((ix+.5)*320/224,(ix+.5)*240/224)


def grid(fig,rows,cols,left,right,bottom,top,gap):
    width=(right-left-(cols-1)*gap)/cols
    height=width*fig.get_figwidth()*.75/fig.get_figheight()
    return np.array([[fig.add_axes([left+c*(width+gap),top if r==0 else bottom,width,height]) for c in range(cols)] for r in range(rows)])


def quiver(ax,v,maximum,scale,key,label):
    f=v[:,ix[:,None],ix[None,:]]
    q=ax.quiver(xx,yy,*f,np.hypot(*f),cmap='viridis',norm=Normalize(0,maximum),angles='xy',scale_units='xy',scale=scale,pivot='tail',width=.004,minlength=0)
    ax.set(xlim=(0,320),ylim=(240,0),aspect='equal',facecolor='#f5f6f8');ax.set_xticks([]);ax.set_yticks([])
    ax.quiverkey(q,.53,.045,key,label,labelpos='E',coordinates='axes',fontproperties={'size':12})
    return q


def save(fig,path):
    for ext in ('png','pdf'):fig.savefig(path.with_suffix('.'+ext),dpi=200,facecolor='white',bbox_inches='tight',pad_inches=.08)
    plt.close(fig)


out=ROOT/'.cache/vae_encoded_deform_reconstruction'
d=np.load(out/'arrays.npz');report=json.loads((out/'summary.json').read_text())
maximum=max(np.linalg.norm(d['centered_gt'],axis=2).max(),np.linalg.norm(d['centered_pred'],axis=2).max())
names=['Linear20','Linear80','Linear160','Linear240','Original nonlinear','Radial nonlinear']
for sample,key in enumerate(('training','test_max')):
    fig=plt.figure(figsize=(25,7.15))
    axes=grid(fig,2,6,.04,.925,.095,.515,.009)
    for c,name in enumerate(names):
        axes[0,c].set_title(name,pad=9)
        for row,array_key in enumerate(('centered_gt','centered_pred')):
            q=quiver(axes[row,c],d[array_key][c,sample],maximum,.02,.25,'0.25 encoded')
        mse=next(r['encoded_deform_mse'] for r in report['metrics'] if r['mode']==str(d['modes'][c]) and r['sample']==key)
        axes[1,c].set_xlabel(f'G/B MSE: {mse:.6f}',labelpad=8,fontsize=17)
    axes[0,0].set_ylabel('Encoded GT',fontsize=20,labelpad=10)
    axes[1,0].set_ylabel('VAE reconstruction',fontsize=20,labelpad=10)
    cax=fig.add_axes([.946,.095,.010,.795])
    cb=fig.colorbar(q,cax=cax);cb.set_label('Encoded deform magnitude',fontsize=18,labelpad=12)
    title='Training VAE Deformation Reconstruction' if sample==0 else 'Test VAE Deformation Reconstruction'
    fig.suptitle(title,fontsize=25,y=.995)
    save(fig,out/f'{key}_encoded_deform')

out=ROOT/'.cache/radial_six_mapping_comparison'
d=np.load(out/'arrays.npz');report=json.loads((out/'summary.json').read_text())
names=['Radial nonlinear','Original nonlinear','Linear20','Linear80','Linear160','Linear240']
colors=['#d64914','#222222','#a5791f','#5a9850','#8c64ab','#2878ad']
for key in ('policy_train','test_max'):
    fig=plt.figure(figsize=(16,9.2))
    axes=grid(fig,2,3,.035,.91,.07,.55,.025)
    for ax,name,color in zip(axes.ravel(),names,colors):
        v=d[f'{key}_{name.replace(" ","_")}'];q=quiver(ax,v,np.sqrt(2),.04,.5,'0.5 response')
        ax.set_title(name,color=color,pad=8)
        m=report['metrics'][key]['modes'][name]
        text=f"Angle change: {m['mean_direction_error_deg_raw_gt5px']:.2f}° mean / {m['p95_direction_error_deg_raw_gt5px']:.2f}° P95\n"
        text+=f"Clipped components: {m['clipped_components_percent']:.2f}%" if name.startswith('Linear') else 'No hard clipping'
        ax.set_xlabel(text,fontsize=15,labelpad=7)
    cax=fig.add_axes([.936,.07,.014,.832])
    cb=fig.colorbar(q,cax=cax);cb.set_label('Centered normalized deform magnitude',fontsize=17,labelpad=12)
    title='Training Normalized Deformation' if key=='policy_train' else 'Test Normalized Deformation'
    fig.suptitle(title,fontsize=24,y=.985)
    save(fig,out/f'{key}_six_mapping_deform')
print('Updated four PNG and PDF figures from saved arrays; no inference rerun.')
