"""Static comparison of nonlinear +/-240 and +/-300 pixel caps."""
from pathlib import Path
import json
import sys

import cv2
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from preview_nonlinear_depth_deform import candidate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'.cache/nonlinear_caps_240_300'
SOURCE = ROOT/'.cache/rollout_depth_deform_comparison/depth_deform_arrays.h5'
EPISODE = ROOT/'.cache/exports/exports/pull_out_key_marker2depth_seed1000000/episode.h5'


def main():
    (OUT/'frames').mkdir(parents=True,exist_ok=True)
    fig, axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax, bound in zip(axes,(60,340)):
        x=np.linspace(-bound,bound,4001)
        for cap in (240,300):
            ax.plot(x,candidate(x,cap),label=f'Nonlinear cap +/-{cap} px',lw=2)
        ax.axvspan(-40,40,color='grey',alpha=.1)
        ax.set(xlabel='Displacement (sensor px)',ylabel='Encoded G / B',ylim=(-.02,1.02))
        ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Same +/-40 linear region (output 0.25..0.75); rational tails with different caps')
    fig.savefig(OUT/'mapping_curves.png',dpi=170);plt.close(fig)
    frames={};metrics=[]
    with h5py.File(SOURCE,'r') as source,h5py.File(EPISODE,'r') as episode:
        actions=source['source_action_index'][:]
        for i,action in enumerate(actions):
            canvas=np.full((558,896,3),22,np.uint8)
            def text(value,x,y,scale=.5):
                cv2.putText(canvas,value,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(240,240,240),1,cv2.LINE_AA)
            text(f'Nonlinear caps | source action {action:03d} | same TPS / depth',10,24,.65)
            for row,side in enumerate(('left','right')):
                old=source[f'{side}/depth_deform_nonlinear'][i]
                field=source[f'{side}/displacement_tps_px'][i]
                rgb=cv2.resize(episode[f'frames/tactile/{side}_tactile/rgb_marker'][i],(224,224),interpolation=cv2.INTER_AREA)
                panels=[rgb]
                for cap in (240,300):
                    gb=candidate(field,cap)
                    assert np.isfinite(gb).all() and gb.min()>=0 and gb.max()<=1
                    image=np.concatenate([old[:1],gb],axis=0)
                    panels.append(np.rint(image.transpose(1,2,0)*255).astype(np.uint8))
                difference=np.abs(candidate(field,240)-candidate(field,300)).mean(axis=0)
                heat=cv2.cvtColor(cv2.applyColorMap(np.rint(np.clip(difference/.02,0,1)*255).astype(np.uint8),cv2.COLORMAP_MAGMA),cv2.COLOR_BGR2RGB)
                panels.append(heat)
                titles=['Marker RGB','Nonlinear cap +/-240','Nonlinear cap +/-300','Mean |GB diff| 0..0.02']
                top=58+248*row
                for col,(panel,title) in enumerate(zip(panels,titles)):
                    text(side[0].upper()+' '+title,col*224+4,top-8,.45)
                    canvas[top:top+224,col*224:(col+1)*224]=panel
                metrics.append(dict(source_action=int(action),side=side,mean_abs_gb_difference=float(difference.mean()),
                    max_abs_gb_difference=float(np.abs(candidate(field,240)-candidate(field,300)).max())))
            text('Both: +/-40 -> 0.25..0.75; slope-matched rational tails; depth baseline 30 mm / range 5 mm',8,549,.46)
            frames[int(action)]=canvas
            Image.fromarray(canvas).save(OUT/'frames'/f'action_{action:03d}.png')
    Image.fromarray(np.concatenate([frames[a] for a in (32,112,176)])).save(OUT/'selected_observations.png')
    train=[]
    for p in sorted((ROOT/'.cache/key_nonlinear_comparison').glob('case_*_arrays.npz')):
        with np.load(p) as data:
            field=data['tps_displacement_px']; depth=data['actual_pipeline'][-1,:1]
            canvas=np.full((270,672,3),22,np.uint8)
            panels=[cv2.resize(data['marker_rgb'],(224,224))]
            panels += [np.rint(np.concatenate([depth,candidate(field,cap)]).transpose(1,2,0)*255).astype(np.uint8) for cap in (240,300)]
            for col,(panel,title) in enumerate(zip(panels,[p.stem,'Nonlinear cap +/-240','Nonlinear cap +/-300'])):
                cv2.putText(canvas,title,(col*224+4,24),cv2.FONT_HERSHEY_SIMPLEX,.5,(240,240,240),1,cv2.LINE_AA)
                canvas[38:262,col*224:(col+1)*224]=panel
            train.append(canvas)
    if train:Image.fromarray(np.concatenate(train)).save(OUT/'training_contacts.png')
    (OUT/'summary.json').write_text(json.dumps(dict(status='static preview; production unchanged',caps=[240,300],
        linear_range_px=40,linear_output=[.25,.75],metrics=metrics),indent=2)+'\n')
    print('Saved static comparison:',OUT)


if __name__=='__main__':main()
