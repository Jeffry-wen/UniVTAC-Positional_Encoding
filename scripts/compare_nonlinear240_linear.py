"""Static nonlinear candidates vs linear80/160/240; no production changes."""
from pathlib import Path
import json

import cv2
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from preview_nonlinear_depth_deform import candidate

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'.cache/nonlinear20_400_comparison'
ENCODED=ROOT/'.cache/rollout_depth_deform_comparison/depth_deform_arrays.h5'
EPISODE=ROOT/'.cache/exports/exports/pull_out_key_marker2depth_seed1000000/episode.h5'
LIMITS=(80,160,240)
VARIANTS=((40,240),(40,300),(20,400))


def linear(field, limit):
    return (128+127*np.clip(field/limit,-1,1))/255


def render(rows, title, controlled):
    canvas=np.full((60+len(rows)*250,1568,3),22,np.uint8)
    def text(value,x,y,scale=.52):
        cv2.putText(canvas,value,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(240,240,240),1,cv2.LINE_AA)
    text(title,10,24,.63)
    text('Shared TPS for every column' if controlled else 'Actual pipelines: nonlinear TPS; linear triangle / IDW',10,46,.52)
    for row,(name,rgb,images) in enumerate(rows):
        top=84+row*250
        panels=[cv2.resize(rgb,(224,224),interpolation=cv2.INTER_AREA)]
        panels += [np.rint(np.clip(im.transpose(1,2,0),0,1)*255).astype(np.uint8) for im in images]
        for col,(panel,label) in enumerate(zip(panels,['Marker RGB','NL +/-40 / cap240','NL +/-40 / cap300','NL +/-20 / cap400','Linear +/-80','Linear +/-160','Linear +/-240'])):
            text(name+' '+label,col*224+4,top-8,.43)
            canvas[top:top+224,col*224:(col+1)*224]=panel
    return canvas


def main():
    for inner,cap in VARIANTS:
        np.testing.assert_array_equal(candidate(np.array([-cap,-inner,0,inner,cap]),cap,inner),[0,.25,.5,.75,1])
        query=np.linspace(-cap-20,cap+20,10001)
        assert np.all(np.diff(candidate(query,cap,inner))>=0)
        epsilon=.01
        slopes=np.array([(candidate(inner+epsilon,cap,inner)-candidate(inner,cap,inner))/epsilon,
                         (candidate(inner,cap,inner)-candidate(inner-epsilon,cap,inner))/epsilon])
        np.testing.assert_allclose(slopes,.25/inner,rtol=.003)
    for sub in ('frames','controlled_tps_frames'):(OUT/sub).mkdir(parents=True,exist_ok=True)
    previews={False:{},True:{}}
    metrics=[]
    with h5py.File(ENCODED,'r') as encoded,h5py.File(EPISODE,'r') as episode:
        actions=encoded['source_action_index'][:]
        for i,action in enumerate(actions):
            rows={False:[],True:[]}
            for side in ('left','right'):
                tps=encoded[f'{side}/displacement_tps_px'][i]
                base=encoded[f'{side}/depth_deform_nonlinear'][i]
                new=[np.concatenate([base[:1],candidate(tps,cap,inner)]) for inner,cap in VARIANTS]
                actual=[encoded[f'{side}/depth_deform_{limit}'][i] for limit in LIMITS]
                controlled=[np.concatenate([base[:1],linear(tps,limit)]) for limit in LIMITS]
                for image in new+actual+controlled:
                    np.testing.assert_array_equal(image[0],base[0])
                    assert np.isfinite(image).all() and image.min()>=0 and image.max()<=1
                rgb=episode[f'frames/tactile/{side}_tactile/rgb_marker'][i]
                rows[False].append((side[0].upper(),rgb,new+actual))
                rows[True].append((side[0].upper(),rgb,new+controlled))
                field=encoded[f'{side}/displacement_linear_idw_px'][i]
                metrics.append(dict(source_action=int(action),side=side,
                    nonlinear240_clipped_component_percent=float(100*(np.abs(tps)>=240).mean()),
                    nonlinear300_clipped_component_percent=float(100*(np.abs(tps)>=300).mean()),
                    nonlinear20_400_clipped_component_percent=float(100*(np.abs(tps)>=400).mean()),
                    **{f'linear{limit}_clipped_component_percent':float(100*(np.abs(field)>limit).mean()) for limit in LIMITS}))
            for controlled in (False,True):
                image=render(rows[controlled],f'Source action {action:03d} | NEW: +/-20 linear -> 0.25..0.75; rational tails to +/-400',controlled)
                sub='controlled_tps_frames' if controlled else 'frames'
                Image.fromarray(image).save(OUT/sub/f'action_{action:03d}.png')
                previews[controlled][int(action)]=image
    for controlled,name in ((False,'selected_observations.png'),(True,'selected_observations_shared_tps.png')):
        Image.fromarray(np.concatenate([previews[controlled][a] for a in (32,112,176)])).save(OUT/name)
    train=[]
    for p in sorted((ROOT/'.cache/key_nonlinear_comparison').glob('case_*_arrays.npz')):
        with np.load(p) as data:
            production=data['actual_pipeline'];tps=data['tps_displacement_px']
            new=[np.concatenate([production[-1,:1],candidate(tps,cap,inner)]) for inner,cap in VARIANTS]
            train.append((p.stem,data['marker_rgb'],new+[production[i] for i in (1,2,3)]))
    if train:Image.fromarray(render(train,'Training contacts | NEW nonlinear +/-20 / cap400 vs previous candidates and linear80/160/240',False)).save(OUT/'training_contacts.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,bound in zip(axes,(60,440)):
        x=np.linspace(-bound,bound,4001)
        ax.plot(x,candidate(x,240),label='NL linear +/-40, cap240',lw=1.8,color='black')
        ax.plot(x,candidate(x,300),label='NL linear +/-40, cap300',lw=1.8,color='purple',linestyle='--')
        ax.plot(x,candidate(x,400,20),label='NEW: NL linear +/-20, cap400',lw=2.8,color='#d84b16')
        for limit in LIMITS:ax.plot(x,linear(x,limit),label=f'Linear +/-{limit}')
        ax.axvspan(-20,20,color='grey',alpha=.1)
        ax.set(xlabel='Signed displacement (sensor px)',ylabel='G / B',ylim=(-.03,1.03))
        ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.suptitle('NEW nonlinear: +/-20 -> 0.25..0.75; rational tails to +/-400')
    fig.savefig(OUT/'mapping_curves.png',dpi=170);plt.close(fig)
    (OUT/'summary.json').write_text(json.dumps(dict(status='static preview; production unchanged',
        nonlinear=[dict(linear_half_width_px=inner,cap_px=cap,linear_output=[.25,.75]) for inner,cap in VARIANTS],
        new_formula='|x|<=20: 0.5+x/80; otherwise 0.5+sign(x)*(0.25+(19/72)*t/(190/9+t)); t=clip(|x|-20,0,380); hard endpoints at +/-400',
        depth='same 30 mm baseline / 5 mm range',
        production_comparison='nonlinear TPS vs linear triangle/IDW; interpolation and encoding both differ',
        controlled_comparison='shared TPS for all six mappings; production linear byte convention retained',metrics=metrics),indent=2)+'\n')
    print('Saved',OUT)


if __name__=='__main__':main()
