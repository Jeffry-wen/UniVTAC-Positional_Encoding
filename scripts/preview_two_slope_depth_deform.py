"""Static preview of continuous two-slope marker displacement encoding."""
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

from preview_adaptive_nonlinear import adaptive,linear,CENTER,GAIN

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'encoder'))
from nonlinear_depth_deform import encode_displacement

OUT=ROOT/'.cache/two_slope_20_240_preview'
SOURCE=ROOT/'.cache/rollout_depth_deform_comparison/depth_deform_arrays.h5'
EPISODE=ROOT/'.cache/exports/exports/pull_out_key_marker2depth_seed1000000/episode.h5'
TITLES=['Marker RGB','NEW two-slope 20/240','Previous smooth map','Original nonlinear','Linear +/-80','Linear +/-160','Linear +/-240']


def two_slope(x):
    """Same Linear40 core as previous preview, continuous linear tails to 240.

    m=clip(abs(x),0,240); amplitude=m/40 when m<=20,
    otherwise amplitude=0.5+(m-20)/440. Preserve production byte convention.
    """
    x=np.asarray(x,dtype=np.float64)
    if not np.isfinite(x).all():raise ValueError('Non-finite displacement')
    m=np.minimum(np.abs(x),240)
    amplitude=np.where(m<=20,m/40,.5+(m-20)/440)
    return (CENTER+GAIN*np.sign(x)*amplitude).astype(np.float32)


def render(rows,title):
    canvas=np.full((60+250*len(rows),224*len(TITLES),3),22,np.uint8)
    def text(value,x,y,scale=.5):
        cv2.putText(canvas,value,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(240,240,240),1,cv2.LINE_AA)
    text(title,10,24,.65)
    text('Same depth + TPS in every column | center +/-20: Linear40 gain | outer 20..240: fixed lower gain',10,46,.54)
    for row,(side,marker,images) in enumerate(rows):
        top=84+250*row
        panels=[cv2.resize(marker,(224,224),interpolation=cv2.INTER_AREA)]
        panels += [np.rint(np.clip(im.transpose(1,2,0),0,1)*255).astype(np.uint8) for im in images]
        for col,(panel,label) in enumerate(zip(panels,TITLES)):
            text(side+' '+label,col*224+4,top-8,.41)
            canvas[top:top+224,col*224:(col+1)*224]=panel
    return canvas


def main():
    (OUT/'frames').mkdir(parents=True,exist_ok=True)
    x=np.linspace(-300,300,120001)
    y=two_slope(x)
    assert np.all(np.diff(y)>=0) and y.min()>=0 and y.max()<=1
    np.testing.assert_allclose(two_slope(np.linspace(-20,20,1001)),linear(np.linspace(-20,20,1001),40),atol=6e-8)
    np.testing.assert_allclose(two_slope(np.array([-240,-20,0,20,240])),[1/255,CENTER-GAIN/2,CENTER,CENTER+GAIN/2,1],atol=6e-8)
    fig,ax=plt.subplots(figsize=(13,7),layout='constrained')
    for limit,color in [(40,'#79a7c4'),(80,'#d7aa48'),(160,'#71a98d'),(240,'#a598bf')]:
        ax.plot(x,linear(x,limit),label=f'Linear{limit}',color=color,lw=1.3,alpha=.75)
    ax.plot(x,encode_displacement(x),label='Original nonlinear',color='#22262a',ls='--',lw=2)
    ax.plot(x,adaptive(x),label='Previous smooth map',color='#2878ad',ls=':',lw=2.3)
    ax.plot(x,y,label='NEW two-slope map',color='#d64914',lw=3,zorder=6)
    ax.axvspan(-20,20,color='#d64914',alpha=.06)
    for bound in (20,240):
        for sign in (-1,1):ax.axvline(sign*bound,color='grey',ls=':',lw=.8,alpha=.5)
    knots=np.array([-240,-20,0,20,240])
    for px,value in zip(knots,two_slope(knots)):
        ax.scatter(px,value,color='#d64914',s=30,zorder=7)
        offset=(-8,14) if px<0 else (8,-22)
        if px==0:offset=(12,15)
        ax.annotate(f'({px:g}, {value:.3f})',(px,value),xytext=offset,textcoords='offset points',fontsize=10,ha='right' if px<0 else 'left')
    ax.set(xlim=(-300,300),ylim=(-.035,1.035),xlabel='Signed displacement dx or dy (sensor px)',ylabel='Encoded G or B',
           title='Continuous two-slope encoding | no deadzone\nWithin +/-20: Linear40 gain; outer 20..240: fixed lower gain; hard cap +/-240')
    ax.set_xticks([-300,-240,-200,-160,-120,-80,-40,-20,0,20,40,80,120,160,200,240,300]);ax.tick_params(axis='x',labelsize=9)
    ax.set_yticks(np.linspace(0,1,11));ax.grid(alpha=.2);ax.legend(loc='upper left',fontsize=9)
    for ext in ('png','pdf'):fig.savefig(OUT/f'complete_mapping_curve.{ext}',dpi=190)
    plt.close(fig)
    np.savetxt(OUT/'mapping_values.csv',np.column_stack([x,y,adaptive(x),encode_displacement(x)]),delimiter=',',
               header='displacement_px,two_slope,previous_smooth,original_nonlinear',comments='')
    frames={}
    with h5py.File(SOURCE,'r') as source,h5py.File(EPISODE,'r') as episode,h5py.File(OUT/'two_slope_arrays.h5','w') as result:
        actions=source['source_action_index'][:]
        result.create_dataset('source_action_index',data=actions)
        result.attrs['encoding']='center=128/255; gain=127/255; amplitude=m/40 if m<=20 else .5+(m-20)/440; m=min(abs(x),240)'
        result.attrs['status']='preview only; production unchanged'
        for side in ('left','right'):result.create_dataset(side,(len(actions),3,224,224),dtype='f4',compression='gzip')
        for i,action in enumerate(actions):
            rows=[]
            for side in ('left','right'):
                field=source[f'{side}/displacement_tps_px'][i]
                original=source[f'{side}/depth_deform_nonlinear'][i]
                mappings=[two_slope(field),adaptive(field),encode_displacement(field)]+[linear(field,u) for u in (80,160,240)]
                images=[np.concatenate([original[:1],gb]) for gb in mappings]
                np.testing.assert_array_equal(images[2],original)
                for image in images:assert np.isfinite(image).all() and image.min()>=0 and image.max()<=1
                result[side][i]=images[0]
                rows.append((side[0].upper(),episode[f'frames/tactile/{side}_tactile/rgb_marker'][i],images))
            frame=render(rows,f'Preview | source action {action:03d} | two-slope vs previous smooth / original nonlinear / linear')
            Image.fromarray(frame).save(OUT/'frames'/f'action_{action:03d}.png');frames[int(action)]=frame
            print(f'Observation {i+1}/{len(actions)}',flush=True)
    Image.fromarray(np.concatenate([frames[a] for a in (32,112,176)])).save(OUT/'selected_observations.png')
    rows=[]
    for path in sorted((ROOT/'.cache/key_nonlinear_comparison').glob('case_*_arrays.npz')):
        with np.load(path) as data:
            field=data['tps_displacement_px'];depth=data['actual_pipeline'][-1,:1]
            mappings=[two_slope(field),adaptive(field),encode_displacement(field)]+[linear(field,u) for u in (80,160,240)]
            rows.append((path.stem,data['marker_rgb'],[np.concatenate([depth,gb]) for gb in mappings]))
    if rows:Image.fromarray(render(rows,'Training contacts | same TPS field, only encoding changes')).save(OUT/'training_contacts.png')
    summary=dict(status='preview only; production unchanged',core_half_width_px=20,cap_px=240,
        core_slope=GAIN/40,outer_slope=GAIN/440,core_slope_relative_Linear80=2,
        outer_slope_relative_Linear240=240/440,
        formula='m=min(abs(x),240); y=128/255+(127/255)*sign(x)*(m/40 if m<=20 else .5+(m-20)/440)',
        notes=['Continuous at +/-20, with an intentional slope change.',
               'All image columns share TPS fields and depth; linear columns isolate encoding and use TPS rather than production linear/IDW.',
               'High core gain preserves small-deformation contrast; fixed tail gain avoids the nearly flat midrange of the prior curve.'])
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    print('Saved',OUT,flush=True)


if __name__=='__main__':main()
