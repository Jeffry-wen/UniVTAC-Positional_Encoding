"""Preview only: +/-40 px uses 50% output; rational tails cap at +/-300 px by default."""
from pathlib import Path
import argparse
from functools import partial
import json
import sys

import cv2
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'encoder'))
from nonlinear_depth_deform import encode_displacement

OUT = None
SOURCE = ROOT/'.cache/rollout_depth_deform_comparison/depth_deform_arrays.h5'
EPISODE = ROOT/'.cache/exports/exports/pull_out_key_marker2depth_seed1000000/episode.h5'


def candidate(field, max_px=300, linear_px=40):
    field = np.asarray(field, dtype=np.float32)
    magnitude = np.abs(field)
    tail_width = float(max_px)-linear_px
    if not np.isfinite(linear_px) or linear_px <= 0 or not np.isfinite(max_px) or tail_width <= linear_px:
        raise ValueError("linear_px must be positive and max_px must exceed twice linear_px")
    tail_scale = linear_px*tail_width/(tail_width-linear_px)
    tail_gain = tail_scale/(4*linear_px)
    t = np.clip(magnitude-linear_px, 0, tail_width)
    amplitude = np.where(magnitude <= linear_px, magnitude/(4*linear_px), .25+tail_gain*t/(tail_scale+t))
    amplitude = np.where(magnitude >= max_px, .5, amplitude)
    return (.5+np.sign(field)*amplitude).astype(np.float32)


def byte(image):
    return np.rint(np.clip(image.transpose(1,2,0),0,1)*255).astype(np.uint8)


def label(canvas, value, x, y, scale=.55):
    cv2.putText(canvas,value,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(240,240,240),1,cv2.LINE_AA)


def save(name, canvas):
    Image.fromarray(canvas).save(OUT/name)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-px', type=float, default=300)
    args = parser.parse_args()
    limit_px = args.max_px
    if not np.isfinite(limit_px) or limit_px <= 80:
        parser.error('--max-px must be finite and exceed 80')
    OUT = ROOT/f'.cache/nonlinear_40_{limit_px:g}_preview'
    encode = partial(candidate, max_px=limit_px)
    width = limit_px-40
    tail_scale = 40*width/(width-40)
    tail_gain = tail_scale/160
    formula = f'|x|<=40: 0.5+x/160; otherwise 0.5+sign(x)*(0.25+{tail_gain:.12g}*t/({tail_scale:.12g}+t)), t=clip(|x|-40,0,{width:g})'
    (OUT/'frames').mkdir(parents=True,exist_ok=True)
    x = np.array([-limit_px-60,-limit_px,-240,-160,-80,-40,-20,0,20,40,80,160,240,limit_px,limit_px+60],dtype=np.float32)
    np.testing.assert_allclose(encode(x), 1-encode(-x), atol=1e-7)
    np.testing.assert_allclose(encode(np.array([-limit_px,-40,0,40,limit_px])), [0,.25,.5,.75,1], atol=1e-7)
    assert np.all(np.diff(encode(np.linspace(-limit_px-60,limit_px+60,72001))) >= 0)
    eps = .1
    slopes = [(float(encode(40+eps))-float(encode(40)))/eps,
              (float(encode(40))-float(encode(40-eps)))/eps]
    np.testing.assert_allclose(slopes, [1/160,1/160],rtol=.004)
    fig, axes = plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
    for ax, limit in zip(axes,(60,limit_px+40)):
        query=np.linspace(-limit,limit,4001)
        ax.plot(query,encode_displacement(query),label='Current: +/-20 -> 0.2..0.8',lw=2)
        ax.plot(query,encode(query),label=f'Candidate: +/-40 -> 0.25..0.75; cap +/-{limit_px:g}',lw=2)
        ax.axvspan(-40,40,color='grey',alpha=.1)
        ax.scatter([-40,0,40],[.25,.5,.75],s=20,color='black',zorder=5)
        ax.set(xlabel='Signed displacement (sensor pixels)',ylabel='Encoded G or B',ylim=(-.03,1.03))
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Preview only | Same TPS displacement field; only the encoding changes')
    fig.savefig(OUT/'mapping_curves.png',dpi=170);plt.close(fig)
    frames=[];metrics=[]
    with h5py.File(SOURCE,'r') as source,h5py.File(EPISODE,'r') as episode,h5py.File(OUT/'candidate_arrays.h5','w') as result:
        actions=source['source_action_index'][:]
        result.create_dataset('source_action_index',data=actions)
        result.attrs['status']='preview only; production encoder unchanged'
        result.attrs['encoding']=json.dumps(dict(linear_range_px=40,linear_output=[.25,.75],max_px=limit_px,
            tail=formula,interpolation='same saved TPS'))
        for side in ('left','right'):
            result.create_dataset(side,(len(actions),3,224,224),dtype='f4',compression='gzip')
        for i, action in enumerate(actions):
            canvas=np.full((556,1120,3),22,np.uint8)
            label(canvas,f'PREVIEW | Source action {action:03d} | same depth and TPS; encoding changes only',12,25,.65)
            for row,side in enumerate(('left','right')):
                old=source[f'{side}/depth_deform_nonlinear'][i]
                field=source[f'{side}/displacement_tps_px'][i]
                np.testing.assert_allclose(old[1:],encode_displacement(field),atol=1e-7)
                new=np.concatenate([old[:1],encode(field)],axis=0)
                np.testing.assert_array_equal(new[0],old[0]); assert np.isfinite(new).all()
                result[side][i]=new
                rgb=cv2.resize(episode[f'frames/tactile/{side}_tactile/rgb_marker'][i],(224,224),interpolation=cv2.INTER_AREA)
                # Controlled linear reference uses TPS and center .5 to isolate encoding.
                controlled=np.concatenate([old[:1],np.clip(.5+field/(2*limit_px),0,1)],axis=0)
                diff=np.abs(new[1:]-old[1:]).mean(axis=0)
                heat=cv2.cvtColor(cv2.applyColorMap(np.rint(np.clip(diff/.25,0,1)*255).astype(np.uint8),cv2.COLORMAP_MAGMA),cv2.COLOR_BGR2RGB)
                panels=[rgb,byte(old),byte(new),byte(controlled),heat]
                titles=['Marker RGB','Current nonlinear',f'Candidate +/-40..{limit_px:g}',f'Linear +/-{limit_px:g} (TPS)','Mean |GB change| 0..0.25']
                y=58+row*248
                for col,(panel,title) in enumerate(zip(panels,titles)):
                    label(canvas,side[0].upper()+' '+title,col*224+4,y-8,.43)
                    canvas[y:y+224,col*224:(col+1)*224]=panel
                metrics.append(dict(source_action=int(action),side=side,
                    tps_max_abs_component_px=float(np.abs(field).max()),
                    clipped_component_percent=float(100*(np.abs(field)>=limit_px).mean()),
                    linear_region_component_percent=float(100*(np.abs(field)<=40).mean()),
                    mean_abs_gb_change=float(diff.mean())))
            label(canvas,f'Depth: clip((30-camera_mm)/5) | +/-40 linear: 0.25..0.75 | rational tails to +/-{limit_px:g}: 0..1',10,549,.52)
            frames.append(canvas);save(f'frames/action_{action:03d}.png',canvas)
            print(f'Preview {i+1}/{len(actions)}',flush=True)
    writer=cv2.VideoWriter(str(OUT/'comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),2,(1120,556))
    if not writer.isOpened():raise RuntimeError('Cannot create preview video')
    for frame in frames:writer.write(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
    writer.release()
    reader=cv2.VideoCapture(str(OUT/'comparison.mp4'))
    assert int(reader.get(cv2.CAP_PROP_FRAME_COUNT))==len(frames)
    ok,_=reader.read();assert ok;reader.release()
    selection=[int(np.flatnonzero(actions==a)[0]) for a in (32,112,176)]
    save('selected_observations.png',np.concatenate([frames[i] for i in selection]))
    # Reuse saved real training cases to compare small and large contacts.
    train_frames=[]
    for p in sorted((ROOT/'.cache/key_nonlinear_comparison').glob('case_*_arrays.npz')):
        with np.load(p) as data:
            field=data['tps_displacement_px'];old=data['actual_pipeline'][-1]
            new=np.concatenate([old[:1],encode(field)],axis=0)
            canvas=np.full((270,672,3),22,np.uint8)
            for j,(im,title) in enumerate(zip([cv2.resize(data['marker_rgb'],(224,224)),byte(old),byte(new)],
                                             ['Training '+p.stem,'Current nonlinear',f'Candidate +/-40..{limit_px:g}'])):
                label(canvas,title,j*224+4,24,.48);canvas[38:262,j*224:(j+1)*224]=im
            train_frames.append(canvas)
    if train_frames:save('training_contacts.png',np.concatenate(train_frames))
    summary=dict(status='preview only; production files and checkpoint specs unchanged',
        formula=formula,
        interpretation=f'Central linear output width 0.5; two nonlinear tails each width 0.25; hard cap +/-{limit_px:g}.',
        connection=f'Value and slope continuous at +/-40; slope 1/160. Hard clipping begins at +/-{limit_px:g}.',
        small_motion_gain_relative_to_current=(1/160)/.015,
        values=[dict(px=float(v),current=float(a),candidate=float(b)) for v,a,b in zip(x,encode_displacement(x),encode(x))],
        clipping_percent_max=max(m['clipped_component_percent'] for m in metrics),metrics=metrics,
        note='No ground-truth quality ranking; visual comparison holds TPS and depth constant. Linear reference panel is controlled TPS, not production linear/IDW.')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='metrics'},indent=2),flush=True)
    print('Saved',OUT,flush=True)


if __name__=='__main__':main()
