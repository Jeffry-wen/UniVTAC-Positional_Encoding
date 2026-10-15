"""Static preview: no deadzone, Linear40 core, positive-gain bridge to Linear240."""
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
OUT=ROOT/'.cache/adaptive_nonlinear_no_deadzone_gain40_preview'
SOURCE=ROOT/'.cache/rollout_depth_deform_comparison/depth_deform_arrays.h5'
EPISODE=ROOT/'.cache/exports/exports/pull_out_key_marker2depth_seed1000000/episode.h5'
CENTER=128/255
GAIN=127/255
CORE=20.
SMALL=40.
DEADZONE=0.
RAMP=3.
JOIN=160.
TAIL=240.
DROP_WIDTH=4.
RISE_WIDTH=12.
TITLES=['Marker RGB','NEW: Linear40 / no DZ','Previous: gain40 / DZ5','Previous NL20 / cap400',
        'Linear +/-80','Linear +/-160','Linear +/-240']


def linear(x,limit):
    return (128+127*np.clip(np.asarray(x)/limit,-1,1))/255


def adaptive(x, *, deadzone=DEADZONE, small_range=SMALL, join_px=JOIN, dtype=np.float32):
    """Monotone smooth encoding; old deadzone mapping retained for comparison.

    No deadzone by default: entire +/-20 px matches Linear40.
    20..160 integrates a positive gain with smoothstep boundary transitions.
    Optional deadzone and onset support comparison with the previous candidate.
    """
    x=np.asarray(x,dtype=np.float64)
    if not np.isfinite(x).all():raise ValueError('Non-finite displacement')
    m=np.abs(x)
    ramp=RAMP if deadzone>0 else 0.
    k=GAIN/small_range
    y0=k*(CORE-deadzone-ramp/2)
    y1=GAIN*join_px/TAIL
    width=join_px-CORE
    height=y1-y0
    if height<=0 or width<=0:raise ValueError("Transition must have positive width and height")
    secant=height/width
    d0,d1=k,GAIN/TAIL
    z=np.clip((m-CORE)/width,0,1)
    blend=z*(1-z)
    transition=y0+height*(secant*z*z+d0*blend)/(secant+(d0+d1-2*secant)*blend)
    if not deadzone:
        # Spread the limited output budget across intermediate motion rather
        # than letting a rational spline become almost flat in the middle.
        baseline=(height-d0*DROP_WIDTH/2-d1*RISE_WIDTH/2)/(width-(DROP_WIDTH+RISE_WIDTH)/2)
        if baseline<=0 or DROP_WIDTH+RISE_WIDTH>=width:
            raise ValueError('Transition does not have enough output budget')
        t=np.clip(m-CORE,0,width)
        first=np.minimum(t,DROP_WIDTH)
        a=first/DROP_WIDTH
        integral=d0*first+(baseline-d0)*DROP_WIDTH*(a**3-.5*a**4)
        integral+=baseline*np.clip(t-DROP_WIDTH,0,width-DROP_WIDTH-RISE_WIDTH)
        last=np.clip(t-(width-RISE_WIDTH),0,RISE_WIDTH)
        a=last/RISE_WIDTH
        integral+=baseline*last+(d1-baseline)*RISE_WIDTH*(a**3-.5*a**4)
        transition=y0+integral
    if ramp:
        t=np.clip(m-deadzone,0,ramp)
        small=np.where(m<=deadzone,0,np.where(m<deadzone+ramp,k*t*t/(2*ramp),k*(m-deadzone-ramp/2)))
    else:
        small=k*m
    amplitude=np.where(m<=CORE,small,np.where(m<join_px,transition,GAIN*np.minimum(m/TAIL,1)))
    return (CENTER+np.sign(x)*amplitude).astype(dtype)


def rgb(chw):
    return np.rint(np.clip(chw.transpose(1,2,0),0,1)*255).astype(np.uint8)


def render(rows,title,controlled):
    canvas=np.full((60+250*len(rows),224*len(TITLES),3),22,np.uint8)
    def text(value,x,y,scale=.52):
        cv2.putText(canvas,value,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(240,240,240),1,cv2.LINE_AA)
    text(title,10,24,.63)
    text('Shared TPS: isolates encoding | NEW: |x|<=20 equals Linear40 (>Linear80 slope); |x|>=160 equals Linear240; NO deadzone' if controlled
         else 'Actual pipelines: NEW and previous nonlinear use TPS; linear uses triangle / IDW',10,46,.52)
    for row,(name,marker,images) in enumerate(rows):
        top=84+250*row
        panels=[cv2.resize(marker,(224,224),interpolation=cv2.INTER_AREA)]+[rgb(im) for im in images]
        for col,(panel,label) in enumerate(zip(panels,TITLES)):
            text(name+' '+label,224*col+4,top-8,.41)
            canvas[top:top+224,224*col:224*(col+1)]=panel
    return canvas


def main():
    for sub in ('frames_shared_tps','frames_actual_pipeline'):(OUT/sub).mkdir(parents=True,exist_ok=True)
    x=np.linspace(-300,300,120001)
    y=adaptive(x)
    assert np.all(np.diff(y)>=0) and y.min()>=0 and y.max()<=1
    small=np.linspace(-20,20,4001)
    np.testing.assert_allclose(adaptive(small),linear(small,SMALL),atol=6e-8)
    small_slopes=np.diff(adaptive(small,dtype=np.float64))/np.diff(small)
    assert np.all(small_slopes>GAIN/80)
    large=np.r_[np.linspace(-300,-JOIN,1001),np.linspace(JOIN,300,1001)]
    np.testing.assert_allclose(adaptive(large),linear(large,240),atol=6e-8)
    np.testing.assert_allclose(y+adaptive(-x),2*CENTER,atol=1.3e-7)
    slope_checks=[]
    for point,expected in ((0,GAIN/SMALL),(CORE,GAIN/SMALL),(JOIN,GAIN/TAIL)):
        # Check the continuous mapping before float32 output quantization.
        eps=.0001
        left=float((adaptive(point,dtype=np.float64)-adaptive(point-eps,dtype=np.float64))/eps)
        right=float((adaptive(point+eps,dtype=np.float64)-adaptive(point,dtype=np.float64))/eps)
        np.testing.assert_allclose([left,right],expected,rtol=.005,atol=2e-6)
        slope_checks.append(dict(px=point,left=left,right=right,expected=expected))
    middle=np.linspace(CORE,JOIN,10001)
    middle_slopes=np.diff(adaptive(middle,dtype=np.float64))/np.diff(middle)
    assert middle_slopes.min()>0
    fig,axes=plt.subplots(1,3,figsize=(17,4.4),layout='constrained')
    for ax,bound in zip(axes[:2],(30,260)):
        q=np.linspace(-bound,bound,5001)
        ax.plot(q,adaptive(q),label='NEW Linear40 core / no deadzone',color='#d84b16',lw=2.8)
        ax.plot(q,adaptive(q,deadzone=5,small_range=40,join_px=120),label='Previous Linear40 core / deadzone5',color='#2878ad',ls=':')
        ax.plot(q,candidate(q,400,20),label='Previous NL20/cap400',color='purple',ls='--')
        for limit in (20,40,80,160,240):ax.plot(q,linear(q,limit),label=f'Linear{limit}',alpha=.8)
        ax.axvspan(-CORE,CORE,color='grey',alpha=.1)
        ax.set(xlabel='Signed displacement (sensor px)',ylabel='G / B',ylim=(-.02,1.02))
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    q=np.linspace(0,240,6001)
    axes[2].plot(q,np.gradient(adaptive(q,dtype=np.float64),q),color='#d84b16',label='NEW local gain')
    for limit in (40,80,240):axes[2].axhline(GAIN/limit,ls='--',label=f'Linear{limit} gain')
    axes[2].set(xlabel='Absolute displacement (px)',ylabel='Output change per pixel',ylim=(0,.014))
    axes[2].grid(alpha=.2);axes[2].legend(fontsize=8)
    fig.suptitle('NO deadzone; entire +/-20 equals Linear40; smooth bridge 20..160; Linear240 beyond 160')
    fig.savefig(OUT/'mapping_curves.png',dpi=170)
    fig.savefig(OUT/'mapping_curves.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
    q=np.linspace(-25,25,10001)
    axes[0].plot(q,adaptive(q),color='#d84b16',lw=2.5,label='NEW gain40 / no deadzone')
    axes[0].plot(q,adaptive(q,deadzone=5,small_range=40,join_px=120),ls=':',label='Previous gain40 / deadzone5')
    for limit in (20,40,80,240):axes[0].plot(q,linear(q,limit),label=f'Linear{limit}',alpha=.8)
    axes[0].axvspan(-CORE,CORE,alpha=.1,color='grey')
    axes[0].set(xlabel='Signed displacement (px)',ylabel='G / B',ylim=(.15,.85))
    q=np.linspace(0,25,10001)
    axes[1].plot(q,np.gradient(adaptive(q,dtype=np.float64),q),color='#d84b16',lw=2.5,label='NEW slope')
    axes[1].plot(q,np.gradient(adaptive(q,deadzone=5,small_range=40,join_px=120,dtype=np.float64),q),ls=':',label='Previous slope')
    for limit in (40,80,240):axes[1].axhline(GAIN/limit,ls='--',label=f'Linear{limit} slope')
    axes[1].set(xlabel='Absolute displacement (px)',ylabel='Output change per pixel',ylim=(0,.014))
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('NO deadzone: entire +/-20 px slope is 2x Linear80 (6x Linear240)')
    fig.savefig(OUT/'small_deform_detail.png',dpi=170);plt.close(fig)
    previews={True:{},False:{}};metrics=[]
    with h5py.File(SOURCE,'r') as source,h5py.File(EPISODE,'r') as episode,h5py.File(OUT/'adaptive_arrays.h5','w') as output:
        actions=source['source_action_index'][:]
        output.create_dataset('source_action_index',data=actions)
        output.attrs['status']='preview only; production unchanged'
        output.attrs['encoding']='no deadzone; Linear40 within +/-20; integrated positive gain 20..160 (smoothstep transitions 20..24 and 148..160); Linear240 beyond 160; TPS'
        for side in ('left','right'):output.create_dataset(side,(len(actions),3,224,224),dtype='f4',compression='gzip')
        for i,action in enumerate(actions):
            rows={True:[],False:[]}
            for side in ('left','right'):
                field=source[f'{side}/displacement_tps_px'][i]
                old=source[f'{side}/depth_deform_nonlinear'][i]
                new=np.concatenate([old[:1],adaptive(field)])
                previous=np.concatenate([old[:1],candidate(field,400,20)])
                previous_deadzone=np.concatenate([old[:1],adaptive(field,deadzone=5,small_range=40,join_px=120)])
                actual=[source[f'{side}/depth_deform_{u}'][i] for u in (80,160,240)]
                controlled=[np.concatenate([old[:1],linear(field,u)]) for u in (80,160,240)]
                for image in [new,previous_deadzone,previous]+actual+controlled:
                    np.testing.assert_array_equal(image[0],old[0])
                    assert np.isfinite(image).all() and image.min()>=0 and image.max()<=1
                output[side][i]=new
                marker=episode[f'frames/tactile/{side}_tactile/rgb_marker'][i]
                rows[True].append((side[0].upper(),marker,[new,previous_deadzone,previous]+controlled))
                rows[False].append((side[0].upper(),marker,[new,previous_deadzone,previous]+actual))
                metrics.append(dict(source_action=int(action),side=side,
                    deadzone_component_percent=float(100*(np.abs(field)<=DEADZONE).mean()),
                    clipped_components_percent=float(100*(np.abs(field)>=TAIL).mean()),
                    previous_near_endpoint_percent=float(100*((previous[1:]<.05)|(previous[1:]>.95)).mean()),
                    new_near_endpoint_percent=float(100*((new[1:]<.05)|(new[1:]>.95)).mean())))
            for controlled,sub in ((True,'frames_shared_tps'),(False,'frames_actual_pipeline')):
                image=render(rows[controlled],f'Preview | source action {action:03d} | new adaptive nonlinear vs previous / linear',controlled)
                Image.fromarray(image).save(OUT/sub/f'action_{action:03d}.png')
                previews[controlled][int(action)]=image
            print(f'Observation {i+1}/{len(actions)}',flush=True)
    for controlled,name in ((True,'selected_observations.png'),(False,'selected_actual_pipeline.png')):
        Image.fromarray(np.concatenate([previews[controlled][a] for a in (32,112,176)])).save(OUT/name)
    training={True:[],False:[]}
    for path in sorted((ROOT/'.cache/key_nonlinear_comparison').glob('case_*_arrays.npz')):
        with np.load(path) as data:
            field=data['tps_displacement_px'];depth=data['actual_pipeline'][-1,:1]
            new=np.concatenate([depth,adaptive(field)]);previous=np.concatenate([depth,candidate(field,400,20)])
            previous_deadzone=np.concatenate([depth,adaptive(field,deadzone=5,small_range=40,join_px=120)])
            controlled=[np.concatenate([depth,linear(field,u)]) for u in (80,160,240)]
            training[True].append((path.stem,data['marker_rgb'],[new,previous_deadzone,previous]+controlled))
            training[False].append((path.stem,data['marker_rgb'],[new,previous_deadzone,previous]+list(data['actual_pipeline'][1:4])))
    for controlled,name in ((True,'training_contacts.png'),(False,'training_actual_pipeline.png')):
        if training[controlled]:Image.fromarray(render(training[controlled],'Training contacts | NEW adaptive nonlinear',controlled)).save(OUT/name)
    values=np.array([0,1,4,5,6,8,10,20,40,80,120,160,240],dtype=float)
    summary=dict(status='preview only; production unchanged',method='integrated positive gain with cubic smoothstep boundary transitions',
        parameters=dict(core_px=CORE,join_px=JOIN,small_gain_equivalent_linear_px=SMALL,deadzone_px=DEADZONE,onset_width_px=0,large_linear_range_px=TAIL,center=CENTER,gain=GAIN),
        rationale='No deadzone; exactly Linear40 within +/-20 px, giving 2x Linear80 slope; exactly Linear240 beyond 160 px.',
        tradeoff='Encoding gain amplifies signal and noise together; no denoising is performed. Midrange compression reconnects high gain to Linear240. Requires training validation; visual contrast alone does not establish signal-to-noise improvement.',
        checks=dict(monotone=True,no_deadzone=True,exact_small_Linear40=True,small_gain_Linear40=True,small_gain_above_Linear80=True,exact_large_Linear240=True,midrange_min_gain=float(middle_slopes.min()),slopes=slope_checks),
        values=[dict(px=float(x),adaptive=float(y),previous_deadzone=float(a),linear240=float(b),previous=float(c)) for x,y,a,b,c in zip(values,adaptive(values),adaptive(values,deadzone=5,small_range=40,join_px=120),linear(values,240),candidate(values,400,20))],metrics=metrics)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='metrics'},indent=2),flush=True)
    print('Saved',OUT,flush=True)


if __name__=='__main__':main()
