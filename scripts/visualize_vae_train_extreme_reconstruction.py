"""Compare six full VAE-S final checkpoints on one train frame and one extreme rollout frame."""
import csv
import json
from pathlib import Path
import random
import sys

import cv2
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'encoder'))
from modality import tactile_from_arrays
from linear_depth_deform import require_depth_deform_encoding
from nonlinear_depth_deform import encode_displacement, _tps_interpolation_map
from radial_nonlinear_depth_deform import decode_radial_displacement
from depth_marker_to_deform import _interpolation_map
from image_io import read_rgb
from train_common import TactileFrames
from VAE.model import VAE

MODES=[f'depth_deform_{n}' for n in (20,80,160,240)]+['depth_deform_nonlinear','depth_deform_nonlinear_radial']
LABELS=['Linear20','Linear80','Linear160','Linear240','Original nonlinear','Radial nonlinear']
OUT=ROOT/'.cache/vae_train_extreme_reconstruction_radial'
EPISODE=ROOT/'.cache/exports/exports/pull_out_key_marker2depth_seed1000000/episode.h5'


def frame_raw(group,step):
    raw={key:group[key][step] for key in ('depth','marker')}
    for key in ('marker_full','press_depth','marker_id','marker_in_view'):
        if key in group:raw[key]=group[key][step]
    value=group['rgb_marker'][step]
    rgb=value if isinstance(value,np.ndarray) and value.ndim==3 else read_rgb(group['rgb_marker'],step)
    return raw,rgb


def stats(raw):
    marker=raw.get('marker_full',raw['marker'])
    motion=np.linalg.norm(marker[1]-marker[0],axis=-1)
    return dict(marker_p95_px=float(np.percentile(motion,95)),marker_max_px=float(motion.max()),
        max_press_depth_mm=float(raw['press_depth'].max()) if 'press_depth' in raw else None,
        camera_depth_min_mm=float(raw['depth'].min()),full_marker_available='marker_full' in raw,
        outside_markers=int((~raw['marker_in_view']).sum()) if 'marker_in_view' in raw else None)


def select_cases(config):
    dataset=TactileFrames([Path(p) for p in config['train_files']],limit=config['train_count'],
                          seed=config['seed'],input_mode='depth_deform_nonlinear_radial')
    candidates=[]
    for path,side,step in random.Random(20261014).sample(dataset.rows,min(128,len(dataset.rows))):
        with h5py.File(path,'r') as f:
            group=f[f'tactile/{side}_tactile']
            raw={key:group[key][step] for key in ('depth','marker')}
            for key in ('press_depth','marker_full'):
                if key in group:raw[key]=group[key][step]
        s=stats(raw)
        candidates.append(dict(path=path,side=side,step=step,stats=s))
    ordinary=[c for c in candidates if 2<=c['stats']['marker_p95_px']<=20 and
              (c['stats']['max_press_depth_mm'] is None or .1<=c['stats']['max_press_depth_mm']<=2)]
    if not ordinary:ordinary=[c for c in candidates if c['stats']['marker_p95_px']>0]
    if not ordinary:raise ValueError('No ordinary nonzero-contact training candidates')
    ordinary.sort(key=lambda c:c['stats']['marker_p95_px'])
    selected=ordinary[len(ordinary)//2]
    row=(selected['path'],selected['side'],selected['step'])
    assert row in dataset.rows and selected['path'] not in config['val_files']
    with h5py.File(selected['path'],'r') as f:raw,rgb=frame_raw(f[f"tactile/{selected['side']}_tactile"],selected['step'])
    train=dict(key='train_normal',label='Ordinary training contact',raw=raw,rgb=rgb,
        metadata=dict(**selected,selection='Median marker-p95 among 128 seeded actual sampled train rows with p95 2..20 px; require press max .1..2 mm when press_depth is available',
                      sampled_train_membership_verified=True))
    ranked=[]
    with h5py.File(EPISODE,'r') as f:
        actions=f['frames/source_action_index'][:]
        for i,action in enumerate(actions):
            for side in ('left','right'):
                raw,rgb=frame_raw(f[f'frames/tactile/{side}_tactile'],i)
                s=stats(raw)
                score=s['marker_p95_px']*s['max_press_depth_mm']
                ranked.append((score,i,int(action),side,s))
        score,i,action,side,s=max(ranked,key=lambda c:c[0])
        raw,rgb=frame_raw(f[f'frames/tactile/{side}_tactile'],i)
    test=dict(key='test_extreme',label='Extreme held-out rollout contact',raw=raw,rgb=rgb,
        metadata=dict(path=str(EPISODE),side=side,step=i,source_action=action,stats=s,
            selection='Max marker-p95 times saved press-depth maximum across all 19 rollout observations and both sensors',score=score))
    return [train,test],candidates


def fields(raw):
    fixed,current=np.asarray(raw.get('marker_full',raw['marker']),dtype=np.float32)
    h,w=raw['depth'].shape
    vertices,weights=_interpolation_map(fixed.tobytes(),len(fixed),h,w,224)
    linear=np.einsum('nvc,nv->nc',(current-fixed)[vertices],weights).reshape(224,224,2).transpose(2,0,1)
    tps=(_tps_interpolation_map(fixed.tobytes(),len(fixed),h,w,224) @ (current-fixed)).reshape(224,224,2).transpose(2,0,1)
    return linear,tps



def inverse(gb,mode):
    """Project predicted components to displayable encoding bounds, then invert.

    Linear bounds are their exact clipping endpoints. Original nonlinear
    is asymptotic: use +/-400 sensor px as an explicit display/inversion bound.
    Save the projection mask so these bounds are not hidden in physical errors.
    """
    gb=np.asarray(gb,dtype=np.float64)
    if mode=='depth_deform_nonlinear_radial':
        decoded,projected=decode_radial_displacement(np.moveaxis(gb,0,-1),max_radius_px=400.)
        return np.moveaxis(decoded,-1,0),np.broadcast_to(projected,gb.shape).copy()
    if mode=='depth_deform_nonlinear':
        lower,upper=encode_displacement([-400,400]).astype(np.float64)
    else:lower,upper=1/255,1.
    projected=(gb<lower)|(gb>upper)
    bounded=np.clip(gb,lower,upper)
    if mode=='depth_deform_nonlinear':
        signed=bounded-.5;a=np.abs(signed)
        tail=20+(40/3)*(.2/np.maximum(.5-a,1e-12)-1)
        decoded=np.sign(signed)*np.where(a<=.3,a/.015,tail)
    else:
        limit=int(mode.rsplit('_',1)[1])
        decoded=(bounded*255-128)/127*limit
    return decoded.astype(np.float32),projected


@torch.inference_mode()
def reconstruct(model,images):
    captured=[]
    hook=model.pixels.register_forward_hook(lambda module,inputs,output:captured.append(output.detach()))
    try:model(images)
    finally:hook.remove()
    return captured[0].reshape(-1,14,14,16,16,3).permute(0,5,1,3,2,4).reshape(-1,3,224,224)


def arrows(ax,field,case,error=False):
    h,w=case['raw']['depth'].shape
    indices=np.arange(12,224,24)
    xx,yy=np.meshgrid((indices+.5)*w/224,(indices+.5)*h/224)
    vector=field[:,indices[:,None],indices[None,:]]
    scale=1 if case['key']=='train_normal' else 4
    maximum=40 if case['key']=='train_normal' else 300
    q=ax.quiver(xx,yy,vector[0],vector[1],np.hypot(*vector),cmap='viridis',norm=Normalize(0,maximum),
        angles='xy',scale_units='xy',scale=scale,pivot='tail',width=.004,minlength=0,headwidth=3.5)
    ax.set(xlim=(0,w),ylim=(h,0),aspect='equal',facecolor='#f5f6f8')
    ax.set_xticks([]);ax.set_yticks([])
    ax.quiverkey(q,.68,.05,10 if scale==1 else 100,'10 px' if scale==1 else '100 px',coordinates='axes',labelpos='E',fontproperties={'size':8})
    return q


def save_figure(fig,name):
    fig.savefig(OUT/f'{name}.png',dpi=150,facecolor='white')
    fig.savefig(OUT/f'{name}.pdf',facecolor='white')
    plt.close(fig)


def render(cases,inputs,predictions,decoded_input,decoded_pred,metrics,checkpoints):
    for sample,case in enumerate(cases):
        title=f"{case['label']} | {Path(case['metadata']['path']).name} | {case['metadata']['side']} | "
        title+=f"source action {case['metadata']['source_action']}" if 'source_action' in case['metadata'] else f"frame {case['metadata']['step']}"
        for kind in ('rgb','depth','deform'):
            fig,axes=plt.subplots(6,3,figsize=(12,17),layout='constrained')
            for row,(mode,label) in enumerate(zip(MODES,LABELS)):
                metric=next(m for m in metrics if m['mode']==mode and m['sample']==case['key'])
                target,pred=inputs[row,sample],predictions[row,sample]
                for col in range(3):
                    ax=axes[row,col]
                    if kind=='rgb':
                        if col<2:ax.imshow((target if col==0 else pred).transpose(1,2,0).clip(0,1))
                        else:im=ax.imshow(np.abs(pred-target).mean(0),cmap='magma',vmin=0,vmax=.15)
                    elif kind=='depth':
                        values=(target[0]*5,pred[0]*5,np.abs(pred[0]-target[0])*5)[col]
                        im=ax.imshow(values,cmap='magma' if col==2 else 'viridis',vmin=0,vmax=1 if col==2 else 5)
                    else:
                        source,estimate=decoded_input[row,sample],decoded_pred[row,sample]
                        im=arrows(ax,(source,estimate,estimate-source)[col],case,error=col==2)
                    ax.set_xticks([]);ax.set_yticks([])
                    if row==0:ax.set_title(('Encoded input','Reconstruction','Mean absolute error')[col] if kind=='rgb' else
                                          ('Input depth','Reconstructed depth','Absolute error')[col] if kind=='depth' else
                                          ('Decoded input deform','Decoded reconstruction','Residual: recon - input')[col],fontsize=11)
                    if col==0:ax.set_ylabel(f"{label}\nfull VAE epoch {checkpoints[row]['epoch']}",fontsize=11)
                axes[row,2].set_xlabel(f"MSE {metric['encoded_mse']:.5f}" if kind=='rgb' else
                    f"Depth MAE {metric['depth_mae_mm']:.3f} mm" if kind=='depth' else
                    f"Vector EPE {metric['vector_epe_vs_encoded_input_px']:.2f} px\nProjected components {metric['inverse_projection_percent']:.2f}%",fontsize=9)
            if kind=='rgb':
                fig.colorbar(im,ax=axes[:,2].tolist(),shrink=.4,label='Mean |RGB error| (normalized)')
                subtitle='R: camera indentation / 5 mm; G/B: encoded dx/dy | RGB display clipped; errors use raw predictions'
            elif kind=='depth':
                fig.colorbar(ScalarMappable(norm=Normalize(0,5),cmap='viridis'),ax=axes[:,:2].ravel().tolist(),shrink=.4,label='Camera indentation (mm)')
                fig.colorbar(im,ax=axes[:,2].tolist(),shrink=.4,label='Absolute depth error (mm)')
                subtitle='Same 0..5 mm depth scale; depth = clip((30-camera depth)/5) | Saved press_depth is not the target'
            else:
                fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.4,label='Displacement magnitude (sensor px)')
                scale=1 if case['key']=='train_normal' else 4
                subtitle=f'Reference-coordinate fields | arrow length = displacement / {scale} | +x right, +y down\nInputs decode each encoding after its clipping; original nonlinear inverse shown within +/-400 px'
            fig.suptitle(title+'\n'+subtitle+'\nFull last.pth checkpoints; deterministic latent means; not best encoder-only checkpoints',fontsize=12)
            save_figure(fig,f"{case['key']}_{kind}")
        fig,axes=plt.subplots(1,4,figsize=(16,4.4),layout='constrained')
        axes[0].imshow(case['rgb']);axes[0].set_title('Marker RGB (current image)')
        depth=inputs[0,sample,0]*5
        im=axes[1].imshow(depth,cmap='viridis',vmin=0,vmax=5);axes[1].set_title('Depth target (0..5 mm)')
        arrows(axes[2],case['linear_field'],case);axes[2].set_title('Raw linear/IDW field')
        q=arrows(axes[3],case['tps_field'],case);axes[3].set_title('Raw TPS field')
        for ax in axes:ax.set_xticks([]);ax.set_yticks([])
        fig.colorbar(im,ax=axes[1],shrink=.65,label='mm');fig.colorbar(q,ax=axes[2:].tolist(),shrink=.65,label='sensor px')
        s=stats(case['raw'])
        fig.suptitle(title+f"\nMarker displacement p95 {s['marker_p95_px']:.1f} px; max {s['marker_max_px']:.1f} px | raw fields before encoding/clipping",fontsize=12)
        save_figure(fig,f"{case['key']}_reference")


def render_radial_summary(cases,inputs,predictions,decoded_input,decoded_pred):
    row=MODES.index('depth_deform_nonlinear_radial')
    fig,axes=plt.subplots(2,6,figsize=(20,7.5),layout='constrained')
    titles=['Depth/deform input','Depth/deform reconstruction','Depth input','Depth reconstruction',
            'Deform input','Deform reconstruction']
    for sample,case in enumerate(cases):
        target,pred=inputs[row,sample],predictions[row,sample]
        for col,image in enumerate((target,pred)):
            axes[sample,col].imshow(image.transpose(1,2,0).clip(0,1))
        for col,values in ((2,target[0]*5),(3,pred[0]*5)):
            axes[sample,col].imshow(values,cmap='viridis',vmin=0,vmax=5)
        for col,field in ((4,decoded_input[row,sample]),(5,decoded_pred[row,sample])):
            q=arrows(axes[sample,col],field,case)
        fig.colorbar(q,ax=axes[sample,4:].tolist(),shrink=.6,label='sensor px')
        for col,ax in enumerate(axes[sample]):
            ax.set_xticks([]);ax.set_yticks([])
            if sample==0:ax.set_title(titles[col],fontsize=12)
        s=stats(case['raw'])
        axes[sample,0].set_ylabel(f"{'TRAIN normal' if sample==0 else 'TEST extreme'}\nMarker p95 {s['marker_p95_px']:.1f} px",fontsize=12)
        axes[sample,4].set_xlabel('Arrow length: 1:1' if sample==0 else 'Arrow length: 1:4',fontsize=10)
    fig.colorbar(ScalarMappable(norm=Normalize(0,5),cmap='viridis'),ax=axes[:,2:4].ravel().tolist(),shrink=.6,label='depth (mm)')
    fig.suptitle('Radial nonlinear VAE-S reconstruction | full final checkpoint | deterministic latent mean\n'
                 'Input vs reconstruction: RGB depth/deform, depth 0..5 mm, and decoded displacement arrows',fontsize=15)
    save_figure(fig,'radial_summary')


def main():
    torch.set_num_threads(4);torch.manual_seed(42)
    OUT.mkdir(parents=True,exist_ok=True)
    runs=[max((p.parent for p in (ROOT/'encoder_results'/mode/'univtac/VAE/S').glob('*/last.pth')),key=lambda p:p.name) for mode in MODES]
    configs=[json.loads((p/'config.json').read_text()) for p in runs]
    for config in configs:
        for key in ('train_files','val_files','seed','train_count'):
            assert config[key]==configs[0][key],key
    cases,candidates=select_cases(configs[0])
    metadata=[dict(key=c['key'],label=c['label'],**c['metadata']) for c in cases]
    (OUT/'samples.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('Selected samples:',json.dumps(metadata,indent=2),flush=True)
    for case in cases:case['linear_field'],case['tps_field']=fields(case['raw'])
    inputs=[];predictions=[];decoded_input=[];decoded_pred=[];projections=[];metrics=[];checkpoints=[]
    for mode,run,config in zip(MODES,runs,configs):
        checkpoint=torch.load(run/'last.pth',map_location='cpu',weights_only=False)
        require_depth_deform_encoding(checkpoint,mode)
        assert checkpoint['input_mode']==mode
        model=VAE('S',3,**config['model_options']).eval()
        model.load_state_dict(checkpoint['training_model'],strict=True)
        images=torch.stack([tactile_from_arrays(c['raw'],mode) for c in cases])
        pred=reconstruct(model,images).numpy();target=images.numpy()
        assert np.isfinite(pred).all()
        target_fields=[];pred_fields=[];projected=[]
        for i,case in enumerate(cases):
            truth,mask_truth=inverse(target[i,1:],mode);estimate,mask_pred=inverse(pred[i,1:],mode)
            raw_field=case['linear_field'] if mode.startswith('depth_deform_') and mode.rsplit('_',1)[1].isdigit() else case['tps_field']
            target_fields.append(truth);pred_fields.append(estimate);projected.append(mask_pred)
            metrics.append(dict(mode=mode,sample=case['key'],checkpoint_epoch=int(checkpoint['epoch']),
                encoded_mse=float(np.mean((pred[i]-target[i])**2)),depth_mae_mm=float(np.abs(pred[i,0]-target[i,0]).mean()*5),
                vector_epe_vs_encoded_input_px=float(np.linalg.norm(estimate-truth,axis=0).mean()),
                encoding_loss_vs_raw_field_px=float(np.linalg.norm(truth-raw_field,axis=0).mean()),
                reconstruction_epe_vs_raw_field_px=float(np.linalg.norm(estimate-raw_field,axis=0).mean()),
                inverse_projection_percent=float(mask_pred.mean()*100),input_inverse_projection_percent=float(mask_truth.mean()*100)))
        inputs.append(target);predictions.append(pred);decoded_input.append(target_fields);decoded_pred.append(pred_fields);projections.append(projected)
        checkpoints.append(dict(mode=mode,path=str(run/'last.pth'),epoch=int(checkpoint['epoch']),val_loss=float(checkpoint['val_loss']),
            best_encoder_only=str(run/'encoder.pth'),note='encoder.pth lacks decoder; full final last.pth used without mixing encoder/decoder epochs'))
        print('Reconstructed',mode,'epoch',checkpoint['epoch'],flush=True)
        del model,checkpoint
    inputs=np.stack(inputs);predictions=np.stack(predictions);decoded_input=np.stack(decoded_input);decoded_pred=np.stack(decoded_pred)
    np.savez_compressed(OUT/'arrays.npz',modes=np.array(MODES),inputs=inputs,reconstructions=predictions,
        decoded_input_px=decoded_input,decoded_reconstruction_px=decoded_pred,inverse_projection_masks=np.asarray(projections),
        raw_linear_fields_px=np.stack([c['linear_field'] for c in cases]),raw_tps_fields_px=np.stack([c['tps_field'] for c in cases]))
    for case in cases:np.savez_compressed(OUT/f"{case['key']}_raw.npz",marker_rgb=case['rgb'],**case['raw'])
    with (OUT/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
    render(cases,inputs,predictions,decoded_input,decoded_pred,metrics,checkpoints)
    render_radial_summary(cases,inputs,predictions,decoded_input,decoded_pred)
    summary=dict(samples=metadata,training_candidates_scanned=len(candidates),checkpoints=checkpoints,metrics=metrics,
        evaluation='Full final VAE-S checkpoints, CPU float32 eval, latent mean (no stochastic latent sampling).',
        notes=['Selected train frame is in the actual 80000-frame sampled training set shared by all six models.',
            'Rollout is a distinct failed pull_out_key trajectory, not a training frame; one extreme observation is not a generalization benchmark.',
            'encoder.pth stores no decoder; reconstruction uses last.pth at the recorded epoch, not the best encoder checkpoint.',
            'Physical displacement inversion uses each mode, not a universal linear formula.',
            'Linear components project to valid endpoints; original nonlinear uses +/-400 px per component, radial nonlinear uses a 400 px vector-radius bound. Projection counts are reported.',
            'Encoded MSE and depth MAE use unclipped predictions. Vector metrics use projected decoded fields and distinguish encoding clipping loss from reconstruction error.',
            'Each case uses its own declared arrow length and color scales; within a case all six models use identical scales.',
            'Training legacy markers have no full out-of-view coordinates unless marker_full is present; none are invented.',
            'Dense displacement is interpolated at reference coordinates; marker RGB is in current coordinates.'])
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('Saved',OUT,flush=True)


if __name__=='__main__':main()
