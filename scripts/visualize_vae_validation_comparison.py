"""Representative actual pretraining-val frames, six VAE-S reconstruction comparisons."""
import csv
import json
from pathlib import Path
import random

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import torch

from visualize_vae_train_extreme_reconstruction import (
    ROOT, MODES, TactileFrames, frame_raw, stats, fields, inverse, reconstruct,
    VAE, tactile_from_arrays, require_depth_deform_encoding,
)

OUT=ROOT/'.cache/vae_validation_comparison_radial'
DISPLAY_MODES=['depth_deform_nonlinear_radial','depth_deform_20','depth_deform_80','depth_deform_160','depth_deform_240','depth_deform_nonlinear']
DISPLAY_LABELS=['Radial nonlinear','Linear20','Linear80','Linear160','Linear240','Original nonlinear']


def select(config):
    val=TactileFrames([Path(p) for p in config['val_files']],limit=config['val_count'],
        seed=config['seed']+1,input_mode='depth_deform_nonlinear_radial')
    groups={}
    for row in val.rows:groups.setdefault(Path(row[0]).parent.parent.name,[]).append(row)
    rng=random.Random(20261014)
    cases=[]
    train_paths=set(config['train_files'])
    for shape,rows in sorted(groups.items()):
        candidates=[]
        for path,side,step in rng.sample(rows,min(16,len(rows))):
            with h5py.File(path,'r') as f:
                g=f[f'tactile/{side}_tactile']
                raw={k:g[k][step] for k in ('depth','marker')}
                for k in ('press_depth','marker_full'):
                    if k in g:raw[k]=g[k][step]
            candidates.append((stats(raw)['marker_p95_px'],path,side,step))
        nonzero=[c for c in candidates if c[0]>.5]
        ranked=sorted(nonzero or candidates)
        score,path,side,step=ranked[len(ranked)//2]
        assert path in config['val_files'] and path not in train_paths
        assert (path,side,step) in rows
        with h5py.File(path,'r') as f:raw,rgb=frame_raw(f[f'tactile/{side}_tactile'],step)
        linear,tps=fields(raw)
        metadata=dict(shape=shape,path=path,side=side,frame=step,stats=stats(raw),
            actual_sampled_val_membership_verified=True,not_in_train_files=True,
            candidates_scanned=len(candidates),selection='Median marker-p95 of up to 16 seeded actual val candidates for this shape; prefer p95 > .5 px; no reconstruction-error selection')
        cases.append(dict(metadata=metadata,raw=raw,rgb=rgb,linear=linear,tps=tps))
        print('Selected val',shape,Path(path).name,side,step,'p95',round(score,2),flush=True)
    return cases,len(val)


def quiver(ax,field,case,norm):
    h,w=case['raw']['depth'].shape
    ix=np.arange(12,224,24)
    xx,yy=np.meshgrid((ix+.5)*w/224,(ix+.5)*h/224)
    f=field[:,ix[:,None],ix[None,:]]
    q=ax.quiver(xx,yy,f[0],f[1],np.hypot(*f),cmap='viridis',norm=norm,
        angles='xy',scale_units='xy',scale=1,pivot='tail',width=.004,minlength=0)
    ax.set(xlim=(0,w),ylim=(h,0),aspect='equal',facecolor='#f5f6f8')
    ax.quiverkey(q,.65,.05,10,'10 px',labelpos='E',coordinates='axes',fontproperties={'size':8})
    return q


def save(fig,name):
    for ext in ('png','pdf'):fig.savefig(OUT/f'{name}.{ext}',dpi=145,facecolor='white')
    plt.close(fig)


def render(cases,arrays,metrics,checkpoints):
    lookup={mode:MODES.index(mode) for mode in MODES}
    epochs={c['mode']:c['epoch'] for c in checkpoints}
    metric_lookup={(m['mode'],m['sample']):m for m in metrics}
    lengths=np.linalg.norm(arrays['decoded_inputs'],axis=2)
    vmax=max(40,float(np.ceil(np.percentile(lengths,99)/10)*10))
    norm=Normalize(0,vmax)
    fig,axes=plt.subplots(2,7,figsize=(21,6.4),layout='constrained')
    for ax,case in zip(axes.ravel(),cases):
        m=case['metadata'];ax.imshow(case['rgb']);ax.axis('off')
        ax.set_title(f"{m['shape']} | {Path(m['path']).name}\n{m['side']} frame {m['frame']} | p95 {m['stats']['marker_p95_px']:.1f}px",fontsize=10)
    fig.suptitle('Actual encoder pretraining validation samples | one representative from each of 14 shapes',fontsize=16)
    save(fig,'val_samples_marker_rgb')
    for start in range(0,len(cases),4):
        indices=list(range(start,min(start+4,len(cases))))
        page=start//4+1
        for kind in ('rgb','depth','deform'):
            fig,axes=plt.subplots(2*len(indices),6,figsize=(22,6.4*len(indices)),squeeze=False,layout='constrained')
            for local,sample in enumerate(indices):
                case=cases[sample];meta=case['metadata']
                for recon in (0,1):
                    row=2*local+recon
                    for col,(mode,label) in enumerate(zip(DISPLAY_MODES,DISPLAY_LABELS)):
                        model=lookup[mode];ax=axes[row,col]
                        image=arrays['predictions' if recon else 'inputs'][model,sample]
                        if kind=='rgb':ax.imshow(image.transpose(1,2,0).clip(0,1))
                        elif kind=='depth':ax.imshow(image[0]*5,cmap='viridis',vmin=0,vmax=5)
                        else:quiver(ax,arrays['decoded_predictions' if recon else 'decoded_inputs'][model,sample],case,norm)
                        ax.set_xticks([]);ax.set_yticks([])
                        if row==0:ax.set_title(f'{label}\nfull VAE epoch {epochs[mode]}',fontsize=16,fontweight='bold',
                            color='#c34222' if col==0 else '#7a4d9a' if col==5 else '#236c99',pad=12)
                        if col==0:
                            ax.set_ylabel(f"VAL {meta['shape']}\n{Path(meta['path']).stem}/{meta['side'][0].upper()}/{meta['frame']} | p95 {meta['stats']['marker_p95_px']:.1f}px\n{'RECONSTRUCTION' if recon else 'INPUT'}",fontsize=11,fontweight='bold')
                            for spine in ax.spines.values():spine.set_edgecolor('#c34222');spine.set_linewidth(2)
                        if recon:
                            m=metric_lookup[(mode,sample)]
                            ax.set_xlabel(f"MSE {m['encoded_mse']:.5f}" if kind=='rgb' else
                                f"Depth MAE {m['depth_mae_mm']:.3f} mm" if kind=='depth' else
                                f"Vector EPE {m['vector_epe_px']:.2f} px",fontsize=10)
            if kind=='depth':fig.colorbar(ScalarMappable(norm=Normalize(0,5),cmap='viridis'),ax=axes.ravel().tolist(),shrink=.55,label='Depth target (mm), same 0..5 scale')
            if kind=='deform':fig.colorbar(ScalarMappable(norm=norm,cmap='viridis'),ax=axes.ravel().tolist(),shrink=.55,label='Displacement magnitude (sensor px), same scale for all val samples')
            view={'rgb':'3-channel depth/deform | fixed RGB [0,1] display',
                  'depth':'Depth single channel | 30 mm camera baseline / 5 mm range',
                  'deform':'Decoded deform | arrow length 1:1 sensor pixels | each input retains its encoding clipping'}[kind]
            fig.suptitle(f'Encoder PRETRAINING VAL | page {page} | Radial nonlinear / Linear20 / Linear80 / Linear160 / Linear240 / Original nonlinear\n'
                +view+'\nEach shape: INPUT row then RECONSTRUCTION row | full final last.pth, not best encoder-only weights',fontsize=16)
            save(fig,f'val_{kind}_page_{page:02d}')
            print('Rendered',kind,'page',page,flush=True)


def main():
    torch.set_num_threads(4);torch.manual_seed(42)
    OUT.mkdir(parents=True,exist_ok=True)
    runs=[max((p.parent for p in (ROOT/'encoder_results'/mode/'univtac/VAE/S').glob('*/last.pth')),key=lambda p:p.name) for mode in MODES]
    configs=[json.loads((p/'config.json').read_text()) for p in runs]
    for c in configs:
        for k in ('val_files','train_files','val_count','seed'):assert c[k]==configs[0][k],k
        assert not set(c['val_files']) & set(c['train_files'])
    cases,val_count=select(configs[0])
    (OUT/'samples.json').write_text(json.dumps([c['metadata'] for c in cases],indent=2)+'\n')
    arrays={k:[] for k in ('inputs','predictions','decoded_inputs','decoded_predictions','projection_masks')}
    metrics=[];checkpoints=[]
    for mode,run,config in zip(MODES,runs,configs):
        checkpoint=torch.load(run/'last.pth',map_location='cpu',weights_only=False)
        require_depth_deform_encoding(checkpoint,mode)
        model=VAE('S',3,**config['model_options']).eval()
        model.load_state_dict(checkpoint['training_model'],strict=True)
        images=torch.stack([tactile_from_arrays(c['raw'],mode) for c in cases])
        predictions=[]
        for start in range(0,len(cases),4):predictions.append(reconstruct(model,images[start:start+4]).numpy())
        pred=np.concatenate(predictions);target=images.numpy()
        assert np.isfinite(pred).all()
        decoded_in=[];decoded_out=[];masks=[]
        for sample,case in enumerate(cases):
            truth,_=inverse(target[sample,1:],mode);estimate,mask=inverse(pred[sample,1:],mode)
            decoded_in.append(truth);decoded_out.append(estimate);masks.append(mask)
            raw_field=case['linear'] if mode.rsplit('_',1)[1].isdigit() else case['tps']
            metrics.append(dict(mode=mode,sample=sample,shape=case['metadata']['shape'],epoch=int(checkpoint['epoch']),
                encoded_mse=float(np.mean((pred[sample]-target[sample])**2)),depth_mae_mm=float(np.abs(pred[sample,0]-target[sample,0]).mean()*5),
                vector_epe_px=float(np.linalg.norm(estimate-truth,axis=0).mean()),
                encoding_loss_vs_raw_px=float(np.linalg.norm(truth-raw_field,axis=0).mean()),
                projected_component_percent=float(mask.mean()*100)))
        for key,value in zip(arrays,(target,pred,decoded_in,decoded_out,masks)):arrays[key].append(value)
        checkpoints.append(dict(mode=mode,path=str(run/'last.pth'),epoch=int(checkpoint['epoch'])))
        print('Reconstructed val',mode,'samples',len(cases),flush=True)
        del model,checkpoint
    arrays={k:np.stack(v) for k,v in arrays.items()}
    for sample in range(len(cases)):
        for model in range(1,len(MODES)):np.testing.assert_array_equal(arrays['inputs'][model,sample,0],arrays['inputs'][0,sample,0])
    np.savez_compressed(OUT/'arrays.npz',modes=np.array(MODES),**arrays)
    with (OUT/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
    aggregate={mode:{key:float(np.mean([m[key] for m in metrics if m['mode']==mode])) for key in
        ('encoded_mse','depth_mae_mm','vector_epe_px')} for mode in MODES}
    summary=dict(split='actual encoder pretraining validation subset',actual_val_count=val_count,visualized_samples=len(cases),
        selection='One median-contact representative per shape from 16 seeded actual sampled val candidates; no model-error ranking.',
        columns=DISPLAY_MODES,checkpoints=checkpoints,metrics_on_selected_samples_only=aggregate,
        notes=['All six modes share the identical 128-trajectory held-out split and 20000-frame sampled validation set.',
               'Every plotted row is verified in the actual sampled val set and outside all train trajectories.',
               'Full final last.pth checkpoints contain decoders; best encoder.pth weights do not. Float32 CPU inference uses latent means.',
               'RGB display is clipped, encoded MSE/depth MAE use raw outputs. Arrows invert each encoding with declared projection bounds; input clipping loss is reported.',
               'These 14 representative samples are for visualization, not a new full-20000-frame validation score.',
               'Original nonlinear arrows project to +/-400 px per component; radial nonlinear projects to vector radius 400 px; linear arrows project to their encoding endpoints.',
               'Arrows use reference coordinates; dense fields are interpolated estimates, not dense FEM ground truth.'])
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    render(cases,arrays,metrics,checkpoints)
    print(json.dumps(aggregate,indent=2),flush=True)
    print('Saved',OUT,flush=True)


if __name__=='__main__':main()
