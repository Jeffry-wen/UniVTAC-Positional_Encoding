"""GT vs reconstructed deform arrows in encoding space, without inverse mapping."""
from pathlib import Path
import csv,json,sys
import h5py,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'encoder'))
from modality import tactile_from_arrays
from linear_depth_deform import require_depth_deform_encoding
from VAE.model import VAE
OUT=ROOT/'.cache/vae_encoded_deform_reconstruction'
MODES=[f'depth_deform_{n}' for n in (20,80,160,240)]+['depth_deform_nonlinear','depth_deform_nonlinear_radial']
LABELS=['Linear20','Linear80','Linear160','Linear240','Original nonlinear','Radial nonlinear']


def read_raw(group,frame):
    raw={k:group[k][frame] for k in ('depth','marker')}
    if 'marker_full' in group:raw['marker_full']=group['marker_full'][frame]
    return raw


@torch.inference_mode()
def reconstruct(model,images):
    captured=[]
    hook=model.pixels.register_forward_hook(lambda module,inputs,output:captured.append(output.detach()))
    try:model(images)
    finally:hook.remove()
    return captured[0].reshape(-1,14,14,16,16,3).permute(0,5,1,3,2,4).reshape(-1,3,224,224)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(42)
    sample=json.loads((ROOT/'.cache/policy_train_deform_preview/summary.json').read_text())['sample']
    with h5py.File(sample['path'],'r') as f:train=read_raw(f[f"tactile/{sample['side']}_tactile"],sample['step'])
    episode=ROOT/'.cache/exports/exports/pull_out_key_marker2depth_seed1000000/episode.h5'
    with h5py.File(episode,'r') as f:
        actions=f['frames/source_action_index'][:]
        ix=int(np.flatnonzero(actions==176)[0])
        test=read_raw(f['frames/tactile/right_tactile'],ix)
    cases=[('training','Training deformation',train),('test_max','Test Deformation',test)]
    inputs=[];predictions=[];checkpoints=[];metrics=[]
    for mode in MODES:
        run=max((p.parent for p in (ROOT/'encoder_results'/mode/'VAE/S').glob('*/last.pth')),key=lambda p:p.name)
        config=json.loads((run/'config.json').read_text())
        checkpoint=torch.load(run/'last.pth',map_location='cpu',weights_only=False)
        require_depth_deform_encoding(checkpoint,mode)
        model=VAE('S',3,**config['model_options']).eval()
        model.load_state_dict(checkpoint['training_model'],strict=True)
        images=torch.stack([tactile_from_arrays(raw,mode) for _,_,raw in cases])
        pred=reconstruct(model,images).float().numpy()
        target=images.numpy()
        assert np.isfinite(pred).all()
        inputs.append(target[:,1:]);predictions.append(pred[:,1:])
        checkpoints.append({'mode':mode,'path':str(run/'last.pth'),'epoch':int(checkpoint['epoch'])})
        for i,(key,_,_) in enumerate(cases):
            metrics.append({'mode':mode,'sample':key,'epoch':int(checkpoint['epoch']),'encoded_deform_mse':float(np.mean((pred[i,1:].astype(float)-target[i,1:].astype(float))**2))})
        print('Reconstructed',mode,'epoch',checkpoint['epoch'],flush=True)
        del model,checkpoint
    inputs=np.stack(inputs);predictions=np.stack(predictions)
    # Only subtract the encoding's zero displacement. No scaling, clipping or
    # physical inverse is applied: arrow coordinates remain encoded GB units.
    centers=np.array([128/255]*4+[.5,.5])[:,None,None,None,None]
    target_arrows=inputs.astype(float)-centers
    pred_arrows=predictions.astype(float)-centers
    maximum=max(float(np.linalg.norm(target_arrows,axis=2).max()),float(np.linalg.norm(pred_arrows,axis=2).max()))
    ix=np.arange(8,224,16);xx,yy=np.meshgrid((ix+.5)*320/224,(ix+.5)*240/224)
    for sample_index,(key,title,raw) in enumerate(cases):
        fig,axes=plt.subplots(2,6,figsize=(25,8.7),layout='constrained')
        for col,(mode,label) in enumerate(zip(MODES,LABELS)):
            for row,array in enumerate((target_arrows,pred_arrows)):
                ax=axes[row,col];v=array[col,sample_index][:,ix[:,None],ix[None,:]]
                q=ax.quiver(xx,yy,*v,np.hypot(*v),cmap='viridis',norm=Normalize(0,maximum),angles='xy',scale_units='xy',scale=.02,pivot='tail',width=.004,minlength=0)
                ax.set(xlim=(0,320),ylim=(240,0),aspect='equal',facecolor='#f5f6f8');ax.set_xticks([]);ax.set_yticks([])
                if row==0:ax.set_title(label,fontsize=14)
                if col==0:ax.set_ylabel('GT' if row==0 else 'VAE reconstruction',fontsize=13)
                ax.quiverkey(q,.55,.04,.25,'0.25 encoded',labelpos='E',coordinates='axes',fontproperties={'size':8})
                if row==1:
                    mse=next(m['encoded_deform_mse'] for m in metrics if m['mode']==mode and m['sample']==key)
                    ax.set_xlabel(f'G/B MSE: {mse:.6f}',fontsize=11)
        fig.colorbar(q,ax=axes.ravel().tolist(),shrink=.85,label='Encoded deform magnitude from zero-displacement center')
        fig.suptitle(title,fontsize=20)
        for ext in ('png','pdf'):fig.savefig(OUT/f'{key}_encoded_deform.{ext}',dpi=180,facecolor='white')
        plt.close(fig)
    with (OUT/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
    report={'checkpoints':checkpoints,'policy_train_sample':sample,'test_sample':{'path':str(episode),'source_action':176,'side':'right'},'notes':['Only GT and VAE reconstructed deform arrows, no depth or RGB panels.','Arrows subtract each encoding zero: 128/255 for linear, 0.5 for nonlinear. They are not inverse-normalized or rescaled by encoding gain.','Fixed arrow scale and color scale across all configurations and both sample figures. Predictions are not clipped or projected.','Actual mode-specific preprocessing: linear uses triangle/IDW; both nonlinear modes use TPS.','CPU float32 deterministic latent-mean inference; all six use final full last.pth.','The training sample is from policy train data, not encoder-pretraining train data. These single-sample MSE values are not full-val scores.'],'metrics':metrics}
    (OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(OUT/'arrays.npz',modes=np.array(MODES),encoded_gt=inputs,encoded_pred=predictions,centered_gt=target_arrows,centered_pred=pred_arrows)
    print(json.dumps(metrics,indent=2),flush=True)


if __name__=='__main__':main()
