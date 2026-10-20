"""Full shared pretraining-val deform MSE for the six supported VAE-S modes."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'encoder'))
from train_common import TactileFrames
from VAE.model import VAE
from linear_depth_deform import require_depth_deform_encoding
from nonlinear_depth_deform import encode_displacement

MODES=[f'depth_deform_{n}' for n in (20,80,160,240)]+['depth_deform_nonlinear','depth_deform_nonlinear_radial']


def inverse(gb,mode):
    gb=gb.double()
    if mode=='depth_deform_nonlinear_radial':
        centered=gb-.5;a=torch.linalg.vector_norm(centered,dim=1)
        maximum=.5-.2/(1+(400-20)/(40/3))
        bounded=a.clamp(max=maximum)
        r=torch.where(bounded<=.3,bounded/.015,20+(40/3)*(.2/(.5-bounded).clamp(min=1e-12)-1))
        gain=torch.where(a>0,r/a.clamp(min=1e-30),torch.zeros_like(a))
        return centered*gain[:,None],(a>maximum)[:,None].expand_as(gb)
    if mode=='depth_deform_nonlinear':
        lower,upper=map(float,encode_displacement([-400,400]))
        bounded=gb.clamp(lower,upper);s=bounded-.5;a=s.abs()
        r=torch.where(a<=.3,a/.015,20+(40/3)*(.2/(.5-a).clamp(min=1e-12)-1))
        return s.sign()*r,(gb<lower)|(gb>upper)
    bounded=gb.clamp(1/255,1.)
    return (255*bounded-128)/127*int(mode.rsplit('_',1)[1]),(gb<1/255)|(gb>1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--output',type=Path,default=ROOT/'.cache/vae_deform_full_val')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(42)
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA required but unavailable')
    runs=[max((r.parent for r in (ROOT/'encoder_results'/mode/'univtac/VAE/S').glob('*/last.pth')),key=lambda r:r.name) for mode in MODES]
    configs=[json.loads((run/'config.json').read_text()) for run in runs]
    for c in configs:
        for key in ('train_files','val_files','val_count','seed'):assert c[key]==configs[0][key],key
        assert not set(c['val_files'])&set(c['train_files'])
    results=[];checkpoints=[];reference_rows=None;started=time.monotonic()
    status={'state':'running','completed_modes':[],'current_mode':None,'output':str(args.output)}
    def save_status():
        temp=args.output/'status.json.tmp';temp.write_text(json.dumps(status,indent=2)+'\n');temp.replace(args.output/'status.json')
    save_status()
    for mode,run,c in zip(MODES,runs,configs):
        mode_started=time.monotonic();status.update(current_mode=mode,current_batch=0);save_status()
        checkpoint=torch.load(run/'last.pth',map_location='cpu',weights_only=False)
        require_depth_deform_encoding(checkpoint,mode);assert checkpoint['input_mode']==mode
        model=VAE('S',3,**c['model_options']).eval()
        model.load_state_dict(checkpoint['training_model'],strict=True);model.to(device)
        data=TactileFrames([Path(path) for path in c['val_files']],limit=c['val_count'],seed=c['seed']+1,input_mode=mode)
        data.rows.sort();assert len(data)==20000
        if reference_rows is None:
            reference_rows=data.rows.copy()
            (args.output/'val_rows.json').write_text(json.dumps(reference_rows)+'\n')
        else:assert data.rows==reference_rows
        loader=DataLoader(data,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=device.type=='cuda')
        # Accumulate [dx SSE, dy SSE, encoded SSE, EPE sum, projected components,
        # target projected components, depth SSE, pixel count] for all/contact regions.
        sums={region:torch.zeros(8,dtype=torch.float64,device=device) for region in ('all_pixels','contact_roi')}
        captured=[];hook=model.pixels.register_forward_hook(lambda module,inputs,output:captured.append(output.detach()))
        seen=0
        with torch.inference_mode():
            for batch_index,images in enumerate(loader):
                images=images.to(device,non_blocking=True)
                captured.clear()
                with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                    model(images)
                pred=captured[0].float().reshape(-1,14,14,16,16,3).permute(0,5,1,3,2,4).reshape(-1,3,224,224)
                if not torch.isfinite(pred).all():raise FloatingPointError(mode)
                truth,target_projection=inverse(images[:,1:],mode)
                estimate,projection=inverse(pred[:,1:],mode)
                error=estimate-truth
                encoded=(pred[:,1:].double()-images[:,1:].double()).square()
                squared=error.square();epe=torch.linalg.vector_norm(error,dim=1)
                depth=(pred[:,0].double()-images[:,0].double()).square()
                for region,mask in (('all_pixels',torch.ones_like(depth,dtype=torch.bool)),('contact_roi',images[:,0]*5>.25)):
                    sums[region]+=torch.stack([squared[:,0][mask].sum(),squared[:,1][mask].sum(),encoded.permute(0,2,3,1)[mask].sum(),epe[mask].sum(),projection.permute(0,2,3,1)[mask].sum().double(),target_projection.permute(0,2,3,1)[mask].sum().double(),depth[mask].sum(),mask.sum().double()])
                seen+=len(images)
                if (batch_index+1)%50==0:
                    status.update(current_batch=batch_index+1,current_samples=seen);save_status()
                    print(json.dumps({'mode':mode,'samples':seen,'total':len(data)}),flush=True)
        hook.remove();assert seen==len(data)
        for region,total in sums.items():
            sx,sy,se,epe,pr,tp,sd,n=total.cpu().tolist()
            if not n:continue
            results.append({'mode':mode,'epoch':int(checkpoint['epoch']),'region':region,'samples':seen,'pixel_count':int(n),'dx_mse_px2':sx/n,'dy_mse_px2':sy/n,'deform_mse_px2':(sx+sy)/(2*n),'deform_rmse_px':np.sqrt((sx+sy)/(2*n)),'deform_mse_normalized':se/(2*n),'vector_epe_px':epe/n,'projected_components_percent':100*pr/(2*n),'target_projected_components_percent':100*tp/(2*n),'depth_mse_normalized':sd/n})
        checkpoints.append({'mode':mode,'path':str(run/'last.pth'),'epoch':int(checkpoint['epoch']),'depth_deform_encoding':checkpoint['depth_deform_encoding']})
        with (args.output/'metrics.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
        print(json.dumps({'mode':mode,'seconds':round(time.monotonic()-mode_started,1),'metrics':results[-2:]}),flush=True)
        status['completed_modes'].append(mode);save_status()
        del model,checkpoint,loader,data
        if device.type=='cuda':torch.cuda.empty_cache()
    report={'scope':'Full 20000 sampled encoder-pretraining validation frames, identical row identities for all six modes','checkpoint_selection':'last.pth for every mode; final complete encoder+decoder, epoch reported per mode','precision':'CUDA bf16 model forward, float64 MSE accumulation; deterministic latent mean, same as training validation precision','formula':'deform_mse_px2 = mean over pixels and dx/dy of (inverse(predicted GB)-inverse(target GB))^2; normalized MSE uses predicted and target GB without clipping','inverse_bounds':'Linear: exact valid endpoints; original nonlinear: +/-400px per component; radial nonlinear: radius400px. Projection counts reported, bounds apply to decoded predictions AND targets.','limitations':['This is reconstruction error against each encoding target. Linear clipping loss is not included; Linear20 can lose large motions before reconstruction.','Linear modes use triangle/IDW fields, nonlinear modes use TPS; dense targets are interpolated marker fields, not independent FEM truth.','Models have different final epochs: four linears and radial epoch5, original nonlinear epoch6.','Physical MSE depends on declared inverse bounds if projection occurs.'],'checkpoints':checkpoints,'metrics':results,'val_row_sha256':hashlib.sha256(json.dumps(reference_rows).encode()).hexdigest(),'seconds':round(time.monotonic()-started,1)}
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    status.update(state='completed',seconds=report['seconds']);save_status()
    print(json.dumps({'state':'completed','output':str(args.output),'seconds':report['seconds']}),flush=True)


if __name__=='__main__':main()
