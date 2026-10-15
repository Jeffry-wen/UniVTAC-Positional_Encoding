"""Separate depth/deform errors for the 14 already-visualized pretraining val frames."""
import csv
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from visualize_vae_train_extreme_reconstruction import fields, frame_raw

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'.cache/vae_validation_comparison_radial'
ORDER=['depth_deform_nonlinear_radial','depth_deform_20','depth_deform_80','depth_deform_160','depth_deform_240','depth_deform_nonlinear']
LABELS=['Radial nonlinear','Linear20','Linear80','Linear160','Linear240','Nonlinear']


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def measures(error_depth_normalized,error_deform_normalized,physical_error,raw_error,encoding_error,mask):
    e_depth=error_depth_normalized[mask]
    e_deform=error_deform_normalized[:,mask]
    e=physical_error[:,mask]
    mse_xy=np.mean(e**2,axis=1)
    return dict(pixel_count=int(mask.sum()),depth_mse_normalized=float(np.mean(e_depth**2)),
        deform_mse_normalized=float(np.mean(e_deform**2)),
        depth_mse_mm2=float(np.mean((e_depth*5)**2)),depth_rmse_mm=float(np.sqrt(np.mean((e_depth*5)**2))),
        depth_mae_mm=float(np.mean(np.abs(e_depth*5))),
        dx_mse_px2=float(mse_xy[0]),dy_mse_px2=float(mse_xy[1]),
        deform_mse_px2=float(mse_xy.mean()),deform_component_rmse_px=float(np.sqrt(mse_xy.mean())),
        vector_mse_px2=float(mse_xy.sum()),vector_rmse_px=float(np.sqrt(mse_xy.sum())),
        vector_epe_px=float(np.linalg.norm(e,axis=0).mean()),
        raw_field_total_deform_mse_px2=float(np.mean(raw_error[:,mask]**2)),
        encoding_only_deform_mse_px2=float(np.mean(encoding_error[:,mask]**2)))


def main():
    samples=json.loads((OUT/'samples.json').read_text())
    source_summary=json.loads((OUT/'summary.json').read_text())
    checkpoints={c['mode']:c for c in source_summary['checkpoints']}
    raw_fields=[]
    for sample in samples:
        with h5py.File(sample['path'],'r') as f:
            raw,_=frame_raw(f[f"tactile/{sample['side']}_tactile"],sample['frame'])
        raw_fields.append(fields(raw))
    rows=[]
    with np.load(OUT/'arrays.npz') as data:
        modes=list(data['modes'])
        inputs=data['inputs'];predictions=data['predictions']
        decoded_in=data['decoded_inputs'];decoded_out=data['decoded_predictions']
        projections=data['projection_masks']
        for mode in ORDER:
            index=modes.index(mode)
            for sample,meta in enumerate(samples):
                target=inputs[index,sample].astype(np.float64)
                pred=predictions[index,sample].astype(np.float64)
                error=pred-target
                true_field=decoded_in[index,sample].astype(np.float64)
                field=decoded_out[index,sample].astype(np.float64)
                raw_field=raw_fields[sample][0 if mode.rsplit('_',1)[1].isdigit() else 1].astype(np.float64)
                physical_error=field-true_field
                for region,mask in (('all_pixels',np.ones(target.shape[1:],dtype=bool)),
                                    ('contact_roi',target[0]*5>.25)):
                    if not mask.any():continue
                    result=measures(error[0],error[1:],physical_error,field-raw_field,true_field-raw_field,mask)
                    np.testing.assert_allclose(result['depth_mse_mm2'],25*result['depth_mse_normalized'],rtol=1e-12)
                    np.testing.assert_allclose(result['deform_mse_px2'],(result['dx_mse_px2']+result['dy_mse_px2'])/2,rtol=1e-12)
                    rows.append(dict(mode=mode,epoch=checkpoints[mode]['epoch'],sample=sample,shape=meta['shape'],region=region,
                        **result,inverse_projected_component_percent=float(projections[index,sample][:,mask].mean()*100)))
                normalized_total=float(np.mean(error**2))
                full=rows[-2] if rows[-1]['region']=='contact_roi' else rows[-1]
                np.testing.assert_allclose(normalized_total,(full['depth_mse_normalized']+2*full['deform_mse_normalized'])/3,rtol=1e-12)
    write_csv(OUT/'quantification_by_sample.csv',rows)
    aggregate=[]
    squared_to_rmse={'depth_rmse_mm':'depth_mse_mm2','deform_component_rmse_px':'deform_mse_px2','vector_rmse_px':'vector_mse_px2'}
    for mode in ORDER:
        for region in ('all_pixels','contact_roi'):
            selected=[r for r in rows if r['mode']==mode and r['region']==region]
            count=sum(r['pixel_count'] for r in selected)
            result=dict(mode=mode,epoch=checkpoints[mode]['epoch'],region=region,samples=len(selected),pixel_count=count)
            for key in selected[0]:
                if key in ('mode','epoch','sample','shape','region','pixel_count') or key in squared_to_rmse:continue
                result[key]=float(sum(r[key]*r['pixel_count'] for r in selected)/count)
            for rmse,mse in squared_to_rmse.items():result[rmse]=float(np.sqrt(result[mse]))
            aggregate.append(result)
    write_csv(OUT/'quantification_summary.csv',aggregate)
    report=dict(scope='14 visualized actual val samples, one per shape; not all 20000 val frames',
        checkpoint_source='Full final last.pth; epoch recorded per sample; not best encoder-only checkpoints',
        formulas=dict(depth_mse_mm2='mean((5*R_pred - 5*R_target)^2)',
            deform_mse_px2='mean over pixels and both dx/dy of (decoded_pred - decoded_encoded_target)^2',
            vector_mse_px2='mean over pixels of (dx_error^2 + dy_error^2) = 2*deform_mse_px2',
            vector_epe_px='mean over pixels of sqrt(dx_error^2+dy_error^2)',
            deform_mse_normalized='mean((G_pred-G_target)^2, (B_pred-B_target)^2)',
            roi='target depth > .25 mm, same camera-depth mask for all models',
            aggregation='pixel-weighted; recompute RMSE from pooled MSE, do not average per-image RMSE'),
        truth='Depth is the normalized 30 mm camera-baseline target, clipped to 0..5 mm, not saved press_depth. Dense displacement truth is marker-interpolated, not independent dense FEM ground truth.',
        inverse='Use saved decoded values: project linear outputs to valid encoding endpoints; original nonlinear to +/-400 px per component and radial nonlinear to radius 400 px before inversion. Projection counts reported.',
        notes=['Encoded deform MSE depends on the mapping/range, so use physical px-squared error for cross-mode interpretation.',
               'raw_field_total_deform_mse_px2 compares to unclipped pre-encoding fields; encoding_only_deform_mse_px2 quantifies encoding clipping loss.',
               'Encoding-only and reconstruction MSE do not generally sum to total MSE because a cross term exists.',
               'Linear modes use linear/IDW reference fields; nonlinear/Radial nonlinear use TPS fields. Their dense physical targets can differ.',
               'No clipping is applied to predicted depth or normalized-channel errors.'],aggregate=aggregate)
    (OUT/'quantification.json').write_text(json.dumps(report,indent=2)+'\n')
    full=[r for r in aggregate if r['region']=='all_pixels']
    colors=['#d64914']+['#2878ad']*4+['#8054a1']
    fig,axes=plt.subplots(1,3,figsize=(16,4.5),layout='constrained')
    for ax,key,title,unit,logarithmic in zip(axes,
        ('depth_mse_mm2','deform_mse_px2','vector_epe_px'),
        ('Depth MSE','Deform component MSE (dx/dy mean)','Deform mean vector error (EPE)'),
        ('mm squared','sensor pixels squared','sensor pixels'),(False,True,False)):
        values=[r[key] for r in full]
        bars=ax.bar(LABELS,values,color=colors)
        if logarithmic:ax.set_yscale('log')
        ax.set(title=title,ylabel=unit+(' (log scale)' if logarithmic else ''))
        ax.tick_params(axis='x',rotation=30);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        ax.bar_label(bars,labels=[f'{v:.4f}' if v<1 else f'{v:.2f}' for v in values],padding=4,fontsize=9)
        ax.margins(y=.18 if not logarithmic else .25)
    fig.suptitle('14 representative encoder pretraining VAL frames | full final VAE checkpoints\nDepth and deform quantified separately; all pixels; decoded displacement error vs encoded target',fontsize=13)
    for ext in ('png','pdf'):fig.savefig(OUT/f'quantification.{ext}',dpi=170,facecolor='white')
    plt.close(fig)
    for r in full:print(json.dumps({k:r[k] for k in ('mode','depth_mse_mm2','depth_rmse_mm','deform_mse_px2','deform_component_rmse_px','vector_epe_px','inverse_projected_component_percent','encoding_only_deform_mse_px2')},ensure_ascii=False))
    print('Saved quantification in',OUT,flush=True)


if __name__=='__main__':main()
