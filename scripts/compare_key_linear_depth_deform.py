"""Compare zero-at-rest linear depth AND displacement ranges on fixed samples."""
import argparse
import csv
import json
from pathlib import Path
import sys

import cv2
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'encoder'))
from linear_depth_deform import encode_linear_fields,convert_linear


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hdf5',type=Path,default=ROOT/'exports/pull_out_key_marker2depth_seed1000000/episode.h5')
    parser.add_argument('--output',type=Path,default=ROOT/'.cache/key-online-deform/linear_zero_rest')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    actions=[64,80,112,176];depth_ranges=[2.,4.,5.];deform_ranges=[80.,160.,240.]
    rows=[];cases=[]
    with h5py.File(args.hdf5) as f:
        sources=f['frames/source_action_index'][:]
        for action in actions:
            i=int(np.flatnonzero(sources==action)[0]);g=f['frames/tactile/right_tactile'];derived=f['derived/right']
            press=g['press_depth'][i];depth=g['depth'][i];marker=g['marker_full'][i];rgb=g['rgb_marker'][i]
            raw=derived['displacement_new_raw'][i];old=derived['depth_deform_new'][i]
            indentation=cv2.resize(press,(224,224),interpolation=cv2.INTER_AREA)
            order=derived['new_marker_order'][i];ordered=marker[:,order]
            direct=convert_linear(depth,ordered,press_depth=press,depth_range_mm=5.,deform_range_px=240.)
            reference=convert_linear(depth,ordered,reference_depth_mm=28.,depth_range_mm=5.,deform_range_px=240.)
            proposed=encode_linear_fields(indentation,raw,depth_range_mm=5.,deform_range_px=240.)
            np.testing.assert_allclose(direct,proposed,atol=1e-6,rtol=0)
            np.testing.assert_allclose(reference,proposed,atol=1e-6,rtol=0)
            images=[]
            for d in depth_ranges:
                for u in deform_ranges:
                    encoded=encode_linear_fields(indentation,raw,d,u)
                    images.append(encoded)
                    rows.append(dict(source_action=action,depth_range_mm=d,deform_half_range_px=u,
                                     depth_exceeded_pixel_fraction=float((indentation>d).mean()),
                                     deform_exceeded_component_fraction=float((np.abs(raw)>u).mean()),
                                     raw_press_max_mm=float(press.max()),raw_displacement_max_abs_px=float(np.abs(raw).max())))
            fig,axes=plt.subplots(3,4,figsize=(14,11),layout='constrained')
            axes[0,0].imshow(rgb);axes[0,0].set_title('Same-state Marker_RGB')
            axes[1,0].imshow(old.transpose(1,2,0));axes[1,0].set_title('Historical full-marker encoding\nR=(34-depth)/10; dx/dy +/-20 px')
            im=axes[2,0].imshow(indentation,vmin=0,vmax=5,cmap='magma');axes[2,0].set_title('Saved positive indentation (mm)');fig.colorbar(im,ax=axes[2,0],shrink=.7)
            for row,d in enumerate(depth_ranges):
                for col,u in enumerate(deform_ranges,1):
                    image=encode_linear_fields(indentation,raw,d,u)
                    axes[row,col].imshow(image.transpose(1,2,0))
                    axes[row,col].set_title(f'Depth 0..{d:g} mm | dx/dy +/-{u:g} px')
                    axes[row,col].set_xlabel(f'Depth exceeded: {100*(indentation>d).mean():.1f}%\nDeform exceeded: {100*(np.abs(raw)>u).mean():.1f}%')
            for ax in axes.ravel():ax.set_xticks([]);ax.set_yticks([]);ax.title.set_fontsize(10)
            fig.suptitle(f'Pull Out Key | input action {action}, right | fixed full-marker reference-coordinate interpolation\n'
                         'Zero indentation -> R=0. Zero displacement -> G=B=128. Fixed linear ranges, no per-frame rescaling.',fontsize=13)
            fig.savefig(args.output/f'action_{action:03d}_depth_deform_ranges.png',dpi=170,bbox_inches='tight');plt.close(fig)
            np.savez_compressed(args.output/f'action_{action:03d}_linear_inputs.npz',depth_ranges_mm=depth_ranges,
                                deform_ranges_px=deform_ranges,encoded=np.stack(images),proposed=proposed,
                                indentation_mm=indentation,raw_displacement_px=raw)
            cases.append(dict(action=action,rgb=rgb,old=old,proposed=proposed,press_max=float(press.max())))
    fig,axes=plt.subplots(len(cases),3,figsize=(11,3.4*len(cases)),layout='constrained')
    for row,c in enumerate(cases):
        for col,(image,title) in enumerate(((c['rgb'],'Marker_RGB'),(c['old'].transpose(1,2,0),'Historical full-marker encoding'),
                                            (c['proposed'].transpose(1,2,0),'Linear: depth 0..5 mm; deform +/-240 px'))):
            axes[row,col].imshow(image);axes[row,col].set_title(title,fontsize=11)
            axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
        axes[row,0].set_ylabel(f'Action {c["action"]}\nmax indentation {c["press_max"]:.2f} mm')
    fig.suptitle('Fixed linear encoding | no-contact value (R,G,B) = (0,128,128)\nFull markers and original interpolation unchanged; only depth zero-point and channel scales change.',fontsize=13)
    fig.savefig(args.output/'zero_rest_linear_overview.png',dpi=180,bbox_inches='tight');plt.close(fig)
    with (args.output/'metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=dict(formulas_byte_space=dict(R='255 * clip(indentation_mm / D, 0, 1)',
                                          GB='128 + 127 * clip(displacement_px / U, -1, 1)'),
                 proposed=dict(depth_range_mm=5.,deform_half_range_px=240.),
                 rest_RGB=[0,128,128],source_actions=actions,depth_ranges_mm=depth_ranges,
                 deform_half_ranges_px=deform_ranges,metrics=rows,
                 notes=['No-contact baseline, not the already-contacting first episode frame.',
                        'Uses recorded press_depth; known 28 mm baseline fallback verified on these samples.',
                        'G/B uses symmetric byte interval 1..255 around 128.',
                        'No production training/inference defaults or shared archive changed.'])
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print('Saved',args.output)


if __name__=='__main__':main()
