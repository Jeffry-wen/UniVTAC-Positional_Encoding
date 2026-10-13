"""Same-state marker RGB / legacy / 3mm-100px depth-deform contact comparison."""
import json
from pathlib import Path
import sys

import cv2
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'encoder'))
from depth_marker_to_deform import convert, _interpolation_map
from modality import tactile_from_arrays
from linear_depth_deform import DEPTH_DEFORM_SPEC

OUTPUT = ROOT/'.cache/key-online-deform/train_vs_extreme_linear100_depth3'
TRAIN = ROOT/'resources/data/isaac51/pull_out_key/hdf5/38.hdf5'
ONLINE = ROOT/'.cache/key-online-deform/full_marker_capture/policy_inputs/1000000_failed.hdf5'
RUN = ROOT/'policy_results/20261008-114326-417910781_starvla_groot_multi_task_all_encode_depth_deform_VAE_S_full_encoder'


def dense(marker, shape):
    marker = np.asarray(marker, np.float32)
    p, q = marker
    vertices, weights = _interpolation_map(p.tobytes(), len(p), *shape, 224)
    return np.einsum('nvc,nv->nc', (q-p)[vertices], weights).reshape(224,224,2).transpose(2,0,1)


def sample(path, step, online):
    with h5py.File(path) as f:
        g = f['tactile/right_tactile']
        data = {key:g[key][step] for key in ('depth','press_depth','marker')}
        if online:
            data['marker_full'] = g['marker_full'][step]
            ids = g['marker_id'][step]
            np.testing.assert_array_equal(ids, np.arange(len(ids)))
            rgb = g['rgb_marker'][step]
            outside = int((~g['marker_in_view'][step]).sum())
            assert int(f['source_action_index'][step]) == step
        else:
            rgb = cv2.imdecode(np.frombuffer(g['rgb_marker'][step],np.uint8),cv2.IMREAD_COLOR)
            if rgb is None: raise ValueError('Cannot decode train marker RGB')
            rgb = cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB)
            outside = None  # Invisible marker coordinates are absent in legacy training HDF5.
            ids = np.array([], dtype=np.int64)
        old = convert(data['depth'],data['marker'][:,:63],224)
        if online:
            np.testing.assert_allclose(old,f['policy_input/observation.images.tac_right'][step],rtol=0,atol=1e-6)
        new = tactile_from_arrays(data,'depth_deform',224).numpy()
    chosen = data.get('marker_full',data['marker'])
    old_field = dense(data['marker'][:,:63],data['depth'].shape)
    new_field = dense(chosen,data['depth'].shape)
    delta = chosen[1]-chosen[0]
    stats = dict(file=str(path),side='right',frame=step,
        marker_source='recorded full markers + stable IDs' if online else 'existing stored markers; full out-of-view positions unavailable',
        full_marker_available=online,stored_slots=int(chosen.shape[1]),
        unique_reference_positions=int(len(np.unique(chosen[0],axis=0))),
        out_of_view_marker_count=outside,
        indentation_max_mm=float(data['press_depth'].max()),
        dx_range_px=[float(delta[:,0].min()),float(delta[:,0].max())],
        dy_range_px=[float(delta[:,1].min()),float(delta[:,1].max())],
        old_deform_clipped_component_percent=float(100*(np.abs(old_field)>20).mean()),
        new_deform_clipped_component_percent=float(100*(np.abs(new_field)>DEPTH_DEFORM_SPEC['deform_range_px']).mean()),
        new_depth_clipped_pixel_percent=float(100*(cv2.resize(data['press_depth'],(224,224),interpolation=cv2.INTER_AREA)>DEPTH_DEFORM_SPEC['depth_range_mm']).mean()))
    return dict(rgb=cv2.resize(rgb,(224,224),interpolation=cv2.INTER_AREA),old=old,new=new,
                marker=chosen,ids=ids,press_depth=data['press_depth'],stats=stats)


def figure(cases, path, minimal=False):
    if minimal:
        fig = plt.figure(figsize=(12, 4.35*len(cases)))
        axes = np.empty((len(cases), 3), dtype=object)
        for row in range(len(cases)):
            for col in range(3):
                axes[row,col] = fig.add_axes([.005+col/3, .01+(len(cases)-1-row)/len(cases),
                                             .325, 3.9/(4.35*len(cases))])
    else:
        fig, axes = plt.subplots(len(cases),3,figsize=(12,4.9*len(cases)),squeeze=False,layout='constrained')
    for row,(label,c) in enumerate(cases):
        s=c['stats']
        outside = f"{s['out_of_view_marker_count']}/63 outside" if s['full_marker_available'] else 'full out-of-view coordinates unavailable'
        marker_note = 'Full recorded positions + stable IDs' if s['full_marker_available'] else 'Stored train positions + new normalization'
        titles=['marker_rgb','depth_deform_old','depth_deform_new']
        captions=[f"{label}\nMax indentation: {s['indentation_max_mm']:.3f} mm\n{outside}",
            f"Old depth scale; dx/dy +/-20 px\nG/B clipped components: {s['old_deform_clipped_component_percent']:.1f}%",
            f"{marker_note}\nDepth 0..3 mm; dx/dy +/-100 px\nG/B clipped components: {s['new_deform_clipped_component_percent']:.1f}%"]
        for col,image in enumerate((c['rgb'],c['old'].transpose(1,2,0),c['new'].transpose(1,2,0))):
            axes[row,col].imshow(image,vmin=0,vmax=1)
            axes[row,col].set_title(titles[col],fontsize=22 if minimal else 13, pad=6)
            if not minimal:
                axes[row,col].set_xlabel(captions[col],fontsize=10)
            axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
    if not minimal:
        fig.suptitle('Pull out key | right tactile | same observation within each row\n'
                     'Fixed encoding ranges; no per-image contrast enhancement or rescaling',fontsize=14)
    fig.savefig(path,dpi=180,bbox_inches='tight',facecolor='white')
    if len(cases)==2:fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight',facecolor='white')
    plt.close(fig)


def single_row(cases, path):
    fig = plt.figure(figsize=(24, 4.35))
    titles = ['Marker RGB', 'Depth Deform (Old)', 'Depth Deform (New)']
    for group, (_, case) in enumerate(cases):
        for col, image in enumerate((case['rgb'], case['old'].transpose(1,2,0), case['new'].transpose(1,2,0))):
            ax = fig.add_axes([.0025+(group*3+col)/6, .01, .1625, 3.9/4.35])
            ax.imshow(image, vmin=0, vmax=1)
            ax.set_title(titles[col], fontsize=22, pad=6)
            ax.set_xticks([]); ax.set_yticks([])
    for ext in ('png', 'pdf'):
        fig.savefig(path.with_suffix('.'+ext), dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    split=json.loads((RUN/'episodes.json').read_text())
    assert TRAIN.name in split['train']['pull_out_key']
    normal=sample(TRAIN,171,False)
    severe=sample(ONLINE,112,True)
    cases=[('TRAIN strong contact | episode 38, frame 171',normal),
           ('ONLINE extreme contact | seed 1000000, action 112',severe)]
    single_row(cases,OUTPUT/'marker_rgb_old_new_comparison_single_row.png')
    figure(cases,OUTPUT/'marker_rgb_old_new_comparison.png')
    figure(cases,OUTPUT/'marker_rgb_old_new_comparison_clean.png',minimal=True)
    for key,case,label in [('train_strong',normal,cases[0][0]),('online_extreme',severe,cases[1][0])]:
        figure([(label,case)],OUTPUT/f'{key}_comparison.png')
        np.savez_compressed(OUTPUT/f'{key}_arrays.npz',marker_rgb=case['rgb'],
            depth_deform_old=case['old'],depth_deform_new=case['new'],
            marker_positions=case['marker'],marker_ids=case['ids'],press_depth_mm=case['press_depth'])
        for name in ('rgb','old','new'):
            image=case[name]
            if name!='rgb':image=np.rint(np.clip(image.transpose(1,2,0),0,1)*255).astype(np.uint8)
            cv2.imwrite(str(OUTPUT/f'{key}_{name}.png'),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
    summary=dict(task='pull_out_key',encoding_new=DEPTH_DEFORM_SPEC,
        encoding_old={'R':'255 * clip((34 - camera_depth_mm) / 10, 0, 1)',
                      'G_B':'255 * clip(displacement_px / 40 + 0.5, 0, 1)',
                      'markers':'legacy filtered/shuffled/padded positions'},
        selection={'train':'Maximum indentation frame in the audited Pull out key training split.',
                   'online':'Near-4mm indentation with severe displacement and 38 out-of-view markers.'},
        cases=[normal['stats'],severe['stats']],
        notes=['These are different contact states, each compared across representations at the same timestamp.',
               'Train has no full marker positions or stable IDs; no missing positions are invented.',
               'Online uses actual recorded full marker positions and stable IDs.',
               'Interpolation formula and reference coordinates are unchanged.',
               'Clipping percentage is over interpolated dx/dy components; not a percentage of markers.',
               'Old online image was checked against the recorded actual policy input.'])
    (OUTPUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    print('Saved',OUTPUT)


if __name__=='__main__': main()
