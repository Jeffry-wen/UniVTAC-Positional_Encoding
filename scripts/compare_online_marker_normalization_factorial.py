"""Factorial old/full marker and old/new normalization comparison on recorded inference inputs."""
import json
from pathlib import Path
import sys

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle
import numpy as np
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'encoder'))
from depth_marker_to_deform import convert, _interpolation_map
from linear_depth_deform import convert_linear, DEPTH_DEFORM_SPEC

SOURCE=ROOT/'.cache/key-online-deform/full_marker_capture/policy_inputs/1000000_failed.hdf5'
OUT=ROOT/'.cache/key-online-deform/old_full_marker_old_new_normalization'
ACTIONS=[32,64,112]


def field(marker,shape):
    p,q=np.asarray(marker,np.float32)
    v,w=_interpolation_map(p.tobytes(),len(p),*shape,224)
    return np.einsum('nvc,nv->nc',(q-p)[v],w).reshape(224,224,2).transpose(2,0,1)


def load_cases():
    cases=[]
    with h5py.File(SOURCE) as f:
        for action in ACTIONS:
            index=int(np.flatnonzero(f['source_action_index'][:]==action)[0])
            g=f['tactile/right_tactile']
            depth=g['depth'][index];press=g['press_depth'][index]
            legacy=g['marker'][index];full=g['marker_full'][index];ids=g['marker_id'][index]
            visible=g['marker_in_view'][index]
            np.testing.assert_array_equal(ids,np.arange(len(ids)))
            assert len(np.unique(full[0],axis=0))==len(ids)
            dist,matching=cKDTree(full[0]).query(legacy[0])
            assert dist.max()<1e-4
            np.testing.assert_allclose(full[:,matching],legacy,atol=1e-4,rtol=0)
            raw=[field(m,depth.shape) for m in (legacy,full)]
            encoded=[];decoded=[];clipped=[]
            for norm_name in ('old','new'):
                for m,r in zip((legacy,full),raw):
                    if norm_name=='old':
                        encoded.append(convert(depth,m,224))
                        decoded.append((encoded[-1][1:]-.5)*40)
                        clipped.append(float(100*(np.abs(r)>20).mean()))
                    else:
                        encoded.append(convert_linear(depth,m,224,press_depth=press))
                        decoded.append((encoded[-1][1:]*255-128)/127*100)
                        clipped.append(float(100*(np.abs(r)>100).mean()))
                    np.testing.assert_allclose(decoded[-1],np.clip(r,-20 if norm_name=='old' else -100,20 if norm_name=='old' else 100),atol=3e-5,rtol=0)
            np.testing.assert_allclose(encoded[0],f['policy_input/observation.images.tac_right'][index],atol=1e-6,rtol=0)
            np.testing.assert_array_equal(encoded[0][0],encoded[1][0])
            np.testing.assert_array_equal(encoded[2][0],encoded[3][0])
            stats=dict(source_action=action,side='right',indentation_max_mm=float(press.max()),
                full_marker_count=len(ids),legacy_unique_markers=int(len(np.unique(legacy[0],axis=0))),
                outside_markers=int((~visible).sum()),
                clipped_component_percent=dict(zip(('old_markers_old_norm','full_markers_old_norm','old_markers_new_norm','full_markers_new_norm'),clipped)),
                raw_old_full_mean_vector_difference_px=float(np.linalg.norm(raw[0]-raw[1],axis=0).mean()),
                raw_old_full_max_vector_difference_px=float(np.linalg.norm(raw[0]-raw[1],axis=0).max()))
            cases.append(dict(action=action,rgb=g['rgb_marker'][index],legacy=legacy,full=full,ids=ids,
                              matching=matching,visible=visible,raw=raw,encoded=encoded,decoded=decoded,stats=stats))
    return cases


def overview(cases,name,arrows=False):
    fig,axes=plt.subplots(len(cases),5,figsize=(20,3.7*len(cases)),layout='constrained',squeeze=False)
    titles=['Marker RGB','Old Markers\nOld Normalization','Full Markers + IDs\nOld Normalization',
            'Old Markers\nNew Normalization','Full Markers + IDs\nNew Normalization']
    norm=Normalize(0,150)
    xs=(np.arange(224,dtype=np.float32)+.5)*320/224
    ys=(np.arange(224,dtype=np.float32)+.5)*240/224
    xx,yy=np.meshgrid(xs,ys);sl=(slice(None,None,8),slice(None,None,8))
    for row,c in enumerate(cases):
        for col,ax in enumerate(axes[row]):
            if row==0:ax.set_title(titles[col],fontsize=15,pad=8)
            if col==0:
                ax.imshow(c['rgb'],extent=(0,320,240,0) if arrows else None)
                ax.set_ylabel(f"Action {c['action']}\n{c['stats']['outside_markers']}/63 Outside",fontsize=13)
            elif arrows:
                d=c['decoded'][col-1];u=d[0][sl];v=d[1][sl]
                ax.quiver(xx[sl],yy[sl],u,v,np.hypot(u,v),cmap='viridis',norm=norm,
                    angles='xy',scale_units='xy',scale=4,width=.0018,headwidth=3.3,
                    headlength=4,headaxislength=3.5,minlength=0,alpha=.85)
            else:ax.imshow(c['encoded'][col-1].transpose(1,2,0),vmin=0,vmax=1)
            if arrows:ax.set(xlim=(0,320),ylim=(240,0),aspect='equal')
            ax.set_xticks([]);ax.set_yticks([])
    if arrows:
        fig.colorbar(ScalarMappable(norm=norm,cmap='viridis'),ax=axes.ravel().tolist(),
            location='bottom',shrink=.55,aspect=70,pad=.02,
            label='Decoded displacement magnitude (px); same scale in all panels; arrow length = displacement / 4')
    for ext in ('png','pdf'):fig.savefig(OUT/f'{name}.{ext}',dpi=180,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def positions(c):
    fig,axes=plt.subplots(1,3,figsize=(16,5.5),layout='constrained')
    axes[0].imshow(c['rgb']);axes[0].set_title('Marker RGB',fontsize=17)
    axes[0].set_xticks([]);axes[0].set_yticks([])
    unique=np.unique(c['matching'])
    for ax,selected,title in [(axes[1],unique,'Old: Visible Markers'),(axes[2],np.arange(63),'New: Full Markers + IDs')]:
        ref=c['full'][0,selected];current=c['full'][1,selected]
        color=np.where(c['visible'][selected],'#265e98','#bb6b24')
        ax.add_patch(Rectangle((0,0),320,240,facecolor='#f1f1f1',edgecolor='#555555',ls='--',lw=1))
        ax.quiver(ref[:,0],ref[:,1],current[:,0]-ref[:,0],current[:,1]-ref[:,1],color='#999999',
            angles='xy',scale_units='xy',scale=1,width=.0016,alpha=.6)
        ax.scatter(current[:,0],current[:,1],c=color,s=17,zorder=3)
        for marker_id,p in zip(c['ids'][selected],current):ax.annotate(str(marker_id),p,xytext=(3,2),textcoords='offset points',fontsize=7)
        ax.set(xlim=(-230,340),ylim=(260,-140),aspect='equal',xlabel='Current marker x (px)',ylabel='Current marker y (px)')
        ax.set_title(title+f' ({len(selected)})',fontsize=16)
    for ext in ('png','pdf'):fig.savefig(OUT/f'action_{c["action"]:03d}_marker_positions.{ext}',dpi=180,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cases=load_cases()
    overview(cases,'encoding_comparison')
    overview(cases,'arrow_comparison',arrows=True)
    for c in cases:
        fig,axes=plt.subplots(2,2,figsize=(8,8),layout='constrained')
        # Rows marker source; columns normalization, matching a 2x2 experiment.
        for row in range(2):
            for col in range(2):
                ax=axes[row,col];ax.imshow(c['encoded'][2*col+row].transpose(1,2,0),vmin=0,vmax=1)
                if row==0:ax.set_title(('Old Normalization','New Normalization')[col],fontsize=17)
                if col==0:ax.set_ylabel(('Old Markers','Full Markers + IDs')[row],fontsize=16)
                ax.set_xticks([]);ax.set_yticks([])
        for ext in ('png','pdf'):fig.savefig(OUT/f'action_{c["action"]:03d}_2x2.{ext}',dpi=180,bbox_inches='tight',facecolor='white')
        plt.close(fig)
        np.savez_compressed(OUT/f'action_{c["action"]:03d}_arrays.npz',marker_legacy=c['legacy'],
            marker_full=c['full'],marker_ids=c['ids'],marker_in_view=c['visible'],
            raw_fields=np.stack(c['raw']),encoded=np.stack(c['encoded']),decoded=np.stack(c['decoded']))
    positions(cases[-1])
    summary=dict(source=str(SOURCE),encoding_new=DEPTH_DEFORM_SPEC,
        encoding_old={'R':'clip((34-camera_depth_mm)/10,0,1)','GB':'clip(displacement_px/40+0.5,0,1)'},
        encoded_order=['old_markers_old_norm','full_markers_old_norm','old_markers_new_norm','full_markers_new_norm'],
        interpolation='Same reference-coordinate linear interpolation + IDW8 formula; marker inputs differ.',
        notes=['Same recorded inference observation across all four variants at each source action.',
               'Full geometry preserves actual out-of-view positions and stable IDs; no image-based marker detection.',
               'Old marker positions are filtered/shuffled/padded; exact old input reproduced against saved policy tensor.',
               'Arrows decode normalized tensors to pixels; no red saturation overlay.',
               'Old R is the historical 34mm baseline/10mm scale, not zero-rest indentation divided by 4.',
               'Comparisons recompute inputs offline; no policy outcome claim.'],
        cases=[c['stats'] for c in cases])
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
