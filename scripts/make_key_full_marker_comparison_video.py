"""Same-state cameras/Marker_RGB/old deform/full-marker deform comparison.

Old deform is the tensor that controlled the captured rollout. New deform is
counterfactual: identical conversion, depth and normalization, but all FEM
markers are retained, including out-of-image current positions. No TPS/mask.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import cv2
import h5py
import numpy as np
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'encoder'))
from depth_marker_to_deform import convert
from make_key_tactile_comparison_video import text


def comparison(h5,index,side):
    g=h5[f'tactile/{side}_tactile']
    old=g['marker'][index].astype(np.float32)
    full=g['marker_full'][index].astype(np.float32)
    ids=g['marker_id'][index]
    np.testing.assert_array_equal(ids,np.arange(full.shape[1]))
    if not np.isfinite(full).all():
        raise ValueError(f'Non-finite full marker coordinates at {index}/{side}')
    # Preserve visible marker order to isolate added points from any Delaunay
    # tie-breaking differences caused solely by reordering the same point set.
    if np.all(old==0):
        retained=np.array([],dtype=int)
    else:
        _,positions=np.unique(old[0],axis=0,return_index=True)
        positions.sort()
        distance,retained=cKDTree(full[0]).query(old[0,positions])
        assert distance.max()<1e-4
        np.testing.assert_allclose(full[:,retained],old[:,positions],rtol=0,atol=1e-4)
    missing=np.setdiff1d(np.arange(full.shape[1]),retained)
    ordered=full[:,np.r_[retained,missing]]
    depth=g['depth'][index]
    old_image=h5[f'policy_input/observation.images.tac_{side}'][index].transpose(1,2,0)
    np.testing.assert_allclose(old_image,convert(depth,old).transpose(1,2,0),atol=1e-6)
    new_image=convert(depth,ordered).transpose(1,2,0)
    np.testing.assert_array_equal(old_image[:,:,0],new_image[:,:,0])
    if not len(missing):
        np.testing.assert_allclose(old_image,new_image,atol=1e-6)
    in_view=g['marker_in_view'][index]
    h,w=depth.shape
    computed=(full[1,:,0]>0)&(full[1,:,0]<w)&(full[1,:,1]>0)&(full[1,:,1]<h)
    np.testing.assert_array_equal(in_view,computed)
    assert len(retained)==int(in_view.sum())
    difference=np.abs(old_image[:,:,1:]-new_image[:,:,1:])
    metrics=dict(side=side,source_action=int(h5['source_action_index'][index]),
                 old_unique_markers=len(retained),new_full_markers=full.shape[1],
                 outside_markers=int((~computed).sum()),
                 gb_mean_abs_difference=float(difference.mean()),gb_max_abs_difference=float(difference.max()),
                 full_marker_max_displacement_px=float(np.linalg.norm(full[1]-full[0],axis=1).max()))
    return g['rgb_marker'][index],old_image,new_image,metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hdf5',type=Path,default=ROOT/'.cache/key-online-deform/full_marker_capture/policy_inputs/1000000_failed.hdf5')
    parser.add_argument('--output',type=Path,default=ROOT/'.cache/key-online-deform/marker_rgb_vs_depth_deform_old_new_with_rgb.mp4')
    args=parser.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    with h5py.File(args.hdf5) as f:
        actions=f['action_index'][:];sources=f['source_action_index'][:]
        fresh=np.flatnonzero(actions==sources)
        cache={};metrics=[];new_images={'left':[],'right':[]}
        for index in fresh:
            source=int(sources[index]);base=np.full((1080,1280,3),18,np.uint8)
            for col,name in enumerate(('head','wrist')):
                image=f[f'observation/{name}/rgb'][index]
                base[56:416,col*640:(col+1)*640]=cv2.resize(image,(640,360),interpolation=cv2.INTER_LINEAR)
                text(base,name.upper()+' RGB - same source observation',(col*640+12,49),.6)
            for row,side in enumerate(('left','right')):
                marker,old,new,m=comparison(f,int(index),side);metrics.append(m)
                new_images[side].append(new.transpose(2,0,1))
                top=424+row*320
                labels=(f'{side.upper()} Marker_RGB ({m["old_unique_markers"]} visible)',
                        f'{side.upper()} Depth_Deform OLD',f'{side.upper()} Depth_Deform NEW ({m["new_full_markers"]} full)')
                for col,(image,label) in enumerate(zip((marker,old,new),labels)):
                    x=col*426+69
                    if image.dtype!=np.uint8:image=np.rint(np.clip(image,0,1)*255).astype(np.uint8)
                    base[top+24:top+312,x:x+288]=cv2.resize(image,(288,288),interpolation=cv2.INTER_AREA)
                    text(base,label,(col*426+12,top+17),.58)
            text(base,'OLD controls this rollout. NEW uses full markers in the same state; same interpolation, depth and normalization.',(12,1074),.52)
            cache[source]=base
        chosen={int(sources[fresh[0]]),min(metrics,key=lambda r:r['old_unique_markers'])['source_action'],
                max(metrics,key=lambda r:r['gb_mean_abs_difference'])['source_action']}
        process=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pixel_format','rgb24',
                                  '-video_size','1280x1080','-framerate','10','-i','-','-an','-c:v','libx264',
                                  '-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(args.output)],stdin=subprocess.PIPE)
        try:
            for action,source in zip(actions,sources):
                frame=cache[int(source)].copy()
                text(frame,f'Pull Out Key | seed 1000000 | action {action:03d} | input source action {source:03d}',(12,24),.67)
                if action!=source:text(frame,'QUEUED ACTION - INPUT HELD',(925,24),.55,(255,195,80))
                process.stdin.write(frame.tobytes())
                if action==source and int(source) in chosen:
                    cv2.imwrite(str(args.output.with_name(args.output.stem+f'_action_{action:03d}.png')),
                                cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
        finally:
            process.stdin.close();code=process.wait()
        if code:raise RuntimeError(f'ffmpeg exited with {code}')
        np.savez_compressed(args.output.with_suffix('.npz'),source_action=sources[fresh],
                            left_new=np.stack(new_images['left']),right_new=np.stack(new_images['right']))
        metadata=dict(source_hdf5=str(args.hdf5.resolve()),checkpoint=str(f.attrs['checkpoint']),
                      video=str(args.output.resolve()),frames=len(actions),fps=10,width=1280,height=1080,
                      fresh_inputs=len(fresh),metrics=metrics,preview_actions=sorted(chosen),
                      layout=['head RGB','wrist RGB','left Marker_RGB','left deform old','left deform new',
                              'right Marker_RGB','right deform old','right deform new'],
                      note='New capture with full marker coordinates; not a reconstruction of the previous MP4. All panels share one state. OLD controlled the policy; NEW is counterfactual. No training changes, TPS or mask channel. Input held during queued actions.')
        args.output.with_suffix('.json').write_text(json.dumps(metadata,indent=2))
        print(json.dumps(metadata,indent=2))


if __name__=='__main__':main()
