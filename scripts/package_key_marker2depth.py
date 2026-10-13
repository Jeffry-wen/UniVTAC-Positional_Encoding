"""Build a standalone, losslessly deduplicated research episode archive."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

import h5py
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'.cache/key-online-deform/full_marker_capture/policy_inputs/1000000_failed.hdf5'
DEST=ROOT/'exports/pull_out_key_marker2depth_seed1000000'

REPRODUCE='''"""Reproduce stored old/new deform and unclipped displacement fields."""
from pathlib import Path
import h5py
import numpy as np
from scipy.spatial import cKDTree
from depth_marker_to_deform import convert, _interpolation_map

def compute(depth, old, full, size=224):
    if np.all(old == 0):
        retained = np.array([], dtype=int)
    else:
        _, positions = np.unique(old[0], axis=0, return_index=True)
        positions.sort()
        distance, retained = cKDTree(full[0]).query(old[0, positions])
        assert distance.max() < 1e-4
        np.testing.assert_allclose(full[:, retained], old[:, positions], atol=1e-4, rtol=0)
    order = np.r_[retained, np.setdiff1d(np.arange(full.shape[1]), retained)]
    new = full[:, order]
    def raw(marker):
        marker = marker.astype(np.float32)
        p, moved = marker
        vertices, weights = _interpolation_map(p.tobytes(), len(p), *depth.shape, size)
        field = np.einsum('nvc,nv->nc', (moved-p)[vertices], weights)
        return field.reshape(size, size, 2).transpose(2,0,1).astype(np.float32)
    return dict(depth_deform_old=convert(depth, old, size),
                depth_deform_new=convert(depth, new, size),
                displacement_old_raw=raw(old), displacement_new_raw=raw(new),
                new_marker_order=order.astype(np.int64))

def main():
    with h5py.File(Path(__file__).with_name('episode.h5')) as f:
        frames = f['frames']
        for side in ('left','right'):
            g = frames['tactile/'+side+'_tactile']
            for i in range(len(frames['source_action_index'])):
                result = compute(g['depth'][i], g['marker'][i], g['marker_full'][i])
                for key, value in result.items():
                    np.testing.assert_allclose(value, f['derived/'+side+'/'+key][i], atol=1e-6, rtol=0)
                actual = frames['policy_input/observation.images.tac_'+side][i]
                np.testing.assert_allclose(result['depth_deform_old'], actual, atol=1e-6, rtol=0)
    print('PASS: all old/new encoded fields and raw displacement fields reproduced.')

if __name__ == '__main__': main()
'''

READER='''"""Read a paired marker/depth sample without Isaac Sim or PyTorch."""
import argparse
from pathlib import Path
import h5py
import numpy as np

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-action', type=int, default=112)
    p.add_argument('--preview', type=Path)
    args = p.parse_args()
    with h5py.File(Path(__file__).with_name('episode.h5')) as f:
        sources = f['frames/source_action_index'][:]
        found = np.flatnonzero(sources == args.source_action)
        if not len(found):
            p.error('Available source actions: '+str(sources.tolist()))
        i = int(found[0])
        print('Unique observations:', len(sources), 'recorded action entries:', len(f['timeline/action_index']))
        print('Reading unique frame', i, 'at source action', int(sources[i]))
        panels = []
        for side in ('left','right'):
            g = f['frames/tactile/'+side+'_tactile']
            marker = g['marker_full'][i]   # [2,63,2]; reference/current, ID, x/y
            displacement = marker[1] - marker[0]
            depth = g['depth'][i]         # [240,320]; camera distance in mm
            indentation = g['press_depth'][i]  # positive gel indentation in mm
            print(side, 'full markers:', marker.shape, 'in view:', int(g['marker_in_view'][i].sum()),
                  'max marker displacement (px):', float(np.linalg.norm(displacement,axis=1).max()),
                  'camera depth range (mm):', [float(depth.min()),float(depth.max())],
                  'max indentation (mm):', float(indentation.max()))
            panels.append([g['rgb_marker'][i],
                           f['derived/'+side+'/depth_deform_old'][i].transpose(1,2,0),
                           f['derived/'+side+'/depth_deform_new'][i].transpose(1,2,0)])
        if args.preview:
            import cv2
            canvas = np.zeros((2*288,3*320,3),np.uint8)
            for row,side in enumerate(('left','right')):
                for col,(image,label) in enumerate(zip(panels[row],('Marker_RGB','old deform','new deform'))):
                    if image.dtype != np.uint8:
                        image = np.rint(np.clip(image,0,1)*255).astype(np.uint8)
                    canvas[row*288+32:(row+1)*288,col*320+32:col*320+288] = cv2.resize(image,(256,256))
                    cv2.putText(canvas,side+' '+label,(col*320+8,row*288+22),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
            args.preview.parent.mkdir(parents=True,exist_ok=True)
            if not cv2.imwrite(str(args.preview),cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR)):
                raise RuntimeError('Could not write preview')
            print('Saved',args.preview)

if __name__ == '__main__': main()
'''

README='''Pull Out Key: paired full-marker / depth research episode
========================================================

This package contains one FAILED simulated Pull Out Key rollout, seed 1000000,
using the StarVLA+VAE Depth_Deform 50k checkpoint. It is the full-marker capture
used for the included old/new comparison video, not the earlier MP4 capture.
No model weights, simulator installation or external accounts are needed to read it.

IMPORTANT TIME CONTRACT
There are 300 recorded action entries but only 19 distinct inference observations.
Queued actions reused their last inference input. episode.h5 stores these 19
observations losslessly under /frames, and the complete 300-entry mapping under
/timeline. This is NOT 300 independently observed sensor frames. Playback at
10 fps is a presentation choice, not a physical timestamp. The recorded arrays
can be expanded exactly using timeline/frame_index. Actual commanded actions,
per-control-step live sensor frames, and restartable simulation state were not
recorded; this package cannot replay the robot simulation by itself.

QUICK START (Python 3.10+)
  python -m pip install -r requirements.txt
  python read_episode.py --source-action 112 --preview preview.png
  python reproduce_deform.py

Minimal direct access:
  import h5py
  with h5py.File('episode.h5') as f:
      i = 7  # source action 112
      g = f['frames/tactile/right_tactile']
      marker = g['marker_full'][i]       # [2,63,2], float32
      ids = g['marker_id'][i]            # [63], int64
      in_view = g['marker_in_view'][i]   # [63], bool
      displacement = marker[1]-marker[0]
      depth_mm = g['depth'][i]           # [240,320], float32
      indentation_mm = g['press_depth'][i]
      j = 120  # original recorded action entry
      unique_frame = f['timeline/frame_index'][j]

FIELDS AND COORDINATES
All /frames datasets use the first dimension T=19. schema.json enumerates every
dataset path, shape, dtype and meaning. All observations in a row share the same
source-action timestamp, including both RGB cameras, both tactile sensors, robot
state, actor poses, and actual prepared policy images.

marker_full: [T,2,63,2], reference/current positions in raw 320x240 image pixels.
  x points right, y points down. IDs are stable WITHIN THIS EPISODE. Full current
  coordinates can be negative or exceed image bounds; they are not clamped.
marker_id: stable-order IDs 0..62, corresponding to marker_full's point axis.
marker_in_view: geometric test 0<x<320 and 0<y<240, not an occlusion/force label.
marker: legacy filtered/shuffled array [T,2,63,2]. Out-of-view markers were removed
  and the last retained point was duplicated to fill 63 slots. Its row index is
  NOT a stable ID. Reference/current pairs within a row remain matched.
depth: raw camera distance, millimetres. 34 mm is the camera far-plane value.
press_depth: positive gel indentation, millimetres. This is not measured force.
  The sampled nominal gel surface is approximately 28 mm from the camera; do not
  confuse it with the 34 mm far plane.
rgb/rgb_marker: uint8 RGB sensor arrays. The latter includes rendered markers.
observation/head/rgb and observation/wrist/rgb: uint8 RGB context camera arrays.
policy_input/*: actual float32 CHW [0,1] image tensors before the model's internal
  processing. Prepared camera colors follow historical training preprocessing,
  which reverses raw sensor RGB channels; both forms are preserved here.
actor/*, embodiment/ee, tactile/*/pose: saved pose convention [x,y,z,qw,qx,qy,qz],
  positions in metres. embodiment/joint is the raw 9-DOF robot joint observation,
  not an executed action command.

OLD/NEW DEFORM AND RAW DISPLACEMENT
/derived/{left,right}/ contains old/new depth_deform [T,3,224,224] and unclipped
displacement fields [T,2,224,224] in original-image pixel displacement units.
OLD controlled the captured rollout. NEW was computed counterfactually in the
same states, using all 63 full markers and the SAME original conversion:
  inside reference-marker hull: Delaunay triangle linear interpolation
  outside reference-marker hull: inverse-square-distance weighting, 8 neighbours
  R = clip((34 - resized_camera_depth_mm)/10, 0, 1)
  G = clip(dx/40 + 0.5, 0, 1)
  B = clip(dy/40 + 0.5, 0, 1)
No TPS and no mask channel are used. The diagnostic marker_in_view field does not
filter NEW. new_marker_order records the full-ID permutation used for conversion:
visible points retain legacy order, missing full IDs are appended. This avoids
Delaunay tie-breaking changes caused only by point reordering. With no missing
points, old/new conversion is identical.

Zero displacement maps to G=B=0.5, NOT zero. RGB (0,0,0) can encode large negative
dx and dy with depth at the far plane. Clipping is irreversible; use the supplied
unclipped fields to study magnitude. These dense fields are interpolated estimates,
NOT dense FEM deformation ground truth. A full marker pair is simulation-derived,
but its interpolated dense field is not an additional measurement.

Displacement fields are sampled at REFERENCE coordinates; depth and rendered RGB
are current image-plane observations. Do not assume a deformed marker at a given
current pixel is the same material point as the field value at that pixel.

RESEARCH USE
For marker-to-depth experiments, marker_full and depth/press_depth are paired
inputs/targets. This is a single temporally correlated failed episode, not an
independent train/test benchmark. No train/test split is provided. This package
contains no added expert-action labels and no normal/tangential force ground truth.
metadata.json describes provenance and schema.json describes array semantics.
SHA256SUMS verifies every packaged file. The compressed tar archive has a separate
SHA256 checksum next to it. Conversion code is included for reproducibility.
'''


def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def meaning(path):
    if path.startswith('timeline/'):
        return {'timeline/action_index':'Recorded action ordinal; not an action command.',
                'timeline/source_action_index':'Ordinal of the inference input used for this action.',
                'timeline/frame_index':'Index into every /frames and /derived dataset.'}[path]
    if path=='frames/source_action_index':return 'Inference-input action ordinal for each unique observation.'
    if '/marker_full' in path:return 'Full paired reference/current xy, raw image pixels; stable point axis.'
    if path.endswith('/marker_id'):return 'Stable within-episode marker IDs.'
    if path.endswith('/marker_in_view'):return 'Geometric in-image flag, not occlusion or force.'
    if path.endswith('/marker'):return 'Legacy filtered/shuffled/padded reference/current xy, pixels.'
    if path.endswith('/press_depth'):return 'Positive gel indentation map, mm; not force.'
    if path.endswith('/depth'):return 'Raw camera depth map, mm; far plane 34 mm.'
    if '/displacement_' in path:return 'Unclipped interpolated dx/dy field in original-image pixel units; not dense GT.'
    if '/depth_deform_' in path:return 'Encoded CHW [0,1] depth/dx/dy representation; see README formulas.'
    if path.endswith('/new_marker_order'):return 'Full marker IDs in conversion order.'
    if '/policy_input/' in path:return 'Actual pre-model float32 CHW image tensor, [0,1].'
    if path.endswith('/rgb') or path.endswith('/rgb_marker'):return 'Raw uint8 HWC RGB sensor image.'
    if '/actor/' in path or path.endswith('/ee') or path.endswith('/pose'):return 'Saved pose [x,y,z,qw,qx,qy,qz]; position metres.'
    if path.endswith('/joint'):return 'Raw 9-DOF robot joint observation; not action command.'
    return 'Original source observation; see README.'


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    (DEST/'README.txt').write_text(README)
    (DEST/'reproduce_deform.py').write_text(REPRODUCE)
    (DEST/'read_episode.py').write_text(READER)
    (DEST/'requirements.txt').write_text('numpy\nh5py\nscipy\nopencv-python\n')
    shutil.copy2(ROOT/'encoder/depth_marker_to_deform.py',DEST/'depth_marker_to_deform.py')
    namespace={};exec(compile(REPRODUCE,str(DEST/'reproduce_deform.py'),'exec'),namespace)
    compute=namespace['compute']
    # Import the bundled conversion rather than relying on an installed module.
    schema=[];verified=[]
    with h5py.File(SOURCE) as source,h5py.File(DEST/'episode.h5','w') as out:
        action=source['action_index'][:];sources=source['source_action_index'][:]
        unique,first=np.unique(sources,return_index=True);mapping=np.searchsorted(unique,sources)
        np.testing.assert_array_equal(action[first],unique)
        def store(path,data):
            data=np.asarray(data)
            out.create_dataset(path,data=data,chunks=(1,*data.shape[1:]),compression='gzip',compression_opts=4,shuffle=True)
            schema.append(dict(path=path,shape=list(data.shape),dtype=str(data.dtype),meaning=meaning(path)))
        store('timeline/action_index',action)
        store('timeline/source_action_index',sources)
        store('timeline/frame_index',mapping.astype(np.int64))
        store('frames/source_action_index',unique)
        datasets=[]
        source.visititems(lambda path,item:datasets.append(path) if isinstance(item,h5py.Dataset) and path not in ('action_index','source_action_index') else None)
        for path in datasets:
            ds=source[path];data=ds[first]
            for start in range(0,len(action),32):
                np.testing.assert_array_equal(ds[start:start+32],data[mapping[start:start+32]])
            store('frames/'+path,data);verified.append(path)
        for side in ('left','right'):
            results=[];g=out['frames/tactile/'+side+'_tactile']
            for i in range(len(unique)):
                result=compute(g['depth'][i],g['marker'][i],g['marker_full'][i])
                np.testing.assert_allclose(result['depth_deform_old'],out['frames/policy_input/observation.images.tac_'+side][i],atol=1e-6,rtol=0)
                results.append(result)
            for key in results[0]:store('derived/'+side+'/'+key,np.stack([r[key] for r in results]))
        out.attrs['schema_version']='1.0'
        out.attrs['task']='pull_out_key';out.attrs['seed']=1000000;out.attrs['result']='failed'
        out.attrs['model_id']='20261008-114326-417910781_starvla_groot_multi_task_all_encode_depth_deform_VAE_S_full_encoder'
        out.attrs['checkpoint_step']=50000
        out.attrs['time_contract']='19 unique inference observations; 300 recorded-action mappings; no physical timestamps.'
        frames=len(unique)
    (DEST/'schema.json').write_text(json.dumps(schema,indent=2))
    metadata=dict(schema_version='1.0',task='pull_out_key',seed=1000000,result='failed',checkpoint_step=50000,
                  model_id='20261008-114326-417910781_starvla_groot_multi_task_all_encode_depth_deform_VAE_S_full_encoder',
                  source_capture='full_marker_capture/policy_inputs/1000000_failed.hdf5',
                  source_sha256=sha256(SOURCE),source_recorded_actions=len(action),unique_inference_observations=frames,
                  source_actions=unique.tolist(),source_datasets_losslessly_verified=verified,
                  repository_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  working_tree_note='Capture used additional full-marker observation fields and actual-input recording. Bundled conversion code is the exact export snapshot.',
                  result_metrics=json.loads((ROOT/'.cache/key-online-deform/full_marker_capture/metadata.json').read_text())['1000000'])
    (DEST/'metadata.json').write_text(json.dumps(metadata,indent=2))
    extras=DEST/'visualizations';extras.mkdir(exist_ok=True)
    base=ROOT/'.cache/key-online-deform'
    for name in ['marker_rgb_vs_depth_deform_old_new_with_rgb.mp4',
                 'marker_rgb_vs_depth_deform_old_new_with_rgb_action_112.png',
                 'marker_rgb_vs_depth_deform_old_new_with_rgb_action_128.png',
                 'full_marker_black_pixel_diagnostic.png']:
        shutil.copy2(base/name,extras/name)
    files=sorted(p for p in DEST.rglob('*') if p.is_file() and p.name!='SHA256SUMS' and '__pycache__' not in p.parts)
    (DEST/'SHA256SUMS').write_text(''.join(f'{sha256(p)}  {p.relative_to(DEST)}\n' for p in files))
    archive=DEST.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz',compresslevel=6) as tar:
        for p in [*files,DEST/'SHA256SUMS']:
            tar.add(p,arcname=str(Path(DEST.name)/p.relative_to(DEST)),recursive=False)
    archive.with_suffix(archive.suffix+'.sha256').write_text(f'{sha256(archive)}  {archive.name}\n')
    print(json.dumps(dict(directory=str(DEST),archive=str(archive),archive_bytes=archive.stat().st_size,
                          unique_frames=frames,recorded_actions=len(action),datasets_losslessly_verified=len(verified)),indent=2))


if __name__=='__main__':
    import sys
    sys.path.insert(0,str(ROOT/'encoder'))
    main()
