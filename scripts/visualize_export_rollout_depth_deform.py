"""Visualize exported unique rollout observations with current production encodings."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import cv2
import h5py
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'encoder'))
from modality import tactile_from_arrays
from linear_depth_deform import depth_deform_spec
from depth_marker_to_deform import _interpolation_map
from nonlinear_depth_deform import _tps_interpolation_map

MODES = [f'depth_deform_{u}' for u in (20, 80, 160, 240)] + ['depth_deform_nonlinear']
LABELS = ['Linear +/-20 px', 'Linear +/-80 px', 'Linear +/-160 px',
          'Linear +/-240 px', 'Nonlinear (TPS)']
SIDES = ('left', 'right')


def label(canvas, value, x, y, scale=.55):
    cv2.putText(canvas, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (235, 235, 235), 1, cv2.LINE_AA)


def byte_rgb(chw):
    return np.rint(np.clip(chw.transpose(1, 2, 0), 0, 1)*255).astype(np.uint8)


def save_png(path, rgb):
    Image.fromarray(rgb).save(path)


def video(path, frames, fps):
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f'Cannot create {path}')
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    reader = cv2.VideoCapture(str(path))
    count = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, _ = reader.read()
    reader.release()
    if not ok or count != len(frames):
        raise RuntimeError(f'Video verification failed: {path}, {count}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=Path, default=ROOT/'.cache/exports/exports/pull_out_key_marker2depth_seed1000000/episode.h5')
    parser.add_argument('--output', type=Path, default=ROOT/'.cache/rollout_depth_deform_comparison')
    args = parser.parse_args()
    out = args.output
    (out/'frames').mkdir(parents=True, exist_ok=True)
    metrics, comparisons = [], []
    individual = {mode: [] for mode in MODES}
    with h5py.File(args.episode, 'r') as source, h5py.File(out/'depth_deform_arrays.h5', 'w') as result:
        sources = source['frames/source_action_index'][:]
        timeline = source['timeline/frame_index'][:]
        actions = source['timeline/action_index'][:]
        np.testing.assert_array_equal(sources[timeline], source['timeline/source_action_index'][:])
        result.create_dataset('source_action_index', data=sources)
        result.create_dataset('timeline/frame_index', data=timeline)
        result.create_dataset('timeline/action_index', data=actions)
        result.attrs['mode_order'] = json.dumps(MODES)
        result.attrs['source'] = str(args.episode.resolve())
        result.attrs['depth_formula'] = 'clip((30-resize(camera_depth_mm))/5,0,1)'
        result.attrs['marker_source'] = 'marker_full; stable IDs; out-of-view positions retained'
        for side in SIDES:
            for mode in MODES:
                ds = result.create_dataset(f'{side}/{mode}', (len(sources), 3, 224, 224), dtype='f4', compression='gzip')
                ds.attrs['encoding'] = json.dumps(depth_deform_spec(mode))
            for kind in ('linear_idw', 'tps'):
                result.create_dataset(f'{side}/displacement_{kind}_px', (len(sources), 2, 224, 224), dtype='f4', compression='gzip')
        for i, action in enumerate(sources):
            canvas = np.full((878, 1568, 3), 22, np.uint8)
            label(canvas, f'Pull Out Key | seed 1000000 | observation {i+1}/{len(sources)} | source action {action}', 12, 25, .7)
            for col, camera in enumerate(('head', 'wrist')):
                image = source[f'frames/observation/{camera}/rgb'][i]
                x = 304 + 480*col
                canvas[62:332, x:x+480] = image
                label(canvas, f'{camera.upper()} RGB', x, 52)
            side_images = {}
            for row, side in enumerate(SIDES):
                g = source[f'frames/tactile/{side}_tactile']
                depth, marker = g['depth'][i], g['marker_full'][i]
                np.testing.assert_array_equal(g['marker_id'][i], np.arange(marker.shape[1]))
                assert np.isfinite(depth).all() and np.isfinite(marker).all()
                fixed, moved = marker.astype(np.float32)
                inside = (moved[:, 0]>0)&(moved[:, 0]<depth.shape[1])&(moved[:, 1]>0)&(moved[:, 1]<depth.shape[0])
                np.testing.assert_array_equal(inside, g['marker_in_view'][i])
                vertices, weights = _interpolation_map(fixed.tobytes(), len(fixed), *depth.shape, 224)
                linear = np.einsum('nvc,nv->nc', (moved-fixed)[vertices], weights).reshape(224,224,2).transpose(2,0,1)
                tps_weights = _tps_interpolation_map(fixed.tobytes(), len(fixed), *depth.shape, 224)
                tps = (tps_weights @ (moved-fixed)).reshape(224,224,2).transpose(2,0,1)
                for kind, field in [('linear_idw', linear), ('tps', tps)]:
                    result[f'{side}/displacement_{kind}_px'][i] = field
                raw = dict(depth=depth, marker_full=marker)
                encoded = [tactile_from_arrays(raw, mode).numpy() for mode in MODES]
                expected_depth = np.clip((30-cv2.resize(depth,(224,224),interpolation=cv2.INTER_AREA))/5,0,1)
                for mode, image in zip(MODES, encoded):
                    assert image.shape == (3,224,224) and np.isfinite(image).all()
                    assert image.min() >= 0 and image.max() <= 1
                    np.testing.assert_array_equal(image[0], expected_depth)
                    result[f'{side}/{mode}'][i] = image
                # Independently check the four linear displacement mappings.
                for limit, image in zip((20,80,160,240), encoded):
                    np.testing.assert_allclose(image[1:], (128+127*np.clip(linear/limit,-1,1))/255, atol=1e-7)
                rgb = cv2.resize(g['rgb_marker'][i], (224,224), interpolation=cv2.INTER_AREA)
                heat = cv2.cvtColor(cv2.applyColorMap(np.rint(expected_depth*255).astype(np.uint8),cv2.COLORMAP_MAGMA),cv2.COLOR_BGR2RGB)
                panels = [rgb, heat] + [byte_rgb(image) for image in encoded]
                side_images[side] = panels
                top = 370 + row*252
                for col, (panel, title) in enumerate(zip(panels, ['Marker RGB', 'Depth: 0..5 mm']+LABELS)):
                    canvas[top+20:top+244, col*224:(col+1)*224] = panel
                    label(canvas, f'{side[0].upper()} {title}', col*224+5, top+13, .46)
                displacement = moved-fixed
                metrics.append(dict(source_action=int(action), side=side, outside_markers=int((~inside).sum()),
                    marker_displacement_p95_px=float(np.percentile(np.linalg.norm(displacement,axis=1),95)),
                    marker_displacement_max_px=float(np.linalg.norm(displacement,axis=1).max()),
                    camera_depth_min_mm=float(depth.min()), press_depth_max_mm=float(g['press_depth'][i].max()),
                    encoded_depth_max_mm=float(expected_depth.max()*5),
                    linear_idw_abs_max_px=float(np.abs(linear).max()), tps_abs_max_px=float(np.abs(tps).max()),
                    **{f'linear_{u}_clipped_percent': float(100*(np.abs(linear)>u).mean()) for u in (20,80,160,240)}))
            label(canvas, 'R: depth 30 mm baseline / 5 mm | G: dx | B: dy | full 63 markers | fixed scales | same observation', 12, 875, .55)
            save_png(out/'frames'/f'action_{action:03d}.png', canvas)
            comparisons.append(canvas)
            for m, mode in enumerate(MODES):
                single = np.full((554,672,3),22,np.uint8)
                label(single, f'{LABELS[m]} | source action {action}', 10,25,.65)
                for row, side in enumerate(SIDES):
                    label(single, side.upper(), 8, 52+row*252)
                    for col, p in enumerate([side_images[side][0], side_images[side][1], side_images[side][m+2]]):
                        single[62+row*246:286+row*246,col*224:(col+1)*224] = p
                individual[mode].append(single)
            print(f'Processed observation {i+1}/{len(sources)} (source action {action})', flush=True)
    video(out/'comparison_unique_observations.mp4', comparisons, 2)
    expanded = []
    for action, i in zip(actions, timeline):
        frame = comparisons[int(i)].copy()
        frame[0:32] = 22
        label(frame, f'Action {action:03d} | observation source {sources[i]:03d} | '+('HELD INPUT' if action!=sources[i] else 'NEW INPUT'), 12,25,.7)
        expanded.append(frame)
    video(out/'comparison_300_actions.mp4', expanded, 10)
    for mode, frames in individual.items():
        video(out/f'{mode}.mp4', frames, 2)
    Image.fromarray(cv2.resize(comparisons[0],(1120,627))).save(out/'comparison.gif', save_all=True,
        append_images=[Image.fromarray(cv2.resize(f,(1120,627))) for f in comparisons[1:]],duration=500,loop=0)
    peak = max(metrics, key=lambda r: r['marker_displacement_max_px'])
    peak_index = int(np.flatnonzero(sources==peak['source_action'])[0])
    save_png(out/'peak_displacement_comparison.png',comparisons[peak_index])
    for action in (112,128):
        save_png(out/f'action_{action:03d}_comparison.png',comparisons[int(np.flatnonzero(sources==action)[0])])
    with (out/'metrics.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(metrics[0])); writer.writeheader(); writer.writerows(metrics)
    summary = dict(source=str(args.episode.resolve()), source_sha256=hashlib.sha256(args.episode.read_bytes()).hexdigest(),
        task='pull_out_key',seed=1000000,result='failed',unique_observations=len(sources),recorded_actions=len(actions),
        source_actions=sources.tolist(),marker_source='marker_full: all 63 stable IDs, out-of-view coordinates retained',
        encodings={mode:depth_deform_spec(mode) for mode in MODES}, peak_displacement=peak,
        notes=['Counterfactual preprocessing of recorded observations; no policy rerun or VAE reconstruction.',
               'Linear uses linear/IDW interpolation; nonlinear uses TPS, so both interpolation and encoding differ.',
               '30 mm camera baseline, 5 mm depth range; saved press_depth is diagnostic only.',
               '2 fps unique playback and 10 fps action playback are presentation rates, not physical timestamps.',
               'Dense displacement is interpolated at reference coordinates; raw RGB/depth are current coordinates.'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    print('Saved',out.resolve(),flush=True)


if __name__ == '__main__':
    main()
