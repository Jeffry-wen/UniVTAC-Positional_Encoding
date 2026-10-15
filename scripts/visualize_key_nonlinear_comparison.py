"""Compare displacement maps on real strong-contact pull_out_key train frames."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'encoder'))
from image_io import read_rgb
from modality import tactile_from_arrays
from nonlinear_depth_deform import encode_displacement, _tps_interpolation_map

LIMITS = (20, 80, 160, 240)


def select_cases(data_root, split):
    """Rank episodes by p95 marker displacement, then take distinct episodes."""
    candidates = []
    for name in split['train']['pull_out_key']:
        path = data_root / name
        with h5py.File(path, 'r') as f:
            for side in ('left', 'right'):
                g = f[f'tactile/{side}_tactile']
                marker = g['marker_full'][:] if 'marker_full' in g else g['marker'][:]
                motion = np.linalg.norm(marker[:, 1] - marker[:, 0], axis=-1)
                scores = np.percentile(motion, 95, axis=1)
                step = int(np.argmax(scores))
                candidates.append((float(scores[step]), name, side, step))
    # Keep the previously audited high-indentation sample for comparison.
    selected = [('38.hdf5', 'right', 171, 'Audited strong depth contact')]
    if selected[0][0] not in split['train']['pull_out_key']:
        selected = []
    for score, name, side, step in sorted(candidates, reverse=True):
        if name not in [c[0] for c in selected]:
            selected.append((name, side, step, 'High p95 marker displacement'))
        if len(selected) == 3:
            break
    return selected, candidates


def load_case(path, side, step, selection):
    with h5py.File(path, 'r') as f:
        g = f[f'tactile/{side}_tactile']
        raw = {key: g[key][step] for key in ('depth', 'marker')}
        for key in ('press_depth', 'marker_full', 'marker_id', 'marker_in_view'):
            if key in g:
                raw[key] = g[key][step]
        rgb = read_rgb(g['rgb_marker'], step)
    marker = raw.get('marker_full', raw['marker'])
    fixed, moved = marker.astype(np.float32)
    weights = _tps_interpolation_map(fixed.tobytes(), len(fixed), *raw['depth'].shape, 224)
    field = (weights @ (moved - fixed)).reshape(224, 224, 2).transpose(2, 0, 1)
    nonlinear = tactile_from_arrays(raw, 'depth_deform_nonlinear').numpy()
    # Float center .5 for all controlled comparisons: isolate mapping only.
    controlled = [np.concatenate((nonlinear[:1], np.clip(.5 + field/(2*u), 0, 1)), axis=0)
                  for u in LIMITS] + [nonlinear]
    production = [tactile_from_arrays(raw, f'depth_deform_{u}').numpy() for u in LIMITS] + [nonlinear]
    for image in production:
        np.testing.assert_array_equal(image[0], nonlinear[0])
    displacement = moved - fixed
    stats = dict(file=str(path.resolve()), episode=path.name, side=side, frame=step,
                 selection=selection, marker_source='marker_full' if 'marker_full' in raw else 'stored marker',
                 full_marker_available='marker_full' in raw,
                 ids_available='marker_id' in raw,
                 p95_marker_displacement_px=float(np.percentile(np.linalg.norm(displacement, axis=1), 95)),
                 max_marker_abs_component_px=float(np.abs(displacement).max()),
                 max_tps_abs_component_px=float(np.abs(field).max()),
                 max_camera_indentation_30mm=float(np.maximum(30-raw['depth'], 0).max()),
                 max_saved_press_depth_mm=float(raw['press_depth'].max()) if 'press_depth' in raw else None,
                 controlled_linear_clip_percent={str(u):float(100*(np.abs(field)>u).mean()) for u in LIMITS},
                 nonlinear_endpoint_percent=float(100*((nonlinear[1:]<=0)|(nonlinear[1:]>=1)).mean()))
    return dict(raw=raw, rgb=rgb, field=field, controlled=controlled, production=production, stats=stats)


def overview(cases, output, production=False):
    fig, axes = plt.subplots(len(cases), 7, figsize=(21, 3.35*len(cases)), squeeze=False, layout='constrained')
    titles = ['Marker RGB', 'Depth: 30 mm baseline / 5 mm'] + [f'Linear +/-{u} px' for u in LIMITS] + ['Nonlinear: linear +/-20 + tails']
    for row, c in enumerate(cases):
        s = c['stats']
        images = c['production' if production else 'controlled']
        axes[row, 0].imshow(c['rgb'])
        im = axes[row, 1].imshow(c['controlled'][-1][0]*5, cmap='magma', vmin=0, vmax=5)
        for col, image in enumerate(images, 2):
            axes[row, col].imshow(image.transpose(1, 2, 0), vmin=0, vmax=1)
            if col < 6:
                u = LIMITS[col-2]
                if production:
                    fraction = float(100*((image[1:] <= 1/255 + 1e-7)|(image[1:] >= 1-1e-7)).mean())
                    caption = f'Endpoint components: {fraction:.1f}%\nDepth: (30-camera) / 5; linear + IDW'
                else:
                    caption = f'Clipped components: {s["controlled_linear_clip_percent"][str(u)]:.1f}%'
            else:
                caption = f'Endpoint components: {s["nonlinear_endpoint_percent"]:.1f}%'
                if production:
                    caption += '\nDepth: (30-camera) / 5; TPS'
            axes[row, col].set_xlabel(caption, fontsize=9)
        axes[row, 0].set_ylabel(f'Episode {Path(s["file"]).stem}, frame {s["frame"]}, {s["side"]}\n'
                               f'Marker p95: {s["p95_marker_displacement_px"]:.1f} px\n'
                               f'Saved press max: {s["max_saved_press_depth_mm"]:.2f} mm', fontsize=10)
        for col, title in enumerate(titles):
            if row == 0:
                axes[row, col].set_title(title, fontsize=11)
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.colorbar(im, ax=axes[:, 1].tolist(), shrink=.45, label='clipped max(30-camera depth,0), mm')
    if production:
        subtitle = 'Unified depth: (30-camera) / 5 in all modes; linear uses linear/IDW interpolation, nonlinear uses TPS'
    else:
        subtitle = 'Controlled comparison: identical 30/5 depth channel and TPS field; only displacement mapping changes'
    fig.suptitle('Pull out key | real training strong contacts\n' + subtitle + '\nFixed scales; no per-image contrast enhancement', fontsize=14)
    stem = 'actual_pipeline_comparison' if production else 'mapping_comparison'
    for ext in ('png', 'pdf'):
        fig.savefig(output / f'{stem}.{ext}', dpi=160, facecolor='white')
    plt.close(fig)


def curves(output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    for ax, limit in zip(axes, (30, 300)):
        x = np.linspace(-limit, limit, 2001)
        for u in LIMITS:
            ax.plot(x, np.clip(.5+x/(2*u), 0, 1), label=f'Linear +/-{u}')
        ax.plot(x, encode_displacement(x), color='black', lw=2.3, label='Nonlinear')
        ax.scatter([-20, 0, 20], [.2, .5, .8], color='black', s=18, zorder=5)
        ax.axvspan(-20, 20, color='grey', alpha=.08)
        ax.set(xlabel='Displacement (original sensor pixels)', ylabel='Normalized G or B', ylim=(-.03, 1.03))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Displacement response | nonlinear preserves small-motion contrast with rational tails')
    fig.savefig(output/'mapping_curves.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'.cache/key_nonlinear_comparison')
    parser.add_argument('--data-root', type=Path, default=ROOT/'resources/data/isaac51/pull_out_key/hdf5')
    parser.add_argument('--split', type=Path, default=ROOT/'policy_results/20261014-031128-890564344_starvla_groot_multi_task_all_none/episodes.json')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    split = json.loads(args.split.read_text())
    selections, candidates = select_cases(args.data_root, split)
    cases = [load_case(args.data_root/name, side, step, reason) for name, side, step, reason in selections]
    overview(cases, args.output)
    overview(cases, args.output, production=True)
    curves(args.output)
    for i, c in enumerate(cases):
        np.testing.assert_allclose(c['controlled'][-1], tactile_from_arrays(c['raw'], 'depth_deform_nonlinear').numpy(), atol=1e-7)
        np.savez_compressed(args.output/f'case_{i}_arrays.npz',
                            **{k:v for k,v in c['raw'].items()}, marker_rgb=c['rgb'],
                            tps_displacement_px=c['field'], controlled=np.stack(c['controlled']),
                            actual_pipeline=np.stack(c['production']))
    summary = dict(task='pull_out_key', split_file=str(args.split.resolve()), train_episode_count=len(split['train']['pull_out_key']),
                   scanned_episode_sides=len(candidates), cases=[c['stats'] for c in cases],
                   controlled_formula='linear: clip(0.5 + u/(2*L),0,1); nonlinear: shared production rational mapping',
                   controlled_depth='clip((30-camera_depth_mm)/5,0,1)', controlled_interpolation='TPS for every column',
                   actual_linear_formula='(128+127*clip(u/L,-1,1))/255; depth (30-camera)/5; linear/IDW',
                   notes=['Linear20/80/160/240 are registered production input modes.',
                          'Legacy train data has no stable marker IDs or full out-of-view coordinates; none are invented.',
                          'No online capture is included because the previously used file is unavailable.'])
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
    with (args.output/'metrics.csv').open('w', newline='') as f:
        rows = [dict(episode=c['stats']['episode'], side=c['stats']['side'], frame=c['stats']['frame'],
                     mapping=f'linear_{u}', clipped_component_percent=c['stats']['controlled_linear_clip_percent'][str(u)])
                for c in cases for u in LIMITS]
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2), flush=True)
    print(f'Saved {args.output}', flush=True)


if __name__ == '__main__':
    main()
