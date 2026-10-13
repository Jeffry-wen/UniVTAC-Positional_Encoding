"""Offline sparse-displacement audit and paired-video index for Pull Out Key.

Run with the univtac Python environment. Outputs CSV/JSON/PNG, without changing
the deployed representation. LOO errors are sparse prediction proxies, not
dense FEM ground-truth errors. Contact thresholds are fitted on train only.
"""
import argparse
import csv
import json
from functools import lru_cache
from pathlib import Path

import cv2
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import Delaunay, cKDTree

ROOT = Path(__file__).resolve().parents[1]
RUN = '20261008-114326-417910781_starvla_groot_multi_task_all_encode_depth_deform_VAE_S_full_encoder'
METHODS = ('linear_idw8', 'idw8', 'thin_plate')


def linear_weights(points, query):
    tri = Delaunay(points)
    s = tri.find_simplex(query)
    w = np.zeros((len(query), len(points)))
    inside = s >= 0
    affine = tri.transform[s[inside]]
    xy = np.einsum('nij,nj->ni', affine[:, :2], query[inside]-affine[:, 2])
    w[np.flatnonzero(inside)[:, None], tri.simplices[s[inside]]] = np.column_stack((xy, 1-xy.sum(axis=1)))
    if (~inside).any():
        d, idx = cKDTree(points).query(query[~inside], k=min(8, len(points)))
        inv = 1 / np.maximum(d, 1e-6)**2
        w[np.flatnonzero(~inside)[:, None], idx] = inv / inv.sum(axis=1, keepdims=True)
    return w


@lru_cache(maxsize=16)
def loo_maps(fixed_bytes, count):
    p = np.frombuffer(fixed_bytes, dtype=np.float32).reshape(count, 2)
    maps = {m: np.zeros((count, count)) for m in METHODS}
    outside = np.zeros(count, dtype=bool)
    for i in range(count):
        keep = np.arange(count) != i
        x, q = p[keep], p[i:i+1]
        maps['linear_idw8'][i, keep] = linear_weights(x, q)[0]
        outside[i] = Delaunay(x).find_simplex(q)[0] < 0
        d, idx = cKDTree(x).query(q, k=min(8, len(x)))
        inv = 1 / np.maximum(d, 1e-6)**2
        row = np.zeros(len(x))
        row[idx[0]] = inv[0] / inv[0].sum()
        maps['idw8'][i, keep] = row
        # Coordinates scaled only for numerical conditioning; no tuning on val.
        maps['thin_plate'][i, keep] = RBFInterpolator(
            x / 320, np.eye(len(x)), kernel='thin_plate_spline', smoothing=0)(q / 320)[0]
    return maps, outside


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit(args):
    splits = json.loads(args.episodes.read_text())
    rows = []
    for split in ('train', 'val'):
        for name in splits[split]['pull_out_key']:
            with h5py.File(args.data / name) as f:
                n = len(f['step'])
                indices = np.unique(np.linspace(0, n-1, min(n, args.frames), dtype=int))
                for side in ('left', 'right'):
                    g = f[f'tactile/{side}_tactile']
                    for t in indices:
                        markers = g['marker'][t]
                        # Simulator shuffles point order each frame; sort fixed/moved together.
                        order = np.lexsort((markers[0, :, 1], markers[0, :, 0]))
                        fixed, moved = markers[:, order].astype(np.float32)
                        raw_count = len(fixed)
                        # Simulator pads missing/out-of-frame markers by repeating a point.
                        # Remove copies before LOO so a held-out target cannot leak back in.
                        _, unique = np.unique(fixed, axis=0, return_index=True)
                        fixed, moved = fixed[unique], moved[unique]
                        if len(fixed) < 4:
                            raise ValueError(f'Too few unique markers: {name}/{side}/{t}')
                        maps, outside = loo_maps(fixed.tobytes(), len(fixed))
                        u = moved - fixed
                        depth = g['depth'][t]
                        press = g['press_depth'][t]
                        xs = (np.arange(32) + .5) * depth.shape[1] / 32
                        ys = (np.arange(24) + .5) * depth.shape[0] / 24
                        xx, yy = np.meshgrid(xs, ys)
                        query = np.column_stack((xx.ravel(), yy.ravel()))
                        out = Delaunay(fixed).find_simplex(query) < 0
                        field = (linear_weights(fixed, query) @ u).reshape(24, 32, 2)
                        gel = depth < 33.9
                        row = dict(split=split, episode=name, side=side, frame=int(t),
                                   press_max_mm=float(press.max()),
                                   contact_area_fraction=float((press > .1).mean()),
                                   marker_disp_p95_px=float(np.percentile(np.linalg.norm(u, axis=1), 95)),
                                   marker_component_clip_fraction=float((np.abs(u) >= 20).mean()),
                                   field_component_clip_fraction=float((np.abs(field) >= 20).mean()),
                                   extrapolated_pixel_fraction=float(out.mean()),
                                   depth_r_mean=float(np.clip((34-depth)/10, 0, 1).mean()),
                                   gel_reference_depth_mm=float(np.median((depth+press)[gel])),
                                   marker_count=len(fixed),
                                   duplicate_marker_fraction=1-len(fixed)/raw_count)
                        for method, weights in maps.items():
                            error = np.linalg.norm(weights @ u - u, axis=1)
                            row[method+'_epe_px'] = float(error.mean())
                            row[method+'_interior_epe_px'] = float(error[~outside].mean())
                            row[method+'_boundary_epe_px'] = float(error[outside].mean())
                        rows.append(row)
            print(f'{split} {name}: {len(rows)} sensor frames', flush=True)
    write_csv(args.output / 'frames.csv', rows)
    train = [r for r in rows if r['split'] == 'train']
    # Independent depth and shear strata; high shear need not mean high normal pressure.
    thresholds = {key: np.quantile([r[key] for r in train], [.5, .9]).tolist()
                  for key in ('press_max_mm', 'marker_disp_p95_px')}
    groups = []
    for split in ('train', 'val'):
        for variable, (q50, q90) in thresholds.items():
            for level in ('low', 'medium', 'high'):
                subset = [r for r in rows if r['split'] == split and
                          (('low' if r[variable] <= q50 else 'medium' if r[variable] <= q90 else 'high') == level)]
                item = dict(split=split, variable=variable, level=level, sensor_frames=len(subset),
                            episodes=len(set(r['episode'] for r in subset)))
                for key in [k for k in rows[0] if k.endswith('_px') or 'fraction' in k or k.endswith('_mm')]:
                    # Macro-average episodes to avoid treating adjacent frames as independent trials.
                    episode_means = [np.mean([r[key] for r in subset if r['episode'] == ep])
                                     for ep in sorted(set(r['episode'] for r in subset))]
                    item[key] = float(np.mean(episode_means)) if episode_means else None
                groups.append(item)
    summary = dict(episodes_file=str(args.episodes.resolve()), data_root=str(args.data.resolve()),
                   sampling=f'{args.frames} uniformly spaced frames per trajectory, both sensors',
                   thresholds_train_only=thresholds, frame_count=len(rows), groups=groups,
                   limitations=['LOO removes one marker; this is not dense deformation ground truth.',
                                'val is the policy held-out demonstration split, not online evaluation rollouts.',
                                'Boundary LOO changes the convex hull and is reported separately.',
                                'No algorithm selected or tuned using validation results.',
                                '32x24 fields used only to estimate clipping and extrapolation fractions.'])
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, variable in zip(axes, thresholds):
        for method in METHODS:
            g = [r for r in groups if r['split'] == 'val' and r['variable'] == variable]
            ax.plot([r['level'] for r in g], [r[method+'_epe_px'] for r in g], 'o-', label=method)
        ax.set(xlabel=variable+' (train quantiles)', ylabel='Episode-macro LOO EPE (px)')
        ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / 'loo_error.png', dpi=160)
    plt.close(fig)


def cases(args):
    videos = {}
    counts = {}
    for label in ('none', 'rgb_only', 'marker_only', 'marker_rgb', 'depth_deform'):
        result = args.campaign / label / 'results/pull_out_key.json'
        if result.exists():
            r = json.loads(result.read_text())
            counts[label] = {k:r[k] for k in ('success', 'evaluated', 'errors')}
        videos[label] = {}
        for p in sorted((args.campaign / label).glob('**/video/*_*.mp4')):
            if '/pull_out_key/' not in str(p):
                continue
            seed, status = p.stem.split('_', 1)
            videos[label][int(seed)] = (status, p)
    rows = []
    for seed, (status, p) in sorted(videos['depth_deform'].items()):
        row = dict(seed=seed, depth_deform=status, depth_deform_video=str(p.resolve()))
        for label in ('none', 'rgb_only', 'marker_only', 'marker_rgb'):
            other, path = videos[label].get(seed, ('missing', None))
            row[label] = other
            row[label+'_video'] = str(path.resolve()) if path else ''
        rows.append(row)
    write_csv(args.output / 'paired_cases.csv', rows)
    selected = [r for r in rows if r['marker_rgb'] == 'success' and r['depth_deform'] == 'failed'][:3]
    for row in selected:
        # Time fractions are navigation aids, not synchronized manipulation phases.
        canvas = np.full((2*220, 5*360, 3), 255, np.uint8)
        meta = []
        for i, label in enumerate(('marker_rgb', 'depth_deform')):
            cap = cv2.VideoCapture(row[label+'_video'])
            n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
            if not cap.isOpened() or n < 1 or fps <= 0:
                raise RuntimeError(row[label+'_video'])
            meta.append(dict(label=label, frames=n, fps=fps, duration_s=n/fps))
            for j, frac in enumerate((0, .25, .5, .75, .99)):
                frame = min(n-1, int(frac*n))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, img = cap.read()
                if not ok:
                    raise RuntimeError(f'Cannot decode {row[label+"_video"]} frame {frame}')
                img = cv2.resize(img, (360, 180))
                canvas[i*220+40:(i+1)*220, j*360:(j+1)*360] = img
                cv2.putText(canvas, f'{label} {frame/fps:.1f}s', (j*360+5, i*220+25),
                            cv2.FONT_HERSHEY_SIMPLEX, .55, (0,0,0), 1)
            cap.release()
        cv2.imwrite(str(args.output / f'case_{row["seed"]}.jpg'), canvas)
        row['video_metadata'] = meta
    (args.output / 'cases.json').write_text(json.dumps(dict(
        counts=counts, selected=selected,
        note='Same seeds can have different rollout histories. Frames shown at relative video times; no dense tactile arrays were saved in this campaign.'), indent=2))


def worst_frame(args):
    with (args.output / 'frames.csv').open() as f:
        rows = [r for r in csv.DictReader(f) if r['split'] == 'val']
    r = max(rows, key=lambda x: float(x['linear_idw8_epe_px']))
    with h5py.File(args.data/r['episode']) as f:
        g = f[f'tactile/{r["side"]}_tactile']
        t = int(r['frame'])
        marker, press = g['marker'][t], g['press_depth'][t]
    _, idx = np.unique(marker[0], axis=0, return_index=True)
    p, moved = marker[:, idx].astype(np.float32)
    u = moved-p
    maps, boundary = loo_maps(p.tobytes(), len(p))
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    im = axes[0].imshow(press, cmap='magma', origin='upper')
    fig.colorbar(im, ax=axes[0], label='Indentation (mm)', shrink=.7)
    axes[0].set_title('Saved press_depth')
    for ax, method in zip(axes[1:], METHODS):
        pred = maps[method] @ u
        err = np.linalg.norm(pred-u, axis=1)
        im = ax.scatter(p[:, 0], p[:, 1], c=err, vmin=0, vmax=8, cmap='magma')
        ax.quiver(p[:, 0], p[:, 1], u[:, 0], u[:, 1], color='gray', angles='xy', scale_units='xy', scale=1)
        ax.quiver(p[:, 0], p[:, 1], pred[:, 0], pred[:, 1], color='cyan', angles='xy', scale_units='xy', scale=1)
        ax.set(xlim=(0, 320), ylim=(240, 0), title=method+' LOO')
        fig.colorbar(im, ax=ax, label='EPE (px)', shrink=.7)
    fig.suptitle(f'Highest baseline LOO error in sampled val: {r["episode"]}, {r["side"]}, frame {t}\n'
                 'Gray: observed marker displacement; cyan: held-out prediction (not dense ground truth)')
    fig.tight_layout()
    fig.savefig(args.output/'worst_val_frame.png', dpi=160)
    plt.close(fig)
    (args.output/'worst_val_frame.json').write_text(json.dumps(r, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'resources/data/isaac51/pull_out_key/hdf5')
    parser.add_argument('--episodes', type=Path, default=ROOT/f'policy_results/{RUN}/episodes.json')
    parser.add_argument('--campaign', type=Path, default=ROOT/'test_results/campaign_20261012_50k_50ep')
    parser.add_argument('--output', type=Path, default=ROOT/'.cache/key-deform-analysis')
    parser.add_argument('--frames', type=int, default=32)
    args = parser.parse_args()
    if args.frames < 2:
        parser.error('--frames must be >= 2')
    args.output.mkdir(parents=True, exist_ok=True)
    audit(args)
    cases(args)
    worst_frame(args)


if __name__ == '__main__':
    main()
