"""Compare observed marker motion with current/TPS fields for Pull Out Key.

Scans all frames for indentation and displacement extremes; only selected
cases receive dense interpolation. Does not change training/deployment code.
Dense field disagreement is NOT ground-truth error. Both interpolants use
all observed markers, so agreement at those same markers is not validation.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import ConvexHull, Delaunay

from analyze_key_deform import ROOT, RUN, linear_weights, loo_maps, write_csv
sys.path.insert(0, str(ROOT/'encoder'))
from depth_marker_to_deform import _interpolation_map, convert


def scan(args):
    rows = []
    splits = json.loads(args.episodes.read_text())
    for split in ('train', 'val'):
        for episode in splits[split]['pull_out_key']:
            with h5py.File(args.data/episode) as f:
                atom = f['atom/id'][:]
                for side in ('left', 'right'):
                    g = f[f'tactile/{side}_tactile']
                    m = g['marker'][:]
                    norms = np.linalg.norm(m[:, 1]-m[:, 0], axis=-1)
                    for start in range(0, len(m), 32):
                        press = g['press_depth'][start:start+32]
                        for j, image in enumerate(press):
                            t = start+j
                            rows.append(dict(split=split, episode=episode, side=side, frame=t,
                                             atom_id=int(atom[t]), press_max_mm=float(image.max()),
                                             press_p99_mm=float(np.percentile(image, 99)),
                                             contact_area_fraction=float((image > .1).mean()),
                                             disp_p95_px=float(np.percentile(norms[t], 95)),
                                             disp_max_px=float(norms[t].max())))
            print(f'{split} {episode}: {len(rows)} frames', flush=True)
    write_csv(args.output/'all_frames.csv', rows)
    return rows


def load(args, r):
    with h5py.File(args.data/r['episode']) as f:
        g = f[f'tactile/{r["side"]}_tactile']
        t = int(r['frame'])
        m, depth, press = g['marker'][t], g['depth'][t], g['press_depth'][t]
        rgb = cv2.imdecode(np.frombuffer(g['rgb_marker'][t], np.uint8), cv2.IMREAD_COLOR)
        if rgb is None:
            raise ValueError(f'Cannot decode marker image for {r}')
    return m, depth, press, cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)


def fields(m, depth):
    size = 224
    h, w = depth.shape
    xx, yy = np.meshgrid((np.arange(size)+.5)*w/size, (np.arange(size)+.5)*h/size)
    q = np.column_stack((xx.ravel(), yy.ravel()))
    p, moved = m.astype(np.float32)
    u = moved-p
    v, weights = _interpolation_map(p.tobytes(), len(p), h, w, size)
    current = np.einsum('nvc,nv->nc', u[v], weights).reshape(size, size, 2)
    _, unique = np.unique(p, axis=0, return_index=True)
    fit = RBFInterpolator(p[unique]/320, u[unique], kernel='thin_plate_spline', smoothing=0)
    tps = fit(q/320).reshape(size, size, 2)
    old_rgb = convert(depth, m).transpose(1, 2, 0)
    new_rgb = old_rgb.copy()
    new_rgb[:, :, 1:] = np.clip(tps/40+.5, 0, 1)
    assert np.allclose(old_rgb[:, :, 1:], np.clip(current/40+.5, 0, 1), atol=1e-6)
    return p, u, unique, q, current, tps, old_rgb, new_rgb, fit


def draw_case(args, r):
    m, depth, press, rgb = load(args, r)
    p, u, unique, q, current, tps, old_rgb, new_rgb, fit = fields(m, depth)
    pu, uu = p[unique], u[unique]
    inside = (Delaunay(pu).find_simplex(q) >= 0).reshape(224, 224)
    hull = pu[ConvexHull(pu).vertices]
    hull = np.vstack((hull, hull[:1]))
    diff = np.linalg.norm(tps-current, axis=-1)
    contact = cv2.resize(press, (224,224), interpolation=cv2.INTER_AREA) > .1
    loo, boundary = loo_maps(pu.astype(np.float32).tobytes(), len(pu))
    metrics = dict(r, unique_markers=len(unique),
                   current_at_marker_max_error_px=float(np.linalg.norm(linear_weights(p, pu)@u-uu, axis=-1).max()),
                   tps_at_marker_max_error_px=float(np.linalg.norm(fit(pu/320)-uu, axis=-1).max()),
                   marker_clipping_epe_px=float(np.linalg.norm(np.clip(uu, -20, 20)-uu, axis=-1).mean()),
                   field_disagreement_inside_mean_px=float(diff[inside].mean()),
                   field_disagreement_outside_mean_px=float(diff[~inside].mean()),
                   field_disagreement_contact_mean_px=float(diff[contact].mean()) if contact.any() else None,
                   extrapolated_contact_fraction=float((~inside)[contact].mean()) if contact.any() else None)
    for name, field in [('current', current), ('tps', tps)]:
        metrics[name+'_clip_component_fraction'] = float((np.abs(field) > 20).mean())
        metrics[name+'_contact_clip_component_fraction'] = float((np.abs(field[contact]) > 20).mean()) if contact.any() else None
        metrics[name+'_max_field_magnitude_px'] = float(np.linalg.norm(field, axis=-1).max())
        metrics[name+'_loo_epe_px'] = float(np.linalg.norm(loo['linear_idw8' if name=='current' else 'thin_plate']@uu-uu, axis=-1).mean())
        metrics[name+'_clip_marker_p95_error_px'] = float(np.percentile(np.linalg.norm(np.clip(uu,-20,20)-uu, axis=-1),95))
    vmax = max(np.linalg.norm(current, axis=-1).max(), np.linalg.norm(tps, axis=-1).max(), np.linalg.norm(u, axis=-1).max())
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes[0, 0].imshow(rgb, extent=(0, 320, 240, 0))
    axes[0, 0].set_title('Saved RGB + markers')
    im = axes[0, 1].scatter(pu[:, 0], pu[:, 1], c=np.linalg.norm(uu, axis=-1), cmap='viridis', vmin=0, vmax=vmax)
    axes[0, 1].quiver(pu[:, 0], pu[:, 1], uu[:, 0], uu[:, 1], angles='xy', scale_units='xy', scale=1, color='black')
    axes[0, 1].scatter((pu+uu)[:, 0], (pu+uu)[:, 1], s=8, color='red')
    axes[0, 1].set_title('Observed motion (arrows at true scale)\nReference: colored; current: red')
    fig.colorbar(im, ax=axes[0, 1], label='Displacement (px)', shrink=.7)
    for ax, field, label in [(axes[0, 2], current, 'Current linear + IDW8'), (axes[0, 3], tps, 'TPS')]:
        im = ax.imshow(np.linalg.norm(field, axis=-1), extent=(0,320,240,0), cmap='viridis', vmin=0, vmax=vmax)
        ax.plot(hull[:, 0], hull[:, 1], 'w--', lw=1)
        ax.quiver(q[::700, 0], q[::700, 1], field.reshape(-1,2)[::700,0], field.reshape(-1,2)[::700,1], angles='xy', scale_units='xy', scale=1, color='white', width=.003)
        ax.set_title(label+' raw field\nDashed: marker hull; outside is extrapolation')
        fig.colorbar(im, ax=ax, label='Displacement (px)', shrink=.7)
    im = axes[1, 0].imshow(press, extent=(0,320,240,0), cmap='magma', vmin=0, vmax=2)
    axes[1, 0].set_title('Saved indentation (not force)')
    fig.colorbar(im, ax=axes[1, 0], label='Indentation (mm)', shrink=.7)
    for ax, image, name in [(axes[1,1],old_rgb,'Current'), (axes[1,2],new_rgb,'TPS')]:
        ax.imshow(image, extent=(0,320,240,0))
        ax.set_title(name+' depth_deform\nSame R; G/B clipped to +/-20 px')
    im = axes[1,3].imshow(diff, extent=(0,320,240,0), cmap='inferno', vmin=0, vmax=15)
    axes[1,3].plot(hull[:,0],hull[:,1],'w--', lw=1)
    axes[1,3].set_title('Raw field disagreement\nNot ground-truth error')
    fig.colorbar(im, ax=axes[1,3], label='TPS - current norm (px)', shrink=.7)
    for ax in axes.ravel():
        ax.set(xlim=(0,320), ylim=(240,0), aspect='equal')
        ax.title.set_fontsize(10)
    fig.suptitle(f'Pull Out Key | {r["reason"]} | {r["split"]} {r["episode"]} {r["side"]} frame {r["frame"]}, atom {r["atom_id"]}\n'
                 f'Peak indentation {r["press_max_mm"]:.3f} mm; marker P95 {r["disp_p95_px"]:.2f} px; unique markers {len(unique)}/{len(p)}')
    fig.tight_layout(rect=(0,0,1,.93))
    stem = f'{r["split"]}_{r["reason"]}_{Path(r["episode"]).stem}_{r["side"]}_{r["frame"]}'
    fig.savefig(args.output/(stem+'.png'), dpi=150)
    plt.close(fig)
    np.savez_compressed(args.output/(stem+'.npz'), marker=m, press_depth=press, depth=depth,
                        current_raw=current, tps_raw=tps, current_depth_deform=old_rgb, tps_depth_deform=new_rgb)
    metrics['figure'] = str((args.output/(stem+'.png')).resolve())
    # Nearby frames expose whether an extreme is transient; no extrapolated time units.
    with h5py.File(args.data/r['episode']) as f:
        n = len(f['step'])
    times = sorted(set(max(0,min(n-1,int(r['frame'])+delta)) for delta in (-8,-4,0,4,8)))
    fig, axes = plt.subplots(3, len(times), figsize=(3.4*len(times), 8))
    for j,t in enumerate(times):
        rr = dict(r, frame=t)
        mm,dd,pp,ii = load(args,rr)
        ff = fields(mm,dd)
        for i,img in enumerate((ii,ff[6],ff[7])):
            axes[i,j].imshow(img)
            axes[i,j].set_title(f'Frame {t}' if i==0 else '')
            axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
        axes[0,j].set_xlabel(f'Indentation max {pp.max():.2f} mm')
    for ax,name in zip(axes[:,0],('Saved marker RGB','Current depth_deform','TPS depth_deform')):
        ax.set_ylabel(name)
    fig.suptitle(f'Pull Out Key | {stem} | nearby recorded frames (same G/B scale)')
    fig.tight_layout()
    fig.savefig(args.output/(stem+'_sequence.png'), dpi=130)
    plt.close(fig)
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT/'resources/data/isaac51/pull_out_key/hdf5')
    p.add_argument('--episodes', type=Path, default=ROOT/f'policy_results/{RUN}/episodes.json')
    p.add_argument('--output', type=Path, default=ROOT/'.cache/key-strong-contact')
    p.add_argument('--reuse-scan', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.reuse_scan:
        with (args.output/'all_frames.csv').open() as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            for k in ('frame','atom_id'):
                r[k] = int(r[k])
            for k in ('press_max_mm','press_p99_mm','contact_area_fraction','disp_p95_px','disp_max_px'):
                r[k] = float(r[k])
    else:
        rows = scan(args)
    train = [r for r in rows if r['split']=='train']
    thresholds = {k:float(np.quantile([r[k] for r in train], .99)) for k in ('press_max_mm','disp_p95_px')}
    selected = []
    for split in ('train','val'):
        candidates = [r for r in rows if r['split']==split]
        used = set()
        for reason, score in [('max_indentation',lambda r:r['press_max_mm']),
                              ('max_displacement',lambda r:r['disp_p95_px']),
                              ('joint_extreme',lambda r:min(r['press_max_mm']/thresholds['press_max_mm'],r['disp_p95_px']/thresholds['disp_p95_px']))]:
            # Different trajectory for each case, rather than adjacent peaks of one episode.
            r = max((r for r in candidates if r['episode'] not in used), key=score)
            used.add(r['episode'])
            selected.append(dict(r, reason=reason))
    metrics = [draw_case(args,r) for r in selected]
    write_csv(args.output/'selected_cases.csv', metrics)
    summary = dict(task='pull_out_key', sensor_frames=len(rows), episodes=len(set(r['episode'] for r in rows)),
                   train_p99_thresholds=thresholds, selected=metrics,
                   selection='All recorded frames; maxima for indentation/displacement and min of train-P99-normalized scores; distinct trajectory per case within each split.',
                   caveats=['Indentation is a geometric proxy, not measured contact force.',
                            'All-marker interpolation is not held-out validation.',
                            'Field disagreement does not identify which field is physically correct.',
                            'Extreme cases are selected examples, not population averages.',
                            'Normal indentation and in-plane marker displacement are selected independently.',
                            'TPS uses unique reference coordinates; current uses original marker array including padding.'])
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary, indent=2),flush=True)


if __name__=='__main__':
    main()
