"""Compare production spatial interpolators on identical saved marker samples."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull, Delaunay

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'encoder'))
from depth_marker_to_deform import _interpolation_map
from nonlinear_depth_deform import _tps_interpolation_map


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'.cache/key_nonlinear_comparison')
    parser.add_argument('--output', type=Path, default=ROOT/'.cache/linear_vs_tps')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = json.loads((args.input/'summary.json').read_text())
    size = 224
    cases = []
    for i, metadata in enumerate(source['cases']):
        with np.load(args.input/f'case_{i}_arrays.npz') as data:
            marker = data['marker_full'] if 'marker_full' in data else data['marker']
            fixed, moved = marker.astype(np.float32)
            h, w = data['depth'].shape
            displacement = moved-fixed
            vertices, weights = _interpolation_map(fixed.tobytes(), len(fixed), h, w, size)
            linear = np.einsum('nvc,nv->nc', displacement[vertices], weights).reshape(size, size, 2)
            tps = (_tps_interpolation_map(fixed.tobytes(), len(fixed), h, w, size) @ displacement).reshape(size, size, 2)
            assert np.isfinite(linear).all() and np.isfinite(tps).all()
            # Confirm we reproduced the previously exported raw TPS field.
            np.testing.assert_allclose(tps.transpose(2, 0, 1), data['tps_displacement_px'], atol=1e-5)
            x = (np.arange(size)+.5)*w/size
            y = (np.arange(size)+.5)*h/size
            xx, yy = np.meshgrid(x, y)
            inside = (Delaunay(fixed).find_simplex(np.c_[xx.ravel(), yy.ravel()]) >= 0).reshape(size, size)
            difference = np.linalg.norm(tps-linear, axis=-1)
            hull = fixed[ConvexHull(fixed).vertices]
            hull = np.vstack([hull, hull[0]])
            cases.append(dict(metadata=metadata, rgb=data['marker_rgb'].copy(), fixed=fixed,
                              displacement=displacement, linear=linear, tps=tps, difference=difference,
                              inside=inside, hull=hull, x=x, y=y, w=w, h=h))
            np.savez_compressed(args.output/f'case_{i}_fields.npz', linear_px=linear, tps_px=tps,
                                difference_norm_px=difference, inside_marker_hull=inside, marker=marker)

    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12, 'figure.facecolor': 'white'})
    vmax = max(float(np.linalg.norm(c[k], axis=-1).max()) for c in cases for k in ('linear', 'tps'))
    dmax = max(float(c['difference'].max()) for c in cases)
    fig, axes = plt.subplots(len(cases), 4, figsize=(16, 3.6*len(cases)), layout='constrained', squeeze=False)
    metrics = []
    for row, c in enumerate(cases):
        m = c['metadata']
        extent = (0, c['w'], c['h'], 0)
        axes[row, 0].imshow(c['rgb'], extent=extent)
        axes[row, 0].quiver(*c['fixed'].T, *c['displacement'].T, angles='xy', scale_units='xy',
                            scale=1, color='#00ffff', width=.004)
        axes[row, 0].set_title(f"Episode {Path(m['episode']).stem} / frame {m['frame']} / {m['side']}")
        for col, key in ((1, 'linear'), (2, 'tps')):
            field = c[key]
            im = axes[row, col].imshow(np.linalg.norm(field, axis=-1), extent=extent, cmap='viridis', vmin=0, vmax=vmax)
            xx, yy = np.meshgrid(c['x'][8::16], c['y'][8::16])
            f = field[8::16, 8::16]
            axes[row, col].quiver(xx, yy, f[..., 0], f[..., 1], angles='xy', scale_units='xy', scale=1,
                                 color='white', alpha=.8, width=.003)
            axes[row, col].set_title('Linear inside hull + IDW outside' if key == 'linear' else 'TPS (smoothing = 0)')
        dim = axes[row, 3].imshow(c['difference'], extent=extent, cmap='magma', vmin=0, vmax=dmax)
        axes[row, 3].set_title(r'Difference $\|u_{TPS}-u_{Linear}\|$')
        stats = dict(episode=m['episode'], frame=m['frame'], side=m['side'],
                     mean_difference_px=float(c['difference'].mean()), max_difference_px=float(c['difference'].max()),
                     mean_inside_hull_px=float(c['difference'][c['inside']].mean()),
                     mean_outside_hull_px=float(c['difference'][~c['inside']].mean()))
        metrics.append(stats)
        axes[row, 3].set_xlabel(f"Mean {stats['mean_difference_px']:.2f} px | max {stats['max_difference_px']:.2f} px")
        for ax in axes[row]:
            ax.plot(*c['hull'].T, '--', color='#ffdc73', lw=1.2)
            ax.set(xlim=(0, c['w']), ylim=(c['h'], 0), xticks=[], yticks=[])
    fig.colorbar(im, ax=axes[:, 1:3].ravel().tolist(), shrink=.65, label='Displacement magnitude (sensor px)')
    fig.colorbar(dim, ax=axes[:, 3].tolist(), shrink=.65, label='Vector difference (sensor px)')
    fig.suptitle('Linear interpolation vs thin-plate spline | real pull_out_key tactile frames\n'
                 'Identical marker pairs and query grid; raw displacement, no channel encoding. '
                 'Dashed line: marker convex hull. Arrows: true scale.', fontsize=14)
    for ext in ('png', 'pdf'):
        fig.savefig(args.output/f'linear_vs_tps.{ext}', dpi=190)
    plt.close(fig)

    # Signed components and a cross-section expose changes in slope directly.
    c = cases[0]
    bound = max(float(np.abs(c[key]).max()) for key in ('linear', 'tps'))
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.4), layout='constrained')
    section = size//2
    for row, component in enumerate(('dx', 'dy')):
        for col, key in enumerate(('linear', 'tps')):
            im = axes[row, col].imshow(c[key][..., row], extent=(0, c['w'], c['h'], 0),
                                       cmap='RdBu_r', vmin=-bound, vmax=bound)
            axes[row, col].plot(*c['hull'].T, '--', color='#333333', lw=1)
            axes[row, col].axhline(c['y'][section], color='#19c897', lw=1.5)
            axes[row, col].set_title(f'{key.upper()}: {component} (sensor px)')
            axes[row, col].set(xticks=[], yticks=[])
        ax = axes[row, 2]
        for key, color in (('linear', '#d66c22'), ('tps', '#275db3')):
            ax.plot(c['x'], c[key][section, :, row], label=key.upper(), color=color, lw=2)
        interior = c['inside'][section]
        if interior.any():
            ax.axvspan(c['x'][interior][0], c['x'][interior][-1], color='gray', alpha=.1, label='Inside marker hull')
        ax.set(title=f'{component} cross-section at y = {c["y"][section]:.1f} px', xlabel='Reference x (sensor px)', ylabel=f'{component} (sensor px)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
    fig.colorbar(im, ax=axes[:, :2].ravel().tolist(), shrink=.8, label='Signed displacement (sensor px)')
    fig.suptitle('Same input, same color scale | component maps and center cross-section\n'
                 'Piecewise linear interpolation changes slope at triangle edges; TPS varies smoothly.', fontsize=14)
    for ext in ('png', 'pdf'):
        fig.savefig(args.output/f'linear_vs_tps_detail.{ext}', dpi=190)
    plt.close(fig)
    (args.output/'metrics.json').write_text(json.dumps(dict(source=str(args.input), cases=metrics,
        notes=['Difference is between methods, not error against ground truth.',
               'Production linear interpolation uses inverse-distance weighting outside the marker hull.',
               'Saved legacy marker correspondences are used as-is; no stable IDs are inferred.']), indent=2))
    print(json.dumps(metrics, indent=2))
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
