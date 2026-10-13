"""Sparse marker versus dense CURRENT deform arrows in strong Pull Out Key cases.

All arrows start at reference image coordinates and show moved - reference.
Display lengths use the same 0.25 multiplier throughout (color is raw px).
The encoded field is decoded from G/B with (value - 0.5)*40.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[1]


def render(r, output, stride):
    source = Path(r['figure']).with_suffix('.npz')
    with np.load(source) as d:
        marker = d['marker']
        raw = d['current_raw']
        encoded = (d['current_depth_deform'][..., 1:]-.5)*40
        depth = d['depth']
    p, moved = marker
    _, unique = np.unique(p, axis=0, return_index=True)
    p, moved = p[unique], moved[unique]
    u = moved-p
    h,w = depth.shape
    size = raw.shape[0]
    yy,xx = np.meshgrid((np.arange(size)+.5)*h/size,
                       (np.arange(size)+.5)*w/size, indexing='ij')
    sl = (slice(None,None,stride),slice(None,None,stride))
    x,y = xx[sl].ravel(), yy[sl].ravel()
    a,b = raw[sl].reshape(-1,2), encoded[sl].reshape(-1,2)
    norm = Normalize(0,45)
    hull = p[ConvexHull(p).vertices]
    hull = np.vstack((hull,hull[:1]))
    # Zoom follows the strongest observed marker motions, without selecting by field difference.
    strong = np.linalg.norm(u,axis=1) >= np.percentile(np.linalg.norm(u,axis=1),80)
    center = np.median(p[strong],axis=0)
    x0 = float(np.clip(center[0]-55,0,w-110))
    y0 = float(np.clip(center[1]-55,0,h-110))
    roi = (x0,x0+110,y0,y0+110)
    stem = source.stem
    for zoom in (False,True):
        fig,axes = plt.subplots(1,4,figsize=(20,5.8),layout='constrained')
        titles = ('Observed marker displacement',
                  f'Current raw deform: {len(x):,} arrows',
                  'Overlay: magenta marker / gray deform',
                  'Current encoded deform: decoded G/B')
        options = dict(angles='xy',scale_units='xy',scale=4,pivot='tail',
                       headwidth=3.2,headlength=4,headaxislength=3.5,minlength=0)
        axes[0].scatter(p[:,0],p[:,1],s=7,color='black',zorder=3)
        q=axes[0].quiver(p[:,0],p[:,1],u[:,0],u[:,1],np.linalg.norm(u,axis=1),
                         cmap='viridis',norm=norm,width=.004,**options)
        axes[1].quiver(x,y,a[:,0],a[:,1],np.linalg.norm(a,axis=1),
                       cmap='viridis',norm=norm,width=.0015,**options)
        axes[2].quiver(x,y,a[:,0],a[:,1],color='#999999',alpha=.7,width=.0013,**options)
        axes[2].scatter(p[:,0],p[:,1],s=7,color='#cf007b',zorder=3)
        axes[2].quiver(p[:,0],p[:,1],u[:,0],u[:,1],color='#cf007b',width=.004,zorder=4,**options)
        axes[3].quiver(x,y,b[:,0],b[:,1],np.linalg.norm(b,axis=1),
                       cmap='viridis',norm=norm,width=.0015,**options)
        for ax,title in zip(axes,titles):
            ax.set_title(title,fontsize=12)
            ax.plot(hull[:,0],hull[:,1],'--',color='#777777',lw=.8,zorder=2)
            ax.set(xlim=(roi[0],roi[1]) if zoom else (0,w),
                   ylim=(roi[3],roi[2]) if zoom else (h,0),aspect='equal',
                   xlabel='Reference image x (px)',ylabel='Reference image y (px)')
        axes[0].quiverkey(q,.60,.97,20,'20 px displacement',labelpos='E',coordinates='axes',fontproperties={'size':8})
        fig.colorbar(ScalarMappable(norm=norm,cmap='viridis'),ax=axes.tolist(),
                     location='bottom',shrink=.55,aspect=55,label='Displacement magnitude (px); shared color scale')
        fig.suptitle(f'Pull Out Key | {r["split"]} {r["episode"]} {r["side"]}, frame {r["frame"]} | '
                     f'indentation max {r["press_max_mm"]:.3f} mm; marker P95 {r["disp_p95_px"]:.2f} px\n'
                     'All arrows drawn at 0.25 x displacement; same scale in every panel. Dashed line: marker hull. '
                     'Dense arrows are interpolated, not additional measurements.',fontsize=12)
        suffix = '_zoom' if zoom else ''
        for ext in ('png','pdf'):
            fig.savefig(output/(stem+suffix+'.'+ext),dpi=200,bbox_inches='tight')
        plt.close(fig)
    return dict(episode=r['episode'],side=r['side'],frame=r['frame'],split=r['split'],
                reason=r['reason'],dense_arrow_count=len(x),marker_arrow_count=len(p),
                zoom_reference_roi=roi,overview=str((output/(stem+'.png')).resolve()),
                zoom=str((output/(stem+'_zoom.png')).resolve()),
                vector_pdf=str((output/(stem+'.pdf')).resolve()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary',type=Path,default=ROOT/'.cache/key-strong-contact/summary.json')
    parser.add_argument('--output',type=Path,default=ROOT/'.cache/key-dense-arrows')
    parser.add_argument('--stride',type=int,default=4)
    args=parser.parse_args()
    if args.stride < 1:
        parser.error('--stride must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    summary=json.loads(args.summary.read_text())
    cases=[render(r,args.output,args.stride) for r in summary['selected']]
    (args.output/'index.json').write_text(json.dumps(dict(
        task='pull_out_key',arrow_display_multiplier=.25,field='current linear + IDW8',
        encoded_decode='(G/B - 0.5) * 40; contains +/-20 px component clipping',
        note='Vectors use reference marker coordinates. Dense arrows do not validate interpolated behavior between markers.',
        cases=cases),indent=2))
    print(json.dumps(cases,indent=2))


if __name__=='__main__':
    main()
