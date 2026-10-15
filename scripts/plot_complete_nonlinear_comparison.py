"""Full-range plot of proposed nonlinear, original production nonlinear, and linear modes."""
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'encoder'))
from nonlinear_depth_deform import encode_displacement
from preview_adaptive_nonlinear import adaptive,linear


def main():
    out=ROOT/'.cache/adaptive_nonlinear_no_deadzone_gain40_preview'
    out.mkdir(parents=True,exist_ok=True)
    x=np.linspace(-300,300,12001)
    old=encode_displacement(x)
    new=adaptive(x)
    fig,ax=plt.subplots(figsize=(13,7),layout='constrained')
    for limit,color in [(20,'#a4acb5'),(40,'#548fbd'),(80,'#dea23b'),(160,'#489470'),(240,'#8873b2')]:
        ax.plot(x,linear(x,limit),label=f'Linear +/-{limit} px',color=color,lw=1.3,alpha=.65)
    ax.plot(x,old,label='OLD nonlinear: +/-20 -> 0.2..0.8; rational tails',color='#20252b',ls='--',lw=2.6,zorder=5)
    ax.plot(x,new,label='NEW nonlinear: Linear40 core / Linear240 tails',color='#d64914',lw=3,zorder=6)
    ax.axvspan(-20,20,color='#d64914',alpha=.06)
    for bound in (20,160,240):
        for sign in (-1,1):ax.axvline(sign*bound,color='#63676d',ls=':',lw=.8,alpha=.55)
    knots=np.array([-240,-160,-20,0,20,160,240],dtype=float)
    for px,value in zip(knots,adaptive(knots)):
        ax.scatter(px,value,color='#d64914',s=25,zorder=7)
        offset=(-8,13) if px<0 else (8,-22)
        if px==0:offset=(12,14)
        ax.annotate(f'({px:g}, {value:.3f})',(px,value),xytext=offset,textcoords='offset points',fontsize=9,
                    ha='right' if px<0 else 'left')
    ax.set(xlim=(-300,300),ylim=(-.035,1.035),xlabel='Signed displacement dx or dy (original sensor pixels)',
           ylabel='Encoded G or B [0, 1]',title='Complete mapping comparison | original nonlinear vs proposed nonlinear\n'
           'OLD: slope 0.015 within +/-20, asymptotic tails | NEW: slope 0.01245 within +/-20, Linear240 beyond +/-160')
    ax.set_xticks([-300,-240,-200,-160,-120,-80,-40,-20,0,20,40,80,120,160,200,240,300])
    ax.tick_params(axis='x',labelsize=9)
    ax.set_yticks(np.linspace(0,1,11));ax.grid(alpha=.2)
    ax.legend(loc='upper left',fontsize=9,framealpha=.95)
    for ext in ('png','pdf'):fig.savefig(out/f'complete_mapping_curve_with_old.{ext}',dpi=190)
    np.savetxt(out/'complete_mapping_curve_with_old.csv',np.column_stack([x,new,old]),delimiter=',',
               header='displacement_px,new_nonlinear,old_production_nonlinear',comments='')
    plt.close(fig)
    print(out/'complete_mapping_curve_with_old.png')


if __name__=='__main__':main()
