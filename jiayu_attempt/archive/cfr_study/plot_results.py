"""Regenerate publication figures from the recorded experiment data."""
from pathlib import Path
import json
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/hanamikoji-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'task4'/'results'
r=json.loads((DATA/'experiments.json').read_text())
plt.rcParams.update({'font.size':12,'axes.titlesize':12,'axes.labelsize':12,
                    'xtick.labelsize':12,'ytick.labelsize':12,'legend.fontsize':12,
                    'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
styles=[('CFR simultaneous','CFR-S','#425466','-'),('CFR alternating','CFR-A','#b7791f','--'),
        ('CFR+ uniform','CFR+-U','#3182a1','-.'),('CFR+ linear','CFR+-L','#24734e','-')]
fig,axes=plt.subplots(1,2,figsize=(6.3,2.9))
for name,label,color,style in styles:
    h=r['scenarios']['neutral']['methods'][name][0]['history'][1:]
    gap=[q['spr'] for q in h]
    axes[0].loglog([q['iteration'] for q in h],gap,label=label,color=color,ls=style,lw=1.8)
    axes[1].loglog([q['elapsed_seconds'] for q in h],gap,label=label,color=color,ls=style,lw=1.8)
for ax in axes:
    ax.axhline(.002,color='#777777',ls=':',lw=1)
    ax.set_ylabel('SPR')
    ax.grid(True,which='major',alpha=.2)
    ax.set_ylim(5e-6,.04)
axes[0].set_xlim(20,2500)
axes[1].set_xlim(.007,1.25)
axes[0].set_xlabel('Iterations')
axes[1].set_xlabel('Solve time (s)')
axes[1].set_ylabel('')
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=4,frameon=False,
           handlelength=1.3,columnspacing=.8,handletextpad=.4)
fig.subplots_adjust(left=.13,right=.96,bottom=.23,top=.81,wspace=.38)
fig.savefig(DATA/'convergence.png',dpi=240)
plt.close(fig)
print(DATA/'convergence.png')
