"""Plot the measured three-family comparison (Figure 1) and the decomposition/DO-cost study (Figure 2)."""
from pathlib import Path
import argparse,json,os
os.environ.setdefault('MPLCONFIGDIR','/tmp/hanamikoji-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ATTEMPT=Path(__file__).resolve().parents[1]
COLORS={'CFR':'#425466','Fictitious play':'#24734e','Double oracle':'#b36b23'}

def median_run(runs):
    return sorted(runs,key=lambda q:q['wall_seconds'])[len(runs)//2]

def style_axes(axes,threshold):
    for ax in axes:
        ax.axhline(threshold,color='#777777',ls=':',lw=1)
        ax.grid(True,which='major',alpha=.2)
        ax.margins(x=.12,y=.15)

def figure_convergence(r,output):
    fig,axes=plt.subplots(1,2,figsize=(6.3,2.9))
    for name,style in (('CFR','-'),('Fictitious play','--'),('Double oracle','-.')):
        run=median_run(r['scenarios']['neutral']['methods'][name])
        h=[q for q in run['history'] if q['iteration']>0]
        values=[max(q['spr'],1e-8) for q in h]
        axes[0].loglog([q['iteration'] for q in h],values,label=name,color=COLORS[name],ls=style,lw=1.8)
        axes[1].loglog([q['elapsed_seconds'] for q in h],values,label=name,color=COLORS[name],ls=style,lw=1.8)
    style_axes(axes,r['configuration']['spr_threshold'])
    axes[0].set_ylabel('SPR')
    axes[0].set_xlabel('Iterations')
    axes[1].set_xlabel('Elapsed time (s)')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False,
               handlelength=1.3,columnspacing=.9,handletextpad=.4)
    fig.subplots_adjust(left=.13,right=.96,bottom=.23,top=.81,wspace=.38)
    fig.savefig(output,dpi=240)
    plt.close(fig)

def figure_decomposition(r,output):
    """(a) CFR is identical on the default game and its halves, DO is not; (b) restricted-LP cost."""
    fig,axes=plt.subplots(1,2,figsize=(6.3,3.0))
    floor=1e-8
    lines=[('CFR','neutral','CFR, default','-',1.0),('CFR','first_player','CFR, P1 starts',(0,(1,1.6)),1.0),
           ('Double oracle','neutral','DO, default','-',1.0),('Double oracle','first_player','DO, P1 starts','--',.75),
           ('Double oracle','second_player','DO, P2 starts',':',.75)]
    for method,scenario,label,style,alpha in lines:
        run=median_run(r['scenarios'][scenario]['methods'][method])
        h=[q for q in run['history'] if 0<q['iteration']<=2000]
        # Numerical certificates (~1e-16) are drawn at the lower edge rather than stretching the axis.
        axes[0].loglog([q['iteration'] for q in h],[max(q['spr'],floor) for q in h],label=label,
                       color=COLORS[method],ls=style,lw=1.8 if style=='-' else 2.2,alpha=alpha)
    style_axes(axes[:1],r['configuration']['spr_threshold'])
    axes[0].set_ylim(floor,1.0)
    axes[0].set_ylabel('SPR')
    axes[0].set_xlabel('Iterations')
    axes[0].set_title('(a) first 2,000 iterations',loc='left',fontsize=11)
    exact=r['analysis']['double_oracle_exact']
    for scenario,label,marker in (('neutral','default','o'),('first_player','P1 starts','^'),('second_player','P2 starts','s')):
        trace=exact[scenario]['restricted_lp_trace']
        axes[1].plot([row[0] for row in trace],[1000*row[2] for row in trace],marker,ms=2.6,
                     color=COLORS['Double oracle'],alpha=1.0 if scenario=='neutral' else .45,label=label)
    axes[1].set_xlabel('Stored P1 plans')
    axes[1].set_ylabel('Restricted LP time (ms)')
    axes[1].set_title('(b) DO restricted-game cost',loc='left',fontsize=11)
    axes[1].grid(True,which='major',alpha=.2)
    axes[1].legend(frameon=False,fontsize=10,handletextpad=.2,loc='upper left')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False,fontsize=10,
               handlelength=1.8,columnspacing=.9,handletextpad=.4)
    fig.subplots_adjust(left=.13,right=.97,bottom=.22,top=.74,wspace=.42)
    fig.savefig(output,dpi=240)
    plt.close(fig)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ATTEMPT/'task4'/'results'/'experiments.json')
    parser.add_argument('--output',type=Path,default=ATTEMPT/'task4'/'results'/'convergence.png')
    parser.add_argument('--output2',type=Path,default=ATTEMPT/'task4'/'results'/'decomposition.png')
    args=parser.parse_args()
    r=json.loads(args.input.read_text())
    plt.rcParams.update({'font.size':12,'axes.labelsize':12,'xtick.labelsize':12,'ytick.labelsize':12,
                        'legend.fontsize':12,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    figure_convergence(r,args.output)
    print(args.output)
    if 'analysis' in r and len(r['scenarios'])>2:
        figure_decomposition(r,args.output2)
        print(args.output2)

if __name__=='__main__':main()
