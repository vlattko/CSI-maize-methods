"""Print-sized real-geometry scaffold and predeclared neighbour-edge figures."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from architecture_analysis import read,dump,sha
from architecture_external_tracking import source_gauge

def run(args):
    out=args.run/'architecture'/'figures';out.mkdir(exist_ok=True);ext=args.run/'architecture'/'external_pheno4d';p=args.external/'original/Pheno4D/Maize01/M01_0324_a.txt';cloud=np.loadtxt(p,dtype='float32');fg=cloud[:,4]>0;origin,height=source_gauge(cloud);points=(cloud[fg,:3]-origin)/height;labels=cloud[fg,4].astype(int);rr=[r for r in read(ext/'scaffold/terminal_scaffold_point_support.csv') if r['scan']=='M01_0324_a'];rng=np.random.default_rng(2026);ix=rng.choice(len(points),min(12000,len(points)),replace=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],'font.size':9,'axes.titlesize':9,'xtick.labelsize':8,'ytick.labelsize':8})
    fig,axes=plt.subplots(1,3,figsize=(6.7,2.8))
    for ax,(a,b,xlabel,ylabel),letter in zip(axes,[(0,2,'Horizontal','Up'),(1,2,'Depth','Up'),(0,1,'Horizontal','Depth')],'ABC'):
        ax.scatter(points[ix,a],points[ix,b],c=labels[ix],cmap='tab10',s=.25,alpha=.35,rasterized=True)
        for r in rr:
            q=np.array([float(r[k]) for k in ['point_x_native_horizontal','point_y_native_depth','point_z_native_up']]);ax.plot(q[a],q[b],'k*',markersize=7);ax.annotate(r['mapped_id_evaluation_only'],(q[a],q[b]),xytext=(3,3),textcoords='offset points',fontsize=8)
        ax.set_xlabel(xlabel);ax.set_ylabel(ylabel);ax.set_aspect('equal');ax.set_title(letter,loc='left',fontweight='bold');ax.margins(.08)
    fig.tight_layout(pad=.6,w_pad=1.2);fig.savefig(out/'terminal_scaffold_publication.png',dpi=300);plt.close(fig)
    edges=[r for r in read(ext/'exposure/fixed_shape_neighbour_removal_edges.csv') if r['experiment']=='primary' and r['scheme']=='leaf_tip' and r['receiver_id']=='3' and r['removed_id']=='4'];fig=plt.figure(figsize=(6.7,3.1));left=fig.add_subplot(121,projection='polar');right=fig.add_subplot(122);colors={'0324':'#265d99','0325':'#b66b1e'}
    for date in ['0324','0325']:
        records=sorted([r for r in edges if r['plant']=='Maize01' and r['date']==date],key=lambda r:int(r['view']));theta=np.radians([float(r['azimuth_deg']) for r in records]+[360]);baseline=[float(r['baseline_visible_fraction']) for r in records];after=[float(r['after_removal_visible_fraction']) for r in records];left.plot(theta,baseline+baseline[:1],color=colors[date],label=date+' baseline');left.plot(theta,after+after[:1],'--',color=colors[date],label=date+' remove 4')
    left.set_ylim(0,1);left.set_yticks([.25,.5,.75,1]);left.set_title('A  Plant 1: receiver 3',pad=14,loc='left');left.legend(loc='lower left',bbox_to_anchor=(-.28,-.42),fontsize=7,ncol=2,frameon=False)
    plants=sorted(set(r['plant'] for r in edges))
    for i,plant in enumerate(plants):
        values=[float(np.mean([float(r['gain_fraction']) for r in edges if r['plant']==plant and r['date']==date])) for date in ['0324','0325']];right.plot([i-.1,i+.1],values,color='#777',linewidth=1)
        for x,date,value in zip([i-.1,i+.1],['0324','0325'],values):right.plot(x,value,'o',color=colors[date],markersize=4)
    right.set_xticks(range(len(plants)),[p[-2:] for p in plants]);right.set_xlabel('Known physical plant');right.set_ylabel('Mean removal gain / potential footprint');right.set_title('B  Receiver 3 / removed 4',loc='left');right.set_ylim(bottom=0);right.spines[['top','right']].set_visible(False);fig.subplots_adjust(left=.055,right=.98,bottom=.28,top=.82,wspace=.65);fig.savefig(out/'known_id_dependency_publication.png',dpi=300);plt.close(fig)
    dump(out/'publication_figure_provenance.json',dict(script_sha256=sha(__file__),source_point_file_sha256=sha(p),scaffold_table_sha256=sha(ext/'scaffold/terminal_scaffold_point_support.csv'),exposure_table_sha256=sha(ext/'exposure/fixed_shape_neighbour_removal_edges.csv'),outputs={name:sha(out/name) for name in ['terminal_scaffold_publication.png','known_id_dependency_publication.png']},scope='Print-sized figures from frozen numerical results; no model selection or omitted case in prespecified receiver3/removed4 pair.'))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--external',type=Path,required=True);p.add_argument('--run',type=Path,required=True);run(p.parse_args())
