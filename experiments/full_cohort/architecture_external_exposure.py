"""True-ID fixed-receiver neighbour-removal geometry on original Pheno4D scans.

Opaque point-splat depth buffers; all regions, dates and40 horizontal nominal
directions. Pixel-potential denominators stay fixed for each receiver/direction
within a removal. These are projected exposure fractions, not irradiance, PAR,
absorption, physiological effects, measured leaf area or physical length units.
"""
import argparse,itertools,json,time,hashlib
from pathlib import Path
import numpy as np
from architecture_analysis import write,dump,sha
from architecture_external_pheno4d import normalized_pair

def raster_layers(xyz,labels,view,ids,size,scale,axis,base,radius=1):
    t=view*2*np.pi/40;horizontal=xyz[:,0]*np.cos(t)+xyz[:,1]*np.sin(t);depth=(-xyz[:,0]*np.sin(t)+xyz[:,1]*np.cos(t)).astype('float32');u=np.rint(axis+scale*horizontal).astype(int);v=np.rint(base-scale*xyz[:,2]).astype(int);layers=np.full((len(ids),size,size),-np.inf,dtype='float32')
    for li,label in enumerate(ids):
        selected=labels==label;uu=u[selected];vv=v[selected];dd=depth[selected];layer=layers[li].ravel()
        for dy in range(-radius,radius+1):
            for dx in range(-radius,radius+1):
                x=uu+dx;y=vv+dy;valid=(x>=0)&(x<size)&(y>=0)&(y<size);np.maximum.at(layer,y[valid]*size+x[valid],dd[valid])
    return layers

def ownership(layers,ids):
    # Exact ties retain the lower annotation ID, consistent for every removal.
    best=np.full(layers.shape[1:],-np.inf,dtype='float32');owner=np.full(best.shape,-1,np.int16)
    for i,label in enumerate(ids):
        take=layers[i]>best;best[take]=layers[i][take];owner[take]=label
    return owner

def removal_exposure(layers,ids,removable):
    owner=ownership(layers,ids);states=[];edges=[];after={}
    for removed in removable:
        keep=[i for i,label in enumerate(ids) if label!=removed];after[removed]=ownership(layers[keep],[ids[i] for i in keep])
    for receiver in removable:
        i=ids.index(receiver);potential=np.isfinite(layers[i]);den=int(potential.sum());baseline=int((owner==receiver).sum());multiplicity=np.zeros(potential.shape,np.uint8)
        for j,label in enumerate(ids):
            if j==i:continue
            # Same precedence as ownership, including exact equal-depth points.
            blocked=potential&np.isfinite(layers[j])&((layers[j]>layers[i])|((layers[j]==layers[i])&(label<receiver)));multiplicity+=blocked
        state=dict(receiver_id=receiver,potential_pixels=den,baseline_visible_pixels=baseline,baseline_visible_fraction=baseline/den if den else None,exclusively_blocked_pixels=int((potential&(multiplicity==1)).sum()),redundantly_blocked_pixels=int((potential&(multiplicity>=2)).sum()))
        assert den==baseline+state['exclusively_blocked_pixels']+state['redundantly_blocked_pixels']
        states.append(state)
        for removed in removable:
            if removed==receiver:continue
            exposed=int((after[removed]==receiver).sum());gain=exposed-baseline
            if gain<0:raise AssertionError('Removing a blocker reduced receiver visibility')
            edges.append(dict(receiver_id=receiver,removed_id=removed,potential_pixels=den,baseline_visible_pixels=baseline,after_removal_visible_pixels=exposed,gain_pixels=gain,baseline_visible_fraction=baseline/den if den else None,after_removal_visible_fraction=exposed/den if den else None,gain_fraction=gain/den if den else None))
        assert sum(r['gain_pixels'] for r in edges if r['receiver_id']==receiver)<=state['exclusively_blocked_pixels']
    return states,edges

def run(args):
    started=time.time();out=args.run/'architecture'/'external_pheno4d'/'exposure';out.mkdir(parents=True,exist_ok=True);manifest=json.loads((args.external/'download_manifest.json').read_text(encoding='utf-8'))
    if manifest['status']!='completed':raise ValueError('External download incomplete')
    by={}
    for r in manifest['files']:
        p=args.external/r['path']
        if sha(p)!=r['sha256']:raise ValueError('Input hash mismatch')
        by.setdefault(p.parent.name,[]).append((p,r))
    if len(by)!=7 or any(len(pair)!=2 for pair in by.values()):raise ValueError('Expected seven complete two-date pairs')
    source=(Path(__file__)).read_bytes();(out/'execution_source.py').write_bytes(source);allstates=[];alledges=[];case_rows=[];gauge_rows=[]
    for pi,(plant,pair) in enumerate(sorted(by.items())):
        pair.sort();a=np.loadtxt(pair[0][0],dtype='float32');b=np.loadtxt(pair[1][0],dtype='float32');na,nb,origin,height=normalized_pair(a,b);union=np.concatenate([na[a[:,4]>0],nb[b[:,4]>0]]);radius=float(np.linalg.norm(union[:,:2],axis=1).max());minz=float(union[:,2].min());maxz=float(union[:,2].max());span=max(2*radius,maxz-minz)*1.04
        gauge_rows.append(dict(plant=plant,shared_earlier_origin=origin.tolist(),shared_earlier_height_gauge=height,normalized_domain_span=span,normalized_min_z=minz,normalized_max_z=maxz,normalization='Earlier scan foreground height; original coordinate units are not assumed to be mm. Shared screenextent and pixel scale across bothdates/all40directions.'))
        for date,cloud,xyz in [('0324',a,na),('0325',b,nb)]:
            # Same geometry bins for the two label conventions, without using
            # instance identity to decide which point represents a spatial bin.
            foreground=cloud[:,4]>0;pp=xyz[foreground];rawlabels=cloud[foreground,3:5].astype(np.int16);_,ix=np.unique(np.floor(pp/.003).astype(np.int32),axis=0,return_index=True);pp=pp[ix];rawlabels=rawlabels[ix]
            for col,scheme in [(0,'leaf_collar'),(1,'leaf_tip')]:
                labels=rawlabels[:,col];ids=sorted(int(i) for i in np.unique(labels) if i>0)
                original_ids=sorted(int(i) for i in np.unique(cloud[foreground,3+col]) if i>0)
                if ids!=original_ids:raise ValueError('Spatial binning removed an annotation instance')
                removable=[i for i in ids if i>=(2 if scheme=='leaf_collar' else 1)];configs=[(256,1,'primary')]
                if plant in ['Maize01','Maize04','Maize07']:configs.extend([(128,0,'grid_and_footprint_sensitivity'),(128,1,'grid_and_footprint_sensitivity')])
                for resolution,splat,experiment in configs:
                    scale=(resolution-6)/span;axis=resolution/2;base=resolution/2+scale*(minz+maxz)/2
                    for view in range(40):
                        layers=raster_layers(pp,labels,view,ids,resolution,scale,axis,base,splat);states,edges=removal_exposure(layers,ids,removable);common=dict(plant=plant,date=date,scheme=scheme,resolution=resolution,splat_radius_px=splat,experiment=experiment,view=view,azimuth_deg=view*9,direction='horizontal_parallel_rays')
                        allstates.extend({**common,**r} for r in states);alledges.extend({**common,**r} for r in edges)
                    case_rows.append(dict(plant=plant,date=date,scheme=scheme,resolution=resolution,splat_radius_px=splat,regions=len(ids),removable_regions=len(removable),stem_separately_retained=scheme=='leaf_collar',directions=40,binned_points=len(pp),source_sha256=pair[0 if date=='0324' else 1][1]['sha256']))
        print(json.dumps(dict(stage='external_exposure',plant=plant,completed=pi+1,total=len(by))),flush=True)
    write(out/'fixed_receiver_visibility.csv',allstates);write(out/'fixed_shape_neighbour_removal_edges.csv',alledges);write(out/'exposure_cases.csv',case_rows);dump(out/'shared_pair_gauges.json',gauge_rows)
    primary=[r for r in alledges if r['experiment']=='primary'];lookup={(r['plant'],r['date'],r['scheme'],r['receiver_id'],r['removed_id'],r['view']):r for r in primary};changes=[]
    for key,early in sorted(lookup.items()):
        plant,date,scheme,receiver,removed,view=key
        if date!='0324':continue
        late=lookup.get((plant,'0325',scheme,receiver,removed,view))
        changes.append(dict(plant=plant,scheme=scheme,receiver_id=receiver,removed_id=removed,view=view,azimuth_deg=view*9,true_instance_ids_present_both_dates=late is not None,early_gain_fraction=early['gain_fraction'],later_gain_fraction=late['gain_fraction'] if late else None,later_minus_earlier_gain_fraction=late['gain_fraction']-early['gain_fraction'] if late and early['gain_fraction'] is not None and late['gain_fraction'] is not None else None,early_receiver_potential_pixels=early['potential_pixels'],later_receiver_potential_pixels=late['potential_pixels'] if late else None,interpretation='Same annotated receiver/blocker IDs; eachdatefixedshape removal. Crossdate changes combine region shape, pose, arrangement and digitization; no isolated growthcause.'))
    write(out/'true_id_cross_date_neighbour_changes.csv',changes)
    sensitivity=[]
    for r in alledges:
        if r['experiment']=='primary':continue
        ref=lookup.get((r['plant'],r['date'],r['scheme'],r['receiver_id'],r['removed_id'],r['view']))
        if ref and r['gain_fraction'] is not None and ref['gain_fraction'] is not None:sensitivity.append(dict(plant=r['plant'],date=r['date'],scheme=r['scheme'],receiver_id=r['receiver_id'],removed_id=r['removed_id'],view=r['view'],resolution=r['resolution'],splat_radius_px=r['splat_radius_px'],reference_gain_fraction=ref['gain_fraction'],alternative_gain_fraction=r['gain_fraction'],gain_difference_from_256_radius1=r['gain_fraction']-ref['gain_fraction'],absolute_gain_difference=abs(r['gain_fraction']-ref['gain_fraction'])))
    write(out/'grid_and_point_footprint_sensitivity.csv',sensitivity)
    aggregate=[]
    for plant,date,scheme in sorted(set((r['plant'],r['date'],r['scheme']) for r in primary)):
        rr=[r for r in primary if (r['plant'],r['date'],r['scheme'])==(plant,date,scheme)]
        for receiver,removed in sorted(set((r['receiver_id'],r['removed_id']) for r in rr)):
            vv=[r['gain_fraction'] for r in rr if r['receiver_id']==receiver and r['removed_id']==removed and r['gain_fraction'] is not None];aggregate.append(dict(plant=plant,date=date,scheme=scheme,receiver_id=receiver,removed_id=removed,directions_with_receiver_footprint=len(vv),mean_gain_fraction=float(np.mean(vv)) if vv else None,max_gain_fraction=float(max(vv)) if vv else None))
    write(out/'mean_directional_neighbour_edges.csv',aggregate)
    summary=dict(plants=len(by),scans=len(manifest['files']),annotation_schemes=2,nominal_horizontal_directions=40,primary_resolution=256,primary_splat_radius_px=1,primary_case_rows=sum(r['resolution']==256 for r in case_rows),primary_visibility_rows=sum(r['experiment']=='primary' for r in allstates),primary_edge_rows=len(primary),true_id_crossdate_rows=len(changes),true_id_crossdate_rows_available=sum(r['true_instance_ids_present_both_dates'] for r in changes),positive_primary_gain_rows=sum(r['gain_pixels']>0 for r in primary),sensitivity_rows=len(sensitivity),max_absolute_gain_fraction_sensitivity=float(max(r['absolute_gain_difference'] for r in sensitivity)) if sensitivity else None,median_absolute_gain_fraction_sensitivity=float(np.median([r['absolute_gain_difference'] for r in sensitivity])) if sensitivity else None,input_manifest_sha256=sha(args.external/'download_manifest.json'),script_sha256=hashlib.sha256(source).hexdigest(),elapsed_seconds=time.time()-started,scope='Known registered laser geometry and supplied annotation regions, opaque point-splat parallel-ray projected exposure. Potential receiver footprint fixed withinremoval; no physicalunits/PAR/absorption/physiology. Tipregionsincludeassignedstemsections; collarstem1remainsblockingandcannotberemoved. Crossdatecomparisondoesnotisolategrowthcause.')
    dump(out/'external_exposure_summary.json',summary);plot(out,allstates,primary,aggregate);print(json.dumps(summary,indent=2),flush=True)

def plot(out,states,edges,aggregate):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],'font.size':10})
    for scheme in ['leaf_collar','leaf_tip']:
        rr=[r for r in aggregate if r['scheme']==scheme];plants=sorted(set(r['plant'] for r in rr));maximum=max([r['mean_gain_fraction'] or 0 for r in rr] or [1]);fig,axes=plt.subplots(2,7,figsize=(16,6.2))
        for col,plant in enumerate(plants):
            ids=sorted(set(r['receiver_id'] for r in rr if r['plant']==plant))
            for row,date in enumerate(['0324','0325']):
                matrix=np.zeros((len(ids),len(ids)))
                for r in rr:
                    if r['plant']==plant and r['date']==date and r['mean_gain_fraction'] is not None:matrix[ids.index(r['receiver_id']),ids.index(r['removed_id'])]=r['mean_gain_fraction']
                im=axes[row,col].imshow(matrix,vmin=0,vmax=maximum,cmap='magma');axes[row,col].set_xticks(range(len(ids)),ids);axes[row,col].set_yticks(range(len(ids)),ids);axes[row,col].set_title(f'{plant}\n{date}',fontsize=9)
                if col==0:axes[row,col].set_ylabel('Receiver ID')
                if row==1:axes[row,col].set_xlabel('Removed ID')
        fig.subplots_adjust(left=.04,right=.9,bottom=.12,top=.77,hspace=.55,wspace=.55);cax=fig.add_axes([.92,.2,.012,.52]);fig.colorbar(im,cax=cax,label='Mean receiver visibility gain / own potential footprint');fig.suptitle(f'Fixed-shape neighbour removals: {scheme.replace("_"," ")} annotation regions\n40 horizontal directions; known point geometry; not physiological absorption',fontsize=13,y=.97);fig.savefig(out/f'neighbour_edges_{scheme}.png',dpi=180);plt.close(fig)
    # Prespecified persistent IDs3(receiver)/4(removed), all seven plants.
    fig,axes=plt.subplots(2,4,figsize=(13,9),subplot_kw={'projection':'polar'});colors={'0324':'#265d99','0325':'#c57523'}
    for ax,plant in zip(axes.ravel(),sorted(set(r['plant'] for r in edges))):
        for date in ['0324','0325']:
            rr=sorted([r for r in edges if r['plant']==plant and r['date']==date and r['scheme']=='leaf_tip' and r['receiver_id']==3 and r['removed_id']==4],key=lambda r:r['view'])
            if rr:
                theta=np.radians([r['azimuth_deg'] for r in rr]+[rr[0]['azimuth_deg']+360]);baseline=[r['baseline_visible_fraction'] or 0 for r in rr];after=[r['after_removal_visible_fraction'] or 0 for r in rr];ax.plot(theta,baseline+baseline[:1],color=colors[date],label=date+' baseline');ax.plot(theta,after+after[:1],'--',color=colors[date],label=date+' remove4')
        ax.set_ylim(0,1);ax.set_title(plant,y=1.12);ax.tick_params(labelsize=8)
    axes.ravel()[-1].axis('off');handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower right',bbox_to_anchor=(.95,.1));fig.suptitle('Known persistent tip-method region3: baseline vs fixed-shape removal of region4\nTwo dates differ in both geometry and arrangement; projected exposure only',fontsize=13,y=.98);fig.subplots_adjust(left=.05,right=.95,bottom=.07,top=.80,hspace=.60,wspace=.4);fig.savefig(out/'persistent_receiver3_removed4_polar.png',dpi=180);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--external',type=Path,required=True);p.add_argument('--run',type=Path,required=True);run(p.parse_args())
