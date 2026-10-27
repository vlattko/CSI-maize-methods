"""Silhouette observability and conditional geometry; no anatomical truth implied.

All angles are nominal index*2pi/40. The visual hull assumes an orthographic,
level camera and uniform turn about an estimated basal axis. Geometry is in
training-image-height gauge, without physical calibration. Held-out views never
estimate camera gauge or enter carving. Skeleton paths are image hypotheses.
"""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='2'
import argparse,csv,hashlib,itertools,json,math,time,datetime,io
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from scipy import ndimage as ndi
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.morphology import skeletonize
from threadpoolctl import threadpool_limits
_THREAD_LIMITER=threadpool_limits(limits=2)
cv2.setNumThreads(1)
KEYS=['treatment','measurement','genotype','replicate','session','camera']
METRICS=['area_fraction','height_fraction','width_fraction','solidity','skeleton_endpoint_proxy','stratum_1_width_fraction','stratum_3_width_fraction','stratum_5_width_fraction']
class MaskIntegrityError(ValueError):pass

def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def dump(path,data):Path(path).write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def number(x):
    try:return float(x)
    except (TypeError,ValueError):return np.nan
def accepted(row):return str(row.get('qc_pass','')).lower() in ['true','1']
def coverage(views):
    v=sorted(set(int(i) for i in views));q=len(set(i//10 for i in v));g=np.diff(v+[v[0]+40]) if v else np.array([40])
    return dict(accepted_views=len(v),quadrants=q,max_circular_gap=int(max(g)),eligible=bool(len(v)>=20 and q==4),strict_eligible=bool(len(v)>=30 and q==4 and max(g)<=4))

def harmonic(views,values,order=2):
    v=np.asarray(views,int);y=np.asarray(values,float);ok=np.isfinite(y);v=v[ok];y=y[ok]
    t=v*2*np.pi/40;a=np.column_stack([np.ones(len(v))]+[f(k*t) for k in range(1,order+1) for f in [np.cos,np.sin]])
    if not coverage(v)['eligible'] or len(v)<max(2*order+2,8) or np.linalg.matrix_rank(a)<a.shape[1]:return dict(status='insufficient_or_aliased',n=len(v))
    b=np.linalg.lstsq(a,y,rcond=None)[0];pred=a@b
    out=dict(status='fit',n=len(v),mean=float(b[0]),rmse=float(np.sqrt(np.mean((y-pred)**2))),condition=float(np.linalg.cond(a)))
    for k in range(1,order+1):out[f'amplitude_{k}']=float(np.hypot(b[2*k-1],b[2*k]))
    # This is prediction of a rotational signal, not independent plants.
    train=v%4!=2;test=~train
    if train.sum()>=a.shape[1]+2 and test.sum()>=2 and np.linalg.matrix_rank(a[train])==a.shape[1]:
        bt=np.linalg.lstsq(a[train],y[train],rcond=None)[0];out['heldout_rmse']=float(np.sqrt(np.mean((y[test]-a[test]@bt)**2)));out['heldout_n']=int(test.sum())
    return out

def mask_candidates(mask,min_length=12):
    """Generic root-to-terminal paths, no inherited seeds or botanical labels."""
    sk=skeletonize(mask);ys,xs=np.where(sk)
    if not len(xs):return []
    lookup=np.full(mask.shape,-1,int);lookup[ys,xs]=np.arange(len(xs));edges=[];weights=[]
    for dy,dx in [(0,1),(1,-1),(1,0),(1,1)]:
        ny=ys+dy;nx=xs+dx;ok=(ny>=0)&(ny<mask.shape[0])&(nx>=0)&(nx<mask.shape[1]);src=np.flatnonzero(ok);dst=lookup[ny[ok],nx[ok]];hit=dst>=0
        edges.extend(zip(src[hit],dst[hit]));weights.extend([np.hypot(dx,dy)]*int(hit.sum()))
    if not edges:return []
    edges=np.asarray(edges);g=csr_matrix((weights,(edges[:,0],edges[:,1])),shape=(len(xs),len(xs)));g=g+g.T;deg=np.diff(g.indptr)
    basal=xs[ys>=np.percentile(ys,95)];axis=float(np.median(basal));root=int(np.argmax(ys-.02*abs(xs-axis)))
    dist,previous=dijkstra(g,indices=root,return_predecessors=True);h=max(float(np.ptp(ys)+1),1);result=[]
    for tip in np.flatnonzero(deg==1):
        if tip==root or not np.isfinite(dist[tip]):continue
        path=[int(tip)];at=int(tip)
        while at!=root and previous[at]>=0:at=int(previous[at]);path.append(at)
        j=next((i for i in path[1:] if deg[i]>=3),root);length=float(dist[tip]-dist[j])
        if length<min_length:continue
        result.append(dict(candidate_id=len(result),terminal_x_px=int(xs[tip]),terminal_y_px=int(ys[tip]),root_x_px=int(xs[root]),root_y_px=int(ys[root]),mask_height_px=h,terminal_x_fraction=float((xs[tip]-axis)/h),terminal_y_fraction=float((ys[tip]-ys.min())/h),root_geodesic_over_height=float(dist[tip]/h),visible_branch_over_height=length/h,censored=bool(xs[tip]<3 or xs[tip]>=mask.shape[1]-3 or ys[tip]<3 or ys[tip]>=mask.shape[0]-3),status='visible_skeleton_path_hypothesis'))
    return result

def correspondence(a,b):
    """All gated pairs plus unmatched alternative; heuristic weights, not posterior."""
    rows=[]
    for i,x in enumerate(a):
        costs=[abs(x['terminal_y_fraction']-y['terminal_y_fraction'])/.15+abs(x['root_geodesic_over_height']-y['root_geodesic_over_height'])/.3 for y in b]
        den=math.exp(-2)+sum(math.exp(-min(c,700)) for c in costs)
        for j,c in enumerate(costs):rows.append(dict(earlier_candidate=i,later_candidate=j,cost=float(c),conditional_weight=math.exp(-min(c,700))/den,unmatched_weight=math.exp(-2)/den,interpretation='same_view_shape_compatibility_only_no_selected_identity'))
    return rows

def visual_hull(masks,axis,base,scale,resolution=64,roll=0,dilation_px=0,allowed_violation_fraction=0):
    """Carve using TRAIN masks only; orthographic normalized coordinates."""
    # This declared rectangular domain is a prior. Boundary occupancy is reported.
    xy=np.linspace(-.8,.8,resolution);yy=np.linspace(0,1.15,resolution)
    x,y,z=np.meshgrid(xy,yy,xy,indexing='ij');grid=(x.astype('float32'),y.astype('float32'),z.astype('float32'));votes=np.zeros(x.shape,np.uint16)
    for view,mask in sorted(masks.items()):
        if dilation_px:mask=cv2.dilate(mask.astype('uint8'),np.ones((2*dilation_px+1,2*dilation_px+1),np.uint8)).astype(bool)
        u,v=coordinates(grid,view,axis,base,scale,roll);ui=np.rint(u).astype(int);vi=np.rint(v).astype(int);valid=(ui>=0)&(ui<mask.shape[1])&(vi>=0)&(vi<mask.shape[0]);inside=np.zeros(x.shape,bool);inside[valid]=mask[vi[valid],ui[valid]];votes+=inside
    threshold=len(masks)-math.floor(len(masks)*allowed_violation_fraction)
    return votes>=threshold,grid
def coordinates(grid,view,axis,base,scale,roll=0):
    x,y,z=grid;t=view*2*np.pi/40;r=np.radians(roll);horizontal=x*np.cos(t)+z*np.sin(t)
    return axis+scale*(horizontal*np.cos(r)+y*np.sin(r)),base+scale*(horizontal*np.sin(r)-y*np.cos(r))
def project_hull(hull,grid,view,shape,axis,base,scale,roll=0):
    u,v=coordinates(grid,view,axis,base,scale,roll);ui=np.rint(u[hull]).astype(int);vi=np.rint(v[hull]).astype(int);ok=(ui>=0)&(ui<shape[1])&(vi>=0)&(vi<shape[0]);out=np.zeros(shape,np.uint8);out[vi[ok],ui[ok]]=1
    # Finite voxels cover a pixel footprint; fixed dilation depends on gauge/grid.
    radius=max(1,int(math.ceil(scale*1.6/(hull.shape[0]-1)/2)));return cv2.dilate(out,np.ones((2*radius+1,2*radius+1),np.uint8)).astype(bool)

def load_mask(row,root,width=256):
    path=Path(row['mask']);path=path if path.is_absolute() else root/path
    content=path.read_bytes()
    if row.get('mask_sha256') and hashlib.sha256(content).hexdigest()!=row['mask_sha256']:raise MaskIntegrityError('Mask hash mismatch: '+str(path))
    with Image.open(io.BytesIO(content)) as im:
        mask=im.convert('L');mask=mask.resize((width,round(mask.height*width/mask.width)),Image.Resampling.NEAREST);return np.asarray(mask)>0

def trait_from_mask(mask):
    ys,xs=np.where(mask);h,w=mask.shape
    if not len(xs):return {}
    result=dict(area_fraction=float(mask.mean()),height_fraction=float(np.ptp(ys)+1)/h,width_fraction=float(np.ptp(xs)+1)/w)
    contours,_=cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);boundary=np.concatenate(contours);ha=cv2.contourArea(cv2.convexHull(boundary));result['solidity']=sum(cv2.contourArea(c) for c in contours)/max(ha,1)
    sk=skeletonize(mask);deg=ndi.convolve(sk.astype('uint8'),np.ones((3,3),np.uint8))-sk;result['skeleton_endpoint_proxy']=int(ndi.label(sk&(deg==1),np.ones((3,3)))[1])
    for i in [0,2,4]:
        widths=[]
        for row in mask[int(ys.min()+(np.ptp(ys)+1)*i/5):int(ys.min()+(np.ptp(ys)+1)*(i+1)/5)]:
            ix=np.flatnonzero(row)
            if len(ix):widths.append(np.ptp(ix)+1)
        result[f'stratum_{i+1}_width_fraction']=float(np.median(widths)/w) if widths else 0.
    return result

def complete_record_snapshot(intake,run,out):
    """Freeze only acquisitions with every expected decodable record present.

    This reads atomic extraction records without modifying the producer's state.
    Both cameras must be complete relative to the immutable intake; raw partial
    acquisitions remain partial. Sessions still processing are excluded entirely.
    """
    known={r['session'] for r in read(intake/'session_decode_qc.csv') if accepted({'qc_pass':r['known_reference_identity']})}
    expected={}
    for row in read(intake/'jpeg_decode_manifest.csv'):
        session=str(Path(row['path']).parent.parent).replace('\\','/')
        if session in known and str(row['decode_ok']).lower()=='true':expected.setdefault(session,set()).add(row['path'])
    records={};hashes={}
    for path in sorted((run/'extraction/records').glob('*.json')):
        r=json.loads(path.read_text(encoding='utf-8'));source=r['source']
        if source in records:raise ValueError('Duplicate extraction record source')
        records[source]=r;hashes[source]=sha(path)
    complete=[s for s,paths in expected.items() if paths and paths.issubset(records)];rows=[records[p] for s in sorted(complete) for p in sorted(expected[s])]
    source=out/'complete_acquisition_partial_snapshot.csv';write(source,rows)
    dump(out/'partial_snapshot_provenance.json',dict(mode='complete_acquisition_partial_snapshot',expected_acquisitions=len(expected),complete_acquisitions=len(complete),completed_records_scanned=len(records),included_rows=len(rows),omitted_acquisitions=len(expected)-len(complete),rule='All expected decodable source records, both cameras, present; incomplete running acquisitions omitted.',included_record_sha256={p:hashes[p] for s in complete for p in expected[s]}))
    return source

def resolve_input(args,out):
    source=getattr(args,'input_table',None)
    if source:return source,args.run,'explicit_partial_snapshot'
    if getattr(args,'records_snapshot',False):return complete_record_snapshot(args.intake,args.run,out),args.run,'complete_acquisition_partial_snapshot'
    source=args.run/'extraction/view_traits.csv'
    if source.exists():return source,args.run,'full_extraction'
    return args.intake/'segmentation_triage/view_diagnostics.csv',args.intake,'quarter_view_diagnostic'

def run(args):
    start=time.time();out=args.run/'architecture'/args.output_name;out.mkdir(parents=True,exist_ok=True)
    execution_source=Path(__file__).read_bytes();(out/'architecture_execution_source.py').write_bytes(execution_source);execution_sha=hashlib.sha256(execution_source).hexdigest()
    source,maskroot,mode=resolve_input(args,out)
    rows=read(source);seen=set();groups={};errors=[];mask_audit=[]
    for row in rows:
        key=tuple(row[k] for k in KEYS);vk=key+(int(row['view']),)
        if vk in seen:raise ValueError('Duplicate camera/view key '+str(vk))
        seen.add(vk);groups.setdefault(key,[]).append(row)
    descriptors=[];harmonics=[];budgets=[];candidates=[];candidate_groups={}
    for gi,(key,group) in enumerate(sorted(groups.items()),1):
        identity=dict(zip(KEYS,key));good=[r for r in group if accepted(r)];descriptors.append({**identity,**coverage([r['view'] for r in good]),'total_rows':len(group),'qc_failed':len(group)-len(good)})
        valid=[]
        for row in good:
            try:
                path=Path(row['mask']);path=path if path.is_absolute() else maskroot/path;actual=sha(path)
                if row.get('mask_sha256') and actual!=row['mask_sha256']:raise MaskIntegrityError('Mask hash mismatch')
                mask=load_mask(row,maskroot);mask_audit.append({**identity,'view':row['view'],'mask':row['mask'],'mask_sha256':actual,'source_sha256':row.get('source_sha256','')})
                if mode=='quarter_view_diagnostic':row={**row,**trait_from_mask(mask)}
                valid.append(row)
                if int(row['view']) in [0,10,20,30]:
                    cc=mask_candidates(mask);candidate_groups[key+(int(row['view']),)]=cc
                    for c in cc:candidates.append({**identity,'view':row['view'],**c})
            except Exception as e:
                if isinstance(e,MaskIntegrityError):raise
                errors.append({**identity,'view':row['view'],'error':str(e)})
        for metric in METRICS:
            v=np.array([int(r['view']) for r in valid]);y=np.array([number(r.get(metric)) for r in valid]);ok=np.isfinite(y);v=v[ok];y=y[ok]
            if not len(y):continue
            h=harmonic(v,y);harmonics.append({**identity,'metric':metric,**h,'observed_min':float(min(y)),'observed_max':float(max(y)),'directional_range_over_mean':float(np.ptp(y)/max(abs(np.mean(y)),1e-12))})
            # Full 40-view baseline is mandatory; missing frames remain excluded.
            if len(v)==40:
                byview=dict(zip(v,y));base=np.median(y)
                for budget in [4,8,12,20]:
                    for phase in range(40):
                        ix=sorted(set((np.rint(np.arange(budget)*40/budget).astype(int)+phase)%40));yy=np.array([byview[i] for i in ix]);budgets.append({**identity,'metric':metric,'budget':budget,'phase':phase,'median_absolute_error':float(abs(np.median(yy)-base)),'median_relative_error':float(abs(np.median(yy)-base)/max(abs(base),1e-12)),'max_underestimate_fraction':float((max(y)-max(yy))/max(abs(max(y)),1e-12))})
        if gi%50==0:print(json.dumps(dict(stage='descriptors',groups=gi,total=len(groups))),flush=True)
    write(out/'session_observability.csv',descriptors);write(out/'harmonic_descriptors.csv',harmonics);write(out/'view_budget_stability.csv',budgets);write(out/'visible_path_hypotheses.csv',candidates);write(out/'mask_hash_audit.csv',mask_audit);write(out/'architecture_errors.csv',errors)
    pairs=[]
    for key,a in candidate_groups.items():
        treatment,measurement,genotype,replicate,session,camera,view=key
        for later,b in candidate_groups.items():
            if later[0]==treatment and later[2]==genotype and later[3]==replicate and later[5]==camera and later[6]==view and int(later[1])==int(measurement)+1:
                for r in correspondence(a,b):pairs.append(dict(treatment=treatment,genotype=genotype,replicate=replicate,camera=camera,view=view,earlier_visit=measurement,later_visit=later[1],earlier_session=session,later_session=later[4],**r))
    write(out/'candidate_correspondence_ambiguity.csv',pairs)
    hullrows=[];hullstatus=[]
    # Prespecified deterministic small CPU case series: first eligible identity per
    # treatment and genotype; at most args.hull_sessions camera acquisitions.
    eligible=[(k,g) for k,g in sorted(groups.items()) if k[-1]=='rgb' and coverage([r['view'] for r in g if accepted(r)])['strict_eligible']]
    identities=[]
    for k,g in eligible:
        identity=(k[0],k[2],k[3])
        if identity not in identities:identities.append(identity)
    selected=[]
    # Round-robin treatment/genotype cases avoid taking only first treatment.
    chosen={}
    for identity in identities:chosen.setdefault(identity[:2],identity)
    for k,g in eligible:
        if (k[0],k[2],k[3]) in chosen.values():selected.append((k,g))
    selected.sort(key=lambda kg:(int(kg[0][1]),kg[0][0],kg[0][2],kg[0][3]))
    for count,(key,group) in enumerate(selected[:args.hull_sessions],1):
        identity=dict(zip(KEYS,key));g=[r for r in group if accepted(r)];train=[r for r in g if int(r['view'])%4!=2];test=[r for r in g if int(r['view'])%4==2]
        if len(train)<20 or len(test)<5:hullstatus.append({**identity,'status':'insufficient_train_or_heldout'});continue
        try:
            tm={int(r['view']):load_mask(r,maskroot,160) for r in train};hm={int(r['view']):load_mask(r,maskroot,160) for r in test};axes=[];bases=[];heights=[]
            for m in tm.values():
                yy,xx=np.where(m);axes.append(float(np.median(xx[yy>=np.percentile(yy,95)])));bases.append(float(yy.max()));heights.append(float(np.ptp(yy)+1))
            axis=float(np.median(axes));base=float(np.median(bases));scale=float(np.median(heights))
            alternatives=[(shift,roll,0,0) for shift,roll in itertools.product([-.03,0,.03],[-2,0,2])]+[(0,0,dilation,fraction) for dilation,fraction in itertools.product([0,1,2],[0,.1]) if dilation or fraction]
            for shift,roll,dilation,fraction in alternatives:
                ax=axis+shift*scale;hull,grid=visual_hull(tm,ax,base,scale,resolution=args.hull_resolution,roll=roll,dilation_px=dilation,allowed_violation_fraction=fraction);boundary=hull[0].sum()+hull[-1].sum()+hull[:,:,0].sum()+hull[:,:,-1].sum()+hull[:,-1,:].sum()
                for view,observed in hm.items():
                    pred=project_hull(hull,grid,view,observed.shape,ax,base,scale,roll);inter=(pred&observed).sum();union=(pred|observed).sum()
                    hullrows.append({**identity,'axis_shift_height_fraction':shift,'roll_deg':roll,'carve_dilation_px_at160':dilation,'allowed_train_view_violation_fraction':fraction,'view':view,'split':'heldout_view_mod4_eq2','iou':float(inter/max(union,1)),'recall':float(inter/max(observed.sum(),1)),'precision':float(inter/max(pred.sum(),1)),'occupied_voxels':int(hull.sum()),'boundary_occupied_count':int(boundary),'gauge_axis_train_px':axis,'gauge_base_train_px':base,'gauge_height_train_px':scale,'geometry':'orthographic_visual_hull_envelope_not_leaf_geometry'})
                    if shift==0 and roll==0 and view==min(hm) and ((dilation==0 and fraction==0) or (dilation==1 and fraction==.1)):
                        image=np.zeros((*observed.shape,3),np.uint8);image[observed]=[230,230,230];image[pred&observed]=[60,190,140];image[pred&~observed]=[240,100,90];Image.fromarray(image).resize((640,4*image.shape[0])).save(out/f'hull_{count:02d}_heldout_dilation{dilation}_violations{fraction}.png')
            hullstatus.append({**identity,'status':'executed','train_views':len(tm),'heldout_views':len(hm),'alternatives':len(alternatives)})
        except Exception as e:hullstatus.append({**identity,'status':'failed','error':str(e)})
        print(json.dumps(dict(stage='conditional_hull',completed=count,total=min(args.hull_sessions,len(selected)))),flush=True)
    write(out/'conditional_hull_heldout.csv',hullrows);write(out/'conditional_hull_cases.csv',hullstatus)
    summary=dict(mode=mode,input_table=str(source),input_sha256=sha(source),script_sha256=execution_sha,source_changed_during_run=sha(__file__)!=execution_sha,input_rows=len(rows),accepted_input_rows=sum(accepted(r) for r in rows),camera_sessions=len(groups),primary_eligible_camera_sessions=sum(r['eligible'] for r in descriptors),strict_eligible_camera_sessions=sum(r['strict_eligible'] for r in descriptors),harmonic_fits=sum(r['status']=='fit' for r in harmonics),view_budget_rows=len(budgets),visible_path_hypotheses=len(candidates),candidate_pair_rows=len(pairs),selected_correspondences=0,conditional_hull_cases=sum(r['status']=='executed' for r in hullstatus),conditional_hull_heldout_rows=len(hullrows),errors=len(errors),elapsed_seconds=time.time()-start,limitations=['Uniform nominal rotation, camera orthography and basal-axis estimates are conditional assumptions.','Held-out silhouette agreement measures predictive compatibility, not 3D anatomical correctness.','No metric scale, anatomical leaf count, physical continuity, true age, per-day growth or physiological absorption inferred.','Geometry grid ranges are sensitivity alternatives, not confidence intervals.','Candidate path compatibility weights are heuristic and include an unmatched state; no anatomical identity selected.','View budgets require all forty masks accepted; this is a selected subset.'])
    dump(out/'architecture_summary.json',summary)
    with (out/'execution.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**summary))+'\n')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--intake',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--output-name',default='');p.add_argument('--input-table',type=Path);p.add_argument('--records-snapshot',action='store_true');p.add_argument('--hull-sessions',type=int,default=18);p.add_argument('--hull-resolution',type=int,default=64);run(p.parse_args())
