"""External correspondence benchmark CONDITIONAL ON supplied instance segmentation.

All seven maize plants, last two labelled scans, both annotation conventions.
Temporal instance IDs partition supplied organs but are replaced by independently
permuted anonymous tokens before feature computation/matching. IDs are reopened
only for scoring. This validates association of supplied segmented 3D organs;
it does not validate CSI silhouette reconstruction or end-to-end leaf tracking.
"""
import argparse,hashlib,itertools,json,math,time
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from architecture_analysis import write,dump,sha

def associate(cost,gap=.75):
    n,m=cost.shape;aug=np.full((n+m,n+m),1e6);aug[:n,:m]=cost
    for i in range(n):aug[i,m+i]=gap
    for j in range(m):aug[n+j,j]=gap
    aug[n:,m:]=0;left,right=linear_sum_assignment(aug);links=[(int(i),int(j)) for i,j in zip(left,right) if i<n and j<m]
    return links,[i for i in range(n) if not any(i==q[0] for q in links)],[j for j in range(m) if not any(j==q[1] for q in links)]

def sequence_associate(cost,gap=.75):
    n,m=cost.shape;dp=np.full((n+1,m+1),np.inf);dp[:,0]=np.arange(n+1)*gap;dp[0,:]=np.arange(m+1)*gap;step=np.zeros((n+1,m+1),int)
    for i in range(1,n+1):
        for j in range(1,m+1):
            options=[dp[i-1,j-1]+cost[i-1,j-1],dp[i-1,j]+gap,dp[i,j-1]+gap];k=int(np.argmin(options));dp[i,j]=options[k];step[i,j]=k
    i,j=n,m;links=[];a=[];b=[]
    while i or j:
        if not i:b.append(j-1);j-=1
        elif not j:a.append(i-1);i-=1
        elif step[i,j]==0:links.append((i-1,j-1));i-=1;j-=1
        elif step[i,j]==1:a.append(i-1);i-=1
        else:b.append(j-1);j-=1
    return links[::-1],a[::-1],b[::-1]

def feature_cost(a,b):return np.sqrt(np.mean((a[:,None,:]-b[None,:,:])**2,axis=2))/.2

def features(points):
    center=np.median(points,axis=0);q=np.percentile(points[:,2],[10,50,90]);radial=np.linalg.norm(points[:,:2],axis=1);dist=np.linalg.norm(points,axis=1);tip=np.median(points[dist>=np.percentile(dist,98)],axis=0);eigen=np.maximum(np.linalg.eigvalsh(np.cov(points.T)),0)
    return np.r_[center,q,tip,np.percentile(radial,95),np.sqrt(eigen[-2:])]

def normalized_pair(first,later):
    # Original files are in a shared scanner reference frame. Native Z is
    # vertical (also confirmed by flat soil in each loaded scan); the published
    # loader rotates native Z to display Y by -90deg about X. Use the earlier
    # basal foreground center and height as the shared scale, without IDs.
    plant=first[first[:,4]>0,:3];bottom=np.percentile(plant[:,2],2);basal=plant[plant[:,2]<=np.percentile(plant[:,2],8)];origin=np.r_[np.median(basal[:,:2],axis=0),bottom];height=float(np.percentile(plant[:,2],99.5)-bottom)
    return (first[:,:3]-origin)/height,(later[:,:3]-origin)/height,origin,height

def components(array,normalized,column,seed):
    labels=array[:,column].astype(int);minimum=2 if column==3 else 1;ids=[int(i) for i in np.unique(labels) if i>=minimum];rng=np.random.default_rng(seed);ids=[ids[i] for i in rng.permutation(len(ids))];output=[];truth=[];pointsizes=[]
    for instance in ids:
        p=normalized[labels==instance];pointsizes.append(len(p))
        if len(p)>20000:p=p[rng.choice(len(p),20000,replace=False)]
        output.append(features(p));truth.append(instance)
    return np.stack(output),truth,pointsizes

def score(links,a,b,truth_a,truth_b):
    expected=set(truth_a)&set(truth_b);correct=sum(truth_a[i]==truth_b[j] for i,j in links);ua={truth_a[i] for i in a};ub={truth_b[j] for j in b};true_a=set(truth_a)-set(truth_b);true_b=set(truth_b)-set(truth_a)
    return dict(true_common_instances=len(expected),predicted_links=len(links),correct_links=correct,incorrect_links=len(links)-correct,link_precision=correct/max(len(links),1),link_recall=correct/max(len(expected),1),earlier_unmatched_truth=len(true_a),later_unmatched_truth=len(true_b),earlier_unmatched_predicted=len(ua),later_unmatched_predicted=len(ub),correct_earlier_unmatched=len(ua&true_a),correct_later_unmatched=len(ub&true_b))

def run(args):
    start=time.time();out=args.run/'architecture'/'external_pheno4d';out.mkdir(parents=True,exist_ok=True);manifest=json.loads((args.external/'download_manifest.json').read_text(encoding='utf-8'))
    if manifest['status']!='completed':raise ValueError('External subset download must finish before scoring')
    files=manifest['files'];by={};inputs=[];results=[];predictions=[];features_rows=[];axis_audit=[];all_features={};source_snapshot=Path(__file__).read_bytes();(out/'execution_source.py').write_bytes(source_snapshot)
    for row in files:
        p=args.external/row['path']
        if sha(p)!=row['sha256']:raise ValueError('External source hash mismatch: '+str(p))
        by.setdefault(p.parent.name,[]).append((p,row));inputs.append(dict(path=row['path'],sha256=row['sha256'],bytes=p.stat().st_size))
    if len(by)!=7 or any(len(v)!=2 for v in by.values()):raise ValueError('Expected all seven plants, exactly two scans each')
    for pi,(plant,pair) in enumerate(sorted(by.items())):
        pair.sort();first=np.loadtxt(pair[0][0],dtype='float32');later=np.loadtxt(pair[1][0],dtype='float32')
        if first.shape[1]!=5 or later.shape[1]!=5:raise ValueError('Expected xyz/collar/tip columns')
        na,nb,origin,height=normalized_pair(first,later)
        for day,cloud in [('0324',first),('0325',later)]:
            soil=cloud[cloud[:,4]==0,:3];axis_audit.append(dict(plant=plant,day=day,points=len(cloud),soil_points=len(soil),soil_xyz_extent=np.ptp(soil,axis=0).tolist(),foreground_height_earlier_gauge=height,origin=origin.tolist()))
        for column,scheme in [(3,'leaf_collar'),(4,'leaf_tip')]:
            fa,ta,sa=components(first,na,column,2100+pi*10);fb,tb,sb=components(later,nb,column,2101+pi*10);all_features[(plant,scheme)]=(fa,fb)
            for date,f,truth,sizes in [('0324',fa,ta,sa),('0325',fb,tb,sb)]:
                for token,(values,label,size) in enumerate(zip(f,truth,sizes)):features_rows.append(dict(plant=plant,scheme=scheme,date=date,anonymous_token=token,points=size,**{f'feature_{i}':float(v) for i,v in enumerate(values)}))
            # Missing-instance controls are chosen by anonymous token, before
            # scoring. They simulate absent segmentation/visibility, not death.
            for scenario in ['original','later_instance_hidden','earlier_instance_hidden']:
                ai=list(range(len(fa)));bi=list(range(len(fb)))
                if scenario=='later_instance_hidden' and bi:bi=bi[1:]
                if scenario=='earlier_instance_hidden' and ai:ai=ai[1:]
                aa=fa[ai];bb=fb[bi];truth_a=[ta[i] for i in ai];truth_b=[tb[i] for i in bi]
                for method in ['centroid_hungarian','geometry_hungarian','geometry_sequence']:
                    cost=np.linalg.norm(aa[:,None,:3]-bb[None,:,:3],axis=2)/.2 if method=='centroid_hungarian' else feature_cost(aa,bb)
                    if method=='geometry_sequence':
                        ia=np.argsort(aa[:,2]);ib=np.argsort(bb[:,2]);ll,ua,ub=sequence_associate(cost[np.ix_(ia,ib)],args.gap);links=[(int(ia[i]),int(ib[j])) for i,j in ll];un_a=[int(ia[i]) for i in ua];un_b=[int(ib[j]) for j in ub]
                    else:links,un_a,un_b=associate(cost,args.gap)
                    metrics=score(links,un_a,un_b,truth_a,truth_b);base=dict(plant=plant,scheme=scheme,scenario=scenario,method=method,earlier_instances=len(aa),later_instances=len(bb),gap_penalty=args.gap);results.append({**base,**metrics})
                    for i,j in links:predictions.append({**base,'earlier_anonymous_token':ai[i],'later_anonymous_token':bi[j],'earlier_truth_for_scoring':truth_a[i],'later_truth_for_scoring':truth_b[j],'correct':truth_a[i]==truth_b[j],'geometry_cost':float(cost[i,j])})
        print(json.dumps(dict(stage='external_correspondence',plant=plant,completed=pi+1,total=len(by))),flush=True)
    write(out/'conditional_correspondence_scores.csv',results);write(out/'predicted_links_and_scoring_truth.csv',predictions);write(out/'anonymous_geometry_features.csv',features_rows);dump(out/'coordinate_and_label_audit.json',axis_audit)
    aggregate=[]
    for scheme,scenario,method in itertools.product(['leaf_collar','leaf_tip'],['original','later_instance_hidden','earlier_instance_hidden'],['centroid_hungarian','geometry_hungarian','geometry_sequence']):
        rr=[r for r in results if (r['scheme'],r['scenario'],r['method'])==(scheme,scenario,method)];correct=sum(r['correct_links'] for r in rr);predicted=sum(r['predicted_links'] for r in rr);common=sum(r['true_common_instances'] for r in rr);aggregate.append(dict(scheme=scheme,scenario=scenario,method=method,plant_pairs=len(rr),correct_links=correct,predicted_links=predicted,true_common_instances=common,micro_precision=correct/max(predicted,1),micro_recall=correct/max(common,1),macro_plant_precision=float(np.mean([r['link_precision'] for r in rr])),macro_plant_recall=float(np.mean([r['link_recall'] for r in rr])),true_unmatched=sum(r['earlier_unmatched_truth']+r['later_unmatched_truth'] for r in rr),correct_unmatched=sum(r['correct_earlier_unmatched']+r['correct_later_unmatched'] for r in rr)))
    write(out/'conditional_correspondence_summary.csv',aggregate)
    summary=dict(dataset='Pheno4D original official files',source='https://www.ipb.uni-bonn.de/data/pheno4d/index.html',paper='https://doi.org/10.1371/journal.pone.0256340',plants=len(by),scans=len(files),selection='Last two labelled maize dates, selected before results; all seven plants retained.',input_manifest_sha256=sha(args.external/'download_manifest.json'),inputs=inputs,script_sha256=hashlib.sha256(source_snapshot).hexdigest(),gap_penalty=args.gap,features='Median xyz, z10/50/90, median most distal2% xyz, radial95%, two PCA scales. Original registered coordinates; earlier basal center/height gauge. No ID/rank/point-count features. <=20000 points per supplied instance sampled with fixed seed.',condition='Supplied ground-truth instance segmentation and plant continuity, registered3D coordinates. Temporal numeric IDs hidden during matching and used only to score. Collar excludes soil0 and stem1; tip excludes soil0. No CSI silhouette/reconstruction/end-to-end validation.',scores=aggregate,elapsed_seconds=time.time()-start,redistribution='Original external data remain separate and are not bundled in own public dataset. Article license is not assumed to establish dataset terms.')
    dump(out/'external_correspondence_summary.json',summary);print(json.dumps(dict(plants=len(by),scans=len(files),scores=[r for r in aggregate if r['scenario']=='original'],elapsed_seconds=time.time()-start),indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--external',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--gap',type=float,default=.75);run(p.parse_args())
