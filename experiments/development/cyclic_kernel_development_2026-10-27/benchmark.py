"""Grouped retrospective model comparison retaining circular view structure.

All scaling and hyperparameters are learned within training folds. Frozen DINO
views are observations, not biological replicates. The cyclic kernel averages
over four quarter shifts and both direction signs; it is not arbitrary-angle
invariance. This study reuses the already explored cohort, not an external test.
"""
import argparse,csv,hashlib,json,os,time
from pathlib import Path
from itertools import product
import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits
from kernel import ViewScaler,image_kernel,trajectory_permutation

FIELDS=('treatment','measurement','genotype','replicate')
TARGETS=('ojip__Fv/Fm','ojip__Pi_Abs','nir__NDVI')
GAMMAS=(.2,1.,5.)
ALPHAS=(.01,.1,1.,10.)
WEIGHTS=(0.,.5,1.)


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def key(row):return tuple(str(row[k]) for k in FIELDS)


def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)


def group_marker(root,label):
    """Keep arbitrary scientific labels out of filesystem path components."""
    return root/('group_'+hashlib.sha256(str(label).encode('utf-8')).hexdigest()+'.json')


def load_fold(marker,signature,test):
    saved=json.loads(marker.read_text());array=marker.with_suffix('.npz')
    if saved['signature']!=signature:raise ValueError('Changed fold configuration')
    if sha(array)!=saved['npz_sha256']:raise ValueError('Changed prediction digest')
    with np.load(array,allow_pickle=False) as data:
        pred=data['prediction']
        if not np.array_equal(data['test_indices'],test):raise ValueError('Changed held-out indices')
    if pred.shape!=(len(test),len(TARGETS)) or not np.isfinite(pred).all():raise ValueError('Invalid cached prediction')
    return pred,saved['parameters']


def load_inputs(intake,run,cohort):
    reference={key(r):r for r in read(intake/'research_audit/reference_master.csv')}
    sessions=read(run/'extraction/session_traits.csv')
    rows=[dict(reference[key(r)],**r) for r in sessions if r['paired_eligible']=='True']
    if len(rows)!=329:raise ValueError('Expected329primary paired acquisitions')
    lookup={key(r):i for i,r in enumerate(rows)};x=np.full((len(rows),2,4,384),np.nan);hashes={}
    signature=json.loads((run/'prediction/encoder/operator.json').read_text())['signature']
    for path in sorted((run/'prediction/encoder/views').glob('*.npz')):
        with np.load(path,allow_pickle=False) as data:meta=json.loads(str(data['metadata']));feature=data['feature'].astype(float)
        if meta['variant']!='recorded' or key(meta) not in lookup:continue
        if meta['operator_signature']!=signature:raise ValueError('Incompatible encoder')
        record=run/'extraction/records'/(hashlib.sha256(meta['source'].encode()).hexdigest()[:24]+'.json')
        if sha(record)!=meta['record_sha256']:raise ValueError('Changed encoder source record')
        camera=('rgb','noir').index(meta['camera']);view=int(meta['view'])//10;index=lookup[key(meta)]
        if np.isfinite(x[index,camera,view]).any():raise ValueError('Duplicate camera/view feature')
        if feature.shape!=(384,) or not np.isfinite(feature).all():raise ValueError('Invalid feature')
        x[index,camera,view]=feature;hashes[path.name]=sha(path)
    complete=np.isfinite(x).all(axis=(1,2,3))
    if cohort=='complete':
        rows=[r for i,r in enumerate(rows) if complete[i]];x=x[complete]
    y=np.array([[float(r[t]) for t in TARGETS] for r in rows])
    return rows,x,y,dict(encoder_signature=signature,view_inputs_sha256=hashes,complete_quarter_rows=int(complete.sum()))


def metadata(rows,mode):
    columns=[(None,str(v)) for v in range(1,6)] if mode=='treatment' else [(t,str(v)) for t in 'CPS' for v in range(1,6)]
    return np.array([[float(r['measurement']==v and (t is None or r['treatment']==t)) for t,v in columns] for r in rows])


def solve_predict(kernel,cross,y,alpha):
    values,vectors=eigh((kernel+kernel.T)/2,check_finite=True)
    if values.min() < -1e-7:raise ValueError('Kernel is not positive semidefinite')
    mean=y.mean(axis=0);coef=vectors@((vectors.T@(y-mean))/(np.maximum(values,0)[:,None]+alpha))
    return cross@coef+mean


def matrices(rows,x,m,train,test,family,gamma,null=False,seed=20261027):
    scaler=ViewScaler().fit(x[train]);a=scaler.transform(x[train]);b=scaler.transform(x[test])
    kt=image_kernel(a,a,family,gamma);kv=image_kernel(b,a,family,gamma)
    if null:
        donor=trajectory_permutation(rows,train,seed)
        positions={int(i):j for j,i in enumerate(train)};order=[positions[int(i)] for i in donor]
        kt=kt[np.ix_(order,order)];kv=kv[:,order]
    return kt,kv,m[train]@m[train].T,m[test]@m[train].T


def fit_fold(rows,x,y,meta,train,test,family,null=False):
    groups=sorted({rows[i]['genotype'] for i in train});choices=list(product(GAMMAS,ALPHAS,WEIGHTS));scores=np.zeros((len(choices),len(TARGETS)))
    for held in groups:
        inside=np.array([i for i in train if rows[i]['genotype']!=held]);valid=np.array([i for i in train if rows[i]['genotype']==held])
        if not len(inside) or not len(valid):raise ValueError('Empty grouped inner fold')
        cache={g:matrices(rows,x,meta,inside,valid,family,g,null) for g in GAMMAS}
        for j,(gamma,alpha,weight) in enumerate(choices):
            k,c,km,cm=cache[gamma];pred=solve_predict((1-weight)*k+weight*km,(1-weight)*c+weight*cm,y[inside],alpha)
            scores[j]+=np.mean((pred-y[valid])**2,axis=0)/len(groups)
    selected=np.argmin(scores,axis=0);prediction=np.empty((len(test),len(TARGETS)));parameters=[];cache={}
    for target,index in enumerate(selected):
        gamma,alpha,weight=choices[int(index)]
        if gamma not in cache:cache[gamma]=matrices(rows,x,meta,train,test,family,gamma,null)
        k,c,km,cm=cache[gamma];pred=solve_predict((1-weight)*k+weight*km,(1-weight)*c+weight*cm,y[train],alpha)
        prediction[:,target]=pred[:,target]
        parameters.append(dict(target=TARGETS[target],gamma=gamma,alpha=alpha,metadata_weight=weight,inner_group_mse=float(scores[index,target])))
    return prediction,parameters


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('intake','run','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--cohort',choices=['primary','complete'],default='primary')
    args=parser.parse_args();root=args.output/args.cohort;root.mkdir(parents=True,exist_ok=True)
    rows,x,y,provenance=load_inputs(args.intake,args.run,args.cohort)
    config=dict(cohort=args.cohort,n=len(rows),targets=TARGETS,source_sha256=sha(__file__),kernel_sha256=sha(Path(__file__).with_name('kernel.py')),
        reference_sha256=sha(args.intake/'research_audit/reference_master.csv'),session_sha256=sha(args.run/'extraction/session_traits.csv'),**provenance,
        gamma=GAMMAS,alpha=ALPHAS,metadata_weights=WEIGHTS,inner_group='genotype',null_seed=20261027,
        scope='Retrospective grouped model development; prior cohort results have been inspected. No external generalization or calibrated spectra inferred.')
    signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest();config['signature']=signature
    if (root/'configuration.json').exists() and json.loads((root/'configuration.json').read_text())!=json.loads(json.dumps(config)):raise ValueError('Changed configuration')
    save(root/'configuration.json',config);(root/'benchmark_execution_source.py').write_bytes(Path(__file__).read_bytes());(root/'kernel_execution_source.py').write_bytes(Path(__file__).with_name('kernel.py').read_bytes())
    summaries=[];start=time.time()
    for mode in ('genotype','treatment'):
        meta=metadata(rows,mode)
        for family,null in [('mean',False),('ordered',False),('cyclic',False),('cyclic',True)]:
            name=mode+'__'+family+('_null' if null else '');dest=root/name;dest.mkdir(exist_ok=True)
            predictions=np.full_like(y,np.nan);parameters=[]
            for held in sorted({r[mode] for r in rows}):
                train=np.array([i for i,r in enumerate(rows) if r[mode]!=held]);test=np.array([i for i,r in enumerate(rows) if r[mode]==held])
                marker=group_marker(dest,held);array=marker.with_suffix('.npz')
                if marker.exists():
                    pred,choices=load_fold(marker,signature,test)
                else:
                    pred,choices=fit_fold(rows,x,y,meta,train,test,family,null)
                    np.savez_compressed(array,prediction=pred,test_indices=test)
                    save(marker,dict(signature=signature,parameters=choices,heldout_group=held,npz_sha256=sha(array)))
                predictions[test]=pred;parameters.extend([dict(heldout_group=held,**r) for r in choices])
                progress=dict(status='running',experiment=name,heldout_group=held,elapsed_seconds=round(time.time()-start,1));save(root/'progress.json',progress);print(json.dumps(progress),flush=True)
            if not np.isfinite(predictions).all():raise ValueError('Missing OOF predictions')
            metrics=[]
            for j,target in enumerate(TARGETS):
                error=predictions[:,j]-y[:,j];denominator=np.sum((y[:,j]-y[:,j].mean())**2)
                metrics.append(dict(mode=mode,family=family,null=null,target=target,n=len(rows),rmse=float(np.sqrt(np.mean(error**2))),r2=float(1-np.sum(error**2)/denominator)))
            np.savez_compressed(dest/'oof.npz',observed=y,prediction=predictions,keys=np.array([key(r) for r in rows]))
            save(dest/'summary.json',dict(status='completed',metrics=metrics,parameters=parameters,oof_sha256=sha(dest/'oof.npz')));summaries.extend(metrics)
    save(root/'summary.json',dict(status='completed',cohort=args.cohort,n=len(rows),metrics=summaries,elapsed_seconds=time.time()-start,signature=signature))
    with (root/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    save(root/'progress.json',dict(status='completed',experiments=8,target_comparisons=len(summaries),elapsed_seconds=time.time()-start))


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
