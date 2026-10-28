"""Group-disjoint harvest prediction using only observations available at a cutoff."""
import numpy as np


def cutoff_columns(columns, cutoff, first_age, history_fields):
    return [c for c in columns if cutoff!=first_age or c not in history_fields]


def partitions(genotypes, treatments, regime):
    g=np.asarray(genotypes);t=np.asarray(treatments)
    if len(g)!=len(t):raise ValueError('Group arrays differ')
    folds=[]
    if regime not in ['genotype','treatment','joint']:raise ValueError('Unknown validation regime')
    pairs=[(x,None) for x in sorted(set(g))] if regime=='genotype' else ([(None,y) for y in sorted(set(t))] if regime=='treatment' else [(x,y) for x in sorted(set(g)) for y in sorted(set(t))])
    for a,b in pairs:
        test=(g==a) if b is None else ((t==b) if a is None else ((g==a)&(t==b)))
        train=(g!=a) if b is None else ((t!=b) if a is None else ((g!=a)&(t!=b)))
        if not test.any():continue
        if not train.any():raise ValueError('Insufficient groups for a disjoint training set')
        folds.append((np.flatnonzero(train),np.flatnonzero(test),str(a or '')+'|'+str(b or '')))
    return folds


def trajectory(days, values, cutoff):
    days=np.asarray(days,float);values=np.asarray(values,float)
    if len(days)!=len(values) or len(days)==0 or np.any(np.diff(days)<=0):raise ValueError('Unique increasing scheduled ages required')
    current=np.flatnonzero(days==cutoff)
    if len(current)!=1:raise ValueError('Cutoff must be a declared observation age')
    ratio=values[current[0]]/values[0] if np.isfinite(values[0]) and values[0]>0 and np.isfinite(values[current[0]]) else np.nan
    usable=(days<=cutoff)&np.isfinite(values)
    x=days[usable];y=values[usable]
    slope=float(np.dot(x-x.mean(),y-y.mean())/np.dot(x-x.mean(),x-x.mean())) if len(x)>=2 else np.nan
    return dict(ratio_to_first=float(ratio),slope_per_day=slope,observed_past_visits=int(usable.sum()))


def train_design(rows, train, test, numeric, categorical=()):
    x=np.array([[float(rows[i].get(c,np.nan)) for c in numeric] for i in train],float).reshape(len(train),len(numeric))
    z=np.array([[float(rows[i].get(c,np.nan)) for c in numeric] for i in test],float).reshape(len(test),len(numeric))
    medians=np.array([np.nanmedian(x[:,j]) if np.isfinite(x[:,j]).any() else 0. for j in range(len(numeric))])
    indicators=np.flatnonzero(~np.isfinite(x).all(axis=0))
    xmissing=~np.isfinite(x);zmissing=~np.isfinite(z)
    x=np.where(xmissing,medians,x);z=np.where(zmissing,medians,z)
    x=np.hstack([x,xmissing[:,indicators]]);z=np.hstack([z,zmissing[:,indicators]])
    mean=x.mean(axis=0);scale=x.std(axis=0);scale[scale==0]=1.
    x=(x-mean)/scale;z=(z-mean)/scale
    categories={}
    for c in categorical:
        levels=sorted({str(rows[i][c]) for i in train});categories[c]=levels
        x=np.hstack([x,np.array([[str(rows[i][c])==level for level in levels] for i in train],float)])
        z=np.hstack([z,np.array([[str(rows[i][c])==level for level in levels] for i in test],float)])
    return x,z,dict(medians=medians.tolist(),indicator_columns=indicators.tolist(),means=mean.tolist(),scales=scale.tolist(),categories=categories)


def ridge_predict(x, z, y, alpha):
    mean=x.mean(axis=0);centered=x-mean;ymean=y.mean(axis=0)
    if x.shape[1]==0:return np.broadcast_to(ymean,(len(z),*np.asarray(ymean).shape)).copy()
    if alpha==0:beta=np.linalg.lstsq(centered,y-ymean,rcond=None)[0]
    elif alpha>0:beta=np.linalg.solve(centered.T@centered+alpha*np.eye(x.shape[1]),centered.T@(y-ymean))
    else:raise ValueError('Alpha must be nonnegative')
    return (z-mean)@beta+ymean
