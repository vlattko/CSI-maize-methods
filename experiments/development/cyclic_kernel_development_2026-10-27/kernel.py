"""Camera-shared scaling and a finite rotation/reflection averaged RBF kernel."""
from collections import defaultdict
import numpy as np


def permutations():
    return [np.array([(sign*v+phase)%4 for v in range(4)]) for sign in (1,-1) for phase in range(4)]


class ViewScaler:
    def fit(self,x):
        x=np.asarray(x,float)
        self.mean=np.nanmean(x,axis=(0,2),keepdims=True)
        self.scale=np.nanstd(x,axis=(0,2),keepdims=True)
        self.mean=np.nan_to_num(self.mean)
        self.scale=np.where(np.isfinite(self.scale)&(self.scale>1e-6),self.scale,1.)
        return self

    def transform(self,x):
        return np.nan_to_num((np.asarray(x,float)-self.mean)/self.scale)


def _raw_kernel(x,y,mode,gamma):
    if mode=='mean':
        a=x.mean(axis=2).reshape(len(x),-1);b=y.mean(axis=2).reshape(len(y),-1)
        distance=np.maximum(np.sum(a*a,1)[:,None]+np.sum(b*b,1)[None,:]-2*a@b.T,0.)/a.shape[1]
        return np.exp(-gamma*distance)
    a=x.reshape(len(x),-1);aa=np.sum(a*a,1)[:,None];result=np.zeros((len(x),len(y)))
    for order in permutations() if mode=='cyclic' else [np.arange(4)]:
        b=y[:,:,order,:].reshape(len(y),-1)
        distance=np.maximum(aa+np.sum(b*b,1)[None,:]-2*a@b.T,0.)/a.shape[1]
        result+=np.exp(-gamma*distance)
    return result/(8 if mode=='cyclic' else 1)


def _diagonal(x,mode,gamma):
    if mode!='cyclic':return np.ones(len(x))
    return sum(np.exp(-gamma*np.mean((x-x[:,:,order,:])**2,axis=(1,2,3))) for order in permutations())/8


def image_kernel(x,y,mode,gamma):
    if mode not in ('mean','ordered','cyclic'):raise ValueError('Unknown view kernel')
    kernel=_raw_kernel(x,y,mode,gamma)
    return kernel/np.sqrt(_diagonal(x,mode,gamma)[:,None]*_diagonal(y,mode,gamma)[None,:])


def trajectory_permutation(rows,indices,seed):
    rng=np.random.default_rng(seed);plants=defaultdict(dict)
    for i in indices:
        row=rows[i];plants[(row['treatment'],row['genotype'],row['replicate'])][row['measurement']]=int(i)
    strata=defaultdict(list)
    for plant,visits in plants.items():strata[(plant[0],tuple(sorted(visits)))].append(plant)
    donor={}
    for members in strata.values():
        shuffled=[members[j] for j in rng.permutation(len(members))]
        for recipient,source in zip(members,shuffled):
            for visit,i in plants[recipient].items():donor[i]=plants[source][visit]
    return np.array([donor[int(i)] for i in indices])
