"""Leave-genotype-out assay baselines using recorded treatment/visit only.

No image features, test-target fitting, tuning or physiological causal claims.
All transformations reduce to training-fold group means.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def save(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--intake',type=Path,required=True)
    root=p.parse_args().intake;dest=root/'metadata_baselines'
    if dest.exists():raise ValueError('Refusing to overwrite a baseline analysis')
    dest.mkdir()
    source=root/'research_audit/reference_master.csv'
    with source.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    targets=['ojip__Fv/Fm','ojip__Pi_Abs','nir__NDVI']
    groups=sorted({r['genotype'] for r in rows})
    if len(groups)!=6 or len(rows)!=360:raise ValueError('Unexpected reference cohort')
    modes={'training_mean':(), 'visit_only':('measurement',),
           'treatment_only':('treatment',),
           'treatment_visit':('treatment','measurement')}
    y=np.array([[float(r[t]) for t in targets] for r in rows])
    if not np.isfinite(y).all():raise ValueError('Nonfinite assay targets')
    predictions=[];folds=[];metrics=[]
    for model,fields in modes.items():
        pred=np.full_like(y,np.nan)
        for held_out in groups:
            train=[i for i,r in enumerate(rows) if r['genotype']!=held_out]
            test=[i for i,r in enumerate(rows) if r['genotype']==held_out]
            assert len(train)==300 and len(test)==60
            assert not ({rows[i]['genotype'] for i in train}&{rows[i]['genotype'] for i in test})
            buckets=defaultdict(list)
            for i in train:buckets[tuple(rows[i][f] for f in fields)].append(i)
            train_means={k:y[v].mean(axis=0) for k,v in buckets.items()}
            for i in test:
                key=tuple(rows[i][f] for f in fields)
                if key not in train_means:raise ValueError('No training support for test metadata')
                pred[i]=train_means[key]
                # Independent direct mean pins the group selection for every prediction.
                explicit=[j for j in train if all(rows[j][f]==rows[i][f] for f in fields)]
                assert np.allclose(pred[i],np.array([y[j] for j in explicit]).mean(axis=0),rtol=0,atol=1e-12)
                for k,t in enumerate(targets):
                    predictions.append(dict(treatment=rows[i]['treatment'],visit=rows[i]['measurement'],
                                            genotype=held_out,replicate=rows[i]['replicate'],
                                            model=model,target=t,observed=float(y[i,k]),predicted=float(pred[i,k]),
                                            training_group_support=len(explicit)))
            folds.append(dict(model=model,held_out_genotype=held_out,training_records=len(train),
                              test_records=len(test),genotype_overlap=False))
        assert np.isfinite(pred).all()
        for k,t in enumerate(targets):
            error=pred[:,k]-y[:,k];denom=float(np.sum((y[:,k]-y[:,k].mean())**2))
            metrics.append(dict(model=model,target=t,records=len(rows),held_out_groups=len(groups),
                                rmse=float(np.sqrt(np.mean(error**2))),mae=float(np.mean(np.abs(error))),
                                r2=float(1-np.sum(error**2)/denom)))
    save(dest/'predictions.csv',predictions);save(dest/'metrics.csv',metrics);save(dest/'folds.csv',folds)
    summary=dict(status='completed',records=len(rows),targets=targets,models=list(modes),
                 outer_partition='leave one genotype out; six folds',images_used=False,
                 reference_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 limitations=['All assays are from one experiment; no external validation.',
                              'Treatment/visit labels are available at prediction time; this is not forecasting.',
                              'Repeated views are not independent samples.',
                              'These controls do not show that imagery predicts physiology.'])
    (dest/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (dest/'execution_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':main()
