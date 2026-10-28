"""Independently replay mass joins, nested tuning and held-out predictions."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from registry import sha


def replay(rows, target, columns, train, test, alpha, shuffled):
    y=np.array([float(r[target]) for r in rows])[train]
    if shuffled:y=np.random.default_rng(20261028+sum(train)).permutation(y)
    if not columns:return np.repeat(y.mean(),len(test))
    xparts=[];zparts=[]
    for column in columns:
        if column in ['genotype','treatment']:
            categories=sorted({rows[i][column] for i in train})
            xparts.append(np.array([[rows[i][column]==c for c in categories] for i in train],float))
            zparts.append(np.array([[rows[i][column]==c for c in categories] for i in test],float))
        else:
            x=np.array([float(rows[i][column]) if rows[i][column] else np.nan for i in train])
            z=np.array([float(rows[i][column]) if rows[i][column] else np.nan for i in test])
            missing=np.isnan(x)
            if missing.any():
                # Indicator columns are unscaled by SimpleImputer, then scaled
                # alongside numerical columns by StandardScaler.
                a=missing.astype(float);b=np.isnan(z).astype(float)
                scale=a.std() or 1.;xparts.append(((a-a.mean())/scale)[:,None]);zparts.append(((b-a.mean())/scale)[:,None])
            median=np.nanmedian(x) if not missing.all() else 0.
            x=np.where(missing,median,x);z=np.where(np.isnan(z),median,z)
            scale=x.std() or 1.;xparts.append(((x-x.mean())/scale)[:,None]);zparts.append(((z-x.mean())/scale)[:,None])
    x=np.hstack(xparts);z=np.hstack(zparts);mean=x.mean(axis=0)
    centered=x-mean
    beta=np.linalg.solve(centered.T@centered+alpha*np.eye(x.shape[1]),centered.T@(y-y.mean()))
    return (z-mean)@beta+y.mean()


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();root=args.output
    manifest=json.loads((root/'mass_validation_manifest.json').read_text())
    for p,digest in manifest['inputs'].items():
        if sha(Path(p))!=digest:raise ValueError('Input changed: '+p)
    for p,digest in manifest['outputs'].items():
        if sha(root/p)!=digest:raise ValueError('Output changed: '+p)
    rows=read(root/'all_pot_mass_image_join.csv');eligible=[r for r in rows if r['benchmark_eligible']=='True']
    if len(rows)!=72 or len({r['plant_uid'] for r in rows})!=72:raise ValueError('Pot coverage error')
    # Re-read original workbook cells, not derived CSV values.
    import openpyxl
    source_mass=next(Path(p) for p in manifest['inputs'] if Path(p).name=='confirmed_mass.csv')
    meta_manifest=json.loads((source_mass.parent/'metadata_manifest.json').read_text())
    original=next(Path(p) for p in meta_manifest['inputs'] if Path(p).name=='MASA_P_6.xlsx')
    sheet=openpyxl.load_workbook(original,data_only=True)['List1']
    for r in rows:
        if float(sheet[r['source_fresh_cell']].value)!=float(r['fresh_mass_g']) or float(sheet[r['source_dry_cell']].value)!=float(r['dry_mass_g']):raise ValueError('Source cell mass mismatch')
        source_row=sheet[r['source_fresh_cell']].row
        if sheet.cell(source_row,2).value!=r['genotype'] or (source_row-2)//6+1!=int(r['replicate']):raise ValueError('Source identity mismatch')
    stored=read(root/'heldout_predictions.csv');pred_lookup={(r['regime'],r['target'],r['model'],r['plant_uid']):float(r['predicted']) for r in stored}
    folds=json.loads((root/'folds.json').read_text());max_error=0.;tuning_error=0.;replayed=0
    for fold in folds:
        field='genotype' if fold['regime']=='unseen_genotype' else 'treatment'
        train=np.array([i for i,r in enumerate(eligible) if r['plant_uid'] in fold['train_uids']])
        test=np.array([i for i,r in enumerate(eligible) if r['plant_uid'] in fold['test_uids']])
        if len(train)+len(test)!=len(eligible) or {eligible[i][field] for i in train}&{eligible[i][field] for i in test}:raise ValueError('Fold partition/leakage')
        cols=manifest['features'][fold['model']];shuffled=fold['model'].startswith('shuffled')
        prediction=replay(eligible,fold['target'],cols,train,test,fold['alpha'],shuffled)
        expected=np.array([pred_lookup[(fold['regime'],fold['target'],fold['model'],eligible[i]['plant_uid'])] for i in test])
        max_error=max(max_error,float(np.max(abs(prediction-expected))))
        if cols:
            y=np.array([float(eligible[i][fold['target']]) for i in train])
            if shuffled:y=np.random.default_rng(20261028+sum(train)).permutation(y)
            # Inner null values are drawn once on the outer train set, then
            # partitioned. They must not be re-permuted within inner folds.
            inner_rows=[dict(eligible[i],**{fold['target']:str(v)}) for i,v in zip(train,y)]
            scores=[]
            for alpha in manifest['alpha_grid']:
                errors=[]
                for group in sorted({r[field] for r in inner_rows}):
                    itrain=np.array([i for i,r in enumerate(inner_rows) if r[field]!=group]);itest=np.array([i for i,r in enumerate(inner_rows) if r[field]==group])
                    pred=replay(inner_rows,fold['target'],cols,itrain,itest,alpha,False)
                    errors.extend((pred-y[itest])**2)
                scores.append(float(np.mean(errors)))
            tuning_error=max(tuning_error,float(np.max(abs(np.array(scores)-fold['inner_mse']))))
            if manifest['alpha_grid'][int(np.argmin(scores))]!=fold['alpha']:raise ValueError('Selected alpha differs under independent replay')
        replayed+=1
    if max_error>1e-9 or tuning_error>1e-9:raise ValueError('Independent numerical replay failed')
    result=dict(status='passed',source_cell_checks=144,pots=len(rows),eligible_pots=len(eligible),outer_fits=replayed,
                predictions=len(stored),max_prediction_replay_error=max_error,max_inner_mse_replay_error=tuning_error,
                manifest_sha256=sha(root/'mass_validation_manifest.json'),verifier_sha256=sha(Path(__file__)))
    (root/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
