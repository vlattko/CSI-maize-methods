"""Rerun frozen index prediction protocol with independently audited strict inputs."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,csv,hashlib,importlib,json,shutil,sys,time
from pathlib import Path
import numpy as np
import strict_features as sf


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');temporary.replace(path)
def csv_rows(path,nominal=False):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            if not nominal or row['pair_kind']=='nominal' and row['method']=='ECC':yield row
def safe_output(output,inputs):
    output=output.resolve()
    for root in inputs:
        root=root.resolve()
        if output.is_relative_to(root) or root.is_relative_to(output):raise ValueError('Output must be separate from all input roots')


def load_operators(args):
    old=args.run/'prediction';pair=args.run/'pairing/full';config=read_json(old/'provenance_indices.json')
    definitions=[('prediction.py',old/'execution_source_images.py',config['hashes']['prediction_operator']),
        ('prediction_indices.py',old/'execution_source_indices.py',config['hashes']['source']),
        ('pairing_aggregate.py',args.protocol_source/'pairing_aggregate.py',read_json(pair/'session_descriptors_manifest.json')['aggregate_source_sha256']),
        ('pairing_dn_controls.py',args.protocol_source/'pairing_dn_controls.py',read_json(pair/'session_camera_dn_manifest.json')['source_sha256'])]
    folder=args.output/'operators';folder.mkdir(parents=True,exist_ok=True);hashes={}
    for name,source,expected in definitions:
        if sha(source)!=expected:raise ValueError('Frozen operator mismatch: '+name)
        target=folder/name
        if target.exists() and sha(target)!=expected:raise ValueError('Operator output already binds another source')
        target.write_bytes(source.read_bytes());hashes[name]=expected
    sys.path.insert(0,str(folder))
    modules=[importlib.import_module(name) for name in ('prediction','prediction_indices','pairing_aggregate','pairing_dn_controls')]
    if any(Path(m.__file__).resolve().parent!=folder.resolve() for m in modules):raise ValueError('Unexpected operator import')
    return (*modules,hashes,config)


def features(args,p,pi,aggregate,dn,operator_hashes,old_config):
    pair=args.run/'pairing/full';audit=args.audit;out=args.output
    summary=read_json(audit/'full/summary.json');rejected=read_json(audit/'rejected58/summary.json')
    if summary['status']!='completed' or summary['failures'] or rejected['status']!='completed':raise ValueError('Completed strict audit required')
    if summary['audited_accepted_anchors']!=1212 or rejected['audited_rejected_nominal_ECC_anchors']!=58:raise ValueError('Unexpected audit denominator')
    for name,key in [('session_descriptors.csv','indices'),('session_camera_dn.csv','dns')]:
        if sha(pair/name)!=old_config['hashes'][key]:raise ValueError('Archived predictor hash changed')
    files={'references':args.intake/'research_audit/reference_master.csv','sessions':args.run/'extraction/session_traits.csv',
        'indices':pair/'session_descriptors.csv','dns':pair/'session_camera_dn.csv','index_manifest':pair/'session_descriptors_manifest.json',
        'dn_manifest':pair/'session_camera_dn_manifest.json','UVIS':args.intake/'research_audit/tables/UVIS_P6_transmitance__List1.csv',
        'NIR':args.intake/'research_audit/tables/NIR_P6_transmitance__List1.csv','source_flags':args.intake/'research_audit/uvis_consistency_issues.json'}
    for name,path in files.items():
        if sha(path)!=old_config['hashes'][name]:raise ValueError('Frozen input changed: '+name)
    for name in ('summary.json','configuration.json','descriptors_27.csv','registration.csv','index_definitions.json'):
        expected=read_json(pair/'session_descriptors_manifest.json')['inputs_sha256'][name]
        if sha(pair/name)!=expected:raise ValueError('Archived pairing input changed: '+name)
    files.update({name:pair/name for name in ('descriptors_27.csv','registration.csv','index_definitions.json','camera_dn_common_views.csv')})
    files.update({'strict_formulas_accepted':audit/'full/formula_changes.csv','strict_formulas_rejected':audit/'rejected58/formula_changes.csv',
        'strict_channels':audit/'full/channel_changes.csv','strict_summary':audit/'full/summary.json','strict_rejected_summary':audit/'rejected58/summary.json',
        'common_dn_replay':audit/'common_dn_replay_verification.json'})
    if not files['common_dn_replay'].exists():files.pop('common_dn_replay')
    hashes={name:sha(path) for name,path in files.items()}
    original=list(csv_rows(pair/'descriptors_27.csv',True));registration=list(csv_rows(pair/'registration.csv',True))
    strict_audits=sf.read(audit/'full/formula_changes.csv')+sf.read(audit/'rejected58/formula_changes.csv')
    strict=sf.replace_formulas(original,strict_audits)
    if len(original)!=34290 or len(registration)!=1270:raise ValueError('Nominal available row counts changed')
    original_dn=sf.read(pair/'camera_dn_common_views.csv');strict_dn=sf.replace_dns(original_dn,sf.read(audit/'full/channel_changes.csv'))
    sessions=sorted([r for r in sf.read(args.intake/'session_decode_qc.csv') if sf.truth(r['known_reference_identity'])],key=lambda r:tuple(r[k] for k in sf.IDENTITY))
    if len(sessions)!=360:raise ValueError('Expected all360 canonical sessions')
    defs=read_json(pair/'index_definitions.json');old_index=aggregate.aggregate(sessions,original,registration,defs,'ECC');new_index=aggregate.aggregate(sessions,strict,registration,defs,'ECC')
    old_dn=dn.aggregate(sessions,original_dn);new_dn=dn.aggregate(sessions,strict_dn)
    sf.compare_tables(sf.read(pair/'session_descriptors.csv'),old_index);sf.compare_tables(sf.read(pair/'session_camera_dn.csv'),old_dn)
    pi.assert_availability(new_index,'paired_index',pi.INDEX_IDS);pi.assert_availability(new_dn,'paired_dn',pi.CHANNELS)
    p.validate_extraction(files['sessions']);references=p.read_keyed(files['references']);traits=p.read_keyed(files['sessions'])
    expected={p.key(r) for r in references};maps=[{p.key(r):r for r in rows} for rows in (traits,new_index,new_dn)]
    if len(expected)!=360 or any(set(m)!=expected for m in maps):raise ValueError('Four-key table mismatch')
    joined=[{**r,**maps[0][p.key(r)],**maps[1][p.key(r)],**maps[2][p.key(r)]} for r in references]
    selected=p.cohort_indices(joined,'paired_primary');rows=[joined[i] for i in selected]
    if len(rows)!=329:raise ValueError('Primary329 cohort changed')
    primary={p.key(r) for r in rows}
    changes,availability=sf.feature_changes(old_index,new_index,pi.INDEX_IDS,'paired_index',primary)
    dc,da=sf.feature_changes(old_dn,new_dn,pi.CHANNELS,'paired_dn',primary);changes+=dc;availability+=da
    feature_root=out/'features';feature_root.mkdir(exist_ok=True)
    for name,data in [('session_descriptors_strict.csv',new_index),('session_camera_dn_strict.csv',new_dn),
                      ('nominal_ECC_descriptors_strict.csv',strict),('camera_dn_common_views_strict.csv',strict_dn),
                      ('feature_changes.csv',changes),('feature_availability.csv',availability)]:p.save_csv(feature_root/name,data)
    config=dict(status='completed',input_sha256=hashes,operator_sha256=operator_hashes,
        source_sha256={name:sha(Path(__file__).with_name(name)) for name in ('strict_run.py','strict_features.py')},
        canonical_sessions=360,primary_paired_sessions=329,accepted_nominal_anchors=1212,rejected_nominal_anchors=58,
        unavailable_nominal_anchors=170,strict_formula_rows=34290,strict_common_dn_rows=7272,legacy_session_replay='exact all fields',
        alphas=list(p.ALPHAS),seeds=list(p.SEEDS),feature_columns=list(pi.INDEX_COLUMNS),dn_columns=list(pi.DN_COLUMNS),
        corrected_table_sha256={name:sha(feature_root/name) for name in ('session_descriptors_strict.csv','session_camera_dn_strict.csv')},
        scope='Only nominal ECC formula/DN statistics replaced. Registration/controls not rerun; archived wrong-pair/moment controls keep original tolerance.')
    signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest();config['signature']=signature
    manifest=out/'feature_manifest.json'
    if manifest.exists() and read_json(manifest)['signature']!=signature:raise ValueError('Sensitivity output binds different inputs/source')
    save(manifest,config)
    snapshots=out/'execution_sources';snapshots.mkdir(exist_ok=True)
    for name in config['source_sha256']:shutil.copyfile(Path(__file__).with_name(name),snapshots/name)
    save(out/'feature_summary.json',dict(status='completed',availability=availability,changed_feature_cells=len(changes),
                                      legacy_aggregation_exact=True,primary_cohort_unchanged=True))
    return rows,config


def domains(rows,p,intake):
    result=[('assay_'+label,(target,),np.array([[p.number(r[target])] for r in rows]),{})
            for target,label in zip(p.TARGETS,('FvFm','PiAbs','NIR_NDVI'))]
    for band in ('UVIS','NIR'):
        waves,values,flags=p.read_spectra(intake,band)
        result.append((band,tuple(str(v) for v in waves),np.array([values[p.key(r)] for r in rows]),flags))
    return result


def experiments(args,rows,config,p,pi):
    results=[];comparisons=[];case_root=args.output/'prediction';case_root.mkdir(exist_ok=True)
    for domain,targets,y,flags in domains(rows,p,args.intake):
        for sensitivity in ('source_retained','source_flag_sensitivity') if flags else ('source_retained',):
            keep=np.isfinite(y).all(axis=1)
            if sensitivity=='source_flag_sensitivity':keep&=np.array([p.key(r) not in flags for r in rows])
            records=[r for i,r in enumerate(rows) if keep[i]];yy=y[keep]
            for mode in ('genotype','treatment'):
                identity=f'indices__paired_primary__{domain}__{sensitivity}__{mode}'
                old=args.run/'prediction'/identity;destination=case_root/identity
                with np.load(old/'oof.npz',allow_pickle=False) as archived:
                    if not np.array_equal(archived['keys'],np.array([p.key(r) for r in records])) or not np.array_equal(archived['observed'],yy) or tuple(archived['targets'])!=targets:
                        raise ValueError('Archived case cohort/target differs: '+identity)
                signature=hashlib.sha256(json.dumps(dict(feature_signature=config['signature'],identity=identity,keys=[p.key(r) for r in records]),sort_keys=True).encode()).hexdigest()
                integrity=destination/'sensitivity_integrity.json'
                if integrity.exists():
                    for name,digest in read_json(integrity)['files_sha256'].items():
                        if sha(destination/name)!=digest:raise ValueError('Sensitivity cached output changed: '+name)
                result=p.run_experiment(destination,records,yy,targets,mode=mode,cohort='paired_primary',domain=domain,image=True,
                    flags=flags,signature=signature,nulls=True,designs=pi.designs(records,mode),null_models_override=['metadata_plus_indices'])
                old_result=read_json(old/'complete.json');old_by={r['model']:r for r in old_result['summary']}
                with np.load(old/'oof.npz',allow_pickle=False) as a,np.load(destination/'oof.npz',allow_pickle=False) as b:
                    for model in ('training_mean','visit_mean','metadata_ridge','sam_traits'):
                        if not np.array_equal(a[model],b[model]):raise ValueError('Unchanged baseline prediction differs: '+model)
                for row in result['summary']:
                    before=old_by[row['model']]
                    comparisons.append(dict(experiment=identity,domain=domain,sensitivity=sensitivity,split=mode,model=row['model'],n=row['n'],
                        target_channels=row['target_channels'],old_rmse=before['rmse'],strict_rmse=row['rmse'],delta_rmse=row['rmse']-before['rmse'],
                        old_r2=before['mean_target_r2'],strict_r2=row['mean_target_r2'],delta_r2=row['mean_target_r2']-before['mean_target_r2']))
                save(integrity,dict(signature=signature,files_sha256={f.name:sha(f) for f in destination.iterdir() if f.is_file() and f.name!=integrity.name},
                    archived_oof_sha256=sha(old/'oof.npz'),archived_complete_sha256=sha(old/'complete.json')))
                results.append(result);save(args.output/'progress.json',dict(status='running',completed_experiments=len(results),last_experiment=identity))
                p.save_csv(args.output/'metrics_comparison.csv',comparisons)
    save(args.output/'summary.json',dict(status='completed',feature_signature=config['signature'],experiments=len(results),results=results,
        limitations=['Archived controls retain original operator scope.','Same retrospectively inspected experiment; no new external validation.','Changes diagnose deterministic JPEG-DN validity only, not physiological or anatomical accuracy.']))
    save(args.output/'progress.json',dict(status='completed',completed_experiments=len(results)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('intake','run','repo','audit','protocol-source','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--features-only',action='store_true');args=parser.parse_args()
    safe_output(args.output,[args.intake,args.run,args.audit,args.protocol_source,args.repo]);args.output.mkdir(parents=True,exist_ok=True)
    started=time.time()
    try:
        p,pi,aggregate,dn,hashes,old_config=load_operators(args)
        rows,config=features(args,p,pi,aggregate,dn,hashes,old_config)
        if not args.features_only:experiments(args,rows,config,p,pi)
        print(json.dumps(dict(status='completed',features_only=args.features_only,elapsed_seconds=time.time()-started),indent=2))
    except Exception as exc:
        save(args.output/'failure.json',dict(status='failed',type=type(exc).__name__,reason=str(exc)));raise


if __name__=='__main__':main()
