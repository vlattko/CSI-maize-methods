"""Run the full-cohort research protocol from explicit local inputs.

The driver does not fetch models or external datasets. Scientific producers
retain their own source, input and output checks, including failed hypotheses.
Use a checkout, a fresh intake and a separate output folder. A completed stage
can be resumed only with identical code/configuration and unchanged anchors.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from contextlib import contextmanager
from importlib import metadata
from extraction_resilient import with_sharing_retry


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def validate_layout(config):
    roots={k:Path(config[k]).resolve() for k in ('raw','intake','run')}
    for a,b in [('raw','intake'),('raw','run'),('intake','run')]:
        if roots[a].is_relative_to(roots[b]) or roots[b].is_relative_to(roots[a]):
            raise ValueError('Raw, intake and run paths must be separate non-nested folders')
    if config.get('device','cuda:0')!='cuda:0':
        raise ValueError('This single-GPU coordinator requires cuda:0; triage and encoder use that device')


def files_below(root,exclude=()):
    for directory,folders,files in os.walk(root):
        folders[:]=[f for f in folders if f not in {'.git','.cache','__pycache__',*exclude}]
        for name in files:
            if not name.endswith('.tmp'):yield Path(directory)/name


def input_hashes(config,target):
    raw=Path(config['raw']);paths=set(files_below(raw/'IMAGES'))|set(raw.rglob('*.xlsx'))
    for root in [Path(config['models'])/'sam2.1-hiera-small',Path(config['dino'])]:
        if not root.is_dir():raise ValueError('Missing supplied model folder: '+str(root))
        paths.update(files_below(root))
    if target=='full':
        paths.update(files_below(Path(config['external'])))
        assets=Path(config['da3_assets'])
        paths.update(files_below(assets/'models/DA3-SMALL'))
        paths.update(files_below(assets/'vendor/Depth-Anything-3/src'))
        paths.update(files_below(assets/'da3_deps'))
    return {str(p):digest(p) for p in sorted(paths)}


def output_hashes(config,previous):
    paths=set(files_below(Path(config['intake']),{'IMAGES'}))
    paths.update(files_below(Path(config['run']),{'pipeline_logs'}))
    result={}
    for path in sorted(paths):
        if path.name in {'cohort_pipeline.json','cohort_pipeline.lock'}:continue
        stat=path.stat();old=previous.get(str(path))
        changed=old is None or old['bytes']!=stat.st_size or old['mtime_ns']!=stat.st_mtime_ns
        result[str(path)]=dict(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,
                              sha256=digest(path) if changed else old['sha256'])
    return result


def verify_outputs(recorded):
    for path,entry in recorded.items():
        if not Path(path).is_file() or digest(path)!=entry['sha256']:
            raise ValueError('Recorded output missing or changed: '+path+
                             '. Use a fresh work directory or restore the recorded outputs before resuming; no changed result is silently reused.')


@contextmanager
def run_lock(root):
    root.mkdir(parents=True,exist_ok=True)
    stream=(root/'cohort_pipeline.lock').open('a+b')
    stream.seek(0,2)
    if stream.tell()==0:stream.write(b'0');stream.flush()
    stream.seek(0)
    try:
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError as error:
        stream.close();raise RuntimeError('Another coordinator holds this run; no recovery or stage was started') from error
    try:yield
    finally:
        stream.seek(0)
        if os.name=='nt':msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
        else:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)
        stream.close()


def claim_folder(folder,state):
    identity=str(folder.resolve())
    if identity in state.setdefault('owned_folders',[]):return
    if folder.exists() and any(folder.iterdir()):
        raise ValueError('Preexisting nonempty derived folder is not owned by this workflow: '+str(folder))
    state['owned_folders'].append(identity)


def save_state(path,state):
    temporary=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    temporary.write_text(json.dumps(state,indent=2),encoding='utf-8')
    with_sharing_retry(temporary.replace,path)


def retire_interrupted(config,name,state):
    if name not in ('intake','triage'):return
    folder=Path(config['intake']) if name=='intake' else Path(config['intake'])/'segmentation_triage'
    if not folder.exists():return
    if str(folder.resolve()) not in state.get('owned_folders',[]):
        raise ValueError('Cannot retire an unowned derived folder')
    parent=folder.parent.resolve();resolved=folder.resolve()
    if resolved.parent!=parent or resolved==Path(config['raw']).resolve():
        raise ValueError('Interrupted output is not the owned derived folder')
    destination=parent/(folder.name+'.interrupted-'+str(time.time_ns()))
    if destination.resolve().parent!=parent:raise ValueError('Recovery destination escapes derived parent')
    # Native pathlib rename preserves the partial evidence and any source junction.
    folder.rename(destination)
    for path in list(state.get('outputs',{})):
        if Path(path).is_relative_to(folder):del state['outputs'][path]
    state.setdefault('retired_partial_outputs',[]).append(str(destination))


def build_plan(config, scripts, target):
    validate_layout(config)
    if target=='full':
        for field in ('external','da3_assets'):
            if not config.get(field):raise ValueError('Full route requires '+field)
    c=config
    run=Path(c['run']);intake=Path(c['intake'])
    common=['--intake',str(intake),'--run',str(run),'--repo',c['repo']]
    stages=[]

    def add(name,script,args,anchor):
        stages.append(dict(name=name,script=str(scripts/script),args=list(map(str,args)),anchor=str(anchor)))

    add('intake','intake/intake_audit.py',['all','--raw',c['raw'],'--output',intake,'--repo',c['repo'],
        '--previous',c.get('previous',str(intake.parent/'unavailable_previous_archive')),'--workers','4'],intake/'intake_summary.json')
    add('triage','intake/segmentation_triage.py',['--intake',intake,'--repo',c['repo'],'--models',c['models'],'--mode','box'],intake/'segmentation_triage/summary.json')
    extract=['extract',*common,'--models',c['models'],'--device',c.get('device','cuda:0'),'--shards','1','--shard','0','--batch-size',str(c.get('batch_size',8))]
    add('extract','extraction_resilient.py',extract,run/'extraction/progress_shard_0.json')
    add('extraction_preflight','prediction_audit.py',[*common,'--stage','extraction-preflight'],run/'prediction/extraction_preflight_audit.json')
    add('aggregate','extraction.py',['summarize',*common,'--shards','1'],run/'extraction/summary.json')
    add('extraction_tables','prediction_audit.py',[*common,'--stage','extraction-tables'],run/'prediction/extraction_tables_audit.json')
    add('architecture','architecture_run.py',common,run/'architecture/architecture_summary.json')
    add('architecture_audit','architecture_verify.py',['--run',run],run/'architecture/architecture_full_run_verification.json')
    add('geometry_domain_repeat','architecture_domain_repeat.py',['--run',run],run/'architecture/geometry_domain_repeat/domain_repeat_summary.json')
    add('known_answer','architecture_known_answer.py',['--run',run],run/'architecture/known_answer/known_answer_summary.json')
    add('pairing_input_audit','pairing_input_audit.py',common,run/'pairing/quarter_input_audit.json')
    add('pairing','pairing_run.py',[*common,'--mode','full'],run/'pairing/full/summary.json')
    add('index_aggregation','pairing_aggregate.py',common,run/'pairing/full/session_descriptors_manifest.json')
    add('common_dn','pairing_dn_controls.py',common,run/'pairing/full/session_camera_dn_manifest.json')
    add('spatial_atlas','pairing_atlas.py',common,run/'pairing/full/atlas_manifest.json')
    add('pairing_report','pairing_report.py',[*common,'--mode','full'],run/'pairing/full/report_verification.json')
    add('pairing_audit','pairing_verify.py',common,run/'pairing/full/final_verification.json')
    add('encoder','prediction_encoders.py',[*common,'--model',c['dino']],run/'prediction/encoder/progress.json')
    add('metadata_predictions','prediction.py',[*common,'--stage','metadata'],run/'prediction/summary_metadata.json')
    add('image_predictions','prediction.py',[*common,'--stage','images','--embeddings',run/'prediction/encoder/session_embeddings.npz'],run/'prediction/summary_images.json')
    add('index_predictions','prediction_indices.py',common,run/'prediction/summary_indices.json')
    add('prediction_audit','prediction_audit.py',common,run/'prediction/audit.json')
    if target=='cohort':
        add('architecture_report','architecture_report.py',['--intake',intake,'--run',run],run/'architecture/architecture_report_manifest.json')
    add('prediction_report','prediction_report.py',common,run/'prediction/prediction_report.html')
    if target=='full':
        ext=['--external',c['external'],'--run',str(run)]
        for name,script,anchor in [
            ('external_correspondence','architecture_external_pheno4d.py','external_correspondence_summary.json'),
            ('external_rendered','architecture_external_rendered.py','rendered/rendered_external_summary.json'),
            ('external_scaffold','architecture_external_scaffold.py','scaffold/external_scaffold_summary.json'),
            ('external_tracking','architecture_external_tracking.py','scaffold/automatic_scaffold_crossdate_summary.json'),
            ('external_exposure','architecture_external_exposure.py','exposure/external_exposure_summary.json')]:
            add(name,script,ext,run/'architecture/external_pheno4d'/anchor)
        add('external_uncertainty','architecture_external_uncertainty.py',['--run',run],run/'architecture/external_pheno4d/uncertainty_and_negative_controls.json')
        add('external_audit','architecture_external_audit.py',ext,run/'architecture/external_pheno4d/external_denominator_audit.json')
        add('depth','depth_trial.py',['--intake',intake,'--run',run,'--assets-base',c['da3_assets'],'--device',c.get('device','cuda:0')],run/'depth_trial/summary.json')
        add('architecture_report','architecture_report.py',['--intake',intake,'--run',run],run/'architecture/architecture_report_manifest.json')
        add('full_report','full_report.py',['--intake',intake,'--run',run],run/'full_report_build.json')
    return stages


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--target',choices=['cohort','full'],default='cohort')
    parser.add_argument('--plan',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    config=json.loads(args.config.read_text(encoding='utf-8'))
    for name in ('raw','intake','run','repo','models','dino','previous','external','da3_assets'):
        if config.get(name):config[name]=str((args.config.resolve().parent/config[name]).resolve())
    base=Path(__file__).parent
    stages=build_plan(config,base,args.target)
    if args.plan:
        print(json.dumps(stages,indent=2));return
    with run_lock(Path(config['run'])):
        execute_workflow(args,config,base,stages)


def execute_workflow(args,config,base,stages):
    for stage in stages:
        if not Path(stage['script']).is_file():raise ValueError('Missing producer: '+stage['script'])
    for name in ('raw','repo','models','dino'):
        if not Path(config[name]).is_dir():raise ValueError('Missing supplied input: '+name)
    if args.target=='full' and not (Path(config['external'])/'download_manifest.json').is_file():
        raise ValueError('Supply a completed external retrieval; this driver does not download it')
    sources={p.relative_to(base).as_posix():digest(p) for p in base.rglob('*.py') if '__pycache__' not in p.parts}
    operators=Path(config['repo'])/'src/csi_maize'
    code={p.relative_to(operators).as_posix():digest(p) for p in operators.rglob('*.py')}
    out=Path(config['run']);statepath=out/'cohort_pipeline.json'
    if not statepath.exists():
        if any(p.name!='cohort_pipeline.lock' for p in out.iterdir()):raise ValueError('Use a new empty output folder')
        intake=Path(config['intake'])
        if intake.exists() and any(intake.iterdir()):raise ValueError('Use a new empty intake folder; existing data are not owned')
    print('Binding original images, workbooks and model assets by SHA256',flush=True)
    context=dict(config=config,target=args.target,sources=sources,operators=code,
                 inputs=input_hashes(config,args.target),python=sys.version,
                 packages=sorted((d.metadata['Name'],d.version) for d in metadata.distributions() if d.metadata['Name']))
    signature=hashlib.sha256(json.dumps(context,sort_keys=True).encode()).hexdigest()
    state=dict(signature=signature,context=context,stages=[])
    if statepath.exists():
        if not args.resume:raise ValueError('Existing workflow requires --resume')
        state=json.loads(statepath.read_text())
        if state['signature']!=signature:raise ValueError('Source/configuration changed; use a new run')
        verify_outputs(state.get('outputs',{}))
    out.mkdir(parents=True,exist_ok=True);logs=out/'pipeline_logs';logs.mkdir(exist_ok=True)
    save_state(statepath,state)
    environment=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    for stage in stages:
        previous=next((r for r in state['stages'] if r['name']==stage['name'] and r['status']=='completed'),None)
        anchor=Path(stage['anchor'])
        if previous:
            if not anchor.is_file():raise ValueError('Completed anchor missing: '+stage['name'])
            continue
        interrupted=any(r['name']==stage['name'] and r['status'] in ('running','failed') for r in state['stages'])
        if stage['name'] in ('intake','triage'):
            folder=Path(config['intake']) if stage['name']=='intake' else Path(config['intake'])/'segmentation_triage'
            claim_folder(folder,state)
        if interrupted:retire_interrupted(config,stage['name'],state)
        started=time.time();command=[sys.executable,'-u','-X','utf8',stage['script'],*stage['args']]
        state['stages'].append(dict(name=stage['name'],status='running',command=command))
        state['status']='running';save_state(statepath,state)
        print('RUN '+stage['name'],flush=True)
        with (logs/(stage['name']+'.log')).open('a',encoding='utf-8') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=environment)
        record=dict(name=stage['name'],command=command,elapsed_seconds=time.time()-started,
                    status='completed' if result.returncode==0 and anchor.is_file() else 'failed',exit_code=result.returncode)
        if record['status']=='completed':record['anchor_sha256']=digest(anchor)
        state['stages'][-1]=record;state['status']=record['status'] if record['status']=='failed' else 'running'
        if record['status']=='completed':state['outputs']=output_hashes(config,state.get('outputs',{}))
        save_state(statepath,state)
        if record['status']=='failed':raise RuntimeError('Stage failed or omitted declared anchor: '+stage['name']+'; inspect its log')
    verify_outputs(state.get('outputs',{}))
    state['status']='completed';save_state(statepath,state)
    print('COMPLETED '+args.target,flush=True)


if __name__=='__main__':main()
