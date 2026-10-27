"""Run finalized full-cohort CPU architecture analyses with immutable inputs."""
import argparse,json,subprocess,sys,time
from pathlib import Path

def main(args):
    table=args.run/'extraction/view_traits.csv'
    if not table.exists():raise ValueError('Final extraction/view_traits.csv is required; no implicit partial run')
    base=Path(__file__).parent;out=args.run/'architecture';out.mkdir(exist_ok=True);start=time.time();common=['--intake',str(args.intake),'--run',str(args.run),'--repo',str(args.repo)];commands=[('architecture_analysis.py',common+['--hull-sessions','0']),('architecture_tracking.py',common),('architecture_longitudinal.py',['--intake',str(args.intake),'--run',str(args.run)]),('architecture_geometry_diagnostics.py',common+['--cases',str(args.geometry_cases)])]
    if args.external:
        commands.append(('architecture_external_scaffold.py',['--external',str(args.external),'--run',str(args.run)]))
        commands.append(('architecture_external_tracking.py',['--external',str(args.external),'--run',str(args.run)]))
    commands.append(('architecture_report.py',['--intake',str(args.intake),'--run',str(args.run)]))
    results=[]
    for script,arguments in commands:
        command=[sys.executable,'-u',str(base/script),*arguments];print(json.dumps(dict(stage='start',script=script)),flush=True)
        with (out/(script.removesuffix('.py')+'.log')).open('w',encoding='utf-8') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        results.append(dict(script=script,exit_code=result.returncode))
        (out/'architecture_pipeline_progress.json').write_text(json.dumps(dict(status='running' if result.returncode==0 else 'failed',completed=results,elapsed_seconds=time.time()-start),indent=2),encoding='utf-8')
        if result.returncode:raise RuntimeError(script+' failed; inspect its retained log')
        print(json.dumps(dict(stage='complete',script=script)),flush=True)
    (out/'architecture_pipeline_progress.json').write_text(json.dumps(dict(status='completed',completed=results,elapsed_seconds=time.time()-start),indent=2),encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--intake',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--external',type=Path);p.add_argument('--geometry-cases',type=int,default=15);main(p.parse_args())
