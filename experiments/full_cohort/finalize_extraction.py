"""Wait for both GPU shards, audit their evidence and finalize cohort tables."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2),encoding='utf-8');temp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('intake','run','repo'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();base=Path(__file__).resolve().parent
    directory=args.run/'extraction';statuspath=directory/'controller_status.json'
    started=time.monotonic();stages=[]
    while True:
        states=[]
        for shard in range(2):
            path=directory/f'progress_shard_{shard}.json'
            if path.exists():
                state=json.loads(path.read_text());states.append(state)
                if state['status']!='completed' and time.time()-path.stat().st_mtime>600:
                    write(statuspath,dict(status='failed',reason='GPU shard has not updated for 10 minutes',shard=shard));raise SystemExit(1)
        value=dict(status='waiting_for_gpu_shards',shards=states,elapsed_seconds=round(time.monotonic()-started,1))
        write(statuspath,value)
        if len(states)==2 and all(s['status']=='completed' for s in states):break
        time.sleep(10)
    common=['--intake',str(args.intake),'--run',str(args.run),'--repo',str(args.repo)]
    for name,script,extra in [
        ('extraction-preflight','prediction_audit.py',['--stage','extraction-preflight']),
        ('aggregate','extraction.py',['summarize']),
        ('extraction-tables','prediction_audit.py',['--stage','extraction-tables'])]:
        source=base/script
        command=[sys.executable,'-u','-X','utf8',str(source),*extra,*common]
        write(statuspath,dict(status='running',stage=name,completed_stages=stages))
        print(json.dumps(dict(stage=name,command=command)),flush=True)
        result=subprocess.run(command,capture_output=True,text=True,encoding='utf-8')
        (directory/f'controller_{name}.log').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
        event=dict(stage=name,exit_code=result.returncode,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        stages.append(event)
        if result.returncode:
            write(statuspath,dict(status='failed',stage=name,completed_stages=stages))
            print(result.stderr,flush=True);raise SystemExit(result.returncode)
        print(result.stdout,flush=True)
    write(statuspath,dict(status='completed',completed_stages=stages,elapsed_seconds=round(time.monotonic()-started,1)))
    print('Full extraction and independently checked cohort tables completed.',flush=True)


if __name__=='__main__':main()
