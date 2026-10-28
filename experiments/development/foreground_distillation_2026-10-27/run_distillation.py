"""Resume bounded preparation-gated pilot, validation, full fits and held-out test."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import foreground_core as core


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('repo','intake','teacher-run','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    scripts=Path(__file__).parent
    common=['--repo',str(args.repo),'--output',str(args.output)]
    extra=['--intake',str(args.intake),'--teacher-run',str(args.teacher_run)]
    print('waiting for completed preparation',flush=True)
    while not (args.output/'preparation.json').exists():
        time.sleep(5)
    steps=[('pilot_train','train_foreground.py',common+['--phase','pilot','--epochs','5']),
           ('pilot_validation','evaluate_foreground.py',common+extra+['--phase','pilot','--epochs','5','--split','val']),
           ('full_train','train_foreground.py',common+['--phase','full','--epochs','25']),
           ('full_validation','evaluate_foreground.py',common+extra+['--phase','full','--epochs','25','--split','val']),
           ('full_test','evaluate_foreground.py',common+extra+['--phase','full','--epochs','25','--split','test'])]
    for name,script,arguments in steps:
        marker=args.output/'pipeline'/f'{name}.json'
        if marker.exists() and json.loads(marker.read_text()).get('returncode')==0:
            continue
        command=[sys.executable,str(scripts/script),*arguments]
        log=args.output/'logs'/f'{name}_{time.strftime("%Y%m%d_%H%M%S")}.log'
        log.parent.mkdir(parents=True,exist_ok=True)
        record=dict(step=name,status='running',command=command,log=str(log),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        core.write_json(args.output/'pipeline/state.json',record)
        print('starting',name,flush=True)
        with open(log,'w',encoding='utf8') as stream:
            process=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        record.update(status='completed' if process.returncode==0 else 'failed',returncode=process.returncode,
                      ended_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        core.write_json(marker,record); core.write_json(args.output/'pipeline/state.json',record)
        if process.returncode:
            raise RuntimeError(f'{name} failed; see {log}')
    core.write_json(args.output/'pipeline/state.json',dict(status='completed',steps=[x[0] for x in steps]))


if __name__=='__main__':
    main()
