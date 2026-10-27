"""Continue the active prediction run when final extraction and encoder arrive."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import prediction as p


def ready(run):
    for path in (run/'extraction'/'summary.json',
                 run/'prediction'/'encoder'/'progress.json'):
        if not path.exists() or json.loads(path.read_text(encoding='utf-8'))['status']!='completed':
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('intake','run','repo'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    destination = args.run/'prediction'
    destination.mkdir(parents=True,exist_ok=True)
    lock = destination/'finalizer.lock'
    with lock.open('x',encoding='utf-8') as f:
        json.dump(dict(pid=os.getpid(),started_unix=time.time()),f)
    state = destination/'finalizer_status.json'
    try:
        p.save_json(state,dict(status='waiting_for_final_inputs',pid=os.getpid()))
        while not ready(args.run):
            time.sleep(20)
        p.save_json(state,dict(status='running_images',pid=os.getpid()))
        script = Path(__file__).parent.resolve()
        common = ['--intake',str(args.intake),'--run',str(args.run),'--repo',str(args.repo)]
        env = dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',
                   OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
        commands = [
            [sys.executable,'-u',str(script/'prediction.py'),*common,'--stage','images',
             '--embeddings',str(destination/'encoder'/'session_embeddings.npz')],
            [sys.executable,'-u',str(script/'prediction_audit.py'),*common],
        ]
        with (destination/'images_execution.log').open('a',encoding='utf-8') as logfile:
            for command in commands:
                print('RUN',json.dumps(command),flush=True)
                subprocess.run(command,stdout=logfile,stderr=subprocess.STDOUT,
                               env=env,check=True)
        p.save_json(state,dict(status='completed',pid=os.getpid(),completed_unix=time.time()))
        print('Full-cohort image prediction and output audit completed',flush=True)
    except Exception as exc:
        p.save_json(state,dict(status='failed',pid=os.getpid(),error_type=type(exc).__name__,
                              error=str(exc)))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__':
    main()
