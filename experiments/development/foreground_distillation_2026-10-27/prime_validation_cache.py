"""Prime only validation processing inputs while the all-angle GPU fit runs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
import foreground_core as core
import evaluate_foreground as evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('intake','teacher-run','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    frames=[r for r in core.read_csv(args.output/'all_views/frame_inventory.csv') if r['split']=='val']
    if any(r['genotype']!='Os6251/15' for r in frames):
        raise ValueError('This cache primer is restricted to the validation genotype')
    evaluate.verify_frame_inputs(frames,args.intake,args.teacher_run)
    signature=core.digest(Path(core.__file__))
    cache=args.output/'evaluation/processing_cache'/signature[:12]
    start=time.perf_counter(); count=0
    def prepare(row):
        return evaluate.prepare_cached_frame(row,args.intake,args.teacher_run,cache)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for entry in pool.map(prepare,frames):
            count+=1
            if count%250==0:
                core.write_json(args.output/'evaluation/validation_cache_status.json',dict(status='running',frames=count,total=len(frames)))
                print('validation cache',count,'of',len(frames),flush=True)
    core.write_json(args.output/'evaluation/validation_cache_status.json',dict(status='completed',frames=count,total=len(frames),
        elapsed_seconds=time.perf_counter()-start,operator_signature=signature,test_frames=0))


if __name__=='__main__':
    main()
