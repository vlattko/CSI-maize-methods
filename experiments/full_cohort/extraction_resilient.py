"""Resume the frozen extractor with bounded Windows sharing-violation retries.

This patches only the atomic JSON writer in memory. The recorded extractor,
models, inputs, scientific operators and configuration signature remain exact.
The launcher and reason for this operational repair are recorded separately.
"""
import hashlib
import json
from pathlib import Path
import sys
import time
import extraction


def with_sharing_retry(function,*args,**kwargs):
    for attempt in range(30):
        try:return function(*args,**kwargs)
        except PermissionError as exc:
            if getattr(exc,'winerror',None) not in (5,32,33) or attempt==29:raise
            time.sleep(min(.05*2**min(attempt,5),1.))


def main():
    original=extraction.atomic_json
    extraction.atomic_json=lambda *a,**k:with_sharing_retry(original,*a,**k)
    run=Path(sys.argv[sys.argv.index('--run')+1])
    shard=int(sys.argv[sys.argv.index('--shard')+1])
    source=Path(extraction.__file__)
    recovery=dict(reason='WinError5 while atomically replacing a progress JSON read by watchers',
                  repaired_operation='bounded retry of atomic JSON replacement only',
                  extractor_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),shard=shard)
    extraction.atomic_json(run/'extraction'/f'io_recovery_shard_{shard}.json',recovery)
    progress=run/'extraction'/f'progress_shard_{shard}.json'
    if progress.exists():
        current=json.loads(progress.read_text());current['phase']='validating_cached_frames_for_resume'
        extraction.atomic_json(progress,current)
    extraction.main()


if __name__=='__main__':main()
