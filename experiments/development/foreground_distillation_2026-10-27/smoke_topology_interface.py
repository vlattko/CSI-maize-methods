"""Validate the frozen cross-model API on two validation-camera frames only."""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import foreground_core as core
import evaluate_foreground as evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('intake','teacher-run','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args(); torch.set_num_threads(2)
    specs=evaluate.topology_specs(args.output/'additional_model_test_gate.json')
    frames=core.read_csv(args.output/'quarter_frames.csv')
    cases=[next(r for r in frames if r['split']=='val' and r['camera']==camera and r['teacher_accepted'].lower()=='true') for camera in ('rgb','noir')]
    destination=args.output/'topology_interface_smoke'; destination.mkdir(exist_ok=True)
    records=[]
    for name,spec in specs.items():
        path=Path(spec['inference_module']); sys.path.insert(0,str(path.parent))
        loader=importlib.util.spec_from_file_location('frozen_topology_smoke',path)
        module=importlib.util.module_from_spec(loader); sys.modules[loader.name]=module; loader.loader.exec_module(module)
        for row in cases:
            rgb=core.processing_image(args.intake/row['source'])
            truth=np.array(Image.open(args.teacher_run/row['mask']))>0
            prediction=module.predict_array(rgb,spec['path'],spec['threshold'],target_shape=truth.shape,device='cuda:1')
            if prediction.dtype!=bool or prediction.shape!=truth.shape:
                raise ValueError('Frozen topology API returned an incompatible mask grid')
            path=destination/f'{name}_{row["camera"]}.png'
            Image.fromarray(prediction.astype(np.uint8)*255).save(path)
            records.append(dict(model=name,camera=row['camera'],split='val',id=row['id'],
                threshold=spec['threshold'],checkpoint_sha256=spec['sha256'],
                **core.mask_metrics(truth,prediction),**core.terminal_metrics(truth,prediction)))
        module.load_model.cache_clear()
    core.write_json(destination/'manifest.json',dict(status='passed',records=records,test_frames=0))
    print(records,flush=True)


if __name__=='__main__':
    main()
