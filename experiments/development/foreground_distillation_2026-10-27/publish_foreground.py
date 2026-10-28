"""Copy hash-bound inference candidates and their validation-only confidence choices."""
import argparse
import json
from pathlib import Path
import shutil
import foreground_core as core


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=['pilot','frozen','final'],required=True)
    args=p.parse_args()
    if args.stage=='final':
        test=json.loads((args.output/'evaluation/comparison_test/manifest.json').read_text())
        if test['status']!='completed' or not test['all_angles']:
            raise ValueError('Final publication requires completed all-angle held-out evaluation')
    phase='pilot' if args.stage=='pilot' else 'comparison'
    root=args.output/'evaluation'/f'{phase}_validation'
    selection=json.loads((root/'selection.json').read_text())
    if args.stage=='pilot' and not json.loads((root/'viability.json').read_text())['viable']:
        raise ValueError('The pilot did not meet its declared agreement viability rule')
    names=('rgb','joint') if args.stage=='pilot' else ('rgb','joint','joint_all40','joint_all40_960')
    cards=[]
    for name in names:
        spec=selection['models'][name]
        source=Path(spec['path'])
        if core.digest(source)!=spec['sha256']:
            raise ValueError('Candidate model changed since confidence selection')
        tag='pilot' if args.stage=='pilot' else 'final'
        destination=args.output/'deployment'/f'{tag}_{name}'
        destination.mkdir(parents=True,exist_ok=True)
        weights=destination/'best.pt'
        shutil.copyfile(source,weights)
        status='usable_validation_only_candidate' if args.stage=='pilot' else ('choices_frozen_pending_independent_test' if args.stage=='frozen' else 'held_out_evaluated_candidate')
        card=dict(name=f'{tag}_{name}',status=status,
            weights=str(weights),weights_sha256=core.digest(weights),confidence=selection['thresholds'][name],
            classes={'0':'plant_foreground'},imgsz=spec.get('imgsz',640),processing_max_dimension=1024,
            cameras=['rgb'] if name=='rgb' else ['rgb','recorded_noir'],
            joint_training_meaning='one3channel model trained on mixed RGB/NoIR frames; each inference uses one camera image independently; no6channel fusion',
            preprocessing='same PIL draft/thumbnail processing1024; BGR numpy input to Ultralytics; retina mask union; black background',
            partition=json.loads((args.output/'preparation.json').read_text())['genotype_splits'],
            validation_selection_sha256=core.digest(root/'selection.json'),
            architecture='Ultralytics YOLO26n-seg, pinned8.4.187 runtime',
            upstream_initialization='Ultralytics assets v8.4.0 yolo26n-seg.pt',
            upstream_license='AGPL-3.0 or applicable separate upstream license; retain upstream attribution',
            inference_cli='infer_foreground.py --repo REPO --weights best.pt --model-card model_card.json --source IMAGE --output DEST --device 1',
            limits=['Teacher-distilled whole-plant foreground; expert foreground accuracy is unmeasured.',
                    'A single plant class does not identify, count or track biological leaves.',
                    'Test evidence, when available, covers one held-out genotype within this experiment.',
                    'NoIR is recorded camera DN imagery, not calibrated spectral reflectance.'])
        outcome=args.output/'all40_extension_decision.json'
        if outcome.exists():
            card['all40_training_outcome']=json.loads(outcome.read_text())
            card['all40_interpretation']='Training on all40views was not uniformly better: the five-epoch640model failed the predeclared competitiveness gate against joint quarter25 on the same4277accepted validation frames. No additional all40epochs were run. Inference at960is a separate sensitivity, not evidence that more training views always improve performance.'
        core.write_json(destination/'model_card.json',card)
        cards.append(card)
    core.write_json(args.output/'deployment'/f'{args.stage}_manifest.json',dict(status='completed',models=cards))
    print(json.dumps(cards,indent=2),flush=True)


if __name__=='__main__':
    main()
