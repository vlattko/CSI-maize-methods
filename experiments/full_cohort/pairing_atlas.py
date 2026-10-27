"""Supporting 27-definition maps from immutable full-cohort registrations."""
import argparse
import base64
import html
import json
from pathlib import Path
import statistics
import sys
import time

base=Path(__file__).resolve().parent
if base.name=='pairing_support':base=base.parent
sys.path.insert(0,str(base))
import pairing_run as run
from pairing_core import atomic_json,sha,digest,read_csv


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('intake','run','repo'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--watch',action='store_true')
    args=p.parse_args()
    root=args.run.resolve()/'pairing/full';intake=args.intake.resolve()
    summary_path=root/'summary.json'
    while True:
        summary=json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.exists() else {}
        if summary.get('status')=='completed':break
        if not args.watch:raise ValueError('Completed full pairing required')
        time.sleep(30)
    if summary.get('mode')!='full' or summary.get('planned_anchor_pairs')!=1440:
        raise ValueError('Cannot use partial or pilot output as a full supporting atlas')
    config=json.loads((root/'configuration.json').read_text(encoding='utf-8'))
    methods=args.repo.resolve()/'src/csi_maize/_methods'
    for name,value in config['scientific_source_sha256'].items():
        if sha(methods/name)!=value:raise ValueError('Public scientific source changed: '+name)
    run.configure(args.repo.resolve())
    np=run.np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Times New Roman','savefig.facecolor':'white'})
    candidates={}
    for path in sorted((root/'records').glob('*.json')):
        record=json.loads(path.read_text(encoding='utf-8'))
        if int(record['view'])!=0 or str(record['measurement']) not in ('2','5'):continue
        ev=[r for r in record['evaluations'] if r['pair_kind']=='nominal' and r['method']=='ECC' and r['cross_support_usable']]
        if not ev:continue
        candidates.setdefault((record['treatment'],str(record['measurement'])),[]).append((path,ev[0]['whole_iou_rgb_grid']))
    decoded={r['path']:r for r in read_csv(intake/'jpeg_decode_manifest.csv')}
    treatments=sorted({r['treatment'] for r in read_csv(intake/'session_decode_qc.csv') if r['known_reference_identity']=='True'})
    expected_strata=[(t,str(v)) for t in treatments for v in (2,5)]
    output=root/'atlas';output.mkdir(exist_ok=True)
    definitions=run.index_definitions();selection=[]
    for stratum,items in sorted(candidates.items()):
        median=statistics.median([v for _,v in items])
        path,iou=min(items,key=lambda pair:(abs(pair[1]-median),str(pair[0])))
        record=json.loads(path.read_text(encoding='utf-8'))
        payload=record.pop('payload_sha256')
        if digest(record)!=payload:raise ValueError('Changed numerical record')
        ev=next(r for r in record['evaluations'] if r['pair_kind']=='nominal' and r['method']=='ECC')
        a,am,_=run.load(record['source_bindings']['rgb']['record'],intake,args.run.resolve(),decoded)
        b,bm,_=run.load(record['source_bindings']['nominal']['record'],intake,args.run.resolve(),decoded)
        matrix=np.asarray(ev['source_to_target'],np.float32)
        stats,channels,maps,supports,wb,common=run.describe(a,am,b,bm,matrix,True)
        expected={r['index']:r for r in ev['descriptors']}
        for row in stats:
            original=expected[row['index']]
            if row['valid_pixels']!=original['valid_pixels'] or row['eligible_pixels']!=original['eligible_pixels']:
                raise ValueError('Re-derived atlas pixel denominator differs from saved numerical result')
            if row['median'] is not None and not np.isclose(row['median'],original['median'],rtol=0,atol=1e-7):
                raise ValueError('Re-derived atlas median differs from saved numerical result')
        wm,_=run.warp_to_rgb(bm.astype(np.uint8),matrix,am.shape,True)
        yy,xx=np.where(am|(wm>0))
        crop=(slice(max(0,int(yy.min())-12),min(am.shape[0],int(yy.max())+13)),
              slice(max(0,int(xx.min())-12),min(am.shape[1],int(xx.max())+13)))
        rgb=a.copy();rgb[~am]=0
        noir=np.clip(wb,0,255).astype(np.uint8);noir[wm==0]=0
        fig,axes=plt.subplots(5,6,figsize=(13,12),layout='constrained')
        axes=axes.ravel()
        for ax,image,label in zip(axes[:2],(rgb,noir),('RGB JPEG DN','Registered NoIR JPEG DN')):
            ax.imshow(image[crop]);ax.set_title(label,fontsize=9);ax.set_axis_off()
        cmap=plt.colormaps['viridis'].copy();cmap.set_bad('black')
        by_index={r['index']:r for r in stats}
        for ax,definition in zip(axes[2:],definitions):
            key=definition['id'];row=by_index[key]
            ax.set_facecolor('black')
            ax.imshow(np.ma.masked_invalid(maps[key][crop]),cmap=cmap,
                      vmin=definition['vmin'],vmax=definition['vmax'])
            ax.set_title(key.replace('__','\n')+'\n'+f"n={row['valid_pixels']}/{row['eligible_pixels']}; range [{definition['vmin']},{definition['vmax']}]",
                         fontsize=8)
            ax.set_axis_off()
        axes[-1].set_axis_off()
        fig.suptitle(f"Supporting camera-DN formulas: treatment {stratum[0]}, visit {stratum[1]}\n"+
                     f"{Path(record['session']).name}; nominal ECC IoU {iou:.3f}; common support is not tissue correspondence",
                     fontsize=12)
        target=output/f'atlas_{stratum[0]}_visit{stratum[1]}.png'
        fig.savefig(target,dpi=170);plt.close(fig)
        selection.append(dict(treatment=stratum[0],measurement=stratum[1],session=record['session'],
                              view=0,source_record=path.name,source_record_payload_sha256=payload,
                              stratum_candidates=len(items),stratum_iou_median=median,selected_iou=iou,
                              figure=target.relative_to(root).as_posix(),figure_sha256=sha(target),
                              verified_definition_statistics=27,definition_availability=stats))
    manifest=dict(status='completed',role='supporting uncalibrated JPEG-DN atlas',
                  selection_policy='For visits2and5, one accepted view0 nominalECC example closest to median maskIoU per treatment/visit; no assay targets',
                  expected_strata=len(expected_strata),rendered_strata=len(selection),selection=selection,
                  unavailable_strata=[dict(treatment=t,measurement=m,reason='no_accepted_view0_nominalECC_common_support')
                                      for t,m in expected_strata if (t,m) not in candidates],
                  producer_source_sha256=sha(__file__),numerical_configuration_sha256=sha(root/'configuration.json'),
                  limitations=['Whole-mask common support does not establish corresponding tissue or leaf identity.',
                               'No calibrated reflectance or independently measured narrow-band channels.',
                               'Fixed original formula display ranges; black pixels are unsupported or invalid.',
                               'Per-frame summary qualification still requires40validpixels; speckles below40are not qualified estimates.'])
    atomic_json(root/'atlas_manifest.json',manifest)
    content='<!doctype html><html><head><meta charset="utf-8"><title>Supporting paired camera-DN atlas</title><style>body{font-family:"Times New Roman",serif;background:white;color:#222;max-width:1250px;margin:30px auto;line-height:1.4}img{max-width:100%}h1,h2{font-weight:normal}</style></head><body><h1>Supporting paired camera-DN atlas</h1><p>Architecture remains the main investigation. These 27 algebraic definitions use recorded JPEG values, with family-specific support and original signal guards. Common whole-plant support does not validate corresponding tissue. No NoIR channel is assigned a calibrated narrow-band NIR response. Unsupported and invalid pixels remain black.</p><p>We selected one accepted view0 example closest to the median maskIoU in each treatment/visit stratum for visits2and5, without assay targets. Every re-derived valid/eligible pixel count and available descriptor median was checked against the retained numerical record. Each map shows its valid/eligible count and original fixed display range. Fewer than40validpixels does not qualify a per-frame summary.</p>'
    for entry in selection:
        image=root/entry['figure']
        content+='<h2>'+html.escape(entry['session'])+'</h2><img src="data:image/png;base64,'+base64.b64encode(image.read_bytes()).decode()+'">'
    content+='<p>Selection, hashes, availability and verification are retained in atlas_manifest.json; the full tables remain in descriptors_27.csv. Technical quarter views do not supply independent biological replicates.</p></body></html>'
    (root/'supporting_atlas.html').write_text(content,encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k!='selection'},indent=2))


if __name__=='__main__':main()
