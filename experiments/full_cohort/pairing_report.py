"""Verify retained pairing records and render the supporting camera-DN audit."""
import argparse
import base64
from collections import Counter, defaultdict
import html
import json
from pathlib import Path
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
from pairing_core import read_csv, write_csv, atomic_json, digest, sha, true


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('intake','run','repo'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--mode',choices=('pilot','full'),required=True)
    args=p.parse_args()
    root=args.run.resolve()/'pairing'/args.mode
    summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    config=json.loads((root/'configuration.json').read_text(encoding='utf-8'))
    registration=read_csv(root/'registration.csv')
    descriptors=read_csv(root/'descriptors_27.csv')
    availability=read_csv(root/'availability.csv')
    definitions=json.loads((root/'index_definitions.json').read_text(encoding='utf-8'))
    if len(availability)!=summary['planned_anchor_pairs']:
        raise ValueError('Reporting requires all planned anchors retained, including failures')
    if len(definitions)!=27 or len(descriptors)!=27*len(registration):
        raise ValueError('Descriptor denominator mismatch')
    verified=0
    for path in (root/'records').glob('*.json'):
        result=json.loads(path.read_text(encoding='utf-8'))
        expected=result.pop('payload_sha256')
        if digest(result)!=expected:
            raise ValueError('Numerical cache payload changed: '+str(path))
        verified+=1
    if verified!=len(availability):
        raise ValueError('Per-anchor audit count mismatch')
    expected_indices={d['id'] for d in definitions}
    groups=defaultdict(list)
    for row in descriptors:
        groups[row['session'],row['view'],row['pair_kind'],row['method']].append(row)
    for key,rows in groups.items():
        if len(rows)!=27 or {r['index'] for r in rows}!=expected_indices:
            raise ValueError('Missing/duplicated descriptor definition: '+str(key))
        for row in rows:
            if int(row['valid_pixels'])>int(row['eligible_pixels']):
                raise ValueError('Valid count exceeds eligible denominator')
            if true(row['qualified']) != (int(row['valid_pixels'])>=40):
                raise ValueError('Qualified/valid count mismatch')
            if not true(row['qualified']) and row['median']:
                raise ValueError('Unqualified descriptor has a median')
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Times New Roman','font.size':10,'savefig.facecolor':'white'})
    planned=summary['planned_anchor_pairs']
    descriptor_summary=[]
    for definition in definitions:
        for method in ('moment','ECC'):
            rows=[r for r in descriptors if r['index']==definition['id'] and
                  r['pair_kind']=='nominal' and r['method']==method]
            qualified=[r for r in rows if true(r['qualified'])]
            fractions=[float(r['valid_fraction']) for r in rows if r['valid_fraction']]
            descriptor_summary.append(dict(index=definition['id'],family=definition['support'],method=method,
                                           planned=planned,executed=len(rows),qualified=len(qualified),
                                           qualified_fraction_of_planned=len(qualified)/planned,
                                           qualified_fraction_of_executed=len(qualified)/len(rows) if rows else None,
                                           median_valid_fraction=float(np.median(fractions)) if fractions else None,
                                           median_of_qualified_view_medians=float(np.median([float(r['median']) for r in qualified]))
                                           if qualified else None))
    write_csv(root/'descriptor_availability_summary.csv',descriptor_summary)
    coverage=[]
    for grouping,fields in [('treatment_visit',('treatment','measurement')),
                            ('genotype',('genotype',))]:
        labels=sorted({tuple(r[k] for k in fields) for r in availability})
        for label in labels:
            anchors=[r for r in availability if tuple(r[k] for k in fields)==label]
            selected_registration=[r for r in registration if r['pair_kind']=='nominal' and
                                   r['method']=='ECC' and tuple(r[k] for k in fields)==label]
            coverage.append(dict(grouping=grouping,label='|'.join(label),planned_quarters=len(anchors),
                                 nominal_ready=sum(true(r['nominal_ready']) for r in anchors),
                                 ECC_executed=len(selected_registration),
                                 cross_support_usable=sum(true(r['cross_support_usable']) for r in selected_registration),
                                 fraction_cross_usable_of_planned=sum(true(r['cross_support_usable'])
                                     for r in selected_registration)/len(anchors)))
    write_csv(root/'coverage_by_group.csv',coverage)
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.5),sharey=True,layout='constrained')
    kinds=['nominal','wrong_view','wrong_plant','wrong_view_fixed','wrong_plant_fixed']
    labels=['Nominal','Index +20\nrefitted','Other replicate\nrefitted','Index +20\nfixed','Other replicate\nfixed']
    for ax,method in zip(axes,('moment','ECC')):
        distributions=[]
        for kind in kinds:
            rows=[r for r in registration if r['pair_kind']==kind and r['method']==method]
            distributions.append([float(r['whole_iou_rgb_grid']) for r in rows])
        if any(distributions):
            ax.boxplot(distributions,tick_labels=labels,showfliers=False,widths=.6)
        ax.axhline(.65,color='#aa4444',linestyle='--',linewidth=.8)
        ax.set_title(method+'; descriptive quarter-view distributions')
        ax.set_ylim(0,1)
        ax.set_ylabel('Whole-mask IoU on the RGB grid')
        for i,values in enumerate(distributions):
            ax.text(i+1,.965,f'n={len(values)}',ha='center',fontsize=8)
    fig.savefig(root/'figures/registration_controls.png',dpi=180)
    plt.close(fig)
    selected=[r for r in descriptor_summary if r['method']=='ECC']
    fig,ax=plt.subplots(figsize=(10,8),layout='constrained')
    colors={'rgb':'#497c3c','noir':'#6864a0','cross':'#bc763c'}
    ax.barh(range(27),[r['qualified_fraction_of_planned'] for r in selected],
            color=[colors[r['family']] for r in selected])
    ax.set_yticks(range(27),[r['index'] for r in selected],fontsize=8)
    ax.invert_yaxis();ax.set_xlim(0,1)
    ax.set_xlabel(f'Qualified nominal ECC descriptors / {planned} planned quarter anchors')
    ax.set_title('Availability under original signal and support guards')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=v,label=k+' support') for k,v in colors.items()],loc='lower center',
              bbox_to_anchor=(.5,-.15),ncol=3,frameon=False)
    fig.savefig(root/'figures/descriptor_availability.png',dpi=180)
    plt.close(fig)
    missing=Counter()
    for row in availability:
        for failure in json.loads(row['failures']):
            missing[failure['pair_kind']+': '+failure['reason']]+=1
    upstream_paths=(list((args.intake.resolve()/'segmentation_triage').glob('configuration.json'))
                    if args.mode=='pilot' else list((args.run.resolve()/'extraction').glob('configuration*.json')))
    provenance=dict(report_source_sha256=sha(__file__),numerical_config_sha256=sha(root/'configuration.json'),
                    upstream_segmentation_configurations={str(p):dict(sha256=sha(p),
                    contents=json.loads(p.read_text(encoding='utf-8-sig'))) for p in upstream_paths},
                    verified_payloads=verified,registration_rows=len(registration),descriptor_rows=len(descriptors),
                    numerical_tables_sha256={name:sha(root/name) for name in
                        ('registration.csv','descriptors_27.csv','channel_qc.csv','availability.csv')})
    atomic_json(root/'report_verification.json',provenance)
    def table(rows,fields):
        return '<table><thead><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in fields)+'</tr></thead><tbody>'+''.join(
            '<tr>'+''.join('<td>'+html.escape(str(r.get(k,'')))+'</td>' for k in fields)+'</tr>' for r in rows)+'</tbody></table>'
    def embedded(path):
        return '<img src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'">'
    title='Fixed-quarter paired-camera '+('pilot' if args.mode=='pilot' else 'full-cohort investigation')
    comparisons=summary['paired_comparisons']
    optimizer_counts=Counter()
    for row in registration:
        audit=json.loads(row.get('ECC_optimizer') or '{}')
        if row['method']=='ECC' and row['pair_kind'] in ('nominal','wrong_view','wrong_plant'):
            optimizer_counts[row['pair_kind'],audit.get('status','not_logged_in_original_pilot')]+=1
    examples=[r for r in registration if r['pair_kind']=='nominal' and r['method']=='ECC' and r['view']=='0']
    examples=sorted(examples,key=lambda r:float(r['whole_iou_rgb_grid']))
    example_html=''
    for row in [examples[0],examples[-1]] if examples else []:
        path=root/'figures'/(Path(row['session']).name+'__000.png')
        if path.exists():
            example_html+='<p>'+html.escape(row['session'])+'; IoU '+format(float(row['whole_iou_rgb_grid']),'.3f')+'</p>'+embedded(path)
    scope=('This pilot uses the completed 720-image, 90-acquisition stratified diagnostic. '
           'It does not represent the full 360-assay cohort.' if args.mode=='pilot' else
           'All 360 assay acquisitions remain in the fixed-quarter denominator, including partial and empty acquisitions.')
    document='''<!doctype html><html><head><meta charset="utf-8"><title>'''+title+'''</title>
<style>body{font-family:"Times New Roman",Times,serif;background:white;color:#222;max-width:1150px;margin:35px auto;line-height:1.4}table{border-collapse:collapse;font-size:12px;margin:15px 0;width:100%}th,td{border:1px solid #bbb;padding:5px;text-align:left}th{background:#eee}img{max-width:100%;display:block;margin:15px 0}code{font-size:12px}h1,h2{font-weight:normal}</style></head><body><h1>'''+title+'''</h1>
<p>'''+scope+''' Architecture remains the main investigation; these camera-DN results are supporting observations.</p>
<p>We retained '''+str(len(availability))+''' planned quarter anchors and verified '''+str(verified)+''' per-anchor payloads. The run status is <strong>'''+html.escape(summary['status'])+'''</strong>. We recalculated every comparison denominator from this run.</p>
<h2>Registration and wrong-pair controls</h2><p>Moment scaling/translation and the original affine ECC alternative are compared on the RGB grid. ECC retains the moment transform when its source-grid IoU does not improve or optimization fails. Its original return status combines those fallback causes; the full run separately records optimizer convergence or the caught OpenCV error. We refit controls independently and also apply the nominal transform without refitting. The wrong-view control uses index(view+20) modulo40; no calibrated physical angle is assumed. The other-replicate control uses a different candidate identity. High overlap can occur for an incorrect pair and does not establish corresponding tissue.</p>
'''+table(summary['experiments'],['pair_kind','method','planned_anchor_pairs','executed','cross_support_usable','whole_iou_median'])+embedded(root/'figures/registration_controls.png')+'''
<p>The boxplots describe repeated technical views. The dashed line is the original 0.65 mask-IoU engineering gate. Pairwise contrasts below use matched anchors; no biological independence or significance is claimed.</p>
'''+table(comparisons,['method','control','paired_anchors','nominal_minus_control_median','nominal_greater_count'])+table([dict(pair_kind=k[0],optimizer_status=k[1],count=v) for k,v in sorted(optimizer_counts.items())],['pair_kind','optimizer_status','count'])+'''
<h2>Twenty-seven camera-DN definitions</h2><p>We used the original fifteen normalized-difference pairs, six RGB formulas and six multichannel formulas. No NoIR channel is designated a calibrated narrow-band NIR measurement. Each family has its own support and every qualifying pixel obeys the original dark, clipping and denominator guards. For NoIR affine interpolation we also require all contributing unregistered 512×385 processing-grid pixels to be valid. Dark/clipping fractions refer to that JPEG processing grid, not native-sensor clipping. Each summary requires at least 40 valid pixels. Cross-camera support is the intersection of eroded whole-plant silhouettes under the engineering gates; it is not anatomical correspondence.</p>
'''+embedded(root/'figures/descriptor_availability.png')+table(selected,['index','family','planned','executed','qualified','qualified_fraction_of_planned','median_valid_fraction'])+'''
<h2>Black-background examples</h2><p>Each strip shows RGB, registered NoIR, ND(rgb_R,noir_B), CAMSUM6, GLICONTRAST6 and ExG, with each definition's original fixed display range. Unsupported or invalid pixels remain black. We show the lowest and highest view-0 nominal ECC mask-IoU examples; this selection illustrates range, not typical anatomy.</p>
'''+example_html+'''
<h2>Retained availability failures</h2><p>Availability is also retained by treatment/visit and genotype, so symptom-dependent selection can be inspected. These counts refer to quarter anchors, not independent plants.</p>'''+table(coverage,['grouping','label','planned_quarters','nominal_ready','ECC_executed','cross_support_usable','fraction_cross_usable_of_planned'])+table([dict(reason=k,count=v) for k,v in missing.items()],['reason','count'])+'''
<h2>Reproducibility and limitations</h2><p>The CSV tables, index definitions, source snapshots, per-anchor source and mask hashes, configuration and report_verification.json are retained beside this report. Segmentation model, revision, checkpoint asset hashes and source operators are recorded in the upstream configurations preserved in the verification file. The portable pairing_README.md supplies the exact commands.</p>
<p>Processed JPEG values do not supply calibrated reflectance. Silhouette registration cannot identify matching leaves or validate correspondence of internal tissue. These results do not establish anatomy, physiological response, exact leaf count or independent band measurements. Forty angular views would be repeated technical observations, and this analysis samples only the four fixed quarters.</p>
</body></html>'''
    (root/'pairing_report.html').write_text(document,encoding='utf-8')
    print(json.dumps(dict(status='verified',anchors=verified,registrations=len(registration),
                          descriptors=len(descriptors),report=str(root/'pairing_report.html')),indent=2))


if __name__=='__main__':main()
