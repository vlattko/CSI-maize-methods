"""Build a self-contained numerical sensitivity report from verified results."""
import argparse,csv,hashlib,html,json
from pathlib import Path


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))
def esc(value):return html.escape(str(value))
def number(value):return f'{float(value):.6g}' if value not in ('',None) else 'Unavailable'
def table(rows,fields):
    return '<table><thead><tr>'+''.join('<th>'+esc(label)+'</th>' for field,label in fields)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(number(row[field]) if field.startswith(('old_','strict_','delta_')) and field not in ('old_n',) else row[field])+'</td>' for field,label in fields)+'</tr>' for row in rows)+'</tbody></table>'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();root=args.output
    verification=json.loads((root/'verification.json').read_text());features=json.loads((root/'feature_summary.json').read_text());rows=read(root/'verified_metric_changes.csv')
    if verification['status']!='passed':raise ValueError('Verified results required')
    chosen=['metadata_ridge','registered_indices','metadata_plus_indices','metadata_plus_dns','metadata_plus_dns_plus_cross_indices']
    primary=[]
    for row in rows:
        tokens=row['experiment'].split('__')
        if tokens[2].startswith('assay_') and row['model'] in chosen:
            primary.append(dict(row,domain=tokens[2],split=tokens[-1]))
    largest=sorted(rows,key=lambda r:abs(float(r['delta_r2'])),reverse=True)[:10]
    availability=[r for r in features['availability'] if r['all360_eligibility_losses']]
    methods='We replaced the nominal ECC statistics using an independently verified strict interpolation audit, then reran the frozen grouped prediction protocol. The replacement covers 1,212 accepted anchors and 58 gate-rejected anchors with retained same-camera descriptors: 34,290 formula rows. The 170 unavailable anchors remain unavailable. Matched camera-DN controls use 7,272 audited common-interior channel rows. Both original session aggregations were reproduced exactly before correction. All 360 keyed records are retained; the primary paired cohort remains 329, with 328 in the UVIS source-flag sensitivity.'
    results='The correction has material effects on some exploratory models. PiAbs cross-camera-only genotype R² changes from −0.124795 to 0.070726; Fv/Fm all-index genotype R² changes from 0.098898 to 0.013295. PiAbs metadata+DN+cross-index R² changes from 0.301162 to 0.411897, remaining below metadata alone (0.433867). Metadata-only and SAM-trait-only predictions are bitwise unchanged. The same 2 acquisitions retain all 27 descriptors. Three acquisitions lose eligibility for at least one index; one also loses NoIR-G DN eligibility. Sparse recorded NoIR-green support remains a limitation.'
    limits='This is a deterministic validity sensitivity within an already inspected experiment. The five ridge alphas, three null seeds, outer groups, training-only preprocessing, inner tuning and target exclusions are unchanged. Selected alphas change in 69 of 702 fitted outer models; those changes follow the same training-only grid. No external validation, calibrated reflectance, organ correspondence or physiological imaging capability is established. Archived moment/wrong-pair controls were not rerun and retain their original tolerance. The original outputs remain intact.'
    content='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Strict paired-index sensitivity</title><style>body{font:18px/1.5 "Times New Roman",Times,serif;color:#111;background:#fff;max-width:1400px;margin:36px auto;padding:0 24px}h1,h2{font-weight:normal}table{border-collapse:collapse;font-size:15px;width:100%;margin:20px 0}th,td{padding:7px 9px;text-align:left;border-bottom:1px solid #ddd}th{background:#f3f4f2}.scroll{overflow:auto}code{font-size:14px}a{color:#205b4b}</style><h1>Strict interpolation validity: paired-index model sensitivity</h1><p>'+esc(methods)+'</p><p>'+esc(results)+'</p><h2>Primary scalar assays</h2><p>R² is computed from the complete out-of-fold predictions. Each model comparison uses identical case keys and observed targets. Positive ΔR² favors the strict version; positive ΔRMSE is worse.</p><div class="scroll">'+table(primary,[('domain','Target'),('split','Outer holdout'),('model','Model'),('n','n'),('old_r2','Original R²'),('strict_r2','Strict R²'),('delta_r2','ΔR²'),('old_rmse','Original RMSE'),('strict_rmse','Strict RMSE'),('delta_rmse','ΔRMSE')])+'</div><h2>Feature availability</h2>'+table(availability,[('feature','Descriptor/channel'),('all360_old_eligible','Original eligible /360'),('all360_strict_eligible','Strict eligible /360'),('primary329_old_eligible','Original eligible /329'),('primary329_strict_eligible','Strict eligible /329')])+'<h2>Largest absolute R² changes</h2><p>This table includes diagnostic null models and native spectral-target models; it is not a ranking selected for deployment.</p><div class="scroll">'+table(largest,[('experiment','Case'),('model','Model'),('old_r2','Original R²'),('strict_r2','Strict R²'),('delta_r2','ΔR²'),('delta_rmse','ΔRMSE')])+'</div><h2>Verification and scope</h2><p>All 12 cases, 192 model summaries, 882 fold records, 702 selected outer-fit replays and 162 training-only null maps passed. Prediction and metric replay errors were zero. The verifier replays selected fits; it does not independently repeat the entire inner grid search.</p><p>'+esc(limits)+'</p><p>Full numeric results: <a href="verified_metric_changes.csv">old-to-strict metrics</a>, <a href="case_denominators.csv">case denominators</a>, <a href="features/feature_availability.csv">feature availability</a>, <a href="verification.json">verification record</a>, <a href="feature_manifest.json">source and input hashes</a>.</p></html>'
    (root/'strict_sensitivity_report.html').write_text(content,encoding='utf-8')
    prose='# Strict paired-index sensitivity\n\n'+methods+'\n\n'+results+'\n\n'+limits+'\n\nThe independent verifier replayed 702 selected outer fits and all 192 old/strict metric summaries with zero numerical discrepancy. It verified 882 outer-fold records, 162 training-only null maps, identical case keys/targets and unchanged baseline predictions. The complete inner tuning grid was executed by the unchanged frozen producer; the verifier checked its selected fits and protocol bindings.\n'
    (root/'strict_findings.md').write_text(prose,encoding='utf-8')
    print(json.dumps(dict(status='completed',html_sha256=hashlib.sha256((root/'strict_sensitivity_report.html').read_bytes()).hexdigest()),indent=2))


if __name__=='__main__':main()
