"""Summarise frozen-tolerance deltas without modifying the primary audit output."""
import argparse,csv,json
from collections import Counter
from pathlib import Path

def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))

def maximum(rows,key):
    values=[abs(float(r[key])) for r in rows if r.get(key) not in ('',None)]
    return max(values,default=None)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--audit',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise ValueError('Use a new summary path')
    rows=read(args.audit/'formula_changes.csv');channels=read(args.audit/'channel_changes.csv');borders=read(args.audit/'field_border_diagnostic.csv')
    by_formula=[]
    for index in sorted({r['index'] for r in rows}):
        rr=[r for r in rows if r['index']==index]
        by_formula.append(dict(index=index,rows=len(rr),old_qualified=sum(r['old_qualified']=='True' for r in rr),strict_qualified=sum(r['strict_qualified']=='True' for r in rr),valid_pixels_lost=sum(int(r['lost_valid_pixels']) for r in rr),changed_count_rows=sum(int(r['lost_valid_pixels'])>0 for r in rr),changed_median_rows=sum(r['delta_median']!='' and float(r['delta_median'])!=0 for r in rr),maximum_absolute_statistic_delta={k:maximum(rr,'delta_'+k) for k in ('median','q10','q25','q75','q90','iqr')}))
    by_channel=[]
    for channel,scope in sorted({(r['channel'],r['scope']) for r in channels}):
        rr=[r for r in channels if r['channel']==channel and r['scope']==scope]
        deltas={key:[abs(float(r['strict_'+key])-float(r['old_'+key])) for r in rr if r['old_'+key]!='' and r['strict_'+key]!=''] for key in ('median','iqr')}
        by_channel.append(dict(channel=channel,scope=scope,rows=len(rr),old_valid_pixels=sum(int(r['old_valid_pixels']) for r in rr),strict_valid_pixels=sum(int(r['strict_valid_pixels']) for r in rr),lost_valid_pixels=sum(int(r['lost_valid_pixels']) for r in rr),old_qualified=sum(r['old_qualified']=='True' for r in rr),strict_qualified=sum(r['strict_qualified']=='True' for r in rr),maximum_absolute_median_delta=max(deltas['median'],default=None),maximum_absolute_iqr_delta=max(deltas['iqr'],default=None)))
    changed=[r for r in rows if int(r['lost_valid_pixels'])>0]
    output=dict(formula_rows=len(rows),changed_formula_rows=len(changed),anchors_with_any_formula_count_change=len({r['record_file'] for r in changed}),maximum_formula_row_pixel_loss=max((int(r['lost_valid_pixels']) for r in rows),default=0),qualification_transitions=Counter((r['old_qualified']+' to '+r['strict_qualified']) for r in rows if r['old_qualified']!=r['strict_qualified']),formula_summaries=by_formula,channel_summaries=by_channel,field_border=dict(anchors=len(borders),anchors_with_field_difference=sum(int(r['differing_field_pixels'])>0 for r in borders),field_differing_pixels=sum(int(r['differing_field_pixels']) for r in borders),field_differing_common_pixels=sum(int(r['differing_field_in_common_pixels']) for r in borders)),note='Counts across formulas/channels are not counts of unique physical pixels. Inputs and operator are the frozen primary audit; this file only summarises its explicit paired rows.')
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False),encoding='utf-8');print(json.dumps({k:v for k,v in output.items() if k not in ('formula_summaries','channel_summaries')},indent=2))

if __name__=='__main__':main()
