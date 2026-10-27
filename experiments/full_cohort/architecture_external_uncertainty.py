"""Decision-margin and wrong-physical-plant controls for supplied organ matching."""
import argparse,itertools,json
from pathlib import Path
import numpy as np
from architecture_analysis import read,write,dump,sha
from architecture_external_pheno4d import associate,feature_cost

def objective(cost,links,a,b,gap):return float(sum(cost[i,j] for i,j in links)+gap*(len(a)+len(b)))
def margins(cost,gap=.75):
    links,a,b=associate(cost,gap);baseline=objective(cost,links,a,b,gap);result=[]
    for i,j in links:
        altered=cost.copy();altered[i,j]=1e6;alternative,ua,ub=associate(altered,gap);result.append((i,j,objective(altered,alternative,ua,ub,gap)-baseline))
    return result

def run(args):
    out=args.run/'architecture'/'external_pheno4d';features=read(out/'anonymous_geometry_features.csv');truth=read(out/'predicted_links_and_scoring_truth.csv');lookup={}
    for r in features:lookup.setdefault((r['plant'],r['scheme'],r['date']),[]).append(r)
    arrays={k:np.array([[float(r[f'feature_{i}']) for i in range(12)] for r in sorted(rr,key=lambda r:int(r['anonymous_token']))]) for k,rr in lookup.items()}
    rows=[];controls=[]
    for plant,scheme in sorted(set((k[0],k[1]) for k in arrays)):
        a=arrays[(plant,scheme,'0324')];b=arrays[(plant,scheme,'0325')];cost=feature_cost(a,b);links,ua,ub=associate(cost,.75)
        for i,j,margin in margins(cost,.75):
            t=next(r for r in truth if r['plant']==plant and r['scheme']==scheme and r['scenario']=='original' and r['method']=='geometry_hungarian' and int(r['earlier_anonymous_token'])==i and int(r['later_anonymous_token'])==j)
            rows.append(dict(plant=plant,scheme=scheme,earlier_token=i,later_token=j,global_assignment_margin=margin,geometry_cost=float(cost[i,j]),correct=str(t['correct']).lower()=='true',interpretation='Alternative-assignment objective gap, not calibrated probability'))
        for other in sorted(set(k[0] for k in arrays)):
            if other==plant:continue
            bb=arrays[(other,scheme,'0325')];cc=feature_cost(a,bb);ll,x,y=associate(cc,.75);controls.append(dict(earlier_plant=plant,later_plant=other,scheme=scheme,physically_true_links=0,predicted_false_physical_links=len(ll),possible_links=min(len(a),len(bb)),mean_selected_cost=float(np.mean([cc[i,j] for i,j in ll])) if ll else None,interpretation='Different known physical plants; every accepted link is physically false. Shows that correspondence assumes continuity.'))
    write(out/'global_assignment_margins.csv',rows);write(out/'wrong_physical_plant_controls.csv',controls)
    summary=dict(known_true_plant_links=len(rows),correct_true_plant_links=sum(r['correct'] for r in rows),wrong_plant_pairs=len(controls),wrong_plant_false_physical_links=sum(r['predicted_false_physical_links'] for r in controls),wrong_plant_possible_links=sum(r['possible_links'] for r in controls),script_sha256=sha(__file__),scope='Wrong-plant controls deliberately violate the known-plant prerequisite. Costs/margins alone cannot establish CSI physical continuity.')
    summary['wrong_pair_annotation_comparisons']=len(controls)
    summary['unique_wrong_physical_ordered_pairs']=len({(r['earlier_plant'],r['later_plant']) for r in controls})
    for threshold in [0,.1,.25,.5]:
        selected=[r for r in rows if r['global_assignment_margin']>=threshold];summary[f'margin_atleast_{threshold}']=dict(links=len(selected),correct=sum(r['correct'] for r in selected))
    dump(out/'uncertainty_and_negative_controls.json',summary);print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);run(p.parse_args())
