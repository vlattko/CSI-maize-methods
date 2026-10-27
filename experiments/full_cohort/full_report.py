"""Self-contained full-data method report with a keyed longitudinal explorer."""
import argparse
import base64
from collections import defaultdict
import csv
import hashlib
import html
import io
import json
from pathlib import Path
from PIL import Image


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def load(path):return json.loads(path.read_text(encoding='utf-8'))


def number(x):
    try:
        value=float(x)
        return value if value==value and abs(value)!=float('inf') else None
    except (ValueError,TypeError):return None


def default_registry(root):
    """Find producer outputs so a fresh full run needs no manual HTML assembly."""
    rows=[dict(section='architecture',path='architecture/figures/directional_observability.png',
               caption='Directional dependence of accepted silhouettes across the cohort. Image width, height and projected area are observation traits; camera views are repeated technical observations.'),
          dict(section='architecture',path='architecture/figures/view_budget.png',
               caption='View-budget and phase sensitivity within the selected subset of 505 camera acquisitions with all forty masks accepted. Complete-view summaries provide the internal reference.'),
          dict(section='pairing',path='pairing/full/figures/registration_controls.png',
               caption='Nominal and deliberately mismatched camera pairs receive equal registration opportunities. These are whole-mask compatibility and engineering-support tests.'),
          dict(section='pairing',path='pairing/full/atlas/atlas_S_visit2.png',
               caption='Supporting spatial atlas with 27 channel formulas. Unsupported pixels remain black; formula-specific valid support and camera DN guards determine availability.')]
    registry=root/'prediction/report_figure_registry_prediction.json'
    if registry.exists():
        wanted={'incremental_paired_contrasts.png','paired_primary_genotype_mean_target_r2.png',
                'paired_primary_treatment_mean_target_r2.png','native_wavelength_errors.png','eligibility_selection.png'}
        for item in load(registry)['figures']:
            path=Path(item['path'])
            if path.name in wanted:
                resolved=path.resolve() if path.is_absolute() or path.is_file() else (root/path).resolve()
                relative=resolved.relative_to(root.resolve()).as_posix()
                rows.append(dict(section='prediction',path=relative,caption=item['caption']))
    return [r for r in rows if (root/r['path']).is_file()]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--intake',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--allow-pending',action='store_true')
    args=p.parse_args();root=args.run.resolve()
    summary=load(root/'extraction/summary.json')
    if summary['status']!='completed':raise ValueError('Finalized extraction required')
    stages={}
    for name,relative in [('Image models','prediction/summary_images.json'),
                          ('Registered-index models','prediction/summary_indices.json'),
                          ('Paired-camera analysis','pairing/full/summary.json'),
                          ('Architecture','architecture/architecture_pipeline_progress.json')]:
        path=root/relative
        stages[name]=load(path).get('status','unknown') if path.exists() else 'pending'
    if not args.allow_pending and any(v!='completed' for v in stages.values()):
        raise ValueError('Full report still has unfinished stages: '+str(stages))
    references=read(args.intake/'research_audit/reference_master.csv')
    sessions=read(root/'extraction/session_traits.csv')
    frames=read(root/'extraction/view_traits.csv')
    keys=('treatment','measurement','genotype','replicate')
    sample_key=lambda r:tuple(str(r[k]) for k in keys)
    traits={sample_key(r):r for r in sessions}
    if len(traits)!=360 or len(references)!=360:raise ValueError('Expected 360 unique keyed samples')
    frame_lookup={(sample_key(r),r['camera'],int(r['view'])):r for r in frames}
    if len(frame_lookup)!=len(frames):raise ValueError('Duplicate frame identity')
    samples=[];pictures={};image_count=0
    for r in references:
        sample=sample_key(r);feature=traits[sample]
        item={k:r[k] for k in keys};item['id']='|'.join(sample)
        item['targets']={t:number(r[t]) for t in ('ojip__Fv/Fm','ojip__Pi_Abs','nir__NDVI')}
        item['traits']={k:number(v) for k,v in feature.items() if k.startswith(('rgb__','noir__'))}
        item['coverage']={c:{'n':int(feature[c+'_usable_views']),'eligible':feature[c+'_eligible']=='True'} for c in ('rgb','noir')}
        item['profiles']={}
        for camera in ('rgb','noir'):
            item['profiles'][camera]=[]
            for view in range(40):
                row=frame_lookup.get((sample,camera,view))
                if not row:continue
                item['profiles'][camera].append([view,number(row.get('area_fraction')),row['qc_pass']=='True',row.get('qc_issues','')])
                if view not in (0,10,20,30):continue
                token=item['id']+'|'+camera+'|'+str(view)
                if row.get('preview') and (root/row['preview']).is_file():
                    with Image.open(root/row['preview']) as im:
                        im=im.convert('RGB');im.thumbnail((384,384));stream=io.BytesIO();im.save(stream,format='JPEG',quality=72)
                    pictures[token]={'src':'data:image/jpeg;base64,'+base64.b64encode(stream.getvalue()).decode('ascii'),
                                     'qc':row['qc_pass']=='True','reason':row.get('qc_issues','')}
                    image_count+=1
        samples.append(item)
    def figure(relative,caption):
        path=root/relative
        if not path.exists():return ''
        data=base64.b64encode(path.read_bytes()).decode('ascii')
        return f'<figure><img src="data:image/png;base64,{data}" alt="{html.escape(caption,quote=True)}"><figcaption>{html.escape(caption)}</figcaption></figure>'
    external=root/'architecture/external_pheno4d'
    audit=load(external/'external_denominator_audit.json')
    if audit['status']!='passed':raise ValueError('External denominator audit failed')
    scaffold=load(external/'scaffold/external_scaffold_summary.json')
    crossdate=load(external/'scaffold/automatic_scaffold_crossdate_summary.json')
    depth=load(root/'depth_trial/summary.json')
    depth_rows=read(root/'depth_trial/results.csv')
    mode_labels={'learned_pose_aligned_to_training_centres':'Learned poses aligned through training centres',
                 'nominal_f1.2_sign1':'Nominal f/W = 1.2, positive rotation',
                 'nominal_f1.2_sign-1':'Nominal f/W = 1.2, negative rotation',
                 'nominal_f2.0_sign1':'Nominal f/W = 2.0, positive rotation',
                 'nominal_f2.0_sign-1':'Nominal f/W = 2.0, negative rotation'}
    import numpy as np
    depth_table=''
    for camera in ('rgb','noir'):
        for mode in dict.fromkeys(r['mode'] for r in depth_rows):
            values=[float(r['heldout_iou']) for r in depth_rows if r['camera']==camera and r['mode']==mode
                    and r['status']=='completed' and r['heldout_mask_qc_pass']=='True']
            depth_table+=f'<tr><td>{camera.upper()}</td><td>{html.escape(mode_labels.get(mode,mode))}</td><td>{len(values)}</td><td>{np.median(values):.3f}</td></tr>' if values else ''
    stage_table=''.join(f'<tr><td>{html.escape(k)}</td><td>{html.escape(v)}</td></tr>' for k,v in stages.items())
    pairing_path=root/'pairing/full/summary.json'
    pair_text='Paired-camera calculations are pending.'
    if pairing_path.exists():
        paired=load(pairing_path)
        nominal=[r for r in paired.get('experiments',[]) if r['pair_kind']=='nominal' and r['method']=='ECC']
        if nominal:
            a=nominal[0];pair_text=f"The nominal ECC calculation executed {a['executed']} of {paired['planned_anchor_pairs']} planned quarter-view anchors. Median whole-silhouette IoU was {a['whole_iou_median']:.3f}; {a['cross_support_usable']} anchors passed the shared-support rule. These are registration and availability diagnostics."
    figure_registry=default_registry(root)
    registry=root/'report_figure_registry.json'
    if registry.exists():figure_registry=load(registry)
    prediction_figures=''.join(figure(r['path'],r['caption']) for r in figure_registry if r.get('section')=='prediction')
    architecture_figures=''.join(figure(r['path'],r['caption']) for r in figure_registry if r.get('section')=='architecture')
    intervention_figures=''.join(figure(r['path'],r['caption']) for r in figure_registry if r.get('section')=='interventions')
    pairing_figures=''.join(figure(r['path'],r['caption']) for r in figure_registry if r.get('section')=='pairing')
    data=json.dumps(dict(samples=samples,pictures=pictures),ensure_ascii=False,allow_nan=False,separators=(',',':')).replace('<','\\u003c')
    pending=any(v!='completed' for v in stages.values())
    banner='<p class="pending">Investigation in progress. The stage table identifies calculations still running; completed external tests and finalized images are available below.</p>' if pending else ''
    page=r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CSI-MAIZE: leaf trajectories and architectural experiments</title><style>
html{font-family:"Times New Roman",Times,serif;background:white;color:#171717;font-size:18px;line-height:1.5}body{max-width:1220px;margin:36px auto;padding:0 24px}h1{font-weight:normal;font-size:2.5rem;line-height:1.12;max-width:1050px}h2{font-weight:normal;font-size:1.65rem;border-top:1px solid #b7b7b7;padding-top:24px;margin-top:44px}h3{font-size:1.15rem}nav{display:flex;flex-wrap:wrap;gap:18px;padding:14px 0;border-bottom:1px solid #bbb}a{color:#275c44}.lead{max-width:1000px;font-size:1.12rem}.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:15px;margin:26px 0}.facts>div{border-top:2px solid #777;padding:10px}.facts strong{font-weight:normal;font-size:2rem;display:block}figure{margin:27px 0}figure img{max-width:100%;height:auto}figcaption,.small{font-size:.92rem;color:#444}table{border-collapse:collapse;width:100%;font-size:.93rem}td,th{padding:9px;text-align:left;border-bottom:1px solid #ccc}select,button{font:inherit;background:white;border:1px solid #999;padding:6px 9px;color:#111}.controls{display:flex;flex-wrap:wrap;gap:15px;align-items:center;margin:18px 0}.photo-scroll{overflow-x:auto}.photos{display:grid;grid-template-columns:repeat(5,minmax(145px,1fr));gap:8px;min-width:780px}.card{margin:0}.card .unavailable{display:grid;place-items:center;aspect-ratio:1024/769;background:#000;color:#ccc;font-size:.8rem;text-align:center}.card img{display:block;width:100%;aspect-ratio:1024/769;object-fit:contain;background:black}.card figcaption{padding:5px 2px;line-height:1.3;font-size:.8rem}.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:14px}.charts svg{width:100%;height:auto}.diagram{width:100%;height:auto}.pending{border-left:3px solid #a36218;padding:10px 18px}.methods{max-width:1040px}code{font-size:.76rem;overflow-wrap:anywhere}@media print{nav,.controls,button{display:none}figure{break-inside:avoid}}@media(max-width:700px){body{padding:0 14px}h1{font-size:2rem}}
</style></head><body>
<p>CSI-MAIZE · Full-data method investigation · Vlatko Galic and Maja Mazur</p>
<h1>Leaf trajectories and virtual architectural experiments from circular plant imaging</h1>
<p class="lead">We ask how much organ-level information repeated views can support, how uncertainty limits correspondence, and how a change in one organ alters the projected exposure of another. Paired RGB and longpass images provide observations of appearance; the architectural experiment holds its receiver and geometric support explicit.</p>
@@BANNER@@
<nav><a href="#method">Method</a><a href="#explorer">Plant trajectories</a><a href="#cohort">Rotation analyses</a><a href="#validation">Leaf validation</a><a href="#interventions">Architectural interventions</a><a href="#observation">RGB–NoIR observations</a><a href="#prediction">Performance prediction</a><a href="#depth">Depth experiment</a><a href="#record">Execution record</a></nav>
<div class="facts"><div><strong>@@FRAMES@@</strong>identified images processed</div><div><strong>360</strong>keyed assay/acquisition records</div><div><strong>@@PAIRED@@</strong>paired summaries eligible at the declared rule</div><div><strong>7</strong>external plants with true temporal organ labels</div></div>
<section id="method"><h2>The method separates observation, identity and intervention</h2>
<svg class="diagram" viewBox="0 0 1200 235" role="img" aria-label="Circular images feed foreground support, candidate trajectories, conditional geometry and fixed-receiver experiments, with independent component validation."><defs><marker id="arrow" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto"><path d="M0,0 L9,3.5 L0,7" fill="#555"/></marker></defs><g fill="white" stroke="#666"><rect x="8" y="20" width="210" height="95"/><rect x="250" y="20" width="210" height="95"/><rect x="492" y="20" width="210" height="95"/><rect x="734" y="20" width="210" height="95"/><rect x="976" y="20" width="215" height="95"/></g><g stroke="#555" marker-end="url(#arrow)"><path d="M218,66H248"/><path d="M460,66H490"/><path d="M702,66H732"/><path d="M944,66H974"/></g><g font-family="Times New Roman,serif" font-size="21" text-anchor="middle"><text x="113" y="55">Circular acquisition</text><text x="113" y="87" font-size="17">Recorded views and identity</text><text x="355" y="55">Foreground support</text><text x="355" y="87" font-size="17">Masks, failures and coverage</text><text x="597" y="55">Organ hypotheses</text><text x="597" y="87" font-size="17">Rotation and temporal links</text><text x="839" y="55">Conditional geometry</text><text x="839" y="87" font-size="17">Camera and shape alternatives</text><text x="1083" y="55">Architectural experiment</text><text x="1083" y="87" font-size="17">Fixed receiver, remove neighbour</text><text x="600" y="160" font-size="19">Validation: held-out views · known-answer scenes · externally labelled maize geometry</text><text x="600" y="201" font-size="18">RGB–NoIR descriptors and fluorescence targets form a supporting observation and prediction layer.</text></g></svg>
<p>We retain individual view failures and all 360 reference records. A primary camera summary requires at least 20 accepted views distributed across all four nominal angular quadrants; stricter 30- and 40-view analyses show selection sensitivity. Model score and seed agreement are operational checks. They do not measure anatomical segmentation accuracy.</p></section>
<section id="explorer"><h2>Inspect each recorded plant trajectory</h2><p>Choose a treatment, genotype and replicate label. Images use the same processing-image canvas across visits; they are not independently enlarged to fill each panel. These labels define candidate physical trajectories. Exact sampling ages and physical pot continuity require protocol confirmation.</p>
<div class="controls"><label>Treatment <select id="treatment"><option value="C">Control</option><option value="P">Priming</option><option value="S">Stress</option></select></label><label>Genotype <select id="genotype"></select></label><label>Replicate <select id="replicate"><option>1</option><option>2</option><option>3</option><option>4</option></select></label><label>Nominal view <select id="view"><option value="0">0</option><option value="10">10</option><option value="20">20</option><option value="30">30</option></select></label><button id="download">Export selected measurements</button></div>
<h3>RGB foreground candidates</h3><div class="photo-scroll"><div class="photos" id="rgbphotos"></div></div><h3>Recorded NoIR foreground candidates</h3><div class="photo-scroll"><div class="photos" id="noirphotos"></div></div>
<p class="small">A review flag marks a failed mask check. A missing image remains missing. Image colour is camera DN; skeleton endpoints and visible path hypotheses remain distinct from true leaf number or identity.</p><div class="charts" id="trajectorycharts"></div>
<div class="controls"><label>Rotation profile at visit <select id="profilevisit"><option>1</option><option selected>2</option><option>3</option><option>4</option><option>5</option></select></label></div><div class="charts" id="profiles"></div></section>
<section id="cohort"><h2>What all recorded rotations support</h2><p>The finalized cohort contains 681 observed camera acquisitions among 720 reference camera rows. Of these, 668 meet the primary coverage rule and support 5,344 harmonic descriptor fits. We retain 144 candidate camera trajectories; 105 have accepted primary summaries at all five visits. These are repeated-label trajectories, pending independent confirmation of physical pot continuity.</p><p>The angular sampling experiment includes 505 camera acquisitions with all 40 masks accepted. Its 646,400 phase and view-budget rows describe the consequences of subsampling this selected population. In the RGB terminal experiment, 339 acquisitions yield 377 candidate tracks. Of 3,721 withheld track-view rows, 2,495 meet the positional gate and 3,523 contain an observed terminal. This is image compatibility, not anatomical leaf accuracy.</p>@@ARCHITECTURE_FIGURES@@</section><section id="validation"><h2>Leaf correspondence with independent organ labels</h2><p>We tested the rotation-to-scaffold component on 14 original Pheno4D scans from seven maize plants, using the last two labelled scans of each plant. We rendered known geometry and camera poses to obtain silhouettes. Organ identities were withheld from hypothesis generation and fitting, then used to score frozen models.</p>
<p>The method recovered @@TRACKS@@ distinct instance-date regions among @@OBSERVABLE@@ regions with sufficient training-view terminal support, out of @@TRUEINSTANCES@@ true instance-date regions. High agreement among retained tracks therefore coexists with substantial missing coverage. The known render camera and supplied foreground are explicit assistance.</p>
@@SCAFFOLD_FIG@@
<p>When automatically recovered scaffold points were joined across dates, @@CORRECT@@ of @@LINKS@@ predicted links were correct. This recovers @@CORRECT@@ of @@COMMON@@ true common instances overall. A separate wrong-physical-plant control demonstrated that plausible geometric matches can occur between different plants. Physical plant identity must therefore come from experimental metadata.</p>
<p>Point-support distances describe proximity to observed labelled regions. They are not errors against independently annotated anatomical leaf tips. Neither the external render nor the scanner validates an uncalibrated reconstruction from the CSI camera images.</p></section>
<section id="interventions"><h2>Track how one neighbour changes another organ's exposure</h2><p>For each receiver region and viewing direction, we retain its potential projected footprint and change one neighbour. We calculate the difference in visible ownership on the same grid. The external reference permits true-ID joins between dates and separates the intervention at a fixed date from the combined shape and arrangement changes between dates.</p>
@@INTERVENTION_FIGURES@@
@@EXPOSURE_FIG@@
@@POLAR_FIG@@
<p>The audited experiment contains @@EDGES@@ fixed-shape removal edges. We retain both maize annotation conventions: the leaf-tip convention includes an associated stem section, whereas the collar convention has a separate stem that remains present during blade removals. Resolution and point-footprint sensitivity accompany the directional effects. These quantities measure geometric projected exposure; physical area, absorbed radiation and photosynthesis require additional calibration and modelling.</p></section>
<section id="observation"><h2>RGB–NoIR registration and the spatial descriptor layer</h2><p>@@PAIRTEXT@@</p><p>We compare moment initialization, ECC and refitted/fixed wrong-view and wrong-plant controls. Each of the 27 formulas retains its own valid support, dark/clipping exclusions and accepted pixel counts. Formulas requiring the often-dark NoIR green output can remain unavailable. The formulas transform recorded channels; they do not create independent narrow spectral bands.</p>
@@PAIRING_FIGURES@@</section>
<section id="prediction"><h2>Test image information against experimental context</h2><p>We compare image traits, frozen DINOv2 descriptors and registered channel formulas with treatment/visit controls on the same eligible rows. Preprocessing and tuning remain inside training folds. Held-out genotype and held-out treatment answer different questions, and complete-case selection is reported against the original assay population.</p>
@@PREDICTION_FIGURES@@
<p>The original all-assay treatment-by-visit baseline reached held-out-genotype R² of 0.560 for Fv/Fm and 0.433 for PIABS without images. Direct image comparisons use their own matched cohort. We report paired changes in out-of-fold error and exploratory genotype-cluster uncertainty rather than treating pixels or camera views as biological replicates.</p></section>
<section id="depth"><h2>Learned depth under explicit camera hypotheses</h2><p>We ran DA3-SMALL on three fixed quarter views, keeping index 30 out of the input. We compared inferred poses aligned through training camera centres with four nominal turntable hypotheses: two focal-length ratios and both rotation signs. All hypotheses remain in the record, including unsuccessful projections; no held-out score selected the camera.</p><table><thead><tr><th>Camera</th><th>Declared hypothesis</th><th>Usable cases</th><th>Median held-out mask IoU</th></tr></thead><tbody>@@DEPTH_TABLE@@</tbody></table><p>The depth experiment executed @@DEPTH_RUN@@ configurations and skipped @@DEPTH_SKIP@@ configurations because training-mask checks failed. These are learned relative geometry and foreground-consistency tests under unmeasured camera assumptions. They do not establish calibrated anatomical reconstruction.</p></section>
<section id="record"><h2>Execution and scientific scope</h2><table><thead><tr><th>Analysis</th><th>Recorded status</th></tr></thead><tbody>@@STAGES@@</tbody></table><p>We retain source hashes, checkpoint identities, per-frame masks, accepted/rejected views, complete out-of-fold predictions and experiment logs. The original manuscript bundle remains the reference demonstration until the full-data analysis has been reviewed. Remaining protocol details are the exact cold/priming schedule, physical pot continuity, destructive-assay crosswalk and camera calibration.</p><p>Data and reference outputs will be made available via Zenodo after publication. Methods code: <a href="https://github.com/vlattko/CSI-maize-methods">github.com/vlattko/CSI-maize-methods</a>.</p><p>Primary method references: <a href="https://doi.org/10.1371/journal.pone.0256340">Pheno4D</a>, <a href="https://doi.org/10.1186/s13007-022-00961-4">PhenoTrack3D</a>, <a href="https://github.com/ByteDance-Seed/Depth-Anything-3">Depth Anything 3</a>. External input files remain separate from the original study archive.</p></section>
<script id="cohortdata" type="application/json">@@DATA@@</script><script>
'use strict';const D=JSON.parse(document.getElementById('cohortdata').textContent);const byId=new Map(D.samples.map(s=>[s.id,s]));const $=id=>document.getElementById(id);const genes=[...new Set(D.samples.map(s=>s.genotype))];for(const g of genes){const o=document.createElement('option');o.value=g;o.textContent=g;$('genotype').append(o)}$('genotype').value='Os6251/15';
function selected(){return [1,2,3,4,5].map(v=>byId.get([$('treatment').value,String(v),$('genotype').value,$('replicate').value].join('|')))}
function svgChart(title,series,ymin,ymax,xmax=5){const w=360,h=220,l=47,r=12,t=32,b=38;const xx=x=>l+(x-(xmax===5?1:0))*(w-l-r)/(xmax===5?4:39);const yy=y=>h-b-(y-ymin)*(h-t-b)/(ymax-ymin||1);let s='<svg viewBox="0 0 '+w+' '+h+'" role="img" aria-label="'+title+'"><text x="'+l+'" y="19" font-size="15">'+title+'</text>';for(let i=0;i<4;i++){const y=ymin+(ymax-ymin)*i/3;s+='<path d="M'+l+','+yy(y)+'H'+(w-r)+'" stroke="#ddd"/><text x="'+(l-5)+'" y="'+(yy(y)+4)+'" text-anchor="end" font-size="11">'+y.toFixed(2)+'</text>'}for(const x of (xmax===5?[1,2,3,4,5]:[0,10,20,30,39]))s+='<text x="'+xx(x)+'" y="'+(h-16)+'" text-anchor="middle" font-size="11">'+x+'</text>';for(const group of series){let segment=[];for(const p of group.points){if(p[1]===null){if(segment.length)s+='<polyline points="'+segment.join(' ')+'" fill="none" stroke="'+group.color+'" stroke-width="2"/>';segment=[];continue}const x=xx(p[0]),y=yy(p[1]);segment.push(x+','+y);s+='<circle cx="'+x+'" cy="'+y+'" r="'+(p[2]===false?3.5:2.5)+'" fill="'+(p[2]===false?'#a1382f':group.color)+'"/>'}if(segment.length)s+='<polyline points="'+segment.join(' ')+'" fill="none" stroke="'+group.color+'" stroke-width="1.5"/>'}return s+'</svg>'}
function render(){const ss=selected(),v=$('view').value;for(const cam of ['rgb','noir']){const panel=$(cam+'photos');panel.replaceChildren();for(let i=0;i<5;i++){const s=ss[i],f=document.createElement('figure');f.className='card';const img=document.createElement('img');img.alt=(s?s.id:'Missing sample')+' '+cam+' view '+v;const picture=s?D.pictures[s.id+'|'+cam+'|'+v]:null;let visual=img;if(picture){img.src=picture.src}else{visual=document.createElement('div');visual.className='unavailable';visual.textContent='Image unavailable';visual.setAttribute('role','img');visual.setAttribute('aria-label',img.alt+' unavailable')};const c=document.createElement('figcaption');c.textContent='Visit '+(i+1)+' · '+(picture?(picture.qc?'QC pass':'Review: '+picture.reason):'Image unavailable')+(s?' · '+s.coverage[cam].n+'/40 accepted':'');f.append(visual,c);panel.append(f)}}const charts=$('trajectorycharts');charts.innerHTML='';for(const [target,title,min,max] of [['ojip__Fv/Fm','Measured Fv/Fm',0,1],['ojip__Pi_Abs','Measured PIABS',0,4.5],['nir__NDVI','Instrument NIR NDVI',.2,.85]]){const el=document.createElement('div');el.innerHTML=svgChart(title,[{color:'#6550a1',points:ss.map((s,i)=>[i+1,s?s.targets[target]:null])}],min,max);charts.append(el)}const el=document.createElement('div');el.innerHTML=svgChart('SAM silhouette area fraction',['rgb','noir'].map((c,i)=>({color:i?'#395fa6':'#287750',points:ss.map((s,i)=>[i+1,s?s.traits[c+'__area_fraction__median']:null])})),0,.15);charts.append(el);renderProfiles()}
function renderProfiles(){const ss=selected(),s=ss[Number($('profilevisit').value)-1];$('profiles').innerHTML='';for(const cam of ['rgb','noir']){const a=s?s.profiles[cam]:[],values=a.filter(p=>p[1]!==null).map(p=>p[1]),max=Math.max(.02,...values)*1.05;const el=document.createElement('div');el.innerHTML=svgChart(cam.toUpperCase()+' area by nominal view',[{color:cam==='rgb'?'#287750':'#395fa6',points:a}],0,max,40);$('profiles').append(el)}}
for(const id of ['treatment','genotype','replicate','view'])$(id).addEventListener('change',render);$('profilevisit').addEventListener('change',renderProfiles);$('download').addEventListener('click',()=>{const fields=['treatment','measurement','genotype','replicate','Fv_Fm','Pi_Abs','instrument_NIR_NDVI','rgb_accepted_views','noir_accepted_views'];const rows=selected().filter(Boolean).map(s=>[s.treatment,s.measurement,s.genotype,s.replicate,s.targets['ojip__Fv/Fm'],s.targets['ojip__Pi_Abs'],s.targets['nir__NDVI'],s.coverage.rgb.n,s.coverage.noir.n]);const csv=[fields,...rows].map(row=>row.map(v=>'"'+String(v??'').replaceAll('"','""')+'"').join(',')).join('\r\n');const u=URL.createObjectURL(new Blob(['\ufeff'+csv],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=u;a.download='CSI_maize_selected_candidate.csv';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)});render();
</script></body></html>'''
    replacements=dict(BANNER=banner,FRAMES=f"{summary['frames']:,}",PAIRED=str(summary['paired_eligible']),
        TRACKS=str(audit['recovered_unique_true_instance_dates']),OBSERVABLE=str(audit['observable_true_instance_dates']),
        TRUEINSTANCES=str(audit['true_instance_dates']),CORRECT=str(crossdate['correct_crossdate_links']),
        LINKS=str(crossdate['predicted_crossdate_links']),COMMON=str(crossdate['all_true_common_instances']),
        EDGES=f"{audit['primary_exposure_edges']:,}",PAIRTEXT=pair_text,DEPTH_TABLE=depth_table,
        DEPTH_RUN=str(depth['successful']),DEPTH_SKIP=str(depth['skipped']),STAGES=stage_table,DATA=data,
        SCAFFOLD_FIG=figure('architecture/external_pheno4d/scaffold/terminal_scaffold_known_geometry.png','Frozen terminal-scaffold points under the known orthographic render gauge. Evaluation uses observed point support and true instance labels; it does not supply anatomical tip coordinates.'),
        EXPOSURE_FIG=figure('architecture/external_pheno4d/exposure/neighbour_edges_leaf_collar.png','Directional neighbour effects under the collar annotation convention. Each removal retains the receiver and its reference footprint. Stem geometry remains present.'),
        POLAR_FIG=figure('architecture/external_pheno4d/exposure/persistent_receiver3_removed4_polar.png','An ID-preserving directional neighbour comparison between the two labelled scans. Cross-date differences combine shape and arrangement changes.'),
        ARCHITECTURE_FIGURES=architecture_figures,INTERVENTION_FIGURES=intervention_figures,PREDICTION_FIGURES=prediction_figures,PAIRING_FIGURES=pairing_figures)
    for k,v in replacements.items():page=page.replace('@@'+k+'@@',v)
    if '@@' in page:raise ValueError('Unresolved report placeholder')
    destination=root/'CSI_maize_full_data_methods.html';destination.write_text(page,encoding='utf-8')
    result=dict(status='draft_pending_stages' if pending else 'completed',frames=len(frames),samples=len(samples),
                embedded_quarter_previews=image_count,html_bytes=destination.stat().st_size,
                html_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),stages=stages)
    (root/'full_report_build.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
