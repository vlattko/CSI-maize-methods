"""Build an offline white/Times research intake report from recorded evidence."""
from pathlib import Path
from collections import Counter
import argparse
import base64
import csv
import html
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--intake',type=Path,required=True)
    args=p.parse_args();root=args.intake
    summary=json.loads((root/'intake_summary.json').read_text(encoding='utf-8'))
    diagnostic=json.loads((root/'segmentation_triage/summary.json').read_text(encoding='utf-8'))
    if diagnostic['status']!='completed':raise ValueError('Diagnostic replay has not completed')
    source=read(root/'research_audit/reference_master.csv')
    qc=read(root/'session_decode_qc.csv')
    views=read(root/'segmentation_triage/view_diagnostics.csv')
    prompted=read(root/'segmentation_prompted/view_diagnostics.csv')
    if len(prompted)!=len(views):raise ValueError('Prompt comparison is incomplete')
    prompt_by_source={r['source']:r for r in prompted}
    assert set(prompt_by_source)=={r['source'] for r in views}
    transitions=Counter((r['qc_pass']=='True',prompt_by_source[r['source']]['qc_pass']=='True') for r in views)
    prompt_pass=sum(r['qc_pass']=='True' for r in prompted)
    baseline_metrics=read(root/'metadata_baselines/metrics.csv')
    figures=root/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
                         'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                         'savefig.facecolor':'white','figure.facecolor':'white'})
    names={'C':'Control','P':'Priming','S':'Stress'}
    colors={'C':'#237153','P':'#a65c08','S':'#6752a2'}
    def export(fig,name):
        fig.savefig(figures/f'{name}.png',dpi=190,bbox_inches='tight')
        fig.savefig(figures/f'{name}.pdf',bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,3.5),layout='constrained')
    values=np.array([[next(r['complete_40_pair_sessions'] for r in summary['coverage']
                          if r['treatment']==t and r['measurement']==v)
                      for v in range(1,6)] for t in 'CPS'])
    im=ax.imshow(values,vmin=0,vmax=24,cmap='YlGn',aspect='auto')
    for i in range(3):
        for j in range(5):ax.text(j,i,f'{values[i,j]}/24',ha='center',va='center',color='white' if values[i,j]>16 else 'black',fontsize=15)
    ax.set(xticks=range(5),xticklabels=[f'Visit {v}' for v in range(1,6)],
           yticks=range(3),yticklabels=[names[t] for t in 'CPS'],
           title='Acquisitions with all 40 indexed RGB–NoIR pairs fully decodable')
    fig.colorbar(im,ax=ax,label='Complete acquisitions');export(fig,'01_acquisition_coverage')
    # Measured variables, never image predictions. Each point is one assay row.
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    for ax,target,title in zip(axes,['ojip__Fv/Fm','ojip__Pi_Abs'],['Fv/Fm','PIABS (workbook Pi_Abs)']):
        for n,t in enumerate('CPS'):
            means=[]
            for visit in range(1,6):
                rows=[r for r in source if r['treatment']==t and int(r['measurement'])==visit]
                vals=np.array([float(r[target]) for r in rows]);means.append(vals.mean())
                jitter=np.linspace(-.065,.065,len(vals))
                ax.scatter(visit+(n-1)*.17+jitter,vals,s=13,alpha=.42,color=colors[t],linewidths=0)
            ax.plot(np.arange(1,6)+(n-1)*.17,means,'o-',color=colors[t],label=names[t],lw=2,ms=5)
        ax.set(xticks=range(1,6),xlabel='Recorded visit',ylabel=title,title=title)
        ax.grid(axis='y',alpha=.18)
    axes[0].set_ylim(0,1);axes[0].legend(frameon=False)
    fig.suptitle('Measured chlorophyll-fluorescence performance: 24 assay records per group and visit')
    export(fig,'02_measured_performance')
    # All candidate identities remain visible, including empty and damaged visits.
    fig,axes=plt.subplots(1,3,figsize=(11.5,9),layout='constrained')
    gen_order=['Os024445','Mo17','B73','Os7798/2','Os6251/15','Os622448']
    for ax,t in zip(axes,'CPS'):
        plantkeys=[(g,str(rep)) for g in gen_order for rep in range(1,5)]
        matrix=np.zeros((24,5),int)
        for i,(g,r) in enumerate(plantkeys):
            for v in range(1,6):
                s=next(s for s in qc if s['treatment']==t and s['genotype']==g and s['replicate']==r and s['measurement']==str(v))
                matrix[i,v-1]=int(s['paired_decodable_views'])
        ax.imshow(matrix,vmin=0,vmax=40,cmap='YlGn',aspect='auto')
        for i in range(24):
            for j in range(5):
                if matrix[i,j]!=40:ax.text(j,i,str(matrix[i,j]),ha='center',va='center',fontsize=8)
        ax.set(xticks=range(5),xticklabels=range(1,6),xlabel='Visit',title=names[t],
               yticks=range(24),yticklabels=[f'L{i//4+1} · rep {i%4+1}' for i in range(24)])
        ax.tick_params(axis='y',labelsize=8)
    fig.suptitle('Available paired views for 72 repeated-label candidate trajectories\nDark green = 40 intact pairs; numbers mark incomplete visits. Physical continuity awaits protocol confirmation.')
    export(fig,'03_candidate_trajectory_coverage')
    # Diagnostic outcomes are proportions of operator QC acceptance, not accuracy.
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.2),layout='constrained')
    diagnostic_table=[]
    for ax,cam in zip(axes,['rgb','noir']):
        matrix=np.zeros((3,5))
        for i,t in enumerate('CPS'):
            for j in range(5):
                rs=[r for r in views if r['camera']==cam and r['treatment']==t and r['measurement']==str(j+1)]
                good=sum(r['qc_pass']=='True' for r in rs)
                matrix[i,j]=good/len(rs)
                diagnostic_table.append(dict(treatment=t,visit=j+1,camera=cam,images=len(rs),qc_pass=good))
                ax.text(j,i,f'{good}/{len(rs)}',ha='center',va='center',color='white' if good/len(rs)>.7 else 'black',fontsize=11)
        ax.imshow(matrix,vmin=0,vmax=1,cmap='YlGn',aspect='auto')
        ax.set(title=cam.upper(),xticks=range(5),xticklabels=range(1,6),xlabel='Visit',
               yticks=range(3),yticklabels=[names[t] for t in 'CPS'])
    fig.suptitle('Published SAM foreground operators: automated QC acceptance\nSix genotype strata per treatment/visit; four fixed views per camera. No annotated accuracy claim.')
    export(fig,'04_segmentation_diagnostic')
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    labels=[];a=[];b=[]
    for t in 'CPS':
        for cam in ['rgb','noir']:
            labels.append(f'{names[t]}\n{cam.upper()}')
            a.append(sum(r['qc_pass']=='True' for r in views if r['treatment']==t and r['camera']==cam))
            b.append(sum(r['qc_pass']=='True' for r in prompted if r['treatment']==t and r['camera']==cam))
    x=np.arange(6)
    ax.bar(x-.18,a,.36,color='#2a7557',label='Box only')
    ax.bar(x+.18,b,.36,color='#a8682b',label='Box and point prompts')
    for i,(av,bv) in enumerate(zip(a,b)):
        ax.text(i-.18,av+1,str(av),ha='center',fontsize=10)
        ax.text(i+.18,bv+1,str(bv),ha='center',fontsize=10)
    ax.set(xticks=x,xticklabels=labels,ylim=(0,136),ylabel='Images passing operator QC (out of 120)',
           title='The existing point-prompt alternative does not improve QC acceptance uniformly')
    ax.legend(frameon=False,loc='lower left');export(fig,'07_prompt_comparison')
    fig,axes=plt.subplots(1,3,figsize=(12,4.5),layout='constrained')
    models=['training_mean','visit_only','treatment_only','treatment_visit']
    for ax,target,title in zip(axes,['ojip__Fv/Fm','ojip__Pi_Abs','nir__NDVI'],['Fv/Fm','PIABS','Instrument NIR NDVI']):
        vals=[float(next(r['r2'] for r in baseline_metrics if r['model']==m and r['target']==target)) for m in models]
        ax.bar(range(4),vals,color=['#999999','#b4c8bd','#9ab6ab','#276b51'])
        ax.axhline(0,color='black',lw=.7)
        for i,v in enumerate(vals):ax.text(i,v+(.025 if v>=0 else -.025),f'{v:.3f}',ha='center',va='bottom' if v>=0 else 'top',fontsize=10)
        ax.set(xticks=range(4),xticklabels=['Mean','Visit','Treatment','Treatment\n× visit'],ylim=(-.17,.68),
               title=title,ylabel='Pooled out-of-fold R²')
        ax.tick_params(axis='x',labelsize=9)
    fig.suptitle('Assay prediction without images: six held-out-genotype folds, 360 records')
    export(fig,'08_metadata_baselines')
    with (root/'segmentation_triage/stratum_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(diagnostic_table[0]));w.writeheader();w.writerows(diagnostic_table)
    # Compare the same recorded genotype/replicate across all visits and treatments.
    for camera in ['rgb','noir']:
        canvas=Image.new('RGB',(5*290,3*310+70),'white');draw=ImageDraw.Draw(canvas)
        title_font=ImageFont.truetype('C:/Windows/Fonts/times.ttf',22)
        label_font=ImageFont.truetype('C:/Windows/Fonts/times.ttf',17)
        draw.text((15,6),f'{camera.upper()}: genotype L5 (Os6251/15), recorded replicate 1; SAM candidate foregrounds',fill='black',font=title_font)
        draw.text((15,36),'Crops are resized separately: image size is not a growth measure. Organ identities are unlabelled.',fill='black',font=label_font)
        for i,t in enumerate('CPS'):
            for j in range(5):
                rs=[r for r in views if r['camera']==camera and r['treatment']==t and r['measurement']==str(j+1)
                    and r['genotype']=='Os6251/15' and r['replicate']=='1' and r['view']=='0']
                x,y=j*290,i*310+70
                draw.rectangle((x,y,x+286,y+277),fill='black')
                label=f'{names[t]} · visit {j+1}'
                if rs and rs[0].get('preview'):
                    row=rs[0]
                    image=Image.open(root/row['preview']).convert('RGB')
                    mask=Image.open(root/row['mask']).getbbox()
                    if mask:
                        b=(max(0,mask[0]-12),max(0,mask[1]-12),min(image.width,mask[2]+12),min(image.height,mask[3]+12))
                        image=image.crop(b)
                    image.thumbnail((276,265))
                    canvas.paste(image,(x+(286-image.width)//2,y+(277-image.height)//2))
                    label+=' · QC '+('pass' if row['qc_pass']=='True' else 'review')
                else:label+=' · no diagnostic preview'
                draw.text((x+7,y+284),label,fill='black',font=label_font)
        canvas.save(figures/f'05_{camera}_candidate_foregrounds.png')
    # Source/mask comparison for adverse diagnostics, rather than only accepted examples.
    failures=[r for r in views if r['qc_pass']!='True' and r.get('preview')]
    selected=[]
    for needle in ['capture_session_S2_3-1','capture_session_C4_2-1']:
        selected.append(next(r for r in failures if r['session'].endswith(needle) and r['camera']=='noir' and r['view']=='0'))
    canvas=Image.new('RGB',(1400,2*440+60),'white');draw=ImageDraw.Draw(canvas)
    for x,title in [(20,'Source image'),(485,'Mask boundary'),(950,'Candidate foreground')]:
        draw.text((x,10),title,fill='black',font=title_font)
    for i,row in enumerate(selected):
        with Image.open(root/row['source']) as original:
            original.draft('RGB',(1024,1024));original.load()
            source_image=original.convert('RGB');source_image.thumbnail((1024,1024))
        a=np.array(source_image);mask=np.array(Image.open(root/row['mask']))>0
        edge=mask.copy();edge[1:,:]&=mask[:-1,:];edge[:-1,:]&=mask[1:,:]
        edge[:,1:]&=mask[:,:-1];edge[:,:-1]&=mask[:,1:];edge=mask&~edge
        overlay=a.copy();overlay[edge]=[255,90,90]
        for j,panel in enumerate([source_image,Image.fromarray(overlay),Image.open(root/row['preview']).convert('RGB')]):
            panel.thumbnail((455,350));canvas.paste(panel,(j*465+7,i*440+50))
        label=Path(row['session']).name+' · NoIR view 0 · '+row['qc_issues']
        draw.text((15,i*440+410),label,fill='black',font=label_font)
    canvas.save(figures/'06_failed_segmentation_review.png')
    def image(name,caption):
        data=base64.b64encode((figures/name).read_bytes()).decode('ascii')
        return f'<figure><img alt="{html.escape(caption,quote=True)}" src="data:image/png;base64,{data}"><figcaption>{html.escape(caption)}</figcaption></figure>'
    cells=''.join('<tr>'+''.join(f'<td>{html.escape(str(r[k]))}</td>' for k in
                 ['treatment','measurement','expected_records','complete_40_pair_sessions','paired_decodable_views'])+'</tr>' for r in summary['coverage'])
    reasons=Counter(reason for r in views for reason in r.get('qc_issues','').split(';') if reason)
    zero_bytes=sum(int(r['bytes'])==0 for r in read(root/'jpeg_decode_failures.csv'))
    bad=read(root/'jpeg_decode_failures.csv')
    failed_nonempty=''.join(f"<li><code>{html.escape(r['path'])}</code>: {html.escape(r['decode_error'])}</li>" for r in bad if int(r['bytes'])>0)
    metadata=json.loads((root/'workbook_cell_comparison.json').read_text(encoding='utf-8'))
    assert len(metadata)==12 and all(r['same_sheet_names_and_cell_values'] for r in metadata)
    full_mask_count=sum(r['qc_pass']=='True' for r in views)
    assert len(views)==720==diagnostic['images']
    assert full_mask_count==diagnostic['qc_pass']
    text=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CSI-MAIZE: full archive intake and analysis priorities</title>
<style>html{{background:white;color:#171717;font-family:"Times New Roman",Times,serif;font-size:18px;line-height:1.55}}body{{max-width:1100px;margin:38px auto;padding:0 24px}}h1{{font-weight:normal;font-size:2.3rem;line-height:1.12;max-width:900px}}h2{{font-weight:normal;border-top:1px solid #ccc;padding-top:22px;margin-top:40px}}h3{{font-size:1.1rem}}nav{{display:flex;flex-wrap:wrap;gap:18px;border-bottom:1px solid #bbb;padding:10px 0}}a{{color:#24543f}}.facts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:15px;margin:26px 0}}.facts div{{border:1px solid #bbb;padding:14px}}.facts strong{{display:block;font-weight:normal;font-size:1.9rem}}img{{max-width:100%;height:auto}}figure{{margin:24px 0}}figcaption{{font-size:.94rem;color:#444}}table{{border-collapse:collapse;width:100%;font-size:.94rem}}td,th{{padding:7px 12px;text-align:left;border-bottom:1px solid #ddd}}code{{font-size:.76rem;overflow-wrap:anywhere}}.note{{border-left:3px solid #ac7130;padding-left:18px}}@media print{{nav{{display:none}}figure{{break-inside:avoid}}}}</style></head><body>
<p>CSI-MAIZE · Research intake · 27 October 2026</p><h1>Five visits, three treatments: what the complete archive lets us test</h1>
<nav><a href="#coverage">Coverage</a><a href="#performance">Measured performance</a><a href="#segmentation">Segmentation</a><a href="#experiments">Analysis programme</a><a href="#integrity">Integrity</a></nav>
<p>We inspected all files in the supplied archive, rebuilt the reference joins from the original workbooks and fully decoded every JPEG. We then replayed the published SAM foreground operators on a stratified diagnostic set. The figures below distinguish acquired data, measured physiology and candidate image masks.</p>
<div class="facts"><div><strong>{summary['image_files']:,}</strong>JPEG paths inspected</div><div><strong>{summary['complete_identified_40_pair_sessions']}</strong>intact, identified paired acquisitions</div><div><strong>{summary['candidate_trajectories_with_five_complete_visits']}/72</strong>candidate identities with five intact visits</div><div><strong>360</strong>keyed assay records</div></div>
<section id="coverage"><h2>Acquisition coverage</h2><p>Each complete acquisition contains indices 0–39 from both cameras. We found all 360 expected labelled session directories and two additional timestamp-only directories. Empty directories do not count as observations. The two extras are fully decodable but lack verified sample identities.</p>
{image('01_acquisition_coverage.png','Completeness is based on full pixel decoding of 80 images per acquisition. It does not establish anatomical reconstruction, mask accuracy or calibration.')}
<table><thead><tr><th>Treatment</th><th>Visit</th><th>Expected</th><th>Intact acquisitions</th><th>Intact indexed pairs</th></tr></thead><tbody>{cells}</tbody></table>
{image('03_candidate_trajectory_coverage.png','Recorded treatment, genotype and replicate labels define 72 candidate trajectories. We retain incomplete visits explicitly. Counts are paired views, not leaves.')}
<p>Twenty of the 26 acquisitions failing the strict paired-rotation criterion belong to replicate 4. We will therefore report missingness by treatment, visit and replicate, and compare complete-case estimates with analyses using all accepted views. A complete-case subset cannot be assumed representative.</p></section>
<section id="performance"><h2>Measured plant performance</h2><p>The assay workbooks contain 24 records per treatment and visit: six genotypes and four replicate labels. All 12 supplied workbooks have the same sheet names and cached cell values as their earlier reference copies. The expanded evidence comes from the additional imagery.</p>
{image('02_measured_performance.png','Dots show individual assay records; connected symbols show arithmetic group means. No significance test or image prediction is shown. Recorded visits are used because camera clocks conflict with the folder age labels.')}
<p>At visit 2, mean Fv/Fm is 0.508 in Stress, 0.603 in Priming and 0.787 in Control. The treated groups increase at later visits. These measurements give us a direct target for evaluating whether image descriptors capture performance differences beyond genotype and visit. The exact exposure schedule and continuity of physical pots require protocol confirmation before we describe treatment effects or recovery rates.</p></section>
<h2>What a predictor can achieve without images</h2>
{image('08_metadata_baselines.png','Each test genotype is excluded from all training group means. Treatment and visit are known metadata, not image-derived variables. Pooled R² is calculated from all 360 out-of-fold predictions. This is an internal experimental baseline, not external validation or forecasting.')}
<p>A treatment-by-visit training mean reaches R² = 0.560 for Fv/Fm and 0.433 for PIABS on held-out genotypes, using no images. The same baseline gives R² = −0.019 for instrument NIR NDVI. Image models must be compared with these context-aware controls and evaluated on identical eligible records. Beating a global-mean predictor alone would leave the physiological information supplied by the imagery unresolved.</p>
<section id="segmentation"><h2>Segmentation across the experimental range</h2><p>We selected the lowest-numbered fully decodable replicate within each treatment, visit and genotype, then processed views 0, 10, 20 and 30 from both cameras. This gives 90 acquisitions and 720 images. We used the pinned SAM 2.1 small checkpoint and the published colour-seeded bounding-box operators without changing their thresholds.</p>
{image('04_segmentation_diagnostic.png','Numbers show automated QC acceptance out of 24 diagnostic images per treatment, visit and camera. Model score and seed agreement do not measure segmentation accuracy.')}
<p>The operators accepted {full_mask_count} of {len(views)} diagnostic images. We retained every failure and its reason. The most frequent flags were {html.escape(', '.join(f'{reason} ({n})' for reason,n in reasons.most_common(4)))}. Flags can co-occur.</p>
<p>Overall acceptance conceals a relevant difference: the operators accepted only 15 of 24 NoIR images from Stress at visit 2, compared with 24 of 24 RGB images in that stratum. Requiring every view to pass could therefore remove acquisitions from a group with low measured Fv/Fm. We will evaluate per-view exclusions and segmentation error before adopting a complete-rotation trait table.</p>
{image('06_failed_segmentation_review.png','Two failed NoIR diagnostics shown beside their source images. The red line marks the candidate mask boundary. These examples show why model confidence and foreground agreement need visual review.')}
{image('07_prompt_comparison.png','The same 720 source images were processed with the existing box-and-point alternative. Checkpoint bytes, source-image bytes, strata and QC thresholds match. QC acceptance remains an internal diagnostic, not mask accuracy.')}
<p>Point-and-box prompting passes QC on {prompt_pass} images, compared with {full_mask_count} for box-only prompting. It changes {transitions[(False,True)]} previously rejected images to passes and {transitions[(True,False)]} previously accepted images to failures. We retain this negative result. Adding the existing prompts does not justify replacing the box-only workflow, and acceptance alone cannot decide which masks are anatomically correct.</p>
{image('05_rgb_candidate_foregrounds.png','Candidate RGB foregrounds for the same recorded genotype and replicate labels. Panels are independently cropped and resized. Visible leaf loss or shortening may reflect segmentation failure and must be checked against the original image.')}
{image('05_noir_candidate_foregrounds.png','NoIR candidates from the corresponding labelled acquisitions. These are broad camera-channel observations. We do not assign narrow spectral bands or measured chlorophyll concentration to them.')}
<p>A foreground model can omit discoloured tissue when its prompt box comes from a conservative colour seed. We need a reviewed reference set containing green, yellow, necrotic, overlapping and partly occluded leaves before using mask-dependent traits as biological outcomes. Foreground masks also do not supply leaf identity.</p></section>
<section id="experiments"><h2>Analysis programme</h2>
<h3>Architecture remains the main experiment</h3><p>We will evaluate the existing reconstruction on additional complete plants and visits, with view splits declared before fitting. Camera and turntable hypotheses will be tested separately from organ shape. Reconstruction error, visibility and alternative acceptable geometries will accompany each virtual leaf removal or repositioning experiment. An apparent growth trajectory must retain its uncertainty when an organ is hidden, truncated or ambiguously matched.</p>
<p>The larger archive permits tests of whether neighbour-dependent exposure changes across visits and treatments, whether those changes persist across acceptable reconstructions, and how many views are required to retain them. We will distinguish projected area and visibility from leaf surface area. Quantities in image pixels or an uncalibrated model coordinate system will retain those units.</p>
<h3>Persistent leaves need explicit correspondence evidence</h3><p>We will maintain a candidate organ graph using insertion order, emergence, geometry and view support. New leaves, senescing leaves, temporary occlusion and unresolved identity will remain distinct states. We will review ranks and correspondences on held-out trajectories. A whole-plant SAM mask or a tracker ID is insufficient evidence that a physical leaf has persisted across visits.</p>
<h3>RGB–NoIR observations support the architecture</h3><p>We will test paired-camera registration on the enlarged cohort and preserve only spatial support that passes overlap, boundary, saturation and correspondence checks. All 27 existing descriptors, including the 15 two-channel normalized differences, remain camera-DN descriptors. Registration alternatives and wrong-pair controls will quantify their sensitivity. Spectral correspondence must be evaluated against native UVIS and NIR measurements, without treating the six camera channels as six calibrated narrow bands.</p>
<h3>Physiological prediction requires stronger baselines</h3><p>We will compare metadata-only, RGB, NoIR and combined predictors for Fv/Fm, PIABS and spectral response. All views from one acquisition will stay together. Unseen-genotype and unseen-treatment evaluations will be separate; preprocessing and tuning will use training data only. Once physical continuity is confirmed, all visits of a plant will share its partition. Within-treatment/visit controls and trajectory-preserving label permutations will test whether a model adds information beyond acquisition and developmental context.</p>
<p>We will retain native instrument grids, report wavelength-specific performance and compare against training-only means and simple regularized models before expanding neural architectures. The 255 existing UVIS consistency flags remain visible. Biomass records remain unjoined until their harvest timing and replicate crosswalk are documented.</p>
<h3>External tests and scientific context</h3><p>PhenoTrack3D establishes a relevant reference for topology-aware maize leaf correspondence and separate validation of leaf rank and geometry. Pheno4D provides temporally labelled maize point clouds for checking correspondence operators against known organ identities. Its laser geometry cannot validate the accuracy of our camera reconstructions. FieldPheno4D adds a recent multi-temporal field dataset for later transfer experiments. These are proposed external checks; we have not run them in this intake.</p>
<p><a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC9730636/">PhenoTrack3D, Plant Methods (2022)</a> · <a href="https://www.ipb.uni-bonn.de/data/pheno4d/index.html">Pheno4D, Schunck et al. (2021)</a> · <a href="https://bonndata.uni-bonn.de/dataset.xhtml?persistentId=doi%3A10.60507%2FFK2%2FHYI2DS">FieldPheno4D, Esser et al. (2026)</a></p></section>
<section id="integrity"><h2>Integrity and reproducibility</h2><p>We decoded {summary['decode_pass']:,} JPEGs successfully. The {summary['decode_fail']:,} failures comprise {zero_bytes:,} zero-byte placeholders and {summary['decode_fail']-zero_bytes} nonempty damaged streams. Hash comparisons found no duplicated nonempty image files. Header checks alone would have accepted two additional incomplete rotations.</p><ul>{failed_nonempty}</ul>
<p>We read the originals through a registered source binding and wrote derived evidence to a separate analysis directory. The file manifest records every JPEG size, SHA256, decoding outcome and available camera date. The diagnostic replay records its checkpoint files, source hashes, prompts through the unchanged operator definition, masks and per-image QC. We retained both the complete audit and the sparse segmentation replay; this report does not represent an all-view model run or a new manuscript analysis.</p>
<p class="note">The current publication bundle remains the reference method demonstration. Full-data biological conclusions will require the declared analyses and validation above. Data and reference outputs will be made available via Zenodo after publication. Methods code: <a href="https://github.com/vlattko/CSI-maize-methods">github.com/vlattko/CSI-maize-methods</a>.</p></section></body></html>'''
    (root/'CSI_maize_full_archive_report.html').write_text(text,encoding='utf-8')
    (root/'report_build_summary.json').write_text(json.dumps(dict(
        status='completed',figures=len(list(figures.glob('*.png'))),
        diagnostic_images=len(views),diagnostic_qc_pass=full_mask_count,
        html_bytes=len(text.encode('utf-8'))),indent=2),encoding='utf-8')
    print(root/'CSI_maize_full_archive_report.html')


if __name__=='__main__':main()
