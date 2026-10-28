"""Source-bound, training-angle camera alignment and spatial DN maps for one capture."""
import argparse,base64,csv,hashlib,html,importlib.util,io,json,sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def contained(root,name):
    root=Path(root).resolve();target=(root/name).resolve()
    if not target.is_relative_to(root):raise ValueError('Path escapes declared root')
    return target


def verify_capture(root):
    root=Path(root);manifest=json.loads((root/'output_manifest.json').read_text())
    if manifest['status']!='completed':raise ValueError('Capture must be completed without source changes')
    names=[item['path'] for item in manifest['files']]
    if 'capture.json' not in names or len(names)!=len(set(names)):raise ValueError('Manifest must bind capture identity once')
    for item in manifest['files']:
        if sha(contained(root,item['path']))!=item['sha256']:raise ValueError('Capture file digest mismatch: '+item['path'])
    return json.loads((root/'capture.json').read_text())


def validate_frame_binding(token,row,sample_id,steps):
    if row.get('sample_id')!=sample_id or row.get('camera') not in ['rgb','noir'] or not isinstance(row.get('view'),int) or not 0<=row['view']<steps or token!=row['camera']+':'+str(row['view']):
        raise ValueError('Frame identity disagrees with capture index')


def configure(repo=None):
    if repo is None:
        spec=importlib.util.find_spec('csi_maize');folder=Path(next(iter(spec.submodule_search_locations)))/'_methods'
    else:folder=Path(repo).resolve()/'src/csi_maize/_methods'
    sys.path.insert(0,str(folder))
    import channel_fusion_core as channel
    import spatial_fusion_core as spatial
    for module in [channel,spatial]:
        if Path(module.__file__).resolve().parent!=folder.resolve():raise ValueError('Unexpected public operator import')
    return channel,spatial,folder


def training_masks(masks):return {v:m for v,m in masks.items() if v%4!=2}


def describe_pair(rgb,rm,noir,nm,matrix,accepted,operators=None):
    channel,spatial,_=operators or configure()
    moved,field=channel.warp_to_rgb(noir.astype(np.float32),matrix,rm.shape)
    warped_mask,_=channel.warp_to_rgb(channel.interior(nm).astype(np.uint8),matrix,rm.shape,True)
    ri=channel.interior(rm);ni=(warped_mask>0)&field;common=ri&ni
    warped_valid,_=channel.warp_to_rgb(((noir>1)&(noir<254)).astype(np.float32),matrix,rm.shape)
    six=np.concatenate([rgb.astype(np.float32),moved],axis=-1)
    valid=np.concatenate([(rgb>1)&(rgb<254),warped_valid>=1-1e-7],axis=-1)
    supports=dict(rgb=ri,noir=ni,cross=common if accepted else np.zeros_like(common))
    maps=spatial.spatial_indices(six,supports,valid)
    stats=[dict(index=d['id'],family=d['support'],**spatial.index_statistics(maps[d['id']],supports[d['support']])) for d in spatial.index_definitions()]
    foreground=rgb.copy();foreground[~rm]=0
    blend=np.clip(.5*foreground+.5*moved*ni[...,None],0,255).astype(np.uint8)
    return maps,stats,blend


def dump(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')
def table(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def montage(rgb,noir,blend,maps,definitions,registered):
    from matplotlib import colormaps,font_manager
    font=ImageFont.truetype(font_manager.findfont('Times New Roman'),17);small=ImageFont.truetype(font_manager.findfont('Times New Roman'),14)
    canvas=Image.new('RGB',(6*256,5*216),'white');draw=ImageDraw.Draw(canvas)
    panels=[('RGB foreground',rgb),('Registered NoIR foreground' if registered else 'NoIR foreground, unregistered',noir),('Overlap projection' if registered else 'Overlap unavailable',blend)]
    for d in definitions:
        values=maps.get(d['id']);tile=np.zeros((192,256,3),dtype=np.uint8)
        if values is not None:
            finite=np.isfinite(values);rgba=colormaps['viridis'](np.clip((np.nan_to_num(values)-d['vmin'])/(d['vmax']-d['vmin']),0,1))
            image=(rgba[:,:,:3]*255).astype(np.uint8);image[~finite]=0;tile=np.asarray(Image.fromarray(image).resize((256,192),Image.Resampling.NEAREST))
        panels.append((d['id'],tile))
    for i,(label,im) in enumerate(panels):
        x=(i%6)*256;y=(i//6)*216
        if im is not None:canvas.paste(Image.fromarray(im).resize((256,192)),(x,y))
        else:draw.rectangle((x,y,x+255,y+191),fill='black')
        draw.text((x+4,y+194),label,fill='black',font=small if len(label)>23 else font)
    return canvas


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['capture','raw','repo','output']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();capture=args.capture.resolve();raw=args.raw.resolve();out=args.output.resolve()
    for source in [capture,raw]:
        if out.is_relative_to(source) or source.is_relative_to(out):raise ValueError('Fusion output must be separate from input roots')
    if out.exists() and any(out.iterdir()):raise ValueError('Use a new or empty fusion output folder')
    result=verify_capture(capture);operators=configure(args.repo);channel,spatial,folder=operators
    sibling=Path(__file__).parent.parent/'camera_geometry_development_2026-10-27';sys.path.insert(0,str(sibling));sys.path.insert(0,str(sibling/'periodic'))
    import camera_geometry as geometry
    import periodic_camera as periodic
    from capture_core import tracking_support
    support=tracking_support(result['session'])
    records={};images={};masks={};small={c:{} for c in ['rgb','noir']}
    for token,relative in result['record_index'].items():
        r=json.loads(contained(capture,relative).read_text());records[token]=r
        validate_frame_binding(token,r,result['session']['sample_id'],result['session']['steps'])
        if r['status']!='inferred' or not r['structural_usability']['usable']:continue
        source=contained(raw,r['source']);mask=contained(capture,r['mask'])
        source_bytes=source.read_bytes();mask_bytes=mask.read_bytes()
        if hashlib.sha256(source_bytes).hexdigest()!=r['source_sha256'] or hashlib.sha256(mask_bytes).hexdigest()!=r['mask_sha256']:raise ValueError('Source/mask digest mismatch')
        with Image.open(io.BytesIO(source_bytes)) as im:im.load();image=np.asarray(im.convert('RGB').resize((512,385),Image.Resampling.LANCZOS))
        with Image.open(io.BytesIO(mask_bytes)) as im:
            if im.size!=(r['processing_width'],r['processing_height']):raise ValueError('Mask canvas differs from capture record')
            mm=np.asarray(im.convert('L').resize((512,385),Image.Resampling.NEAREST))>0
            tiny=np.asarray(im.convert('L').resize((256,192),Image.Resampling.NEAREST))>0
        images[token]=image;masks[token]=mm;small[r['camera']][r['view']]=tiny
    shared=geometry.fit_session(training_masks(small['rgb']),training_masks(small['noir'])) if support['supported'] else dict(selected=None,status='unsupported_rotation')
    periodic_fit=periodic.fit_periodic(training_masks(small['rgb']),training_masks(small['noir']),shared) if support['supported'] else dict(selection=dict(accepted=False),model=None)
    decision=geometry.deployment_decision(shared);selected=shared.get('selected');periodic_used=periodic_fit['selection']['accepted']
    if periodic_used:decision=dict(deployment_abstain=False,deployment_reasons=[],deployment_status='accepted_periodic_camera_surrogate',deployment_gate_uses_heldout=False)
    out.mkdir(parents=True,exist_ok=True);(out/'maps').mkdir();(out/'previews').mkdir()
    definitions=spatial.index_definitions();descriptors=[];pairs=[];gallery=[]
    for view in range(result['session']['steps']):
        noir_view=(selected['sign']*view+selected['phase'])%40 if selected else view
        rgb_token='rgb:'+str(view);noir_token='noir:'+str(noir_view);pair=dict(view=view,noir_view=noir_view,split='heldout' if view%4==2 else 'train',deployment_abstain=decision['deployment_abstain'])
        values={};stats=[];a=b=blend=None
        if selected and rgb_token in images and noir_token in images:
            matrix=periodic.predict(periodic_fit['model'],view) if periodic_used else np.array(selected['matrix'])
            pair['camera_diagnostic_iou_256']=geometry.iou_warp(small['rgb'][view],small['noir'][noir_view],matrix)
            matrix=geometry.rescale_matrix(matrix,(256,192),(512,385),(256,192),(512,385));pair['matrix_rgb_to_noir']=matrix.tolist()
            a=images[rgb_token];b=images[noir_token];am=masks[rgb_token];bm=masks[noir_token]
            registered_mask,_=channel.warp_to_rgb(bm.astype('uint8'),matrix,am.shape,True);union=int((am|(registered_mask>0)).sum())
            pair['map_grid_iou_512']=float((am&(registered_mask>0)).sum()/union) if union else 0.
            moved,field=channel.warp_to_rgb(channel.interior(bm).astype('uint8'),matrix,am.shape,True);common=channel.interior(am)&(moved>0)&field
            fraction=float(common.sum()/max(1,min(int(channel.interior(am).sum()),int(((moved>0)&field).sum()))))
            reasons=[]
            if pair['map_grid_iou_512']<channel.POLICY['whole_iou_min']:reasons.append('whole_mask_IoU_below_gate')
            if int(common.sum())<channel.POLICY['min_common_interior_pixels']:reasons.append('too_few_common_interior_pixels')
            if fraction<channel.POLICY['min_interior_overlap_fraction']:reasons.append('common_interior_fraction_below_gate')
            if decision['deployment_abstain']:reasons.append('camera_deployment_abstained')
            pair.update(common_pixels=int(common.sum()),common_fraction=fraction,cross_formula_reasons=reasons,cross_formulas_usable=not reasons,status='scored_camera_hypothesis')
            if not decision['deployment_abstain']:
                values,stats,blend=describe_pair(a,am,b,bm,matrix,not reasons,operators)
                a=a.copy();a[~am]=0;b,_=channel.warp_to_rgb(b.astype('float32'),matrix,am.shape);b=np.clip(b*(moved>0)[...,None],0,255).astype('uint8')
                np.savez_compressed(out/f'maps/view_{view:03d}.npz',**values)
            else:
                a=a.copy();a[~am]=0;b=b.copy();b[~bm]=0
        else:pair.update(status='unavailable_pair_or_camera',cross_formulas_usable=False,cross_formula_reasons=['no_usable_pair_or_camera'])
        stats_by={r['index']:r for r in stats}
        for d in definitions:descriptors.append(dict(view=view,noir_view=noir_view,**stats_by.get(d['id'],dict(index=d['id'],family=d['support'],eligible_pixels=0,valid_pixels=0,qualified=False,median=None,q10=None,q25=None,q75=None,q90=None,iqr=None,valid_fraction=None))))
        pairs.append(pair);picture=montage(a,b,blend,values,definitions,not decision['deployment_abstain']);path=out/f'previews/view_{view:03d}.png';picture.save(path)
        gallery.append(dict(view=view,image='data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode(),pair=pair,statistics=[r for r in descriptors if r['view']==view]))
    table(out/'pairs.csv',[{k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in pairs]);table(out/'index_statistics.csv',descriptors)
    provenance=dict(builder_sha256=sha(Path(__file__)),capture_manifest_sha256=sha(capture/'output_manifest.json'),capture_sha256=sha(capture/'capture.json'),
        public_operators={name:sha(folder/name) for name in ['channel_fusion_core.py','spatial_fusion_core.py']},geometry_operators={str(p.relative_to(sibling)):sha(p) for p in [sibling/'camera_geometry.py',sibling/'periodic/periodic_camera.py']},policy=channel.POLICY)
    summary=dict(status='completed',sample_id=result['session']['sample_id'],planned_pairs=len(pairs),descriptor_rows=len(descriptors),qualified_descriptor_rows=sum(r['qualified'] for r in descriptors),usable_cross_pairs=sum(r['cross_formulas_usable'] for r in pairs),
        camera_decision=decision,camera_model='first_harmonic' if periodic_used else 'shared' if selected else 'unavailable',tracking_angle_support=support,
        heldout_scored_pairs=sum(r['split']=='heldout' and 'camera_diagnostic_iou_256' in r for r in pairs),heldout_median_iou_256=float(np.median([r['camera_diagnostic_iou_256'] for r in pairs if r['split']=='heldout' and 'camera_diagnostic_iou_256' in r])) if any(r['split']=='heldout' and 'camera_diagnostic_iou_256' in r for r in pairs) else None,
        heldout_median_iou_512=float(np.median([r['map_grid_iou_512'] for r in pairs if r['split']=='heldout' and 'map_grid_iou_512' in r])) if any(r['split']=='heldout' and 'map_grid_iou_512' in r for r in pairs) else None,
        camera_diagnostic_grid=[256,192],map_grid=[512,385],provenance=provenance,channel_validity_policy='All nonzero bilinear contributors valid: warped validity >=1-1e-7; stricter than frozen cohort >0.999 tolerance',limits=['Camera-DN heuristics, not calibrated reflectance or chlorophyll.','Image alignment is not tissue or organ correspondence.','Camera choice excludes outer held-out indices; individual map support uses the observed pair.'])
    dump(out/'camera_fit.json',dict(shared=shared,periodic=periodic_fit,decision=decision));dump(out/'summary.json',summary);dump(out/'index_definitions.json',definitions)
    payload=json.dumps(gallery,allow_nan=False).replace('<','\\u003c');title=html.escape(summary['sample_id'])
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Capture fusion</title><style>body{font:19px/1.5 "Times New Roman",Times,serif;background:white;color:#151515;max-width:1320px;margin:35px auto;padding:0 24px}h1{font-weight:normal}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:16px}td,th{text-align:left;padding:7px;border-bottom:1px solid #ddd}button,input{font:inherit}pre{white-space:pre-wrap;font-size:14px}a{color:#245c4e}</style><h1>RGB–NoIR capture fusion: '''+title+'''</h1><p>Camera fitting uses training rotation angles; held-out angles are scored afterwards. Cross-camera maps require an accepted camera and sufficient observed foreground overlap. Black pixels in a map mean unavailable support, not a measured zero. Fixed display ranges are recorded in the formula definitions; numerical arrays retain unclipped values.</p><label>Rotation index <input id="view" type="range" min="0" max="'''+str(len(gallery)-1)+'''" value="0"></label><span id="label"></span><p id="status"></p><img id="gallery" alt="Source foregrounds, paired overlap and twenty-seven spatial index maps"><table><thead><tr><th>Formula</th><th>Valid pixels</th><th>Median DN heuristic</th><th>Qualified</th></tr></thead><tbody id="stats"></tbody></table><h2>Camera and provenance</h2><pre>'''+html.escape(json.dumps(summary,indent=2))+'''</pre><p>Data and reference outputs will be made available through Zenodo after publication.</p><script id="payload" type="application/json">'''+payload+'''</script><script>const records=JSON.parse(document.getElementById('payload').textContent);document.getElementById('payload').remove();const $=id=>document.getElementById(id);function render(){const r=records[Number($('view').value)];$('label').textContent=' '+r.view+' / '+(records.length-1);$('gallery').src=r.image;$('status').textContent=JSON.stringify(r.pair);$('stats').replaceChildren();for(const row of r.statistics){const tr=document.createElement('tr');for(const value of [row.index,row.valid_pixels,row.median===null?'Unavailable':row.median.toFixed(4),row.qualified?'Yes':'No']){const td=document.createElement('td');td.textContent=value;tr.append(td)}$('stats').append(tr)}}$('view').addEventListener('input',render);render();</script></html>'''
    (out/'report.html').write_text(page,encoding='utf-8');dump(out/'output_manifest.json',dict(status='completed',files=[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in out.rglob('*') if p.is_file() and p.name!='output_manifest.json']))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    sys.path.insert(0,str(Path(__file__).parent.parent/'new_session_pipeline_development_2026-10-27'))
    main()
