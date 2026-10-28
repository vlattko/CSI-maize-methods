"""Build one self-contained workbench from verified sprites and candidate graphs.

No source images are synthesized. Missing frames retain their original indices.
Assets are embedded as compressed data; only selected sprite pairs are decoded
by the browser. The output works without a local server or external requests.
"""
import argparse,base64,csv,gzip,hashlib,json,mimetypes,time
from pathlib import Path


def sha(data):return hashlib.sha256(data).hexdigest()
def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def key(row):return tuple(str(row[k]) for k in ['treatment','measurement','genotype','replicate'])


def compact_graph(graph):
    # Scientific JSON remains untouched. Avoid repeating every node within its
    # session and every source frame inside each node's40 angular observations.
    for session in graph['sessions']:
        session['node_ids']=[n['node_id'] for n in session.pop('nodes',[])]
    for node in graph['nodes']:
        for observation in node.get('angle_evidence',[]):
            if 'source' in observation:
                observation.pop('source');observation['source_view']=observation['view']
    return graph


def build(args):
    started=time.time();manifest_data=args.sprites.read_bytes();manifest=json.loads(manifest_data)
    if manifest.get('status')!='completed':raise ValueError('Sprite manifest must be completed')
    slots=manifest['slots']
    if len(slots)!=720 or manifest['reference_rows']!=360:raise ValueError('Expected all720camera slots and360reference records')
    slotkeys=[(*key(s['key']),s['camera']) for s in slots]
    if len(set(slotkeys))!=len(slotkeys):raise ValueError('Duplicate four-key camera slot')
    if len({s['id'] for s in slots})!=len(slots):raise ValueError('Sprite IDs must be unique across cameras and identities')
    slot_lookup={(*key(s['key']),s['camera']):s for s in slots}
    for slot in slots:
        if len(slot['frames'])!=40 or [f['view'] for f in slot['frames']]!=list(range(40)):raise ValueError('Every slot must retain40ordered frame indices')
    assetroot=args.asset_root or (args.sprites.parent.parent if args.sprites.parent.name=='sprites' else args.sprites.parent)
    assets={};asset_digests={}
    for slot in slots:
        for field,hashfield in [('original','original_sha256'),('mask','mask_sha256')]:
            rel=slot[field];path=assetroot/rel;data=path.read_bytes();digest=sha(data)
            if slot.get(hashfield) and slot[hashfield]!=digest:raise ValueError('Sprite digest mismatch: '+rel)
            if rel not in assets:assets[rel]='data:'+(mimetypes.guess_type(rel)[0] or 'application/octet-stream')+';base64,'+base64.b64encode(data).decode('ascii')
            asset_digests[rel]=digest
    index=load(args.tracking);graphs=[];graph_digests={};crosschecks=0
    for row in index['candidates']:
        path=args.tracking.parent/row['path'];data=path.read_bytes();g=json.loads(data)
        if len(g['sessions'])!=5:raise ValueError('Missing reference visits in '+g['candidate_id'])
        for session in g['sessions']:
            identity={**g['identity'],'measurement':str(session['visit'])}
            slot=slot_lookup[(*key(identity),'rgb')]
            for frame in session.get('frames',[]):
                displayed=slot['frames'][frame['view']]
                if not displayed.get('present') or displayed.get('source_sha256')!=frame.get('source_sha256'):
                    raise ValueError('Tracking/source sprite mismatch: '+g['candidate_id']+' visit'+str(session['visit'])+' view'+str(frame['view']))
                if displayed.get('mask_sha256') and frame.get('mask_sha256') and displayed['mask_sha256']!=frame['mask_sha256']:
                    raise ValueError('Tracking/mask sprite mismatch: '+g['candidate_id'])
                crosschecks+=1
        graphs.append(compact_graph(g));graph_digests[row['path']]=sha(data)
    if len(graphs)!=72 or len({g['candidate_id'] for g in graphs})!=72:raise ValueError('Expected all72candidate identity graphs')
    with args.reference.open(encoding='utf-8-sig',newline='') as f:reference=list(csv.DictReader(f))
    if len(reference)!=360 or len({key(r) for r in reference})!=360:raise ValueError('Expected360unique four-key reference records')
    if {key(s['key']) for s in slots}!={key(r) for r in reference}:raise ValueError('Sprite and assay reference identities differ')
    identities=[];identity_digest=None
    registry_path=getattr(args,'identity_registry',None)
    if registry_path:
        registry=load(registry_path);identities=registry['records'];identity_digest=sha(registry_path.read_bytes())
        if registry['status']!='verified_experimenter_confirmed_identity_overlay' or len(identities)!=360 or {key(r) for r in identities}!={key(r) for r in reference}:
            raise ValueError('Identity registry must match every reference record')
        if any(r['physical_identity_status']!='experimenter_confirmed_same_pot' for r in identities):raise ValueError('Unexpected identity authority')
    cameras=[];camera_digest=None
    if args.camera:
        c=load(args.camera);cameras=c.get('rows',c) if isinstance(c,dict) else c
        if not isinstance(cameras,list):raise ValueError('Camera lookup must provide a row array')
        camera_digest=sha(args.camera.read_bytes())
        if len({key(r) for r in cameras})!=len(cameras):raise ValueError('Duplicate camera lookup key')
    metadata=dict(title='Maize rotation and tracking workbench',built_unix=time.time(),
        sprite_manifest_sha256=sha(manifest_data),tracking_index_sha256=sha(args.tracking.read_bytes()),reference_sha256=sha(args.reference.read_bytes()),
        camera_lookup_sha256=camera_digest,camera_rows=len(cameras),reference_rows=len(reference),camera_slots=len(slots),candidate_identities=len(graphs),
        tracking_sprite_frame_crosschecks=crosschecks,
        tracking_summary=index.get('summary'),source_graph_sha256=graph_digests,sprite_asset_sha256=asset_digests,
        selection_identity='Experimenter-confirmed pot continuity; anatomical leaf associations remain hypotheses' if identities else 'treatment + recorded visit + genotype + replicate; physical continuity unverified',
        identity_registry_sha256=identity_digest,
        playback='Fixed image indices0..39, not measured servo timing',
        coordinate_scope='Terminal points conditional on nominal orthographic projection and training-image-height gauge; no leaf mesh',
        asset_policy='All assets embedded; selected camera pair decoded lazily; no external requests')
    payload=dict(metadata=metadata,sprite={k:manifest[k] for k in ['frame_width','frame_height','columns','rows']},slots=slots,graphs=graphs,reference=reference,camera=cameras,identityRegistry=identities,assets=assets)
    raw=json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf-8');compressed=gzip.compress(raw,compresslevel=6,mtime=0)
    template=Path(__file__).with_name('workbench.html').read_text(encoding='utf-8')
    if template.count('__WORKBENCH_PAYLOAD__')!=1:raise ValueError('Unexpected template placeholder count')
    html=template.replace('__WORKBENCH_PAYLOAD__',base64.b64encode(compressed).decode('ascii'));args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(html,encoding='utf-8')
    report=dict(status='completed',output=str(args.output),output_sha256=sha(args.output.read_bytes()),html_bytes=args.output.stat().st_size,
        payload_uncompressed_bytes=len(raw),payload_gzip_bytes=len(compressed),asset_files=len(assets),candidate_identities=len(graphs),reference_rows=len(reference),camera_rows=len(cameras),
        script_sha256=sha(Path(__file__).read_bytes()),template_sha256=sha(template.encode()),elapsed_seconds=time.time()-started)
    args.output.with_suffix('.build.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sprites',type=Path,required=True);p.add_argument('--tracking',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--camera',type=Path);p.add_argument('--identity-registry',type=Path);p.add_argument('--asset-root',type=Path);p.add_argument('--output',type=Path,required=True);build(p.parse_args())
