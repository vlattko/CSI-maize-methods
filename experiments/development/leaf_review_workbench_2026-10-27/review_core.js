(function(root,factory){if(typeof module==='object'&&module.exports)module.exports=factory();else root.LeafReviewCore=factory();})(typeof globalThis!=='undefined'?globalThis:this,function(){
  'use strict';
  const clone=x=>JSON.parse(JSON.stringify(x));
  const statuses=new Set(['unreviewed','accepted','rejected','uncertain']);
  const extents=new Set(['unknown','partial','complete_visible_blade']);
  const now=()=>new Date().toISOString();
  function frame(dataset,id){const f=dataset.frames.find(x=>x.frame_id===id);if(!f)throw Error('Unknown source frame ID');return f;}
  function stateFrame(dataset,state,id){frame(dataset,id);if(!state.frames[id])throw Error('Missing review frame');return state.frames[id];}
  function region(dataset,state,fid,rid){const f=stateFrame(dataset,state,fid);if(!f.regions[rid])throw Error('Unknown candidate ID');return f.regions[rid];}
  function event(state,action,detail){state.history.push({utc:now(),action,...detail});state.modified_utc=now();}
  function invalidate(f){f.all_visible_leaves_reviewed=false;}
  function createState(dataset){
    const state={schema:'csi-leaf-review-session',schema_version:1,dataset_id:dataset.dataset_id,
      rater:'',created_utc:now(),modified_utc:now(),manual_counter:0,frames:{},history:[]};
    for(const f of dataset.frames){
      const review={frame_id:f.frame_id,source_sha256:f.source_sha256,native_width:f.native_width,native_height:f.native_height,
        treatment:f.treatment,measurement:f.measurement,genotype:f.genotype,replicate:f.replicate,session:f.session,camera:f.camera,view:f.view,
        all_visible_leaves_reviewed:false,verified_empty:false,regions:{}};
      for(const candidate of f.candidates){
        const choices=[...candidate.proposals].sort((a,b)=>(b.mode==='distal_points_negatives')-(a.mode==='distal_points_negatives')||
          (b.score??-1)-(a.score??-1)||a.alternative-b.alternative);
        const proposal=choices[0];
        review.regions[candidate.candidate_id]={region_id:candidate.candidate_id,source_candidate_id:candidate.candidate_id,
          source_type:'proposal',status:'unreviewed',extent:'unknown',selected_proposal_id:proposal?.proposal_id??null,
          geometry_mask_kind:'clipped',polygons:clone(proposal?.polygons_clipped??[]),geometry_edited:false,notes:''};
      }
      state.frames[f.frame_id]=review;
    }
    return state;
  }
  function polygonArea(ring){let area=0;for(let i=0;i<ring.length;i++){const a=ring[i],b=ring[(i+1)%ring.length];area+=a[0]*b[1]-b[0]*a[1];}return Math.abs(area)/2;}
  function validatePolygons(polygons,width,height){
    if(!Array.isArray(polygons)||!polygons.length)throw Error('A nonempty polygon is required');
    let area=0;
    for(const ring of polygons){
      if(!Array.isArray(ring)||ring.length<3)throw Error('Polygon requires at least three vertices');
      for(const point of ring)if(!Array.isArray(point)||point.length!==2||!point.every(Number.isFinite)||point[0]<0||point[1]<0||point[0]>width||point[1]>height)throw Error('Polygon coordinate outside native source dimensions');
      area+=polygonArea(ring);
    }
    if(area<=1e-6)throw Error('Polygon has zero area');return true;
  }
  function setStatus(dataset,state,fid,rid,status){
    if(!statuses.has(status))throw Error('Unknown review decision');const f=stateFrame(dataset,state,fid),r=region(dataset,state,fid,rid);
    if(status==='accepted'){const source=frame(dataset,fid);validatePolygons(r.polygons,source.native_width,source.native_height);f.verified_empty=false;}
    r.status=status;invalidate(f);event(state,'region_decision',{frame_id:fid,region_id:rid,status});return state;
  }
  function setExtent(dataset,state,fid,rid,extent){
    if(!extents.has(extent))throw Error('Unknown region extent');const f=stateFrame(dataset,state,fid),r=region(dataset,state,fid,rid);
    if(r.extent!==extent&&r.status==='accepted')r.status='unreviewed';r.extent=extent;invalidate(f);
    event(state,'region_extent',{frame_id:fid,region_id:rid,extent});return state;
  }
  function setPolygons(dataset,state,fid,rid,polygons){
    const source=frame(dataset,fid);validatePolygons(polygons,source.native_width,source.native_height);
    const f=stateFrame(dataset,state,fid),r=region(dataset,state,fid,rid);r.polygons=clone(polygons);r.geometry_edited=true;r.status='unreviewed';invalidate(f);
    event(state,'geometry_edit',{frame_id:fid,region_id:rid});return state;
  }
  function useProposal(dataset,state,fid,rid,proposalId,kind='clipped'){
    if(!['raw','clipped'].includes(kind))throw Error('Unknown mask kind');const source=frame(dataset,fid);
    const candidate=source.candidates.find(c=>c.candidate_id===rid);const proposal=candidate?.proposals.find(p=>p.proposal_id===proposalId);
    if(!proposal)throw Error('Proposal does not belong to this source candidate');
    const f=stateFrame(dataset,state,fid),r=region(dataset,state,fid,rid);
    r.selected_proposal_id=proposalId;r.geometry_mask_kind=kind;r.polygons=clone(proposal['polygons_'+kind]);
    r.geometry_edited=false;r.status='unreviewed';r.extent='unknown';invalidate(f);
    event(state,'proposal_geometry_selected',{frame_id:fid,region_id:rid,proposal_id:proposalId,kind});return state;
  }
  function addManual(dataset,state,fid,polygons){
    const source=frame(dataset,fid);validatePolygons(polygons,source.native_width,source.native_height);const f=stateFrame(dataset,state,fid);
    let rid;do{rid='manual:'+fid+':'+(++state.manual_counter);}while(f.regions[rid]);
    f.regions[rid]={region_id:rid,source_candidate_id:null,source_type:'manual',status:'unreviewed',extent:'unknown',
      selected_proposal_id:null,geometry_mask_kind:null,polygons:clone(polygons),geometry_edited:true,notes:''};
    invalidate(f);event(state,'manual_polygon',{frame_id:fid,region_id:rid});return rid;
  }
  function blockers(dataset,state,fid,ignoreCompleteness=false){
    const f=stateFrame(dataset,state,fid),source=frame(dataset,fid),reasons=[];
    if(!state.rater?.trim())reasons.push('Named rater required');
    if(!ignoreCompleteness&&!f.all_visible_leaves_reviewed)reasons.push('Image not complete-reviewed');
    let accepted=0;
    for(const r of Object.values(f.regions)){
      if(r.status==='unreviewed'||r.status==='uncertain')reasons.push('Unresolved '+r.status+' region '+r.region_id);
      if(r.status==='accepted'){
        accepted++;
        if(r.extent!=='complete_visible_blade')reasons.push('Accepted partial/unknown region is not a complete visible blade');
        try{validatePolygons(r.polygons,source.native_width,source.native_height);}catch(e){reasons.push(e.message);}
      }
    }
    if(!accepted&&!f.verified_empty)reasons.push('No complete instances: explicit verified-empty decision required');
    if(accepted&&f.verified_empty)reasons.push('Verified-empty conflicts with accepted regions');
    return [...new Set(reasons)];
  }
  function setComplete(dataset,state,fid,value){
    const f=stateFrame(dataset,state,fid);
    if(value){const reasons=blockers(dataset,state,fid,true);if(reasons.length)throw Error('Cannot mark complete: '+reasons.join('; '));}
    f.all_visible_leaves_reviewed=Boolean(value);if(value){f.completed_by=state.rater.trim();f.completed_utc=now();}
    event(state,'image_completeness',{frame_id:fid,value:Boolean(value)});return state;
  }
  function setVerifiedEmpty(dataset,state,fid,value){
    const f=stateFrame(dataset,state,fid);
    if(value&&Object.values(f.regions).some(r=>r.status==='accepted'))throw Error('An image with accepted regions cannot be verified empty');
    f.verified_empty=Boolean(value);invalidate(f);event(state,'verified_empty',{frame_id:fid,value:Boolean(value)});return state;
  }
  function serializeReview(dataset,state){
    if(state.dataset_id!==dataset.dataset_id)throw Error('Review dataset mismatch');return clone(state);
  }
  function importReview(dataset,payload){
    if(payload.schema!=='csi-leaf-review-session'||payload.schema_version!==1)throw Error('Unsupported review JSON schema');
    if(payload.dataset_id!==dataset.dataset_id)throw Error('Review dataset ID mismatch');
    const state=createState(dataset),expected=dataset.frames.map(f=>f.frame_id).sort(),provided=Object.keys(payload.frames??{}).sort();
    if(JSON.stringify(expected)!==JSON.stringify(provided))throw Error('Review source frame IDs differ');
    state.rater=String(payload.rater??'');state.created_utc=String(payload.created_utc??now());state.history=Array.isArray(payload.history)?clone(payload.history):[];
    state.manual_counter=Number(payload.manual_counter)||0;
    for(const source of dataset.frames){
      const incoming=payload.frames[source.frame_id];
      if(incoming.source_sha256!==source.source_sha256)throw Error('Review source hash mismatch: '+source.frame_id);
      if(incoming.native_width!==source.native_width||incoming.native_height!==source.native_height)throw Error('Review native dimensions mismatch');
      for(const key of ['treatment','measurement','genotype','replicate','session','camera','view'])if(String(incoming[key])!==String(source[key]))throw Error('Review source identity mismatch: '+key);
      const known=new Set(source.candidates.map(c=>c.candidate_id));
      for(const [rid,r] of Object.entries(incoming.regions??{})){
        if(!known.has(rid)&&!rid.startsWith('manual:'+source.frame_id+':'))throw Error('Unknown source candidate ID');
        if(!statuses.has(r.status)||!extents.has(r.extent))throw Error('Invalid review decision or extent');
        if(r.selected_proposal_id){const c=source.candidates.find(c=>c.candidate_id===rid);if(!c?.proposals.some(p=>p.proposal_id===r.selected_proposal_id))throw Error('Mismatched candidate proposal ID');}
        if(r.polygons?.length)validatePolygons(r.polygons,source.native_width,source.native_height);
      }
      for(const rid of known)if(!incoming.regions[rid])throw Error('Review missing source candidate');
      state.frames[source.frame_id]=clone(incoming);
      if(incoming.all_visible_leaves_reviewed){const why=blockers(dataset,state,source.frame_id,true);if(why.length)throw Error('Imported completeness conflicts with partial/unresolved regions: '+why.join('; '));}
    }
    event(state,'review_import',{frames:expected.length});return state;
  }
  function exportReviewed(dataset,state){
    const eligible=dataset.frames.filter(f=>blockers(dataset,state,f.frame_id).length===0);
    if(!eligible.length)throw Error('No complete-reviewed eligible frames. Resolve regions, confirm completeness, and explicitly verify empty images before export.');
    const coco={info:{description:'Human-reviewed complete visible-blade annotations',dataset_id:dataset.dataset_id,
      rater:state.rater,coordinate_system:'native original-image continuous pixel-edge coordinates',
      partial_regions_excluded:true,unreviewed_predictions_are_separate:true},images:[],annotations:[],
      categories:[{id:1,name:'complete_visible_blade'}]};
    const reviewed=[];
    for(const source of eligible){
      const f=state.frames[source.frame_id],imageId=coco.images.length+1;
      const image={id:imageId,file_name:source.asset_path||source.asset,width:source.native_width,height:source.native_height,
        frame_id:source.frame_id,source_sha256:source.source_sha256,verified_empty:f.verified_empty,
        treatment:source.treatment,measurement:source.measurement,genotype:source.genotype,replicate:source.replicate,
        session:source.session,camera:source.camera,view:source.view,all_visible_leaves_reviewed:true,
        reviewed_by:f.completed_by||state.rater,reviewed_utc:f.completed_utc||null};
      coco.images.push(image);const annotations=[];
      for(const r of Object.values(f.regions).filter(r=>r.status==='accepted'&&r.extent==='complete_visible_blade')){
        const points=r.polygons.flat(),xs=points.map(p=>p[0]),ys=points.map(p=>p[1]);
        const x=Math.min(...xs),y=Math.min(...ys),width=Math.max(...xs)-x,height=Math.max(...ys)-y;
        const annotation={id:coco.annotations.length+1,image_id:imageId,category_id:1,iscrowd:0,
          segmentation:r.polygons.map(ring=>ring.flat()),bbox:[x,y,width,height],area:r.polygons.reduce((s,p)=>s+polygonArea(p),0),
          region_id:r.region_id,source_candidate_id:r.source_candidate_id,source_proposal_id:r.selected_proposal_id,
          geometry_edited:r.geometry_edited,review_extent:r.extent,review_status:'accepted',
          cross_date_identity_verified:false};
        coco.annotations.push(annotation);annotations.push(annotation);
      }
      reviewed.push({image,annotations});
    }
    return {coco,reviewed_json:{schema:'csi-reviewed-visible-blades',schema_version:1,dataset_id:dataset.dataset_id,
      rater:state.rater,frames:reviewed,training_export_requires_complete_review:true},
      excluded_frames:dataset.frames.filter(f=>!eligible.includes(f)).map(f=>({frame_id:f.frame_id,reasons:blockers(dataset,state,f.frame_id)}))};
  }
  function processingPointToOriginal(point,pw,ph,ow,oh){return [(point[0]+.5)*ow/pw,(point[1]+.5)*oh/ph];}
  function originalToProcessing(point,pw,ph,ow,oh){return [point[0]*pw/ow,point[1]*ph/oh];}
  function screenToImage(point,rect,view){return [(point[0]-rect.left-view.x)/view.scale,(point[1]-rect.top-view.y)/view.scale];}
  return {createState,setStatus,setExtent,setPolygons,useProposal,addManual,setComplete,setVerifiedEmpty,serializeReview,
    importReview,exportReviewed,blockers,validatePolygons,polygonArea,processingPointToOriginal,originalToProcessing,screenToImage};
});
