(function(){
  'use strict';
  const dataNode=document.getElementById('dataset');let payloadText=dataNode.textContent;
  dataNode.textContent='';dataNode.remove();const D=JSON.parse(payloadText);payloadText=null;
  const C=window.LeafReviewCore,$=id=>document.getElementById(id);
  let S=C.createState(D),frameIndex=0,regionId=null,draft=null,drag=null,space=false,version=0;
  let image=null,overlay=null,view={x:0,y:0,scale:1},previewKind='clipped',hoverVertex=null;
  const canvas=$('canvas'),ctx=canvas.getContext('2d'),wrap=$('canvasWrap');
  const current=()=>D.frames[frameIndex],review=()=>S.frames[current().frame_id],selected=()=>review().regions[regionId];
  function message(text,type=''){const el=$('message');el.textContent=text;el.className=type;}
  function act(fn,success){try{fn();if(success)message(success,'success');refresh();}catch(e){message(e.message,'error');refresh();}}
  function dirty(messageText){message(messageText||'Unsaved review changes. Save the session JSON to resume later.');}
  function getProposal(){const f=current(),r=selected();return f.candidates.find(c=>c.candidate_id===regionId)?.proposals.find(p=>p.proposal_id===r?.selected_proposal_id);}
  function loadImage(src,token,assign){const im=new Image();im.onload=()=>{if(token===version){assign(im);draw();}};im.onerror=()=>{if(token===version)message('Image asset could not load. Keep the HTML with its assets, or use the standalone HTML.','error');};im.src=src;}
  function loadSource(){const token=++version;image=null;overlay=null;const f=current();loadImage($('sourceKind').value==='foreground'?f.foreground_data:f.asset,token,im=>image=im);loadOverlay();}
  function loadOverlay(){overlay=null;const proposal=getProposal();if(proposal&&previewKind!=='none')loadImage(proposal['overlay_'+previewKind],version,im=>overlay=im);else draw();}
  function resize(){const dpr=window.devicePixelRatio||1;canvas.width=Math.round(wrap.clientWidth*dpr);canvas.height=Math.round(wrap.clientHeight*dpr);draw();}
  function fit(){const f=current(),w=wrap.clientWidth,h=wrap.clientHeight;view.scale=Math.min(w/f.native_width,h/f.native_height)*.95;view.x=(w-f.native_width*view.scale)/2;view.y=(h-f.native_height*view.scale)/2;draw();}
  function zoom(factor,cx=wrap.clientWidth/2,cy=wrap.clientHeight/2){const ix=(cx-view.x)/view.scale,iy=(cy-view.y)/view.scale;view.scale=Math.max(.03,Math.min(20,view.scale*factor));view.x=cx-ix*view.scale;view.y=cy-iy*view.scale;draw();}
  function point(event){return C.screenToImage([event.clientX,event.clientY],canvas.getBoundingClientRect(),view);}
  function bounded(xy){return [Math.max(0,Math.min(current().native_width,xy[0])),Math.max(0,Math.min(current().native_height,xy[1]))];}
  function drawRing(ring){ctx.beginPath();ring.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.closePath();}
  function draw(){
    const dpr=window.devicePixelRatio||1,w=wrap.clientWidth,h=wrap.clientHeight;ctx.setTransform(dpr,0,0,dpr,0,0);ctx.fillStyle='#101414';ctx.fillRect(0,0,w,h);
    ctx.setTransform(dpr*view.scale,0,0,dpr*view.scale,dpr*view.x,dpr*view.y);const f=current();
    if(image)ctx.drawImage(image,0,0,f.native_width,f.native_height);
    if(overlay)ctx.drawImage(overlay,0,0,f.native_width,f.native_height);
    for(const [rid,r] of Object.entries(review().regions)){
      if(r.status==='rejected'&&rid!==regionId)continue;
      const active=rid===regionId,color=r.status==='accepted'?'#35dc87':r.status==='rejected'?'#ff6565':r.status==='uncertain'?'#ffcc55':active?'#48d6ff':'#bfcfc9';
      ctx.strokeStyle=color;ctx.fillStyle=color+'23';ctx.lineWidth=(active?2.5:1.2)/view.scale;
      const polygons=active&&drag?.kind==='vertex'&&drag.regionId===rid?drag.preview:r.polygons;
      for(const ring of polygons){drawRing(ring);ctx.fill();ctx.stroke();if(active)for(const p of ring){ctx.fillStyle=color;ctx.beginPath();ctx.arc(p[0],p[1],3.5/view.scale,0,Math.PI*2);ctx.fill();}}
    }
    if($('showPrompts').checked){const p=getProposal();if(p){for(let i=0;i<p.prompt.points.length;i++){
      const xy=C.processingPointToOriginal(p.prompt.points[i],p.processing_width,p.processing_height,f.native_width,f.native_height),positive=p.prompt.labels[i]===1;
      ctx.strokeStyle=positive?'#45ff95':'#ff5252';ctx.fillStyle=ctx.strokeStyle;ctx.lineWidth=2/view.scale;
      if(positive){ctx.beginPath();ctx.arc(xy[0],xy[1],4/view.scale,0,2*Math.PI);ctx.fill();}else{const k=6/view.scale;ctx.beginPath();ctx.moveTo(xy[0]-k,xy[1]-k);ctx.lineTo(xy[0]+k,xy[1]+k);ctx.moveTo(xy[0]-k,xy[1]+k);ctx.lineTo(xy[0]+k,xy[1]-k);ctx.stroke();}
    }}}
    if(draft){ctx.strokeStyle='#fff06a';ctx.lineWidth=2/view.scale;ctx.beginPath();draft.forEach((p,i)=>{if(i)ctx.lineTo(...p);else ctx.moveTo(...p);});ctx.stroke();for(const p of draft){ctx.fillStyle='#fff06a';ctx.beginPath();ctx.arc(...p,4/view.scale,0,Math.PI*2);ctx.fill();}}
    $('coordinates').textContent=`Native source ${f.native_width} × ${f.native_height} · zoom ${(view.scale*100).toFixed(1)}% · coordinates are original-image pixel edges`;
  }
  function refresh(){
    const f=current(),r=selected();$('frameSelect').value=String(frameIndex);$('rater').value=S.rater;
    $('imageIdentity').textContent=`${f.treatment} · visit ${f.measurement} · ${f.genotype} · replicate ${f.replicate} · ${f.camera.toUpperCase()} · index ${f.view}`;
    const list=$('regionList');list.innerHTML='';const ids=Object.keys(review().regions);
    ids.forEach((rid,i)=>{const item=review().regions[rid],button=document.createElement('button');button.className='regionrow'+(rid===regionId?' selected':'');
      button.textContent=`${item.source_type==='manual'?'Manual region':'Terminal hypothesis'} ${i+1} · ${item.status} · ${item.extent==='complete_visible_blade'?'complete visible blade':item.extent}`;
      button.title=rid;button.onclick=()=>{regionId=rid;draft=null;loadOverlay();refresh();};list.appendChild(button);});
    $('regionControls').style.display=r?'block':'none';
    if(r){const p=getProposal();$('extent').value=r.extent;$('notes').value=r.notes||'';
      $('regionStatus').textContent=`${r.status.toUpperCase()}${r.geometry_edited?' · edited geometry':''}`;
      $('proposalMode').disabled=!p;$('adoptMask').disabled=!p||previewKind==='none';
      if(p)$('proposalMode').value=p.mode;
      document.querySelectorAll('.alternative').forEach(b=>{b.disabled=!p;b.classList.toggle('active',Boolean(p&&p.alternative===Number(b.dataset.alt)));});
      $('modelInfo').textContent=p?`Model score ${p.score?.toFixed(3)??'unavailable'} · ${p.processing_width} × ${p.processing_height} proposal grid. Score is not anatomical accuracy.`:'Manual polygon on the original source grid.';
    }
    $('complete').checked=review().all_visible_leaves_reviewed;$('verifiedEmpty').checked=review().verified_empty;
    const reasons=C.blockers(D,S,f.frame_id);$('blockers').textContent=reasons.length?'Training export blocked:\n'+reasons.join('\n'):'This image is eligible for reviewed training export.';
    const complete=D.frames.filter(f=>C.blockers(D,S,f.frame_id).length===0).length;
    const decided=Object.values(S.frames).flatMap(f=>Object.values(f.regions)).filter(r=>r.status!=='unreviewed').length;
    $('progress').textContent=`Image ${frameIndex+1}/${D.frames.length} · ${decided} region decisions · ${complete} complete-reviewed images`;
    $('exportCoco').disabled=$('exportReviewed').disabled=complete===0;
    $('previous').disabled=frameIndex===0;$('next').disabled=frameIndex===D.frames.length-1;
    $('provenance').textContent=JSON.stringify({dataset_id:D.dataset_id,frame_id:f.frame_id,source:f.source,source_sha256:f.source_sha256,
      native_width:f.native_width,native_height:f.native_height,candidate_id:regionId,proposal:getProposal()?{proposal_id:getProposal().proposal_id,
      raw_sha256:getProposal().raw_sha256,clipped_sha256:getProposal().clipped_sha256,prompt_origin:getProposal().prompt_origin}:null},null,2);
    $('canvasNote').textContent=draft?'Drawing: click vertices · Enter finishes · Backspace undoes · Escape cancels':'Wheel: zoom · Space/drag or middle button: pan · Drag vertices to edit · D: new polygon';draw();
  }
  function navigate(index){frameIndex=Math.max(0,Math.min(D.frames.length-1,index));regionId=Object.keys(review().regions)[0]||null;draft=null;drag=null;hoverVertex=null;fit();loadSource();refresh();}
  function chooseAlternative(alt){const r=selected();if(!r)return;const proposal=current().candidates.find(c=>c.candidate_id===regionId)?.proposals.find(p=>p.mode===$('proposalMode').value&&p.alternative===alt);
    if(proposal)act(()=>{C.useProposal(D,S,current().frame_id,regionId,proposal.proposal_id,'clipped');loadOverlay();dirty('Alternative changed. Acceptance and image completeness were reset.');});}
  function startPolygon(){draft=[];drag=null;refresh();message('Click polygon vertices on the original source. Enter finishes; Escape cancels.');}
  function finishPolygon(){if(!draft)return;if(draft.length<3){message('Add at least three vertices.','error');return;}
    act(()=>{regionId=C.addManual(D,S,current().frame_id,[draft]);draft=null;loadOverlay();dirty('Manual polygon added as unreviewed. Choose its extent and make an explicit decision.');});}
  function findVertex(xy){const r=selected();if(!r)return null;let found=null,best=9/view.scale;r.polygons.forEach((ring,ri)=>ring.forEach((p,pi)=>{const d=Math.hypot(p[0]-xy[0],p[1]-xy[1]);if(d<best){best=d;found={ring:ri,point:pi};}}));return found;}
  canvas.addEventListener('pointerdown',e=>{canvas.setPointerCapture(e.pointerId);const xy=bounded(point(e));
    if(e.button===1||space){drag={kind:'pan',sx:e.clientX,sy:e.clientY,x:view.x,y:view.y};return;}
    if(draft){draft.push(xy);draw();return;}const vertex=findVertex(xy);
    if(vertex){drag={kind:'vertex',...vertex,frameId:current().frame_id,regionId,preview:JSON.parse(JSON.stringify(selected().polygons))};hoverVertex=vertex;}
  });
  canvas.addEventListener('pointermove',e=>{const xy=bounded(point(e));if(drag?.kind==='pan'){view.x=drag.x+e.clientX-drag.sx;view.y=drag.y+e.clientY-drag.sy;draw();}
    else if(drag?.kind==='vertex'){drag.preview[drag.ring][drag.point]=xy;draw();}
  });
  canvas.addEventListener('pointerup',()=>{const change=drag;drag=null;if(change?.kind==='vertex'&&change.frameId===current().frame_id&&change.regionId===regionId){
      act(()=>{C.setPolygons(D,S,change.frameId,change.regionId,change.preview);dirty('Geometry edited. Acceptance and image completeness were reset.');});}});
  canvas.addEventListener('pointercancel',()=>{drag=null;hoverVertex=null;draw();});
  canvas.addEventListener('dblclick',e=>{if(draft)return;const r=selected();if(!r)return;const xy=bounded(point(e));let best=null,distance=12/view.scale;
    r.polygons.forEach((ring,ri)=>ring.forEach((a,i)=>{const b=ring[(i+1)%ring.length],dx=b[0]-a[0],dy=b[1]-a[1],t=Math.max(0,Math.min(1,((xy[0]-a[0])*dx+(xy[1]-a[1])*dy)/(dx*dx+dy*dy||1))),d=Math.hypot(xy[0]-a[0]-t*dx,xy[1]-a[1]-t*dy);if(d<distance){distance=d;best={ring:ri,index:i+1};}}));
    if(best)act(()=>{const polygons=JSON.parse(JSON.stringify(r.polygons));polygons[best.ring].splice(best.index,0,xy);C.setPolygons(D,S,current().frame_id,regionId,polygons);dirty();});
  });
  canvas.addEventListener('wheel',e=>{e.preventDefault();const rect=canvas.getBoundingClientRect();zoom(Math.exp(-e.deltaY*.0015),e.clientX-rect.left,e.clientY-rect.top);},{passive:false});
  canvas.addEventListener('contextmenu',e=>e.preventDefault());
  $('previous').onclick=()=>navigate(frameIndex-1);$('next').onclick=()=>navigate(frameIndex+1);$('frameSelect').onchange=()=>navigate(Number($('frameSelect').value));
  $('fit').onclick=fit;$('zoomIn').onclick=()=>zoom(1.25);$('zoomOut').onclick=()=>zoom(.8);$('sourceKind').onchange=loadSource;$('showPrompts').onchange=draw;
  $('maskKind').onchange=()=>{previewKind=$('maskKind').value;loadOverlay();refresh();};
  $('proposalMode').onchange=()=>chooseAlternative(0);document.querySelectorAll('.alternative').forEach(b=>b.onclick=()=>chooseAlternative(Number(b.dataset.alt)));
  $('adoptMask').onclick=()=>act(()=>{const p=getProposal();C.useProposal(D,S,current().frame_id,regionId,p.proposal_id,previewKind);dirty('Displayed mask adopted as unreviewed editable geometry.');});
  $('extent').onchange=()=>act(()=>{C.setExtent(D,S,current().frame_id,regionId,$('extent').value);dirty();});
  for(const [id,status] of [['accept','accepted'],['reject','rejected'],['uncertain','uncertain']])$(id).onclick=()=>act(()=>{if(selected())C.setStatus(D,S,current().frame_id,regionId,status);dirty();});
  $('notes').onchange=()=>{if(selected())selected().notes=$('notes').value;dirty();};
  $('rater').oninput=()=>{S.rater=$('rater').value;refresh();};
  $('complete').onchange=()=>act(()=>C.setComplete(D,S,current().frame_id,$('complete').checked),'Image completeness updated explicitly.');
  $('verifiedEmpty').onchange=()=>act(()=>C.setVerifiedEmpty(D,S,current().frame_id,$('verifiedEmpty').checked),'Verified-empty decision updated. Image completeness remains a separate step.');
  $('newPolygon').onclick=startPolygon;$('undoDraft').onclick=()=>{if(draft){draft.pop();draw();}};$('cancelDraft').onclick=()=>{draft=null;refresh();};
  function download(name,value){const blob=new Blob([JSON.stringify(value,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),2000);}
  $('saveReview').onclick=()=>act(()=>download('leaf_review_session.json',C.serializeReview(D,S)),'Review session downloaded. This file includes partial and unreviewed audit state; it is not a training export.');
  $('exportCoco').onclick=()=>act(()=>download('reviewed_complete_visible_blades_coco.json',C.exportReviewed(D,S).coco),'Only eligible complete-reviewed images were exported.');
  $('exportReviewed').onclick=()=>act(()=>download('reviewed_complete_visible_blades.json',C.exportReviewed(D,S).reviewed_json),'Only eligible complete-reviewed images were exported.');
  $('importReview').onclick=()=>$('importFile').click();$('importFile').onchange=async()=>{const file=$('importFile').files[0];if(!file)return;try{const imported=C.importReview(D,JSON.parse(await file.text()));S=imported;regionId=Object.keys(review().regions)[0]||null;loadOverlay();message('Review resumed after source hash, identity and native-dimension validation.','success');refresh();}catch(e){message('Import rejected: '+e.message,'error');}$('importFile').value='';};
  document.addEventListener('keydown',e=>{if(['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName))return;const key=e.key.toLowerCase();
    if(e.code==='Space'){space=true;e.preventDefault();return;}if(key==='arrowright'){navigate(frameIndex+1);e.preventDefault();}else if(key==='arrowleft'){navigate(frameIndex-1);e.preventDefault();}
    else if(key==='d')startPolygon();else if(key==='enter')finishPolygon();else if(key==='escape'){draft=null;drag=null;refresh();}
    else if(key==='backspace'&&draft){draft.pop();draw();e.preventDefault();}else if(['1','2','3'].includes(key))chooseAlternative(Number(key)-1);
    else if(['a','r','u'].includes(key))$(key==='a'?'accept':key==='r'?'reject':'uncertain').click();
    else if(key==='p'||key==='c'){if(selected())act(()=>C.setExtent(D,S,current().frame_id,regionId,key==='p'?'partial':'complete_visible_blade'));}
    else if(key==='f')fit();else if(key==='+'||key==='=')zoom(1.25);else if(key==='-')zoom(.8);
    else if(key==='j'||key==='k'){const ids=Object.keys(review().regions),i=ids.indexOf(regionId);regionId=ids[Math.max(0,Math.min(ids.length-1,i+(key==='j'?1:-1)))]||null;loadOverlay();refresh();}
    else if(key==='delete'&&hoverVertex&&selected()){const polygons=JSON.parse(JSON.stringify(selected().polygons));if(polygons[hoverVertex.ring]?.length>3){polygons[hoverVertex.ring].splice(hoverVertex.point,1);act(()=>C.setPolygons(D,S,current().frame_id,regionId,polygons));}hoverVertex=null;}
  });
  document.addEventListener('keyup',e=>{if(e.code==='Space')space=false;});window.addEventListener('blur',()=>space=false);
  D.frames.forEach((f,i)=>{const option=document.createElement('option');option.value=i;option.textContent=`${i+1}. ${f.treatment} visit ${f.measurement} · ${f.genotype} r${f.replicate} · ${f.camera} index ${f.view}`;$('frameSelect').appendChild(option);});
  new ResizeObserver(()=>resize()).observe(wrap);navigate(0);
  window.LeafReviewApp={dataset:D,getState:()=>C.serializeReview(D,S),getCurrentFrame:()=>current().frame_id};
})();
