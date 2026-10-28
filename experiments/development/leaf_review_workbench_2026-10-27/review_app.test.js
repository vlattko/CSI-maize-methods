// Actual application handlers exercised against a synthetic, offline DOM fixture.
const test=require('node:test'),assert=require('node:assert/strict'),vm=require('node:vm'),fs=require('node:fs'),path=require('node:path');
const C=require('./review_core.js');
function harness(){
  const data={schema:1,dataset_id:'ui-fixture',frames:[{frame_id:'f',source_sha256:'known-source',treatment:'C',measurement:'1',genotype:'B73',replicate:'1',session:'known/session',camera:'rgb',view:0,native_width:1000,native_height:500,asset:'fixture.jpg',foreground_data:'fixture.png',source:'known/source.jpg',candidates:[{candidate_id:'candidate',proposals:[{proposal_id:'proposal',mode:'distal_points_negatives',alternative:0,score:.9,processing_width:100,processing_height:50,prompt:{points:[],labels:[]},overlay_clipped:'fixture.png',overlay_raw:'fixture.png',polygons_clipped:[[[100,100],[200,100],[200,200],[100,200]]],polygons_raw:[[[100,100],[200,100],[200,200],[100,200]]]}]}]}]};
  const elements=new Map(),documentEvents={};
  class El{constructor(id='',tag='DIV'){this.id=id;this.tagName=tag;this.style={};this.dataset={};this.children=[];this.events={};this.classList={toggle(){}};this.value='';this.checked=false;this.disabled=false;this.textContent='';this.clientWidth=900;this.clientHeight=600;}
    set innerHTML(v){this.children=[];}get innerHTML(){return '';}
    appendChild(el){this.children.push(el);}remove(){this.removed=true;}
    addEventListener(name,fn){this.events[name]=fn;}click(){if(this.onclick)this.onclick({});}
    getBoundingClientRect(){return {left:0,top:0,width:900,height:600};}setPointerCapture(){}
    getContext(){return {setTransform(){},fillRect(){},drawImage(){},beginPath(){},moveTo(){},lineTo(){},closePath(){},fill(){},stroke(){},arc(){}};}}
  const shell=fs.readFileSync(path.join(__dirname,'workbench_shell.html'),'utf8');
  for(const m of shell.matchAll(/<(\w+)[^>]*\bid="([^"]+)"/g))elements.set(m[2],new El(m[2],m[1].toUpperCase()));
  const datasetNode=new El('dataset','SCRIPT');datasetNode.textContent=JSON.stringify(data);elements.set('dataset',datasetNode);
  const alternatives=[0,1,2].map(n=>{const e=new El();e.dataset.alt=String(n);return e;});
  elements.get('sourceKind').value='original';elements.get('proposalMode').value='distal_points_negatives';elements.get('maskKind').value='clipped';
  const document={getElementById:id=>elements.get(id),createElement:tag=>new El('',tag.toUpperCase()),querySelectorAll:s=>s==='.alternative'?alternatives:[],addEventListener:(name,fn)=>documentEvents[name]=fn,activeElement:{tagName:'BODY'}};
  class Img{set src(v){this._src=v;if(this.onload)this.onload();}}
  const window={LeafReviewCore:C,devicePixelRatio:1,addEventListener(){}};
  const context={window,document,Image:Img,ResizeObserver:class{constructor(cb){this.cb=cb;}observe(){this.cb();}},Blob,URL:{createObjectURL:()=> 'blob:test',revokeObjectURL(){}},setTimeout:fn=>fn(),console};
  vm.runInNewContext(fs.readFileSync(path.join(__dirname,'review_app.js'),'utf8'),context,{filename:'review_app.js'});
  return {elements,window,events:documentEvents,getState:()=>window.LeafReviewApp.getState(),source:data};
}
test('UI initialization removes embedded payload and leaves all decisions false',()=>{
  const h=harness();assert.equal(h.elements.get('dataset').textContent,'');assert.equal(h.elements.get('dataset').removed,true);
  assert.equal(h.getState().frames.f.all_visible_leaves_reviewed,false);assert.equal(h.elements.get('exportCoco').disabled,true);
});
test('actual UI partial acceptance cannot mark the image complete or enable export',()=>{
  const h=harness();h.elements.get('rater').value='Fixture only';h.elements.get('rater').oninput();
  h.elements.get('extent').value='partial';h.elements.get('extent').onchange();h.elements.get('accept').click();
  h.elements.get('complete').checked=true;h.elements.get('complete').onchange();
  assert.equal(h.getState().frames.f.all_visible_leaves_reviewed,false);assert.equal(h.elements.get('exportCoco').disabled,true);
  assert.match(h.elements.get('message').textContent,/partial/);
});
test('manual polygon clicks are stored in native original coordinates and remain unreviewed',()=>{
  const h=harness();h.elements.get('newPolygon').click();const canvas=h.elements.get('canvas');
  for(const [x,y] of [[108,171.75],[193.5,171.75],[193.5,257.25]])canvas.events.pointerdown({button:0,pointerId:1,clientX:x,clientY:y});
  h.events.keydown({key:'Enter',code:'Enter',preventDefault(){}});
  const manual=Object.values(h.getState().frames.f.regions).find(r=>r.source_type==='manual');
  assert.ok(manual);assert.ok(Math.abs(manual.polygons[0][0][0]-100)<1e-9);assert.ok(Math.abs(manual.polygons[0][2][1]-200)<1e-9);
  assert.equal(manual.status,'unreviewed');assert.equal(h.getState().frames.f.all_visible_leaves_reviewed,false);
});
test('cancelling a vertex drag cannot silently edit accepted geometry',()=>{
  const h=harness();h.elements.get('rater').value='Fixture only';h.elements.get('rater').oninput();
  h.elements.get('extent').value='complete_visible_blade';h.elements.get('extent').onchange();h.elements.get('accept').click();
  h.elements.get('complete').checked=true;h.elements.get('complete').onchange();const before=JSON.stringify(h.getState().frames.f);
  const canvas=h.elements.get('canvas');canvas.events.pointerdown({button:0,pointerId:1,clientX:108,clientY:171.75});
  canvas.events.pointermove({clientX:150.75,clientY:214.5});
  assert.equal(JSON.stringify(h.getState().frames.f),before);
  canvas.events.pointercancel({});assert.equal(JSON.stringify(h.getState().frames.f),before);
});
test('actual file-import UI rejects a mismatched source without replacing review state',async()=>{
  const h=harness(),payload=h.getState();payload.frames.f.source_sha256='mismatch';
  h.elements.get('importFile').files=[{text:async()=>JSON.stringify(payload)}];await h.elements.get('importFile').onchange();
  assert.match(h.elements.get('message').textContent,/Import rejected:.*source hash/);assert.equal(h.getState().frames.f.source_sha256,'known-source');
});
