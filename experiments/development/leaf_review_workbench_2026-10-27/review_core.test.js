const test=require('node:test');
const assert=require('node:assert/strict');
const C=require('./review_core.js');
const polygon=[[[10,10],[40,10],[40,40],[10,40]]];
function fixture(){return {schema:1,dataset_id:'known-fixture',frames:[{frame_id:'frame-A',source_sha256:'sha-A',
  treatment:'C',measurement:'1',genotype:'B73',replicate:'2',session:'unique/session',camera:'rgb',view:0,
  native_width:1000,native_height:500,asset:'assets/original.jpg',candidates:[{candidate_id:'node-A',
  proposals:[{proposal_id:'proposal-A',mode:'distal_points_negatives',alternative:0,score:.9,polygons_clipped:polygon,
  polygons_raw:polygon,review_status:'accepted'}]}]}]};}
function accepted(){const d=fixture(),s=C.createState(d);s.rater='Known fixture rater';
  C.setExtent(d,s,'frame-A','node-A','complete_visible_blade');C.setStatus(d,s,'frame-A','node-A','accepted');
  return {d,s};}

test('new source proposals cannot become accepted or complete without rater actions',()=>{
  const s=C.createState(fixture());assert.equal(s.frames['frame-A'].all_visible_leaves_reviewed,false);
  assert.equal(s.frames['frame-A'].verified_empty,false);assert.equal(s.frames['frame-A'].regions['node-A'].status,'unreviewed');
});
test('accepted complete object still cannot export from incomplete image',()=>{
  const {d,s}=accepted();assert.throws(()=>C.exportReviewed(d,s),/No complete-reviewed/);
});
test('accepted partial region blocks complete image and training export',()=>{
  const {d,s}=accepted();C.setExtent(d,s,'frame-A','node-A','partial');
  assert.throws(()=>C.setComplete(d,s,'frame-A',true),/partial|complete/);
  s.frames['frame-A'].all_visible_leaves_reviewed=true;
  assert.throws(()=>C.exportReviewed(d,s),/No complete-reviewed/);
});
test('complete-reviewed accepted complete blade exports native image and polygon',()=>{
  const {d,s}=accepted();C.setComplete(d,s,'frame-A',true);const result=C.exportReviewed(d,s);
  assert.equal(result.coco.images[0].width,1000);assert.equal(result.coco.images[0].height,500);
  assert.deepEqual(result.coco.annotations[0].bbox,[10,10,30,30]);assert.equal(result.coco.annotations[0].area,900);
});
test('verified negative needs resolved regions, explicit empty and completeness',()=>{
  const d=fixture(),s=C.createState(d);s.rater='Known fixture rater';
  C.setStatus(d,s,'frame-A','node-A','rejected');assert.throws(()=>C.setComplete(d,s,'frame-A',true),/empty/);
  C.setVerifiedEmpty(d,s,'frame-A',true);assert.throws(()=>C.exportReviewed(d,s),/No complete-reviewed/);
  C.setComplete(d,s,'frame-A',true);const result=C.exportReviewed(d,s);
  assert.equal(result.coco.images.length,1);assert.equal(result.coco.annotations.length,0);
  assert.equal(result.coco.images[0].verified_empty,true);
});
test('editing accepted geometry resets acceptance and completeness',()=>{
  const {d,s}=accepted();C.setComplete(d,s,'frame-A',true);
  C.setPolygons(d,s,'frame-A','node-A',[[[20,20],[50,20],[50,50]]]);
  assert.equal(s.frames['frame-A'].regions['node-A'].status,'unreviewed');
  assert.equal(s.frames['frame-A'].all_visible_leaves_reviewed,false);
});
test('review import rejects mismatched source hash instead of joining filenames',()=>{
  const d=fixture(),s=C.createState(d);const payload=C.serializeReview(d,s);
  payload.frames['frame-A'].source_sha256='different';
  assert.throws(()=>C.importReview(d,payload),/source hash/);
});
test('review serialization does not accept or complete anything',()=>{
  const d=fixture(),s=C.createState(d),before=JSON.stringify(s);C.serializeReview(d,s);
  assert.equal(JSON.stringify(s),before);
});
test('coordinate conversion uses native dimensions and declared pixel centers',()=>{
  assert.deepEqual(C.processingPointToOriginal([0,0],100,50,1000,500),[5,5]);
  assert.deepEqual(C.originalToProcessing([250,100],100,50,1000,500),[25,10]);
  assert.deepEqual(C.screenToImage([220,160],{left:20,top:10},{x:100,y:50,scale:2}),[50,50]);
});
test('choosing a different mask alternative resets acceptance and completeness',()=>{
  const {d,s}=accepted();d.frames[0].candidates[0].proposals.push({...d.frames[0].candidates[0].proposals[0],proposal_id:'proposal-B',alternative:1});
  C.setComplete(d,s,'frame-A',true);C.useProposal(d,s,'frame-A','node-A','proposal-B');
  assert.equal(s.frames['frame-A'].regions['node-A'].status,'unreviewed');assert.equal(s.frames['frame-A'].all_visible_leaves_reviewed,false);
});
test('stale imported manual counter cannot overwrite an existing manual region',()=>{
  const d=fixture(),s=C.createState(d);const first=C.addManual(d,s,'frame-A',polygon);s.manual_counter=0;
  const second=C.addManual(d,s,'frame-A',polygon);assert.notEqual(first,second);assert.equal(Object.keys(s.frames['frame-A'].regions).length,3);
});
test('standalone embedded image does not replace the stable COCO asset filename',()=>{
  const {d,s}=accepted();d.frames[0].asset='data:image/jpeg;base64,fixture';d.frames[0].asset_path='assets/native-original.jpg';
  C.setComplete(d,s,'frame-A',true);assert.equal(C.exportReviewed(d,s).coco.images[0].file_name,'assets/native-original.jpg');
});
