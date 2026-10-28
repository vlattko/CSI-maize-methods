import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import zlib from 'node:zlib';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';

const root=path.dirname(fileURLToPath(import.meta.url));
const file=process.argv[2]||path.join(root,'workbench.html');
const html=fs.readFileSync(file,'utf8');
const source=html.match(/<script id="workbench-program">([\s\S]*?)<\/script>/)?.[1];
assert.ok(source,'Inline application program exists');
assert.ok(source.indexOf("$('workbench-payload').remove()")>0&&source.indexOf("$('workbench-payload').remove()")<source.indexOf("const bytes=Uint8Array"),'Payload DOM node is released before decompression');
assert.ok(/<option value="foreground" selected>/.test(html),'Foreground is the default image display');
const sandbox={console};vm.createContext(sandbox);vm.runInContext(source,sandbox);
const L=sandbox.WorkbenchLogic;assert.ok(L,'Pure selection logic is exported');
assert.equal(L.keyOf({treatment:'C',measurement:2,genotype:'Os6251/15',replicate:4}),'["C","2","Os6251/15","4"]');
assert.equal(L.candidateKey({treatment:'C',genotype:'B73',replicate:1}),'C|B73|1');
assert.equal(JSON.stringify(L.frameBox(39,384,288,8)),JSON.stringify([2688,1152,384,288]));
assert.throws(()=>L.frameBox(40));
const row={deployment_abstain:false,view_matrices_rgb_to_noir:{17:[[2,0,10],[0,2,20]]}};
assert.equal(JSON.stringify(L.cameraMatrix(row,17)),'[[2,0,10],[0,2,20]]');
assert.equal(L.cameraMatrix({...row,deployment_abstain:true},17),null);
const inv=L.inverseScaledAffine([[2,0,10],[0,2,20]],384,288,256,192,256,192);
const p=L.applyAffine(inv,[45,60]);assert.ok(Math.abs(p[0]-15)<1e-9&&Math.abs(p[1]-15)<1e-9);
assert.equal(L.noirView({selected_sign:-1,selected_phase:3},5),38);
assert.equal(L.canProjectPaths({...row,selected_phase:3},17,17),false);
assert.equal(L.canProjectPaths({...row,selected_phase:3},17,20),true);

let payload;const text=html.match(/<script id="workbench-payload" type="application\/octet-stream">([\s\S]*?)<\/script>/)?.[1]?.trim();
if(text&&!text.startsWith('__')){
 payload=JSON.parse(zlib.gunzipSync(Buffer.from(text,'base64')).toString('utf8'));
}else{
 const graphs=[],slots=[],reference=[];
 for(const treatment of ['C','P','S'])for(const genotype of ['a','b','c','d','e','f'])for(const replicate of ['1','2','3','4']){
  const candidate_id=[treatment,genotype,replicate].join('|');const sessions=[];
  for(let measurement=1;measurement<=5;measurement++){
   const key={treatment,genotype,replicate,measurement:String(measurement)};reference.push({...key,'ojip__Fv/Fm':'0.8','ojip__Pi_Abs':'2','nir__NDVI':'0.5'});
   sessions.push({visit:measurement,nodes:[]});
   for(const camera of ['rgb','noir'])slots.push({id:L.keyOf(key)+camera,key,camera,frames:Array.from({length:40},(_,view)=>({view,present:view!==7,qc_pass:true}))});
  }graphs.push({candidate_id,identity:{treatment,genotype,replicate},sessions,nodes:[],edges:[],pairs:[],chains:[]});
 }payload={graphs,slots,reference,assets:{},camera:[],metadata:{fixture:true}};
}
const model=L.buildModel(payload);assert.equal(model.graphs.size,72,'Every candidate identity is present');
let selectionCount=0;
for(const graph of model.graphs.values())for(let measurement=1;measurement<=5;measurement++){
 const state={...graph.identity,measurement:String(measurement),view:39};const s=L.selectRecord(model,state);
 assert.equal(s.graph.candidate_id,graph.candidate_id);assert.ok(s.reference,'Reference row is retained');
 for(const camera of ['rgb','noir']){assert.equal(s.slots[camera].frames.length,40);assert.equal(s.slots[camera].frames[39].view,39);}
 const exported=JSON.parse(JSON.stringify(L.exportSelection(model,state)));
 assert.equal(exported.identity.measurement,String(measurement));assert.equal(exported.view,39);
 const identity=payload.identityRegistry?.find(r=>L.keyOf(r)===L.keyOf(state));
 assert.equal(exported.physical_identity_status,identity?.physical_identity_status||'metadata_candidate_unverified');selectionCount++;
}
assert.equal(selectionCount,360);
const confirmed={...payload.reference[0],plant_uid:'confirmed-pot',declared_das:13,physical_identity_status:'experimenter_confirmed_same_pot'};
const cm=L.buildModel({...payload,identityRegistry:[confirmed]});
const ce=L.exportSelection(cm,{...confirmed,view:0});
assert.equal(ce.physical_identity_status,'experimenter_confirmed_same_pot');
assert.equal(ce.plant_uid,'confirmed-pot');assert.equal(ce.declared_das,13);
assert.ok(!/\bfetch\s*\(|XMLHttpRequest|https?:\/\/[^\s'"<]+\.(js|css)/.test(source),'Application makes no external requests');
assert.ok(!/<(?:script|link)[^>]+(?:src|href)=/i.test(html),'No external scripts or styles');
console.log(JSON.stringify({status:'passed',file,candidate_selections:72,reference_selections:selectionCount,fixed_frames:40,affine_inverse_checked:true,no_external_requests:true,payload_assets:Object.keys(payload.assets||{}).length}));
