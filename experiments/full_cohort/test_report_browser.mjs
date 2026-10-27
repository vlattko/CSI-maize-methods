// Deterministic generated-HTML regression check without network or model assets.
import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';

const page = fs.readFileSync(process.argv[2], 'utf8');
const data = page.match(/<script id="cohortdata" type="application\/json">([\s\S]*?)<\/script>/)[1];
const script = page.match(/<script>([\s\S]*?)<\/script>/)[1];
const elements = new Map();
class Element {
  constructor(tag = 'div') { this.tag = tag; this.children = []; this.value = ''; this.listeners = {}; }
  append(...items) { this.children.push(...items); }
  replaceChildren(...items) { this.children = items; }
  setAttribute(name, value) { this[name] = value; }
  addEventListener(name, fn) { this.listeners[name] = fn; }
  click() { this.listeners.click?.(); }
  set innerHTML(value) { this.html = value; if (!value) this.children = []; }
}
for (const id of ['cohortdata','treatment','genotype','replicate','view','download','rgbphotos','noirphotos','trajectorycharts','profiles','profilevisit']) elements.set(id,new Element());
elements.get('cohortdata').textContent = data;
for (const [key,value] of Object.entries({treatment:'C',replicate:'1',view:'0',profilevisit:'2'})) elements.get(key).value = value;
let downloaded;
const context = vm.createContext({
  document:{getElementById:id=>elements.get(id),createElement:tag=>new Element(tag)},
  Blob,URL:{createObjectURL:blob=>{downloaded=blob;return 'blob:local';},revokeObjectURL:()=>{}},
  setTimeout:()=>{},
});
vm.runInContext(script,context,{timeout:10000});
assert.equal(elements.get('genotype').children.length,6);
let combinations=0;
for (const treatment of ['C','P','S']) for (const option of elements.get('genotype').children) for (const replicate of ['1','2','3','4']) {
  for (const [id,value] of Object.entries({treatment,genotype:option.value,replicate})) elements.get(id).value=value;
  elements.get('treatment').listeners.change();
  for (const camera of ['rgb','noir']) {
    const cards=elements.get(camera+'photos').children;
    assert.equal(cards.length,5);
    for (const card of cards) {
      const visual=card.children[0];
      if (visual.tag==='img') assert.ok(visual.src.startsWith('data:image/jpeg;base64,'));
      else assert.equal(visual.textContent,'Image unavailable');
    }
  }
  assert.equal(elements.get('trajectorycharts').children.length,4);
  assert.equal(elements.get('profiles').children.length,2);
  combinations++;
}
elements.get('download').click();
const csv=await downloaded.text();
assert.equal(csv.trim().split('\r\n').length,6);
assert.ok(csv.includes('"Fv_Fm"'));
assert.equal(combinations,72);
console.log(JSON.stringify({status:'passed',candidate_combinations:combinations,csv_rows:5,images:'embedded or explicit unavailable'}));
