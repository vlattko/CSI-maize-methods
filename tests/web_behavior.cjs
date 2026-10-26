const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const web = path.join(__dirname, '../src/csi_maize/web');

function bridgeChecks() {
  let navigate, exportClick, sent;
  let scrolled = false;
  const parent = {postMessage(value) {sent = value;}};
  const button = {addEventListener(type, handler) {assert.equal(type, 'click'); exportClick = handler;}};
  const destination = {scrollIntoView() {scrolled = true;}};
  const document = {
    addEventListener(type, handler) {assert.equal(type, 'click'); navigate = handler;},
    getElementById(id) {return id === 'download' ? button : id === 'explorer' ? destination : null;},
    querySelector() {return {textContent: 'Investigation Δ\nFixed receiver.'};},
  };
  vm.runInNewContext(fs.readFileSync(path.join(web, 'atlas-bridge.js'), 'utf8'), {
    document, window: {parent},
  });
  let prevented = false;
  const internal = {getAttribute() {return '#explorer';}};
  navigate({target: {closest() {return internal;}}, preventDefault() {prevented = true;}});
  assert.equal(prevented, true, 'embedded fragments must not navigate to the parent report URL');
  assert.equal(scrolled, true);
  assert.equal(internal.target, undefined);
  const external = {getAttribute() {return 'https://github.com/vlattko/CSI-maize-methods';}};
  navigate({target: {closest() {return external;}}, preventDefault() {throw Error('external navigation blocked');}});
  assert.equal(external.target, '_blank');
  assert.equal(external.rel, 'noopener noreferrer');
  let stopped = false;
  exportClick({preventDefault() {}, stopImmediatePropagation() {stopped = true;}});
  assert.equal(stopped, true, 'legacy blob handler must not produce a second export');
  assert.equal(sent.type, 'csi-maize-investigation-export');
  assert.equal(sent.text, 'Investigation Δ\nFixed receiver.');
}

async function hostChecks() {
  let receive;
  let frameAvailable = true;
  const source = {};
  const frame = {contentWindow: source};
  const blobs = [];
  const downloads = [];
  let removed = 0, revoked = 0;
  const document = {
    getElementById(id) {return id === 'architecture-data' ? {textContent: '{"mesh":null,"reposition":[]}'} : null;},
    querySelector() {return frameAvailable ? frame : null;},
    createElement(tag) {
      assert.equal(tag, 'a');
      return {click() {downloads.push({name: this.download, href: this.href});}, remove() {removed++;}};
    },
    body: {appendChild() {}},
  };
  const window = {addEventListener(type, handler) {if (type === 'message') receive = handler;}};
  vm.runInNewContext(fs.readFileSync(path.join(web, 'report.js'), 'utf8'), {
    document, window, Blob,
    URL: {createObjectURL(blob) {blobs.push(blob); return 'blob:verified-export';}, revokeObjectURL() {revoked++;}},
    setTimeout(callback) {callback();},
  });
  const text = 'Investigation Δ\nFixed receiver.';
  const valid = {type: 'csi-maize-investigation-export', text};
  receive({source: {}, data: valid});
  receive({source, data: {type: 'other', text}});
  receive({source, data: {type: valid.type, text: 12}});
  receive({source, data: {type: valid.type, text: 'x'.repeat(1000001)}});
  frameAvailable = false;
  receive({source, data: valid});
  assert.equal(downloads.length, 0, 'only the active atlas may request a bounded text export');
  frameAvailable = true;
  receive({source, data: valid});
  assert.equal(downloads.length, 1);
  assert.equal(downloads[0].name, 'CSI_maize_spatial_fusion_methods.txt');
  assert.equal(await blobs[0].text(), text);
  assert.equal(blobs[0].type, 'text/plain;charset=utf-8');
  assert.equal(removed, 1);
  assert.equal(revoked, 1);
}

(async () => {
  bridgeChecks();
  await hostChecks();
  console.log('Web interaction checks passed: embedded navigation, isolated export and message guards');
})().catch(error => {console.error(error); process.exitCode = 1;});
