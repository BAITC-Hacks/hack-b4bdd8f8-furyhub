// Логика клиента без браузера. Canvas/vis API заменены небольшими заглушками;
// этот тест не проверяет визуальное отображение самой библиотеки vis-network.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const html = fs.readFileSync(path.join(__dirname, '../out/graph.html'), 'utf8');
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)];
const script = scripts.at(-1)[1];
class DataSet {
    constructor() { this.data = new Map(); }
    clear() { this.data.clear(); }
    add(items) { for(const item of items) this.data.set(item.id,item); }
    get(id) { return this.data.get(id); }
    get length() { return this.data.size; }
}
class Network {
    constructor() { this.events = {}; }
    unselectAll() { this.selection=[]; }
    fit() {}
    selectNodes(ids) { this.selection=ids; }
    focus(gid) { this.focused=gid; }
    on(name, callback) { this.events[name]=callback; }
}
const elements = new Map();
const document = {
    createElement() { return {textContent:''}; },
    getElementById(id) {
        if(!elements.has(id)) elements.set(id, {
            textContent:'',value:'',hidden:false,handlers:{},classList:{toggle(){}},
            addEventListener(event,fn) { this.handlers[event]=fn; }
        });
        return elements.get(id);
    }
};
const context = vm.createContext({document, vis:{DataSet, Network}, assert});
vm.runInContext(script, context);
vm.runInContext(`
assert.equal(nodes.length, initialIds.length);
assert.ok(nodes.length < allNodes.length);
assert.ok(allNodes.every(n => typeof n.id === 'string'));
const submit = gid => {
    document.getElementById('gid').value=gid;
    document.getElementById('search-form').handlers.submit({preventDefault(){}});
};
// Hidden node search changes only the canvas subset and centers exact gid.
const hidden = allNodes.find(n => !nodes.get(n.id) && adjacency.get(n.id).size > 0).id;
submit(hidden);
assert.equal(network.focused, hidden);
assert.equal(network.selection[0], hidden);
assert.equal(nodes.length, neighborhood(hidden).size);
for(const e of edges.data.values()) assert.ok(nodes.get(e.from) && nodes.get(e.to));
assert.equal(document.getElementById('detail-gid').textContent, 'gid '+hidden);
const previousCount=nodes.length;
submit('definitely-missing');
assert.equal(nodes.length, previousCount);
assert.ok(document.getElementById('status').textContent.includes('не найден'));
const isolated=allNodes.find(n => adjacency.get(n.id).size===0).id;
submit(isolated);
assert.equal(network.focused,isolated);
assert.equal(nodes.length,1);
assert.equal(edges.length,0);
document.getElementById('reset').handlers.click();
assert.equal(nodes.length,initialIds.length);
submit(initialIds[0]);
assert.equal(network.focused,initialIds[0]);
assert.equal(nodes.length,initialIds.length);
document.getElementById('neighborhood').handlers.click();
assert.equal(nodes.length,neighborhood(initialIds[0]).size);
`, context);
console.log('OK: exact gid, hidden/visible/isolated search, two-hop selection, missing gid, reset.');
