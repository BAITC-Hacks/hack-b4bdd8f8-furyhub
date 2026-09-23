// Клиент выполняется без браузера: проверяем переходы и данные canvas,
// а визуальную читаемость vis-network проверяем отдельно в браузере.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');

const html = fs.readFileSync(path.join(__dirname, '../out/graph.html'), 'utf8');
const scripts = [...html.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script>/g)];
const script = scripts.at(-1)[1];
assert.match(script, /const DATA\s*=/, 'Сгенерируйте новый out/graph.html: python viz.py');

class Element {
    constructor(tag = 'div') {
        this.tagName = tag.toUpperCase();
        this.children = [];
        this.parentElement = null;
        this.dataset = {};
        this.style = {};
        this.attributes = {};
        this.handlers = {};
        this.hidden = false;
        this.disabled = false;
        this.value = '';
        this._text = '';
        this._classes = new Set();
        this.classList = {
            add: (...classes) => classes.forEach(name => this._classes.add(name)),
            remove: (...classes) => classes.forEach(name => this._classes.delete(name)),
            contains: name => this._classes.has(name),
            toggle: (name, force) => {
                const enabled = force === undefined ? !this._classes.has(name) : force;
                if (enabled) this._classes.add(name); else this._classes.delete(name);
                return enabled;
            },
        };
    }
    set className(value) { this._classes = new Set(String(value).split(/\s+/).filter(Boolean)); }
    get className() { return [...this._classes].join(' '); }
    set textContent(value) { this._text = String(value ?? ''); this.children = []; }
    get textContent() { return this._text + this.children.map(child => child.textContent).join(''); }
    set innerHTML(_) { throw new Error('Текст из данных нельзя вставлять через innerHTML'); }
    append(...children) {
        for (const child of children) {
            const element = typeof child === 'string' ? Object.assign(new Element('#text'), {_text: child}) : child;
            element.parentElement = this;
            this.children.push(element);
        }
    }
    appendChild(child) { this.append(child); return child; }
    replaceChildren(...children) { this._text = ''; this.children = []; this.append(...children); }
    removeChild(child) { this.children.splice(this.children.indexOf(child), 1); return child; }
    get firstChild() { return this.children[0] || null; }
    setAttribute(name, value) {
        this.attributes[name] = String(value);
        if (name === 'class') this.className = value;
        if (name.startsWith('data-')) this.dataset[name.slice(5)] = String(value);
    }
    getAttribute(name) { return this.attributes[name] ?? null; }
    removeAttribute(name) { delete this.attributes[name]; }
    addEventListener(name, callback) { (this.handlers[name] ||= []).push(callback); }
    dispatch(name, detail = {}) {
        const event = {target: this, currentTarget: this, preventDefault() {},
            stopPropagation() { this.stopped = true; }, ...detail};
        for (let element = this; element; element = element.parentElement) {
            event.currentTarget = element;
            for (const callback of element.handlers[name] || []) callback(event);
            if (event.stopped) break;
        }
        return event;
    }
    click() { if (!this.disabled) this.dispatch('click'); }
    focus() { this.focused = true; }
    scrollIntoView(options) { this.scrolled = options; }
    matches(selector) {
        const compound = selector.match(/^([a-z]+)(\[.*\])$/i);
        if (compound) return this.matches(compound[1]) && this.matches(compound[2]);
        if (selector === '[data-gid]') return this.dataset.gid !== undefined;
        if (selector.startsWith('.')) return this.classList.contains(selector.slice(1));
        if (selector.startsWith('#')) return this.id === selector.slice(1);
        const gid = selector.match(/^\[data-gid=["']?(.*?)["']?\]$/);
        if (gid) return this.dataset.gid === gid[1];
        return this.tagName.toLowerCase() === selector.toLowerCase();
    }
    closest(selector) { return this.matches(selector) ? this : this.parentElement?.closest(selector) || null; }
    querySelectorAll(selector) {
        return this.children.flatMap(child => [
            ...(child.matches(selector) ? [child] : []), ...child.querySelectorAll(selector),
        ]);
    }
    querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
}

class DataSet {
    constructor(items = []) { this.data = new Map(); this.add(items); }
    clear() { this.data.clear(); }
    add(items) {
        for (const item of Array.isArray(items) ? items : [items]) {
            assert.equal(typeof item.id, 'string', 'Идентификатор canvas должен быть строкой');
            assert.ok(!this.data.has(item.id), `Повторный id ${item.id}`);
            this.data.set(item.id, item);
        }
    }
    update(items) { for (const item of Array.isArray(items) ? items : [items]) this.data.set(item.id, {...this.data.get(item.id), ...item}); }
    get(id) { return id === undefined ? [...this.data.values()] : this.data.get(id); }
    getIds() { return [...this.data.keys()]; }
    forEach(callback) { this.data.forEach(callback); }
    get length() { return this.data.size; }
}

class Network {
    constructor(container, data, options) { this.events = {}; this.options = options; this.scale = 1.2; }
    unselectAll() { this.selection = []; }
    fit() { this.scale = 1.2; this.fitCalls = (this.fitCalls || 0) + 1; }
    getScale() { return this.scale; }
    moveTo(options) { if (options.scale !== undefined) this.scale = options.scale; }
    selectNodes(ids, connected) { this.selection = ids; this.connected = connected; }
    focus(gid, options = {}) { this.focused = gid; if (options.scale) this.scale = options.scale; }
    on(name, callback) { this.events[name] = callback; }
    once(name, callback) { callback(); }
    setOptions(options) { this.options = {...this.options, ...options}; }
    redraw() {}
}

function load(data) {
    const elements = new Map();
    const document = {
        createElement(tag) { return new Element(tag); },
        createTextNode(text) { const node = new Element('#text'); node.textContent = text; return node; },
        getElementById(id) {
            if (!elements.has(id)) { const element = new Element(); element.id = id; elements.set(id, element); }
            return elements.get(id);
        },
        querySelectorAll(selector) { return [...elements.values()].flatMap(element => element.querySelectorAll(selector)); },
    };
    // Сохраняем реальные типы статических элементов, в частности кнопок и формы.
    for (const match of html.matchAll(/<([a-z][\w-]*)\b[^>]*\bid="([^"]+)"[^>]*>/gi)) {
        const element = document.getElementById(match[2]);
        element.tagName = match[1].toUpperCase();
        element.hidden = /\bhidden(?:\s|>|=)/.test(match[0]);
        const leaf = html.slice(match.index + match[0].length).match(new RegExp(`^([^<]*)</${match[1]}>`));
        if (leaf) element.textContent = leaf[1];
    }
    const context = vm.createContext({document, vis: {DataSet, Network}, console,
        navigator: {clipboard: {writeText: async () => {}}},
        requestAnimationFrame: callback => callback(), setTimeout: callback => callback(), clearTimeout() {},
    });
    const client = data ? script.replace(/const DATA\s*=\s*[^\n]*;(?=\s*\n)/, () => `const DATA = ${JSON.stringify(data)};`) : script;
    vm.runInContext(client, context, {filename: 'graph-client.js'});
    const evaluate = expression => vm.runInContext(expression, context);
    return {
        evaluate, document, elements,
        get(id) { return document.getElementById(id); },
        submit(query) {
            document.getElementById('gid').value = query;
            document.getElementById('search-form').dispatch('submit');
        },
        current() { return evaluate('currentGid'); },
        mode() { return evaluate('currentMode'); },
        canvas() { return evaluate('nodes'); },
        edges() { return evaluate('edges'); },
        network() { return evaluate('network'); },
        queue() { return document.getElementById('queue').querySelectorAll('button').filter(button => button.dataset.gid); },
    };
}

const roleLabels = {coordinator: 'Координатор', distributor: 'Распределитель', consolidator: 'Консолидатор', transit: 'Транзит', terminal: 'Конечный получатель', peripheral: 'Периферия'};
function fixture() {
    const makeId = n => (910000000000000000n + BigInt(n) * 1000n + 100n).toString();
    const center = makeId(1), isolated = makeId(99), outside = makeId(98);
    const incoming = Array.from({length: 15}, (_, i) => makeId(10 + i));
    const outgoing = Array.from({length: 15}, (_, i) => makeId(30 + i));
    const mutual = Array.from({length: 7}, (_, i) => makeId(50 + i));
    const ambiguous = ['910000000001231234', '910000000008881234'];
    const ids = [center, ...incoming, ...outgoing, ...mutual, outside, isolated, ...ambiguous];
    const nodes = ids.map((id, index) => ({
        id, gid: id, role: index === 0 ? 'coordinator' : 'transit',
        role_label: index === 0 ? roleLabels.coordinator : roleLabels.transit,
        role_meaning: index === 0 ? 'связывает несколько групп, кандидат в организаторы' : 'пропускает деньги дальше, не удерживая',
        color: index === 0 ? '#e15759' : '#4e9ee8', role_score: .66,
        priority_score: 1 - index / 100, evidence: 'Гипотеза: 3 перевода требуют проверки.',
        is_seed: index === 0, cluster_id: '2', depth: index === 0 ? 0 : 1,
        truncated_by_depth: false, in_kzt: 0, out_kzt: 0, in_deg: 0, out_deg: 0,
        in_tx: 0, out_tx: 0, x: index * 5, y: index * 3,
    }));
    const edges = [];
    const add = (from, to, sum_kzt, n_tx = 3) => edges.push({id: `e${edges.length}`, from, to, sum_kzt, n_tx});
    incoming.forEach((id, i) => add(id, center, (i + 1) * 1000));
    outgoing.forEach((id, i) => add(center, id, (i + 1) * 2000));
    mutual.forEach((id, i) => { add(id, center, (i + 1) * 3000); add(center, id, (i + 1) * 4000); });
    add(incoming[14], outgoing[14], 999999); // В окружении запрещено ребро между соседями.
    add(outside, incoming[14], 850, 1); // Узел вне топа на втором шаге от центра.
    const byId = new Map(nodes.map(node => [node.id, node]));
    for (const edge of edges) {
        const source = byId.get(edge.from), target = byId.get(edge.to);
        source.out_kzt += edge.sum_kzt; source.out_tx += edge.n_tx; source.out_deg++;
        target.in_kzt += edge.sum_kzt; target.in_tx += edge.n_tx; target.in_deg++;
    }
    byId.get(isolated).role = 'peripheral'; byId.get(isolated).role_label = roleLabels.peripheral;
    byId.get(isolated).truncated_by_depth = true; byId.get(isolated).depth = 4;
    byId.get(outside).evidence = '</script><img src=x onerror="alert(1)">Гипотеза: 1 перевод требует проверки.';
    const top = [{gid: center, rank: 1, why: 'Проверить 22 плательщика.'}, {gid: incoming[0], rank: 2, why: 'Проверить 3 перевода.'}];
    let shortLen = 1;
    while (new Set(ids.map(id => id.slice(-shortLen))).size !== ids.length) shortLen++;
    return {
        data: {nodes, edges, top, funnel: {seeds: 1, nodes: nodes.length, top: top.length},
            shortLen, initialGid: center, initialMode: 'neighborhood', overviewIds: ids,
            overviewLabel: 'Вся сеть', hints: {[center]: {attention: 'Сопоставить 22 входящих потока.', next_request: 'Запросить время переводов.'}}},
        center, isolated, outside, incoming, outgoing, mutual, ambiguous,
    };
}

function assertSelected(app, gid) {
    assert.equal(app.current(), gid);
    assert.match(app.get('detail-gid').textContent, new RegExp(gid));
}
function assertNeighborhood(app, gid) {
    assert.ok(app.canvas().get(gid), 'Центр отсутствует на canvas');
    assert.ok(app.canvas().length <= 30, 'Лимит: центр + 12 плательщиков + 12 получателей + 5 встречных');
    for (const edge of app.edges().get()) {
        assert.ok(edge.from === gid || edge.to === gid, 'В окружении ребро должно проходить через центр');
        assert.ok(app.canvas().get(edge.from) && app.canvas().get(edge.to));
    }
}
function assertNoTechnicalWords(app) {
    const banned = /\b(?:priority_score|role_score|pass_through|sum_kzt|n_tx|in_deg|out_deg|coordinator|consolidator|distributor|terminal|peripheral|transit)\b|log\s*\(\s*1\s*\+|∝/i;
    const staticMarkup = html.replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, '').replace(/<style\b[^>]*>[\s\S]*?<\/style>/gi, '').replace(/<[^>]*>/g, '');
    assert.doesNotMatch(staticMarkup, banned, 'Технические слова в видимой разметке');
    for (const element of app.elements.values()) {
        if (!element.hidden) assert.doesNotMatch(element.textContent, banned, `Технические слова в #${element.id}`);
    }
    for (const item of [...app.canvas().get(), ...app.edges().get()]) {
        if (item.label) assert.doesNotMatch(item.label, banned);
        if (item.title) assert.doesNotMatch(item.title.textContent || item.title, banned);
    }
}

// Реальный HTML — отдельный smoke: убеждаемся, что все 2248 gid дошли до клиента строками.
const real = load();
const realData = real.evaluate('DATA');
assert.ok(realData.nodes.length > 0);
assert.ok(realData.nodes.every(node => typeof node.id === 'string'));
assert.equal(new Set(realData.nodes.map(node => node.id.slice(-realData.shortLen))).size, realData.nodes.length);
const first = [...realData.top].sort((a, b) => a.rank - b.rank)[0];
assertSelected(real, first.gid);
assertNeighborhood(real, first.gid);
assert.equal(real.queue().length, realData.top.length);
assert.equal(real.queue()[0].dataset.gid, first.gid);
assertNoTechnicalWords(real);

// Угловые случаи не зависят от текущей контрольной выгрузки.
const sample = fixture();
const app = load(sample.data);
assertSelected(app, sample.center);
assert.equal(app.canvas().length, 30);
assertNeighborhood(app, sample.center);
assert.equal(app.canvas().get(sample.center).x, 0);
assert.equal(app.canvas().get(sample.center).y, 0);
assert.equal(app.canvas().get(sample.center).size, 44);
assert.equal(app.canvas().get(sample.incoming[14]).x, -450);
assert.equal(app.canvas().get(sample.outgoing[14]).x, 450);
assert.ok(app.canvas().get(sample.mutual[6]).y < 0);
assert.equal(app.canvas().get(sample.incoming[14]).shape, 'box');
assert.ok(!app.canvas().get(sample.incoming[0]), 'Избыточные мелкие плательщики скрыты');
assert.ok(!app.canvas().get(sample.outgoing[0]), 'Избыточные мелкие получатели скрыты');
assert.ok(!app.canvas().get(sample.mutual[0]), 'Встречных контрагентов не больше пяти');
assert.match(app.get('hidden-peers').textContent, /показаны крупнейшие/);
assert.equal(app.network().options.interaction.selectConnectedEdges, false);
assert.equal(app.network().options.interaction.hoverConnectedEdges, false);
assert.ok(app.network().scale <= .8);
assert.equal(app.get('detail-why').textContent, sample.data.top[0].why);
assert.match(app.get('detail-confidence').textContent, /66%/);
assert.equal(app.get('hints').hidden, false);

const second = app.queue()[1];
assert.equal(second.tagName, 'BUTTON', 'Нативная кнопка поддерживает Enter без собственного keydown');
second.click();
assertSelected(app, sample.incoming[0]);
assert.ok(second.scrolled, 'Выбранная строка очереди прокручена в видимую область');
assert.ok(second.classList.contains('selected') || second.classList.contains('active') || second.getAttribute('aria-current') === 'true', 'Выбранная строка подсвечена');
app.get('back').click();
assertSelected(app, sample.center);

app.submit(sample.outside);
assertSelected(app, sample.outside);
assertNeighborhood(app, sample.outside);
assert.match(app.get('status').textContent, /вне топа/);
assert.equal(app.get('hints').hidden, true);
assert.equal(app.get('detail-why').textContent, sample.data.nodes.find(node => node.id === sample.outside).evidence);
app.submit(sample.center.slice(-sample.data.shortLen));
assertSelected(app, sample.center);
assert.match(app.get('status').textContent, /№1/);
app.submit(sample.outside.slice(-sample.data.shortLen));
assertSelected(app, sample.outside);
for (const [query, message] of [['1234', /Найдено 2 gid/], ['77777777777777', /не найден/]]) {
    const nodesBefore = app.canvas().getIds().join(',');
    app.submit(query);
    assertSelected(app, sample.outside);
    assert.equal(app.canvas().getIds().join(','), nodesBefore);
    assert.match(app.get('status').textContent, message);
    assert.ok(app.get('status').classList.contains('error'));
}

app.submit(sample.isolated);
assertSelected(app, sample.isolated);
assert.equal(app.canvas().length, 1);
assert.equal(app.edges().length, 0);
assert.equal(app.get('isolated').hidden, false);
assert.match(app.get('isolated').textContent, /Видимых переводов у этого клиента нет/);
assert.match(app.get('flow-ratio').textContent, /Входящих переводов в выборке нет/);
assert.match(app.get('caveats').textContent, /нулевой выход не доказывает/i);
app.get('overview').click();
assert.equal(app.mode(), 'overview');
assert.equal(app.get('directions').hidden, true);
assert.equal(app.canvas().length, sample.data.overviewIds.length);
assert.match(app.get('overview').textContent, /К окружению/);
app.network().events.click({nodes: [sample.center]});
assertSelected(app, sample.center);
assert.equal(app.mode(), 'overview', 'Один клик в обзоре меняет только карточку');
app.get('overview').click();
assertSelected(app, sample.center);
assert.equal(app.get('directions').hidden, false);
assertNeighborhood(app, sample.center);
app.get('overview').click();
app.network().events.doubleClick({nodes: [sample.incoming[14]]});
assertSelected(app, sample.incoming[14]);
assertNeighborhood(app, sample.incoming[14]);
app.get('back').click();
assert.equal(app.mode(), 'overview', 'Назад из окружения восстанавливает обзор');
app.get('overview').click();
app.submit(sample.center);
app.network().events.click({nodes: [sample.outgoing[14]]});
assertSelected(app, sample.outgoing[14]);
assertNeighborhood(app, sample.outgoing[14]);
app.get('back').click();
assertSelected(app, sample.center);
assertNoTechnicalWords(app);

assert.equal(app.evaluate('money(850)'), '850 ₸');
assert.match(app.evaluate('money(4200000)'), /4,2 млн ₸/);
assert.match(app.evaluate('money(393000)'), /393 тыс ₸/);

const byGid = load({...sample.data, initialGid: sample.outside});
assertSelected(byGid, sample.outside);
assertNeighborhood(byGid, sample.outside);
const cluster = load({...sample.data, initialMode: 'overview', overviewIds: [sample.center, sample.incoming[14]]});
assert.equal(cluster.mode(), 'overview');
assert.equal(cluster.canvas().length, 2);
assert.equal(cluster.get('directions').hidden, true);
console.log('OK: real HTML, rank 1, queue/Enter button, back, full/suffix search, errors, isolated, overview, neighbor, limits, directions, safe Russian UI.');
