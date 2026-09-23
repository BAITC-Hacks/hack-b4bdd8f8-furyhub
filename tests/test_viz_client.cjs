// Клиент выполняется без браузера: проверяем переходы и данные canvas,
// а визуальную читаемость vis-network проверяем отдельно в браузере.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const os = require('node:os');
const {execFileSync} = require('node:child_process');

function readHtml() {
    if (process.argv[2]) return fs.readFileSync(path.resolve(process.argv[2]), 'utf8');
    const root = path.resolve(__dirname, '..');
    const localPython = path.join(root, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
    const python = process.env.PYTHON || (fs.existsSync(localPython) ? localPython : 'python');
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'furyhub-viz-test-'));
    const output = path.join(directory, 'graph.html');
    try {
        execFileSync(python, ['-B', 'viz.py', '--no-llm', '--output', output], {
            cwd: root, encoding: 'utf8', timeout: 60000, maxBuffer: 1024 * 1024,
        });
        return fs.readFileSync(output, 'utf8');
    } finally {
        if (fs.existsSync(output)) fs.unlinkSync(output);
        fs.rmdirSync(directory);
    }
}

// По умолчанию проверяем свежий viz.py, не сохранённый артефакт в out/.
// Для отдельного HTML: node tests/test_viz_client.cjs /path/to/graph.html.
const html = readHtml();
const scripts = [...html.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script>/g)];
assert.ok(scripts.length, 'В HTML отсутствует клиентский JavaScript');
const script = scripts.at(-1)[1];
assert.match(script, /const DATA\s*=/, 'HTML должен быть собран актуальным viz.py');

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
    constructor(container, data, options) {
        assert.ok(container, 'В HTML отсутствует контейнер графа');
        this.events = {}; this.options = options; this.scale = 1.2;
    }
    unselectAll() { this.selection = []; }
    fit() { this.scale = 1.2; this.fitCalls = (this.fitCalls || 0) + 1; }
    getScale() { return this.scale; }
    moveTo(options) { if (options.scale !== undefined) this.scale = options.scale; }
    selectNodes(ids, connected) { this.selection = ids; this.connected = connected; }
    focus(gid, options = {}) { this.focused = gid; if (options.scale) this.scale = options.scale; }
    on(name, callback) { this.events[name] = callback; }
    once(name, callback) { callback(); }
    setOptions(options) { this.options = {...this.options, ...options}; }
    redraw() { this.redrawCalls = (this.redrawCalls || 0) + 1; }
}

function load(data) {
    const elements = new Map();
    const document = {
        createElement(tag) { return new Element(tag); },
        createTextNode(text) { const node = new Element('#text'); node.textContent = text; return node; },
        getElementById(id) { return elements.get(id) || null; },
        querySelectorAll(selector) { return [...elements.values()].flatMap(element => element.querySelectorAll(selector)); },
    };
    // Только реальные id из разметки: опечатка в клиенте должна падать, как в браузере.
    const markup = html.replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, '').replace(/<style\b[^>]*>[\s\S]*?<\/style>/gi, '');
    for (const match of markup.matchAll(/<([a-z][\w-]*)\b[^>]*\bid="([^"]+)"[^>]*>/gi)) {
        assert.ok(!elements.has(match[2]), `Повторный HTML id ${match[2]}`);
        const element = new Element(match[1]);
        element.id = match[2];
        elements.set(element.id, element);
        element.hidden = /\bhidden(?:\s|>|=)/.test(match[0]);
        element.disabled = /\bdisabled(?:\s|>|=)/.test(match[0]);
        const leaf = markup.slice(match.index + match[0].length).match(new RegExp(`^([^<]*)</${match[1]}>`));
        if (leaf) element.textContent = leaf[1];
    }
    const messages = [];
    const window = {location: {protocol: 'http:', origin: 'http://127.0.0.1:8765'},
        parent: {postMessage: (message, origin) => messages.push({message, origin})}};
    const context = vm.createContext({document, window, vis: {DataSet, Network}, console,
        navigator: {clipboard: {writeText: async () => {}}},
        requestAnimationFrame: callback => callback(), setTimeout: callback => callback(), clearTimeout() {},
    });
    const client = data ? script.replace(/const DATA\s*=\s*[^\n]*;(?=\s*\n)/, () => `const DATA = ${JSON.stringify(data)};`) : script;
    vm.runInContext(client, context, {filename: 'graph-client.js'});
    const evaluate = expression => vm.runInContext(expression, context);
    return {
        evaluate, document, elements, messages,
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
        role_meaning: index === 0 ? 'связность узла требует проверки' : 'входящие и исходящие потоки требуют сопоставления',
        color: index === 0 ? '#e15759' : '#4e9ee8', role_score: .66,
        priority_score: 1 - index / 100, evidence: 'Гипотеза: 3 перевода требуют проверки.',
        is_seed: index === 0, cluster_id: '2', depth: index === 0 ? 0 : 1,
        truncated_by_depth: false, incoming_incomplete: index === 0,
        role_rule: index === 0 ? 'coordinator' : 'transit',
        in_kzt: 0, out_kzt: 0, in_deg: 0, out_deg: 0,
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
assertSelected(real, realData.initialGid);
if (realData.initialMode === 'overview') {
    assert.equal(real.mode(), 'overview');
    assert.ok(real.canvas().get(realData.initialGid));
} else {
    assertNeighborhood(real, realData.initialGid);
}
if (!process.argv[2]) assertSelected(real, first.gid);
assert.equal(real.queue().length, realData.top.length);
assert.equal(real.queue()[0].dataset.gid, first.gid);
assertNoTechnicalWords(real);
assert.equal(real.document.getElementById('nonexistent-id'), null, 'Fake DOM не должен создавать отсутствующие элементы');

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
assert.match(app.get('detail-confidence').textContent, /сила правила\s*0,66\s*\/\s*1/i);
assert.doesNotMatch(app.get('detail-confidence').textContent, /уверенность/i);
assert.equal(app.get('hints').hidden, false);

// Полные списки доступны независимо от лимита соседей на canvas.
const expanded = load(sample.data);
for (const [target, expected] of [['payers', [...sample.incoming, ...sample.mutual]], ['recipients', [...sample.outgoing, ...sample.mutual]]]) {
    assert.equal(expanded.get(target).querySelectorAll('button[data-gid]').length, 5);
    const more = expanded.get(`${target}-more`);
    assert.equal(more.tagName, 'BUTTON');
    assert.equal(more.hidden, false);
    more.click();
    const shown = expanded.get(target).querySelectorAll('button[data-gid]').map(row => row.dataset.gid);
    assert.deepEqual(new Set(shown), new Set(expected), 'Раскрытие должно показывать всех контрагентов');
    assert.equal(more.hidden, true, 'После последней страницы кнопка скрыта');
}
const hiddenPayer = sample.incoming[1];
assert.ok(!expanded.canvas().get(hiddenPayer), 'Мелкий плательщик остаётся за лимитом canvas');
expanded.get('payers').querySelector(`[data-gid="${hiddenPayer}"]`).click();
assertSelected(expanded, hiddenPayer);
assertNeighborhood(expanded, hiddenPayer);

const filtered = load(sample.data);
for (const [target, gid] of [['payers', sample.incoming[1]], ['recipients', sample.outgoing[1]]]) {
    const input = filtered.get(`${target}-filter`);
    input.value = gid;
    input.dispatch('input');
    const rows = filtered.get(target).querySelectorAll('button[data-gid]');
    assert.equal(rows.length, 1, 'Поиск проверяет полный список, включая скрытых на canvas соседей');
    assert.equal(rows[0].dataset.gid, gid);
    assert.equal(filtered.get(`${target}-more`).hidden, true);
    input.value = 'not-a-gid';
    input.dispatch('input');
    assert.equal(filtered.get(target).querySelectorAll('button[data-gid]').length, 0);
    assert.match(filtered.get(target).textContent, /совпадений нет/i);
    input.value = '';
    input.dispatch('input');
    assert.equal(filtered.get(target).querySelectorAll('button[data-gid]').length, 5);
}

const incomplete = load({...sample.data, initialGid: sample.outside,
    nodes: sample.data.nodes.map(node => node.id === sample.outside ? {...node, incoming_incomplete: true} : node)});
assert.match(incomplete.get('badges').textContent, /неполный видимый вход/i);
assert.match(incomplete.get('caveats').textContent, /не подтверждает транзит/i);

// Возврат между карточками обзора должен сбрасывать поиск и страницу контрагентов.
const historySample = fixture();
const historyNodes = new Map(historySample.data.nodes.map(node => [node.id, node]));
historySample.outgoing.forEach((id, index) => {
    historySample.data.edges.push({id: `history-${index}`, from: historySample.outside, to: id, sum_kzt: 1000, n_tx: 1});
    const source = historyNodes.get(historySample.outside), target = historyNodes.get(id);
    source.out_kzt += 1000; source.out_tx++; source.out_deg++;
    target.in_kzt += 1000; target.in_tx++; target.in_deg++;
});
const restored = load({...historySample.data, initialMode: 'overview'});
restored.network().events.click({nodes: [historySample.outside]});
const recipientFilter = restored.get('recipients-filter');
recipientFilter.value = historySample.outside.slice(0, 8);
recipientFilter.dispatch('input');
assert.equal(restored.get('recipients-more').hidden, false);
restored.get('recipients-more').click();
assert.ok(restored.get('recipients').querySelectorAll('button[data-gid]').length > 5);
restored.get('back').click();
assertSelected(restored, historySample.center);
assert.equal(restored.mode(), 'overview');
assert.equal(recipientFilter.value, '', 'Назад к другому клиенту очищает фильтр его контрагентов');
assert.equal(restored.get('recipients').querySelectorAll('button[data-gid]').length, 5, 'Назад к другому клиенту возвращает первую страницу');
assert.equal(restored.get('recipients-more').hidden, false);

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
const reciprocal = app.edges().get().filter(edge =>
    (edge.from === sample.center && edge.to === sample.mutual[0]) ||
    (edge.to === sample.center && edge.from === sample.mutual[0]));
assert.equal(reciprocal.length, 2);
assert.ok(reciprocal.every(edge => edge.smooth && edge.smooth.enabled), 'Встречные потоки в обзоре должны идти разными дугами');
const oneWay = app.edges().get().find(edge => edge.from === sample.incoming[0] && edge.to === sample.center);
assert.ok(!oneWay.smooth || !oneWay.smooth.enabled, 'Одностороннему потоку дуга не нужна');
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
assert.ok(cluster.canvas().get(cluster.current()), 'Выбранная карточка должна присутствовать в кластере');

// Фокус уменьшает визуальный шум, сохраняя все узлы и переводы обзора.
const overview = load({...sample.data, initialMode: 'overview'});
const overviewNodeIds = new Set(overview.canvas().getIds());
const overviewEdgeIds = new Set(overview.edges().getIds());
const payloadBeforeFocus = overview.evaluate('JSON.stringify(DATA)');
function overviewZoom(scale) {
    overview.network().moveTo({scale});
    overview.network().events.zoom({scale});
}
overviewZoom(.2);
assert.ok(overview.canvas().get(sample.center).label, 'При отдалении у лидера очереди остаётся подпись');
assert.equal(overview.canvas().get(sample.outgoing[0]).label, '', 'Мелкие подписи скрыты в общем плане');
assert.ok(overview.edges().get().every(edge => edge.color.opacity < .25), 'Без фокуса связи служат фоном');
overviewZoom(.8);
assert.ok(overview.canvas().get(sample.outgoing[0]).label, 'При приближении появляются остальные подписи');
overviewZoom(.2);
overview.network().events.click({nodes: [sample.center]});
assertSelected(overview, sample.center);
assert.deepEqual(new Set(overview.canvas().getIds()), overviewNodeIds);
assert.deepEqual(new Set(overview.edges().getIds()), overviewEdgeIds);
const directNeighbors = [...sample.incoming, ...sample.outgoing, ...sample.mutual];
assert.ok(overview.canvas().get(sample.center).label);
assert.equal(directNeighbors.filter(gid => overview.canvas().get(gid).label).length, 12, 'Издалека подписаны только крупнейшие соседи');
assert.ok(overview.canvas().get(sample.mutual[6]).label, 'Крупный встречный поток входит в подписи общего плана');
assert.equal(overview.canvas().get(sample.incoming[0]).label, '', 'Мелкий сосед не загромождает общий план');
assert.ok(directNeighbors.every(gid => overview.canvas().get(gid).opacity === 1), 'Подсветка сохраняет всех соседей даже без подписи');
overviewZoom(.8);
for (const gid of [sample.center, ...sample.incoming, ...sample.outgoing, ...sample.mutual]) {
    assert.ok(overview.canvas().get(gid).label, 'Вблизи подписаны выбранный клиент и все прямые соседи');
    assert.equal(overview.canvas().get(gid).opacity, 1);
}
for (const gid of [sample.outside, sample.isolated]) {
    assert.equal(overview.canvas().get(gid).label, '', 'Клиент за пределами прямого окружения не получает подпись');
    assert.equal(overview.canvas().get(gid).opacity, .22);
}
const focusedIncoming = overview.edges().get().find(edge => edge.from === sample.incoming[0] && edge.to === sample.center);
const focusedOutgoing = overview.edges().get().find(edge => edge.from === sample.center && edge.to === sample.outgoing[0]);
assert.notEqual(focusedIncoming.color.color, focusedOutgoing.color.color, 'Входящие и исходящие различаются цветом');
for (const edge of overview.edges().get()) {
    const incident = edge.from === sample.center || edge.to === sample.center;
    assert.equal(edge.color.opacity, incident ? .95 : .08, 'Яркость выделяет только связи выбранного клиента');
}
assert.equal(overview.evaluate('JSON.stringify(DATA)'), payloadBeforeFocus, 'Фокус не меняет исходные данные');

overviewZoom(.2);
overview.network().events.click({nodes: []});
assert.equal(overview.mode(), 'overview');
assertSelected(overview, sample.center);
assert.ok(overview.canvas().get().every(node => node.opacity === 1), 'Пустой клик снимает затенение узлов');
assert.ok(overview.edges().get().every(edge => edge.color.opacity < .25));
assert.equal(overview.canvas().get(sample.outgoing[0]).label, '');
assert.deepEqual(new Set(overview.canvas().getIds()), overviewNodeIds);

// При возврате из окружения обзор открывается без оставшегося фокуса.
overview.network().events.click({nodes: [sample.center]});
overview.get('overview').click();
assertNeighborhood(overview, sample.center);
overview.get('back').click();
assert.equal(overview.mode(), 'overview');
assert.ok(overview.canvas().get().every(node => node.opacity === 1));
assert.ok(overview.edges().get().every(edge => edge.color.opacity < .25));
assert.deepEqual(new Set(overview.canvas().getIds()), overviewNodeIds);

for (const id of ['zoom-in', 'zoom-out', 'fit-view']) assert.equal(overview.get(id).tagName, 'BUTTON');
const beforeZoom = overview.network().getScale();
overview.get('zoom-in').click();
assert.ok(overview.network().getScale() > beforeZoom);
const afterZoom = overview.network().getScale();
overview.get('zoom-out').click();
assert.ok(overview.network().getScale() < afterZoom);
for (let index = 0; index < 30; index++) overview.get('zoom-in').click();
assert.ok(overview.network().getScale() <= 2.5, 'Увеличение имеет верхний предел');
for (let index = 0; index < 60; index++) overview.get('zoom-out').click();
assert.ok(overview.network().getScale() >= .06, 'Уменьшение имеет нижний предел');
const fitsBefore = overview.network().fitCalls;
overview.get('fit-view').click();
assert.ok(overview.network().fitCalls > fitsBefore, 'Кнопка обзора подгоняет граф к экрану');
assert.deepEqual(new Set(overview.canvas().getIds()), overviewNodeIds);
assert.deepEqual(new Set(overview.edges().getIds()), overviewEdgeIds);
assert.equal(overview.evaluate('JSON.stringify(DATA)'), payloadBeforeFocus);
assert.equal(overview.messages.at(-1).message.type, 'furyhub:node-selected');
assert.equal(overview.messages.at(-1).message.gid, overview.current());
assert.equal(overview.messages.at(-1).origin, 'http://127.0.0.1:8765');
console.log('OK: fresh HTML, strict DOM, navigation, search, cluster, reciprocal flows, complete peers, overview focus/zoom, safe Russian UI.');
