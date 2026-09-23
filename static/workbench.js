'use strict';

(() => {
    const el = id => document.getElementById(id);
    const text = (id, value) => { el(id).textContent = value ?? ''; };
    const visible = (id, show) => { el(id).hidden = !show; };
    const make = (tag, value, className) => {
        const node = document.createElement(tag);
        if (value !== undefined && value !== null) node.textContent = String(value);
        if (className) node.className = className;
        return node;
    };
    const state = {
        token: '', limits: {}, ready: false, upload: null, uploadSequence: 0, importBusy: false,
        datasets: [], runs: [], datasetId: null, runId: null, resultRunId: null,
        historyTab: 'runs', view: 'import', resultTab: 'graph', pollTimer: null,
        nodes: [], cases: [], onlyCases: false, gid: null, caseValue: null,
        nodeSequence: 0, caseSequence: 0, searchTimer: null, drafts: new Map(), savingCase: false,
    };
    const roles = {coordinator: 'Координатор', distributor: 'Распределитель', consolidator: 'Консолидатор', transit: 'Транзит', terminal: 'Конечный получатель', peripheral: 'Периферия'};
    const runStatuses = {queued: 'В очереди', running: 'Выполняется', completed: 'Готово', failed: 'Ошибка'};
    const caseStatuses = {new: 'Новая', in_progress: 'В работе', needs_info: 'Нужны данные', closed: 'Закрыта'};
    const fields = [
        ['src', 'Отправитель', true], ['dst', 'Получатель', true], ['sum_kzt', 'Сумма перевода', true],
        ['date', 'Дата перевода', true], ['currency', 'Валюта', false], ['transaction_id', 'ID перевода', false],
    ];
    const number = value => Number(value || 0).toLocaleString('ru-RU', {maximumFractionDigits: 2});
    const date = (value, includeTime = false) => {
        if (!value) return '—';
        const parsed = new Date(value);
        if (Number.isNaN(parsed.getTime())) return String(value);
        return parsed.toLocaleString('ru-RU', includeTime
            ? {day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit'}
            : {day: '2-digit', month: 'short', year: 'numeric'});
    };
    const nameOf = item => item?.name || item?.metadata?.name || 'Набор переводов';
    const isActive = run => run && ['queued', 'running'].includes(run.status);
    const currentRun = () => state.runs.find(run => String(run.id) === state.runId);
    const currentDataset = id => state.datasets.find(dataset => String(dataset.id) === String(id));
    const draftKey = (runId = state.runId, gid = state.gid) => `${runId}:${gid}`;

    async function api(url, options = {}) {
        const method = (options.method || 'GET').toUpperCase();
        const headers = {Accept: 'application/json'};
        if (!['GET', 'HEAD'].includes(method)) {
            if (!state.token) throw new Error('Соединение ещё не готово. Обновите страницу и повторите действие.');
            headers['X-FuryHub-Token'] = state.token;
        }
        let body = options.body;
        if (body !== undefined && !(body instanceof FormData)) {
            headers['Content-Type'] = 'application/json';
            body = JSON.stringify(body);
        }
        let response;
        try {
            response = await fetch(url, {method, headers, body, credentials: 'same-origin'});
        } catch {
            throw new Error('Нет связи с локальным сервером. Проверьте, что FuryHub запущен, и повторите действие.');
        }
        let payload = {};
        try { payload = await response.json(); } catch { /* Текстовая ошибка сервера не является разметкой UI. */ }
        if (!response.ok) {
            const message = payload.message || (typeof payload.error === 'string' ? payload.error : payload.error?.message)
                || (response.status === 413 ? 'Файл слишком большой для загрузки.' : `Не удалось выполнить действие (код ${response.status}).`);
            const error = new Error(message);
            error.details = payload.errors || payload.validation_errors || payload.error?.errors || payload.details || [];
            throw error;
        }
        return payload;
    }

    function notify(message, type = '') {
        const banner = el('global-message');
        banner.textContent = message;
        banner.className = `notice ${type}`;
        banner.setAttribute('role', type === 'error' ? 'alert' : 'status');
        banner.hidden = !message;
    }

    function setView(view) {
        state.view = view;
        visible('import-panel', view === 'import');
        visible('welcome', view === 'import' && !state.datasets.length);
        visible('dataset-panel', view === 'dataset');
        visible('run-panel', view === 'run');
        renderHistory();
    }

    function setHistoryTab(tab) {
        state.historyTab = tab;
        for (const name of ['runs', 'datasets']) {
            el(`${name}-tab`).setAttribute('aria-selected', String(tab === name));
            visible(`${name}-history`, tab === name);
        }
    }

    function setResultTab(tab) {
        state.resultTab = tab;
        for (const name of ['graph', 'cases']) {
            el(`${name}-tab`).setAttribute('aria-selected', String(tab === name));
            visible(`${name}-panel`, tab === name);
        }
    }

    function chip(status, labels) {
        const known = Object.hasOwn(labels, status);
        return make('span', known ? labels[status] : 'Неизвестно', `status-chip ${known ? status : ''}`);
    }

    function historyItem(item, kind) {
        const active = kind === 'run' ? state.view === 'run' && String(item.id) === state.runId
            : state.view === 'dataset' && String(item.id) === state.datasetId;
        const button = make('button', null, `history-item${active ? ' active' : ''}`);
        button.type = 'button';
        button.dataset.historyKey = `${kind}:${item.id}`;
        button.setAttribute('aria-pressed', String(active));
        button.append(make('span', nameOf(item), 'history-title'));
        const meta = make('span', null, 'history-meta');
        meta.append(make('span', date(item.created_at)));
        meta.append(kind === 'run' ? chip(item.status, runStatuses) : make('span', `${number(item.summary?.row_count)} строк`));
        button.append(meta);
        if (kind === 'run' && isActive(item)) button.append(make('span', item.phase || 'Ожидание обработки', 'history-subtitle'));
        button.addEventListener('click', () => {
            if (kind === 'run') void selectRun(String(item.id));
            else selectDataset(String(item.id));
        });
        return button;
    }

    function renderHistory() {
        const focusKey = document.activeElement?.dataset?.historyKey;
        text('runs-count', state.runs.length);
        text('datasets-count', state.datasets.length);
        const sorted = values => [...values].sort((a, b) => String(b.created_at || '').localeCompare(String(a.created_at || '')));
        const runs = el('runs-history'), datasets = el('datasets-history');
        runs.replaceChildren(...sorted(state.runs).map(run => historyItem(run, 'run')));
        datasets.replaceChildren(...sorted(state.datasets).map(dataset => historyItem(dataset, 'dataset')));
        if (!state.runs.length) runs.append(make('p', 'Здесь появятся запуски анализа. Импортируйте файл или откройте демо.', 'empty-small'));
        if (!state.datasets.length) datasets.append(make('p', 'Сохранённые наборы останутся здесь для повторного анализа.', 'empty-small'));
        if (focusKey) for (const item of document.querySelectorAll('[data-history-key]')) {
            if (item.dataset.historyKey === focusKey) item.focus({preventScroll: true});
        }
    }

    function renderSummary(target, dataset) {
        const summary = dataset?.summary || {};
        const period = summary.period_from || summary.period_to
            ? `${date(summary.period_from)} — ${date(summary.period_to)}` : 'Не указан';
        const tiles = [
            ['Переводы', number(summary.row_count), 'строк в наборе'],
            ['Участники сети', number(summary.node_count), `${number(summary.edge_count)} связей · ${number(summary.seed_count)} исходных клиентов`],
            ['Объём переводов', `${number(summary.total_kzt)} ₸`, summary.currency || 'KZT'],
            ['Период данных', period, (summary.coverage || dataset?.metadata?.coverage || dataset?.coverage) === 'complete' ? 'Полнота исходящих подтверждена' : 'Полнота исходящих неизвестна'],
        ];
        el(target).replaceChildren(...tiles.map(([label, value, detail]) => {
            const tile = make('div', null, 'summary-item');
            tile.append(make('span', label, 'summary-label'), make('strong', value, 'summary-value'), make('span', detail, 'summary-detail'));
            return tile;
        }));
    }

    function renderWarnings(target, dataset) {
        const warnings = dataset?.summary?.warnings || dataset?.warnings || [];
        const list = Array.isArray(warnings) ? warnings : [warnings];
        el(target).replaceChildren(...list.map(warning => make('p', typeof warning === 'string' ? warning : warning.message || String(warning))));
        visible(target, list.length > 0);
    }

    function selectDataset(id) {
        const dataset = currentDataset(id);
        if (!dataset) return;
        state.datasetId = id;
        setView('dataset');
        text('dataset-title', nameOf(dataset));
        text('dataset-description', [dataset.metadata?.source || dataset.source, date(dataset.created_at, true)].filter(Boolean).join(' · '));
        renderSummary('dataset-summary', dataset);
        renderWarnings('dataset-warnings', dataset);
        notify('');
    }

    function mergeRun(run) {
        const index = state.runs.findIndex(item => String(item.id) === String(run.id));
        if (index < 0) state.runs.unshift(run); else state.runs[index] = run;
    }

    function mergeDataset(dataset) {
        const index = state.datasets.findIndex(item => String(item.id) === String(dataset.id));
        if (index < 0) state.datasets.unshift(dataset); else state.datasets[index] = dataset;
    }

    async function refreshHistory() {
        const [datasets, runs] = await Promise.all([api('/api/datasets'), api('/api/runs')]);
        state.datasets = datasets.datasets || [];
        state.runs = runs.runs || [];
        renderHistory();
        if (state.view === 'run' && currentRun()) renderRun(currentRun());
        schedulePoll();
    }

    async function selectRun(id) {
        const changed = state.runId !== id;
        state.runId = id;
        if (changed) {
            state.resultRunId = null;
            state.gid = null;
            state.caseValue = null;
            state.cases = [];
            state.nodes = [];
            state.nodeSequence++;
            state.caseSequence++;
            text('node-query', '');
            el('node-query').value = '';
            visible('case-content', false);
            visible('case-empty', true);
            visible('open-selected-case', false);
            visible('run-results', false);
            el('graph-frame').src = 'about:blank';
            setResultTab('graph');
        }
        setView('run');
        notify('');
        const cached = currentRun();
        if (cached) renderRun(cached);
        try {
            const payload = await api(`/api/runs/${encodeURIComponent(id)}`);
            mergeRun(payload.run);
            if (state.runId === id) renderRun(payload.run);
            renderHistory();
            schedulePoll();
        } catch (error) { if (state.runId === id) notify(error.message, 'error'); }
    }

    function renderRun(run) {
        if (!run || String(run.id) !== state.runId) return;
        const dataset = currentDataset(run.dataset_id);
        text('run-title', run.name || nameOf(dataset));
        text('run-meta', [date(run.created_at, true), dataset?.metadata?.source || dataset?.source].filter(Boolean).join(' · '));
        const status = el('run-status');
        status.textContent = runStatuses[run.status] || 'Неизвестно';
        status.className = `status-chip ${Object.hasOwn(runStatuses, run.status) ? run.status : ''}`;
        renderSummary('run-summary', dataset);
        renderWarnings('run-warnings', dataset);
        visible('run-progress', isActive(run));
        visible('run-error', run.status === 'failed');
        visible('run-results', run.status === 'completed');
        if (isActive(run)) {
            const progress = Math.max(0, Math.min(100, Number(run.progress) || 0));
            text('run-phase', run.phase || (run.status === 'queued' ? 'Ожидание запуска' : 'Анализируем переводы'));
            text('run-percent', `${Math.round(progress)}%`);
            el('progress-bar').value = progress;
        }
        if (run.status === 'failed') text('run-error-text', typeof run.error === 'string' ? run.error : run.error?.message || 'Сервер не смог завершить анализ.');
        if (run.status === 'completed' && state.resultRunId !== String(run.id)) {
            state.resultRunId = String(run.id);
            el('graph-frame').src = `/runs/${encodeURIComponent(run.id)}/graph`;
            const files = [['nodes_roles.csv', 'Все узлы и роли'], ['clusters.csv', 'Группы участников'], ['top_nodes.csv', 'Очередь проверки']];
            el('download-links').replaceChildren(...files.map(([file, label]) => {
                const link = make('a', `${label} · CSV`);
                link.href = `/api/runs/${encodeURIComponent(run.id)}/files/${file}`;
                link.setAttribute('download', file);
                return link;
            }));
            void loadCases();
            void loadNodes();
        }
    }

    function schedulePoll() {
        clearTimeout(state.pollTimer);
        if (!state.runs.some(isActive)) return;
        state.pollTimer = setTimeout(async () => {
            try {
                const payload = await api('/api/runs');
                state.runs = payload.runs || [];
                renderHistory();
                if (state.view === 'run') renderRun(currentRun());
                schedulePoll();
            } catch (error) {
                notify(`${error.message} Статус обновится после восстановления соединения.`, 'error');
                state.pollTimer = setTimeout(schedulePoll, 3500);
            }
        }, 1600);
    }

    function updateImportButtons() {
        const unavailable = !state.ready || state.importBusy;
        el('upload-file').disabled = unavailable;
        el('start-import').disabled = unavailable || !state.upload;
        el('save-dataset').disabled = unavailable || !state.upload;
    }

    function validation(error) {
        visible('validation-panel', true);
        text('validation-message', error.message);
        const details = Array.isArray(error.details) ? error.details : [];
        el('validation-errors').replaceChildren(...details.map(detail => {
            const row = make('tr');
            if (typeof detail === 'string') detail = {message: detail};
            row.append(make('td', detail.row ?? '—'), make('td', detail.column || '—'), make('td', detail.message || 'Некорректное значение'));
            return row;
        }));
        el('validation-errors').closest('.table-scroll').hidden = !details.length;
        el('validation-panel').scrollIntoView({behavior: 'smooth', block: 'nearest'});
    }

    function renderUpload(upload) {
        const columns = (upload.columns || []).map(String);
        const suggested = upload.suggested_mapping || {};
        el('mapping-fields').replaceChildren(...fields.map(([canonical, labelText, required]) => {
            const label = make('label', `${labelText}${required ? ' *' : ''}`);
            const select = make('select');
            select.id = `mapping-${canonical}`;
            select.required = required;
            select.append(new Option(required ? 'Выберите колонку' : 'Не используется', ''));
            for (const column of columns) select.append(new Option(column, column));
            if (suggested[canonical] && columns.includes(String(suggested[canonical]))) select.value = String(suggested[canonical]);
            label.append(select);
            return label;
        }));
        const table = el('preview-table');
        const heading = make('tr');
        heading.append(...columns.map(column => make('th', column)));
        table.querySelector('thead').replaceChildren(heading);
        const preview = Array.isArray(upload.preview) ? upload.preview : [];
        table.querySelector('tbody').replaceChildren(...preview.slice(0, 8).map(record => {
            const row = make('tr');
            row.append(...columns.map((column, index) => {
                const value = Array.isArray(record) ? record[index] : record[column];
                const cell = make('td', value === null || value === undefined || value === '' ? '—' : value);
                cell.title = cell.textContent;
                return cell;
            }));
            return row;
        }));
        text('preview-count', `${Math.min(preview.length, 8)} из ${number(upload.row_count)} строк`);
        text('upload-title', upload.filename || 'Файл загружен');
        text('upload-status', `${number(upload.row_count)} строк · ${columns.length} колонок${upload.sha256 ? ` · отпечаток ${String(upload.sha256).slice(0, 12)}` : ''}`);
        el('upload-status').className = 'upload-status success';
        visible('upload-details', true);
    }

    async function uploadFile(file) {
        if (!file || state.importBusy || !state.ready) return;
        visible('validation-panel', false);
        if (!/\.(csv|parquet)$/i.test(file.name)) {
            validation(new Error('Выберите файл в формате CSV или Parquet.'));
            return;
        }
        if (state.limits.upload_mb && file.size > state.limits.upload_mb * 1024 * 1024) {
            validation(new Error(`Размер файла превышает ${number(state.limits.upload_mb)} МБ.`));
            return;
        }
        const sequence = ++state.uploadSequence;
        state.upload = null;
        state.importBusy = true;
        updateImportButtons();
        visible('upload-details', false);
        text('upload-status', 'Читаем файл и готовим предварительный просмотр…');
        el('upload-status').className = 'upload-status';
        const form = new FormData();
        form.append('file', file);
        try {
            const payload = await api('/api/uploads', {method: 'POST', body: form});
            if (sequence !== state.uploadSequence) return;
            state.upload = payload;
            renderUpload(payload);
            if (!el('dataset-name').value.trim()) el('dataset-name').value = file.name.replace(/\.(csv|parquet)$/i, '');
        } catch (error) {
            text('upload-status', 'Файл не загружен. Исправьте проблему и попробуйте снова.');
            validation(error);
        } finally { state.importBusy = false; updateImportButtons(); }
    }

    async function createDataset(analyze) {
        if (!state.upload || state.importBusy || !el('import-form').reportValidity()) return;
        state.importBusy = true;
        updateImportButtons();
        visible('validation-panel', false);
        text('upload-status', 'Проверяем строки и сохраняем набор…');
        const mapping = {};
        for (const [canonical] of fields) {
            const value = el(`mapping-${canonical}`).value;
            if (value) mapping[canonical] = value;
        }
        let dataset;
        try {
            const payload = await api('/api/datasets', {method: 'POST', body: {
                upload_id: state.upload.upload_id,
                name: el('dataset-name').value.trim(), source: el('dataset-source').value.trim(),
                coverage: el('dataset-coverage').value, currency: 'KZT',
                seed_gids: [...new Set(el('seed-gids').value.split(/[\s,;]+/).filter(Boolean))], mapping,
            }});
            dataset = payload.dataset;
            mergeDataset(dataset);
            renderHistory();
            selectDataset(String(dataset.id));
            text('upload-status', 'Набор сохранён.');
        } catch (error) { validation(error); }
        finally { state.importBusy = false; updateImportButtons(); }
        if (dataset && analyze) await launchRun(String(dataset.id));
    }

    async function launchRun(datasetId) {
        const controls = [el('analyze-dataset'), el('retry-run')];
        if (controls.some(button => button.disabled)) return;
        controls.forEach(button => { button.disabled = true; });
        try {
            const payload = await api('/api/runs', {method: 'POST', body: {dataset_id: datasetId}});
            mergeRun(payload.run);
            setHistoryTab('runs');
            await selectRun(String(payload.run.id));
            schedulePoll();
        } catch (error) { notify(error.message, 'error'); }
        finally { controls.forEach(button => { button.disabled = false; }); }
    }

    async function loadNodes() {
        const runId = state.runId;
        if (!runId || currentRun()?.status !== 'completed') return;
        if (state.onlyCases) { renderNodeList(); return; }
        const sequence = ++state.nodeSequence;
        text('node-result-count', 'Ищем участников…');
        try {
            const query = new URLSearchParams({q: el('node-query').value.trim(), limit: '50'});
            const payload = await api(`/api/runs/${encodeURIComponent(runId)}/nodes?${query}`);
            if (state.runId !== runId || sequence !== state.nodeSequence || state.onlyCases) return;
            state.nodes = payload.nodes || [];
            state.nodeTotal = payload.total ?? state.nodes.length;
            renderNodeList();
        } catch (error) { if (state.runId === runId && sequence === state.nodeSequence) text('node-result-count', error.message); }
    }

    async function loadCases() {
        const runId = state.runId;
        if (!runId) return;
        try {
            const payload = await api(`/api/cases?${new URLSearchParams({run_id: runId})}`);
            if (state.runId !== runId) return;
            state.cases = payload.cases || [];
            text('cases-count', state.cases.length);
            renderNodeList();
        } catch (error) { if (state.runId === runId) notify(error.message, 'error'); }
    }

    function renderNodeList() {
        const query = el('node-query').value.trim();
        const list = state.onlyCases ? state.cases.filter(item => String(item.gid).includes(query)) : state.nodes;
        const total = state.onlyCases ? list.length : state.nodeTotal || 0;
        text('node-result-count', state.onlyCases ? `Проверок: ${number(total)}` : `Показано ${list.length} из ${number(total)} · по приоритету`);
        el('node-list').replaceChildren(...list.map(item => {
            const gid = String(item.gid);
            const button = make('button', null, `node-row${gid === state.gid ? ' active' : ''}`);
            button.type = 'button';
            button.setAttribute('aria-pressed', String(gid === state.gid));
            const top = make('span', null, 'node-row-top');
            top.append(make('span', gid, 'node-gid'));
            const saved = state.cases.find(value => String(value.gid) === gid);
            if (saved) top.append(chip(saved.status, caseStatuses));
            else if (item.priority_score !== undefined) top.append(make('span', number(item.priority_score), 'node-score'));
            button.append(top);
            if (item.role) button.append(make('span', roles[item.role] || item.role, 'node-role'));
            if (item.evidence || item.note) button.append(make('span', item.evidence || item.note, 'node-evidence'));
            button.addEventListener('click', () => { void selectNode(gid, item); });
            return button;
        }));
        if (!list.length) el('node-list').append(make('p', state.onlyCases ? 'Сохранённых проверок пока нет.' : 'Узлы не найдены. Уточните gid.', 'empty-small'));
    }

    function renderEvents(events) {
        const values = Array.isArray(events) ? [...events] : [];
        values.sort((a, b) => String(b.created_at || b.at || '').localeCompare(String(a.created_at || a.at || '')));
        el('case-events').replaceChildren(...values.map(event => {
            const item = make('li', null, 'case-event');
            const heading = make('div', null, 'event-heading');
            const status = event.status || event.to_status || event.new_status;
            heading.append(make('span', caseStatuses[status] || 'Проверка обновлена'), make('span', date(event.created_at || event.at, true), 'event-date'));
            item.append(heading);
            if (event.note || event.new_note) item.append(make('p', event.note || event.new_note, 'event-note'));
            return item;
        }));
        if (!values.length) el('case-events').append(make('li', 'Изменения появятся после первого сохранения.', 'form-help'));
    }

    function renderCase(caseValue, events) {
        state.caseValue = caseValue;
        const draft = state.drafts.get(draftKey());
        el('case-status').value = draft?.status || caseValue?.status || 'new';
        el('case-note').value = draft?.note ?? caseValue?.note ?? '';
        el('case-fields').disabled = false;
        text('case-save-state', draft ? 'Есть несохранённые изменения' : caseValue?.updated_at ? `Сохранено ${date(caseValue.updated_at, true)}` : 'Ещё не сохранено');
        renderEvents(events);
    }

    async function selectNode(gid, node = null) {
        const runId = state.runId;
        if (!runId || currentRun()?.status !== 'completed') return;
        state.gid = String(gid);
        const sequence = ++state.caseSequence;
        node = node || state.nodes.find(item => String(item.gid) === state.gid);
        text('case-gid', state.gid);
        text('case-evidence', node?.evidence || 'Графовая роль — гипотеза. Сопоставьте её с доступными переводами и дополнительными данными.');
        text('copy-case-gid', 'Копировать gid');
        text('case-save-state', 'Загружаем проверку…');
        el('case-fields').disabled = true;
        visible('case-empty', false);
        visible('case-content', true);
        text('open-selected-case', `Открыть проверку · ${state.gid} →`);
        visible('open-selected-case', true);
        renderNodeList();
        try {
            const payload = await api(`/api/runs/${encodeURIComponent(runId)}/cases/${encodeURIComponent(gid)}`);
            if (runId !== state.runId || sequence !== state.caseSequence) return;
            renderCase(payload.case, payload.events);
        } catch (error) {
            if (runId === state.runId && sequence === state.caseSequence) {
                text('case-save-state', 'Не удалось загрузить проверку. Выберите узел ещё раз.');
                notify(error.message, 'error');
            }
        }
    }

    function rememberDraft() {
        if (!state.gid || el('case-fields').disabled) return;
        const draft = {status: el('case-status').value, note: el('case-note').value};
        if (draft.status === (state.caseValue?.status || 'new') && draft.note === (state.caseValue?.note || '')) {
            state.drafts.delete(draftKey());
            text('case-save-state', state.caseValue?.updated_at ? `Сохранено ${date(state.caseValue.updated_at, true)}` : 'Ещё не сохранено');
        } else {
            state.drafts.set(draftKey(), draft);
            text('case-save-state', 'Есть несохранённые изменения');
        }
    }

    async function saveCase(event) {
        event.preventDefault();
        if (!state.gid || state.savingCase || el('case-fields').disabled) return;
        const runId = state.runId, gid = state.gid, key = draftKey();
        const body = {status: el('case-status').value, note: el('case-note').value};
        state.savingCase = true;
        el('save-case').disabled = true;
        text('case-save-state', 'Сохраняем…');
        try {
            const payload = await api(`/api/runs/${encodeURIComponent(runId)}/cases/${encodeURIComponent(gid)}`, {method: 'PUT', body});
            const draft = state.drafts.get(key);
            if (!draft || (draft.status === body.status && draft.note === body.note)) state.drafts.delete(key);
            if (runId === state.runId && gid === state.gid) {
                renderCase(payload.case || {...body, gid}, payload.events);
                if (!state.drafts.has(key)) text('case-save-state', 'Проверка сохранена');
                await loadCases();
            }
        } catch (error) {
            if (runId === state.runId && gid === state.gid) text('case-save-state', 'Не сохранено. Повторите действие.');
            notify(error.message, 'error');
        } finally { state.savingCase = false; el('save-case').disabled = false; }
    }

    el('runs-tab').addEventListener('click', () => setHistoryTab('runs'));
    el('datasets-tab').addEventListener('click', () => setHistoryTab('datasets'));
    el('graph-tab').addEventListener('click', () => setResultTab('graph'));
    el('cases-tab').addEventListener('click', () => setResultTab('cases'));
    el('open-selected-case').addEventListener('click', () => setResultTab('cases'));
    el('new-import').addEventListener('click', () => { setView('import'); notify(''); el('import-heading').scrollIntoView({behavior: 'smooth', block: 'start'}); });
    el('refresh-history').addEventListener('click', async () => {
        el('refresh-history').disabled = true;
        try { await refreshHistory(); notify(''); } catch (error) { notify(error.message, 'error'); }
        finally { el('refresh-history').disabled = false; }
    });
    el('upload-file').addEventListener('change', event => { void uploadFile(event.target.files[0]); });
    for (const eventName of ['dragenter', 'dragover']) el('drop-zone').addEventListener(eventName, event => {
        event.preventDefault();
        if (!state.importBusy) el('drop-zone').classList.add('drag-over');
    });
    for (const eventName of ['dragleave', 'drop']) el('drop-zone').addEventListener(eventName, event => {
        event.preventDefault();
        el('drop-zone').classList.remove('drag-over');
        if (eventName === 'drop') void uploadFile(event.dataTransfer.files[0]);
    });
    el('dataset-coverage').addEventListener('change', () => text('coverage-help', el('dataset-coverage').value === 'complete'
        ? 'Выбирайте полную выгрузку только когда все исходящие переводы за заявленный период включены в файл. Эта отметка влияет на интерпретацию отсутствующего выхода.'
        : 'Если полнота неизвестна, отсутствие исходящих переводов не означает, что средства остались у получателя.'));
    el('import-form').addEventListener('submit', event => { event.preventDefault(); void createDataset(true); });
    el('save-dataset').addEventListener('click', () => { void createDataset(false); });
    el('analyze-dataset').addEventListener('click', () => { if (state.datasetId) void launchRun(state.datasetId); });
    el('retry-run').addEventListener('click', () => { if (currentRun()) void launchRun(String(currentRun().dataset_id)); });
    el('demo-button').addEventListener('click', async () => {
        el('demo-button').disabled = true;
        notify('Подготавливаем демо на исходной выборке…');
        try {
            const payload = await api('/api/demo', {method: 'POST', body: {}});
            mergeDataset(payload.dataset); mergeRun(payload.run); renderHistory();
            setHistoryTab('runs');
            await selectRun(String(payload.run.id));
            schedulePoll();
        } catch (error) { notify(error.message, 'error'); }
        finally { el('demo-button').disabled = false; }
    });
    el('node-search-form').addEventListener('submit', event => { event.preventDefault(); clearTimeout(state.searchTimer); void loadNodes(); });
    el('node-query').addEventListener('input', () => { clearTimeout(state.searchTimer); state.searchTimer = setTimeout(() => { void loadNodes(); }, 250); });
    for (const [id, onlyCases] of [['all-nodes-filter', false], ['saved-cases-filter', true]]) el(id).addEventListener('click', () => {
        state.onlyCases = onlyCases;
        state.nodeSequence++;
        for (const [buttonId, active] of [['all-nodes-filter', !onlyCases], ['saved-cases-filter', onlyCases]]) {
            el(buttonId).classList.toggle('active', active); el(buttonId).setAttribute('aria-pressed', String(active));
        }
        if (onlyCases) renderNodeList(); else void loadNodes();
    });
    el('case-status').addEventListener('change', rememberDraft);
    el('case-note').addEventListener('input', rememberDraft);
    el('case-form').addEventListener('submit', saveCase);
    el('copy-case-gid').addEventListener('click', async () => {
        try { await navigator.clipboard.writeText(state.gid); text('copy-case-gid', 'Скопировано'); }
        catch {
            const selection = window.getSelection(), range = document.createRange();
            range.selectNodeContents(el('case-gid')); selection.removeAllRanges(); selection.addRange(range);
            text('copy-case-gid', 'Нажмите Ctrl/Cmd+C');
        }
    });
    window.addEventListener('message', event => {
        if (event.origin !== window.location.origin || event.source !== el('graph-frame').contentWindow) return;
        if (event.data?.type !== 'furyhub:node-selected' || typeof event.data.gid !== 'string') return;
        if (state.view === 'run' && state.resultRunId === state.runId) void selectNode(event.data.gid);
    });
    window.addEventListener('beforeunload', event => {
        if (state.drafts.size) { event.preventDefault(); event.returnValue = ''; }
    });

    async function initialize() {
        try {
            const bootstrap = await api('/api/bootstrap');
            state.token = bootstrap.csrf_token;
            state.limits = bootstrap.limits || {};
            state.ready = Boolean(state.token);
            const limits = [state.limits.upload_mb && `до ${number(state.limits.upload_mb)} МБ`, state.limits.max_rows && `до ${number(state.limits.max_rows)} строк`, state.limits.max_nodes && `до ${number(state.limits.max_nodes)} узлов`].filter(Boolean);
            text('upload-help', `CSV или Parquet${limits.length ? ` · ${limits.join(' · ')}` : ''}`);
            for (const id of ['new-import', 'demo-button', 'refresh-history']) el(id).disabled = !state.ready;
            updateImportButtons();
            await refreshHistory();
            const latest = [...state.runs].sort((a, b) => String(b.created_at || '').localeCompare(String(a.created_at || '')))[0];
            if (latest) await selectRun(String(latest.id));
            else setView('import');
        } catch (error) {
            notify(error.message, 'error');
            text('runs-history', 'Не удалось загрузить историю. Обновите страницу после запуска сервера.');
        }
    }
    void initialize();
})();
