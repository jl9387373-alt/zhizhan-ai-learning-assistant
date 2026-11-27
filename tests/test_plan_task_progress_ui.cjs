// 使用真实前端函数模拟快速勾选和网络延迟，无需操作用户的计划记录。
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../static/app.js'), 'utf8');
const helpers = source.slice(source.indexOf('function planTaskSummary'), source.indexOf('function planPhases'));
const render = source.slice(source.indexOf('function renderPlanDetail('), source.indexOf('function collapsePlanDetail('));
const progressEntries = source.slice(source.indexOf('function renderProgressPlanEntries('), source.indexOf('async function loadProgressPlanEntries('));
const tick = () => new Promise(resolve => setImmediate(resolve));
const clone = value => JSON.parse(JSON.stringify(value));
function fixture() {
    return {subject: 'Java', created_at: 'p1', duration_days: 61, start_date: '2026-11-25',
        weekly_plan: Array.from({length: 9}, (_, i) => ({week: i + 1, tasks: i ? [] : Array.from({length: 7}, (_, j) => `任务${j}`)})),
        task_completion: {'1': Array(7).fill(false)}};
}
function container() {
    let count = 0, html = '', checks = [];
    const status = {textContent: ''};
    return {
        get renderCount() { return count; }, get innerHTML() { return html; },
        set innerHTML(value) {
            html = value; count++;
            checks = Array.from(value.matchAll(/<input type="checkbox" data-week="([^"]+)" data-idx="([^"]+)" ([^>]*)>/g), match => ({
                checked: match[3].includes('checked'), week: match[1], index: match[2],
                getAttribute(name) { return name === 'data-week' ? this.week : this.index; },
                addEventListener(_, listener) { this.change = listener; }
            }));
        },
        get checks() { return checks; }, get status() { return status.textContent; },
        querySelector() { return status; },
        querySelectorAll(selector) { return selector.startsWith('input') ? checks : []; }
    };
}
function environment() {
    let saved = fixture(), requests = [], maximum = 0, inFlight = 0;
    const errors = [];
    const ctx = vm.createContext({console, JSON, Map, Array, Math, Number, String, Intl, Date,
        planCurrentWeek: () => 1, getCurrentUserId: () => 'u', escapePlanText: s => String(s),
        planConditionMessage: () => '', renderPlanPhases: () => '',
        document: {getElementById: () => null},
        showError: msg => errors.push(msg), loadPlans() {}, loadProgressPlanEntries() {},
        loadProgressOverview() {}, updateDashboardData() {}, loadProgress() {},
        apiRequest: (url, options) => {
            if (!options) return Promise.resolve({data: {plan: clone(saved)}});
            const payload = JSON.parse(options.body);
            inFlight++; maximum = Math.max(maximum, inFlight);
            return new Promise((resolve, reject) => requests.push({payload,
                resolve() { saved.task_completion[payload.week][payload.task_index] = payload.completed; inFlight--; resolve({status: 'success', data: clone(saved)}); },
                reject() { inFlight--; reject(new Error('保存失败')); }
            }));
        }});
    vm.runInContext(helpers + render + progressEntries, ctx);
    return {ctx, requests, errors, get saved() { return clone(saved); }, get maximum() { return maximum; }};
}
async function run() {
    let tests = 0;
    {
        const e = environment(), panel = container();
        e.ctx.renderPlanDetail(panel, 'Java', fixture());
        for (let i = 0; i < 5; i++) { panel.checks[i].checked = true; panel.checks[i].change(); }
        assert.equal(e.requests.length, 1);
        for (let i = 0; i < 5; i++) {
            assert.equal(panel.renderCount, 1, '未全部保存前不重绘');
            e.requests[i].resolve(); await tick();
        }
        assert.equal(e.maximum, 1);
        assert.equal(panel.renderCount, 2);
        assert.equal(e.saved.task_completion['1'].filter(Boolean).length, 5);
        assert.match(panel.innerHTML, /本周任务完成/);
        assert.match(panel.innerHTML, /5\/7 项 · 71%/);
        assert.match(panel.innerHTML, /已细化 1\/9 周/);
        assert.doesNotMatch(panel.innerHTML, /计划进度/);
        assert.equal(panel.status, '勾选已保存');
        tests++;
    }
    {
        const e = environment(), panel = container();
        e.ctx.renderPlanDetail(panel, 'Java', fixture());
        panel.checks[0].checked = true; panel.checks[0].change();
        panel.checks[0].checked = false; panel.checks[0].change();
        panel.checks[0].checked = true; panel.checks[0].change();
        e.requests[0].resolve(); await tick();
        assert.equal(e.requests[1].payload.completed, true);
        e.requests[1].resolve(); await tick();
        assert.equal(e.saved.task_completion['1'].filter(Boolean).length, 1);
        tests++;
    }
    {
        const e = environment(), panel = container();
        e.ctx.renderPlanDetail(panel, 'Java', fixture());
        panel.checks[0].checked = true; panel.checks[0].change();
        panel.checks[1].checked = true; panel.checks[1].change();
        e.requests[0].reject(); await tick();
        e.requests[1].resolve(); await tick();
        assert.equal(panel.checks[0].checked, false);
        assert.equal(panel.checks[1].checked, true);
        assert.equal(e.saved.task_completion['1'].filter(Boolean).length, 1);
        assert.match(panel.status, /未确认保存/);
        tests++;
    }
    {
        const e = environment(), panel = container();
        e.ctx.renderPlanDetail(panel, 'Java', fixture());
        panel.checks[0].checked = true; panel.checks[0].change();
        e.ctx.renderPlanDetail(panel, 'Python', {...fixture(), subject: 'Python'});
        const renders = panel.renderCount;
        e.requests[0].resolve(); await tick();
        assert.equal(panel.renderCount, renders);
        assert.match(panel.innerHTML, /Python · 计划详情/);
        tests++;
    }
    {
        const e = environment(), plan = fixture();
        plan.task_completion['1'] = [true, true, false, false, false, true, true, true];
        const html = e.ctx.renderProgressPlanEntries([plan]);
        assert.match(html, /本周已完成 4\/7 项任务/);
        assert.equal(e.ctx.planTaskSummary(plan).percentage, 57);
        tests++;
    }
    {
        const e = environment(), panel = container();
        e.ctx.renderPlanDetail(panel, 'Java', fixture());
        panel.checks[0].checked = true; panel.checks[0].change();
        panel.checks[1].checked = true; panel.checks[1].change();
        e.ctx.renderPlanDetail(panel, 'Java', fixture());
        assert.equal(panel.checks[0].checked, true);
        assert.equal(panel.checks[1].checked, true);
        panel.checks[2].checked = true; panel.checks[2].change();
        for (let i = 0; i < 3; i++) { e.requests[i].resolve(); await tick(); }
        assert.equal(e.maximum, 1);
        assert.equal(e.saved.task_completion['1'].filter(Boolean).length, 3);
        tests++;
    }
    {
        const e = environment(), panel = container(), plan = fixture();
        plan.duration_days = 10;
        plan.weekly_plan = [{week: 1, tasks: ['任务1', '任务2']}, {week: 2, tasks: ['任务3']}];
        plan.task_completion = {'1': [true, true], '2': [false]};
        e.ctx.renderPlanDetail(panel, 'Java', plan);
        assert.match(panel.innerHTML, /2\/3 项 · 66%/);
        assert.doesNotMatch(panel.innerHTML, /本周任务完成|后续任务尚未安排/);
        tests++;
    }
    console.log(JSON.stringify({tests, failures: 0}));
}
run().catch(error => { console.error(error); process.exit(1); });
