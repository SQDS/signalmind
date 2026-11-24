import {api, esc, json, modal, num} from './ui.js';

const states = {
    accepted: '修补已接受', preserved: '保留原样', unrecoverable: '停用',
    need_evidence: '需要证据', pending_budget: '等待送审',
    hard_gate_rejected: '程序硬门退回', review_rejected: '独立复查退回'
};
const support = {
    evaluated_no_alert: '已评估', suspected_artifact: '疑似伪迹', possible_physiology: '可能是生理事件',
    insufficient_support: '支持不足', not_applicable: '不适用', failed: '检查失败', pending: '检查待完成'
};

export async function qualityPage(el, context) {
    if (!context.run) {
        el.textContent = '请选择一个整库运行。';
        return;
    }
    const pageSize = 40;
    const offset = Math.max(0, Number(context.options.offset) || 0);
    const state = context.options.state || '';
    const params = new URLSearchParams({offset, limit: pageSize + 1});
    if (state) params.set('state', state);
    const host = document.createElement('section');
    el.replaceChildren(host);
    const data = await api('/api/runs/' + encodeURIComponent(context.run) + '/quality?' + params);
    if (data.status !== 'published') {
        host.textContent = '这个运行尚无已发布的候选账本。';
        return;
    }
    const navigate = (next, filter) => context.navigate('quality', {run: context.run, offset: next, state: filter});
    const items = data.items.slice(0, pageSize);
    const options = Object.entries(states).map(([key, label]) =>
        '<option value="' + key + '"' + (state === key ? ' selected' : '') + '>' + label + '</option>').join('');
    const rows = items.map((row, i) => {
        const times = row.source_time_s.map(v => num(v, 3)).join(' – ');
        return '<tr><td>' + esc(row.record_id) + '</td><td>' + esc(row.channel_id) + '</td><td>' + times +
            '</td><td>' + esc(row.detector) + '</td><td>' + esc(support[row.support_status] || row.support_status) +
            '</td><td>' + esc(states[row.state] || row.state) + '</td><td><button data-detail="' + i + '">证据</button></td></tr>';
    }).join('');
    host.innerHTML = '<label>处置状态 <select><option value="">全部</option>' + options + '</select></label>' +
        '<p>全库共 ' + num(data.candidates) + ' 条候选。当前筛选显示 ' + num(items.length) + ' 条。</p>' +
        '<table><thead><tr><th>记录</th><th>通道</th><th>源时间（秒）</th><th>检测项</th><th>证据支持</th>' +
        '<th>处置</th><th></th></tr></thead><tbody>' + rows + '</tbody></table><p><button data-prev>上一页</button> ' +
        '<button data-next>下一页</button></p>';
    host.querySelector('select').onchange = event => navigate(0, event.target.value);
    const prev = host.querySelector('[data-prev]');
    prev.disabled = offset === 0;
    prev.onclick = () => navigate(Math.max(0, offset - pageSize), state);
    const next = host.querySelector('[data-next]');
    next.disabled = data.items.length <= pageSize;
    next.onclick = () => navigate(offset + pageSize, state);
    host.querySelectorAll('[data-detail]').forEach(button => {
        button.onclick = () => modal('候选证据与版本', json(items[Number(button.dataset.detail)]));
    });
}
