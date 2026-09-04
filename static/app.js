let traces = [];
let currentTrace = null;
let currentMetrics = null;
let currentTab = 'answers';
let chartZoom = null;
let chartDrag = { active:false, startX:0, startVs:0 };
let testCases = [];
let diagnosisBusy = false;
let viewMode = "traces";
let currentRun = null;
let chartSelection = null;   // [startMs, endMs]
let selectedSpan = null;     // selected span object
let selectedEventIndices = null; // 选中节点对应的原始事件下标（显式匹配）
let clickGuard = false;
let selectingRange = false;
let selectionStartX = 0;

function sanitizeHtml(html, allowEvents = false) {
  const text = String(html == null ? '' : html);
  const doc = new DOMParser().parseFromString(text, 'text/html');
  doc.querySelectorAll('script,style,iframe,object,embed,link,meta,base,svg,math,form').forEach(n => n.remove());
  doc.querySelectorAll('*').forEach(n => {
    for (const attr of Array.from(n.attributes)) {
      const name = attr.name.toLowerCase();
      const value = (attr.value || '').trim().toLowerCase();
      if (!allowEvents && name.startsWith('on')) {
        n.removeAttribute(attr.name);
        continue;
      }
      if (name === 'srcdoc' || value.startsWith('javascript:') || value.startsWith('data:text/html')) {
        n.removeAttribute(attr.name);
      }
    }
  });
  return doc.body.innerHTML;
}



function switchView(mode) {
  viewMode = mode;
  document.getElementById('navTraces').classList.toggle('active', mode==='traces');
  document.getElementById('navRuns').classList.toggle('active', mode==='runs');
  if (mode==='runs') renderRunsList(); else { refreshList(); }
}

async function renderRunsList() {
  const res = await fetch('/api/runs');
  const runs = await res.json();
  const box = document.getElementById('traceList');
  box.innerHTML=sanitizeHtml(runs.map(r => `
    <div class="trace-item ${currentRun && currentRun.run_id===r.run_id ? 'active' : ''}" onclick="selectRun('${r.run_id}')">
      <div class="q">${escapeHtml(r.name || r.run_id)}</div>
      <div class="meta"><span class="badge found">${r.summary.passed_cases ?? 0}/${r.summary.total_cases ?? 0}</span><span>${formatMs(r.summary.avg_duration_ms)}</span></div>
    </div>`).join('') || '<div style="color:var(--muted);padding:16px">暂无 Run</div>', true);
}

async function selectRun(runId) {
  const res = await fetch('/api/runs/' + runId);
  currentRun = await res.json();
  currentTrace = null;
  renderRunsList();
  renderRunOverview(currentRun);
}

function renderRunOverview(run) {
  document.getElementById('traceTitle').textContent = run.name || run.run_id;
  const s = run.summary || {};
  document.getElementById('traceMeta').innerHTML=sanitizeHtml(`
    <span>通过 ${s.passed_cases ?? 0}/${s.total_cases ?? 0}</span>
    <span>通过率 ${s.pass_rate ?? 0}%</span>
    <span>平均耗时 ${formatMs(s.avg_duration_ms)}</span>`, true);
  const rows = (run.results || []).map(r => `
    <tr>
      <td>${escapeHtml(r.case_id)}</td>
      <td>${r.passed ? '✅' : '❌'}</td>
      <td>${escapeHtml((r.question || '').slice(0, 50))}</td>
      <td>${r.trace_id ? `<a href="javascript:void(0)" onclick="selectTrace('${r.trace_id}')">详情</a>` : '无Trace'}</td>
      <td><button class="tab" onclick="openRunReport('${run.run_id}','${r.case_id}')">报告</button></td>
      <td>${r.passed ? '<span style="color:var(--muted)">—</span>' : `<button class="tab doctor" onclick="openRunDoctor('${run.run_id}','${r.case_id}')">🩺 医生</button>`}</td>
    </tr>`).join('');
  document.getElementById('content').innerHTML=sanitizeHtml(`
    <div class="run-overview">
      <div class="metrics-grid">
        <div class="metric-card"><div class="metric-value">${s.total_cases ?? 0}</div><div class="metric-label">总题数</div></div>
        <div class="metric-card"><div class="metric-value">${s.passed_cases ?? 0}</div><div class="metric-label">通过</div></div>
        <div class="metric-card"><div class="metric-value">${s.failed_cases ?? 0}</div><div class="metric-label">失败</div></div>
        <div class="metric-card"><div class="metric-value">${s.pass_rate ?? 0}%</div><div class="metric-label">通过率</div></div>
      </div>
      <h3>题目结果</h3>
      <table><tr><th>题号</th><th>结果</th><th>题目</th><th>Trace</th><th>报告</th><th>医生</th></tr>${rows}</table>
    </div>`, true);
}

async function openRunReport(runId, caseId) {
  let res = await fetch(`/api/runs/${runId}/report/${caseId}`);
  let d;
  if (res.status === 200) d = await res.json();
  else {
    res = await fetch(`/api/runs/${runId}/report/${caseId}`, {method:'POST'});
    d = await res.json();
  }
  if (d.report_html) {
    const box = document.getElementById('content');
    box.innerHTML=sanitizeHtml(`<h2>${escapeHtml(caseId)} 运行分析报告</h2><div class="report-html">${d.report_html}</div><button class="tab" onclick="selectRun('${runId}')">返回</button>`, false);
  }
}


async function openRunDoctor(runId, caseId) {
  const box = document.getElementById('content');
  box.innerHTML=sanitizeHtml(`<h2>${escapeHtml(caseId)} 项目医生诊断</h2><div class="diag-result">正在执行强制检查单（可能需要 1~3 分钟，期间会只读调用原项目知识库检索）……</div>`, true);
  try {
    let res = await fetch(`/api/runs/${runId}/doctor/${caseId}`);
    let d;
    if (res.status === 200) {
      d = await res.json();
    } else {
      res = await fetch(`/api/runs/${runId}/doctor/${caseId}`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({})
      });
      d = await res.json();
    }
    renderDoctorView(runId, caseId, d.payload || d);
  } catch (e) {
    box.innerHTML=sanitizeHtml(`<h2>${escapeHtml(caseId)} 项目医生诊断</h2><div class="diag-result">请求失败：${escapeHtml(e)}</div><button class="tab" onclick="selectRun('${runId}')">返回</button>`, true);
  }
}

function renderDoctorView(runId, caseId, d) {
  const box = document.getElementById('content');
  const rep = d.report || {};
  const diag = rep.diagnosis || {};
  const cov = d.coverage || rep._coverage || {};
  const pres = rep.prescriptions || [];
  const presHtml = pres.length ? pres.map((p, i) => `
    <div class="doctor-pres">
      <div class="panel-title">处方 ${i + 1} · ${escapeHtml(p.severity || '')}</div>
      <table>
        <tr><th>问题</th><td>${escapeHtml(p.issue || '')}</td></tr>
        <tr><th>根因</th><td>${escapeHtml(p.root_cause || '')}</td></tr>
        <tr><th>证据</th><td>${(p.evidence_ids || []).map(x => `<code>${escapeHtml(x)}</code>`).join(' ')}</td></tr>
        <tr><th>证据等级</th><td>${escapeHtml(p.evidence_level || '—')}</td></tr>
        <tr><th>建议</th><td>${escapeHtml(p.suggestion || '')}</td></tr>
        <tr><th>目标文件</th><td>${p.target_file ? `<code>${escapeHtml(p.target_file)}</code>` : '—'}</td></tr>
        <tr><th>预期效果</th><td>${escapeHtml(p.expected_effect || '')}</td></tr>
      </table>
    </div>`).join('') : '<div class="empty">（无处方）</div>';
  const orders = d.lab_orders || [];
  const orderRows = orders.map(o => `
    <tr><td><code>${escapeHtml(o.id)}</code></td><td>${escapeHtml(o.title)}</td><td>${escapeHtml(o.category)}</td></tr>`).join('');
  const evidenceDigest = d.evidence_by_order || {};
  const evRows = Object.entries(evidenceDigest).map(([oid, evs]) => `
    <tr><td><code>${escapeHtml(oid)}</code></td><td>${(evs || []).map(e => escapeHtml(e.summary || e.status || '')).join('<br>')}</td></tr>`).join('');
  box.innerHTML=sanitizeHtml(`
    <div class="doctor-view">
      <div class="diagnosis-head">
        <h2>${escapeHtml(caseId)} 项目医生诊断</h2>
        <button class="tab" onclick="selectRun('${runId}')">返回</button>
      </div>
      <div class="metrics-grid">
        <div class="metric-card"><div class="metric-value">${cov.completed_orders ?? 0}/${cov.total_orders ?? 0}</div><div class="metric-label">检查单覆盖</div></div>
        <div class="metric-card"><div class="metric-value">${Math.round((cov.coverage ?? 0) * 100)}%</div><div class="metric-label">覆盖闸门</div></div>
        <div class="metric-card"><div class="metric-value">${escapeHtml(d.model || 'qwen3.7-max')}</div><div class="metric-label">医生模型</div></div>
        <div class="metric-card"><div class="metric-value">${Math.round((rep.confidence ?? 0) * 100)}%</div><div class="metric-label">置信度</div></div>
      </div>
      <h3>诊断结论</h3>
      <table>
        <tr><th>结论</th><td>${escapeHtml(diag.summary || '')}</td></tr>
        <tr><th>主根因</th><td>${escapeHtml(diag.primary_root_cause || '')}</td></tr>
        <tr><th>问题分类</th><td>${escapeHtml(diag.issue_classification || '')} / ${escapeHtml(diag.data_vs_recall || '')}</td></tr>
        <tr><th>关键证据</th><td>${(diag.key_evidence || []).map(x => `<code>${escapeHtml(x)}</code>`).join(' ')}</td></tr>
      </table>
      <h3>处方</h3>${presHtml}
      <h3>强制检查单</h3>
      <table><tr><th>ID</th><th>标题</th><th>类别</th></tr>${orderRows || '<tr><td colspan="3">无</td></tr>'}</table>
      <h3>证据摘要</h3>
      <table><tr><th>证据ID</th><th>摘要</th></tr>${evRows || '<tr><td colspan="2">无</td></tr>'}</table>
      <h3>医生长期记忆</h3>
      ${(d.verified_claims || []).map(c => `<div class="doctor-memory"><code>${escapeHtml(c.id)}</code> ${escapeHtml(c.claim)} <span class="muted">证据: ${(c.evidence_ids || []).map(x => `<code>${escapeHtml(x)}</code>`).join(' ')}</span></div>`).join('') || ''}
      ${(d.pinned_facts || []).map(f => `<div class="doctor-memory"><code>${escapeHtml(f.id)}</code> ${escapeHtml(f.text)} <span class="muted">证据: <code>${escapeHtml(f.evidence_id)}</code></span></div>`).join('') || ''}
      ${(!d.verified_claims || !d.verified_claims.length) && (!d.pinned_facts || !d.pinned_facts.length) ? '<div class="empty">（无长期记忆记录）</div>' : ''}
    </div>`, true);
}

async function loadTestCases() {
  const res = await fetch('/api/testcases');
  testCases = await res.json();
}

async function refreshList() {
  const res = await fetch('/api/traces');
  traces = await res.json();
  renderList();
}

function renderList() {
  const q = (document.getElementById('searchInput').value || '').trim().toLowerCase();
  const box = document.getElementById('traceList');
  const filtered = traces.filter(t => (t.question || '').toLowerCase().includes(q));
  box.innerHTML=sanitizeHtml(filtered.map(t => `
    <div class="trace-item ${currentTrace && currentTrace.trace_id === t.trace_id ? 'active' : ''}" onclick="selectTrace('${t.trace_id}')">
      <div class="q">${escapeHtml(t.question || '')}</div>
      <div class="meta">
        <span class="trace-id">${escapeHtml((t.trace_id || '').slice(-8))}</span>
        <span class="badge ${t.execution_mode === 'L1' ? 'l1' : 'l2'}">${t.execution_mode || '?'}</span>
        <span class="badge ${t.response_mode === 'found' ? 'found' : 'not_found'}">${t.response_mode || ''}</span>
        <span>${formatMs(t.duration_ms)}</span>
        <span>${escapeHtml((t.created_at || '').slice(5, 16))}</span>
      </div>
    </div>
  `).join('') || '<div style="color:var(--muted);padding:16px">暂无 Trace</div>', true);
}

async function selectTrace(id) {
  const [traceRes, metricsRes] = await Promise.all([
    fetch('/api/traces/' + id),
    fetch('/api/traces/' + id + '/metrics')
  ]);
  currentTrace = await traceRes.json();
  currentMetrics = await metricsRes.json();
  chartZoom = null;
  chartSelection = null;
  selectedSpan = null;
  selectedEventIndices = null;
  renderList();
  renderTrace(currentTrace);
}

function renderTrace(t) {
  document.getElementById('traceTitle').textContent = t.question || t.trace_id;
  const intent = (t.metadata && t.metadata.intent_labels || []).join(', ');
  const m = currentMetrics || {};
  document.getElementById('traceMeta').innerHTML=sanitizeHtml(`
    <span class="badge ${t.metadata && t.metadata.execution_mode === 'L1' ? 'l1' : 'l2'}">${t.metadata && t.metadata.execution_mode || '?'}</span>
    <span class="badge ${t.metadata && t.metadata.response_mode === 'found' ? 'found' : 'not_found'}">${t.metadata && t.metadata.response_mode || ''}</span>
    ${intent ? `<span class="badge l2">${escapeHtml(intent)}</span>` : ''}
    <span>耗时 ${formatMs(t.duration_ms)}</span>
    <span>工具 ${m.tool_count ?? 0}</span>
    <span>未找到 ${m.not_found_count ?? 0}</span>
    <span>拦截 ${m.intercepted_count ?? 0}</span>
    ${m.meltdown_count ? `<span class="badge not_found">熔断 ${m.meltdown_count}</span>` : ''}
  `, true);
  const timelineHtml = `
    <div class="span-card">
      <div class="span-header">
        <div class="span-icon">🤖</div>
        <div class="span-name">${escapeHtml(t.root_span.name)}</div>
        <div class="span-detail"><span>${t.root_span.status}</span></div>
      </div>
    </div>
    <div class="tree">${(t.root_span.children || []).map(s => renderSpan(s, 0)).join('')}</div>
  `;
  document.getElementById('content').innerHTML=sanitizeHtml(`
    <div class="tabs">
      <button class="tab ${currentTab === 'answers' ? 'active' : ''}" onclick="setTab('answers')">答案</button>
      <button class="tab ${currentTab === 'timeline' ? 'active' : ''}" onclick="setTab('timeline')">时间线</button>
      <button class="tab ${currentTab === 'chart' ? 'active' : ''}" onclick="setTab('chart')">耗时图</button>
      <button class="tab ${currentTab === 'metrics' ? 'active' : ''}" onclick="setTab('metrics')">指标</button>
    </div>
    <div id="tabTimeline" style="${currentTab === 'timeline' ? '' : 'display:none'}">${timelineHtml}</div>
    <div id="tabChart" style="${currentTab === 'chart' ? '' : 'display:none'}">${renderChart(t)}</div>
    <div id="tabMetrics" style="${currentTab === 'metrics' ? '' : 'display:none'}">${renderMetricsTab(m)}</div>
    <div id="tabAnswers" style="${currentTab === 'answers' ? '' : 'display:none'}">${renderAnswersTab(t)}</div>
  `, true);
  bindToggle();
  bindChartWheel();
}

function renderChart(t) {
  const root = t.root_span || {};
  const spans = [];
  (function walk(s) {
    if (s.start_time && s.end_time) spans.push(s);
    (s.children || []).forEach(walk);
  })(root);

  if (!spans.length) return '<div class="empty">没有可用的时间数据</div>';

  let allStart = Infinity, allEnd = 0;
  spans.forEach(s => {
    const a = new Date(s.start_time).getTime();
    const b = new Date(s.end_time).getTime();
    if (a < allStart) allStart = a;
    if (b > allEnd) allEnd = b;
  });
  if (allEnd <= allStart) return '<div class="empty">时间范围异常</div>';

  if (!chartZoom) chartZoom = [allStart, allEnd];
  let [vs, ve] = chartZoom;
  vs = Math.max(allStart, vs);
  ve = Math.min(allEnd, ve);
  const total = Math.max(1, ve - vs);

  const groups = [
    { key: 'agent', label: 'Agent', color: 'agent', filter: s => s.span_type === 'agent' },
    { key: 'phase', label: '阶段', color: 'phase', filter: s => ['rewrite','assess','router'].includes(s.span_type) },
    { key: 'model', label: 'Model', color: 'model', filter: s => s.span_type === 'llm' || s.span_type === 'answer' },
    { key: 'tools', label: 'Tools', color: 'tools', filter: s => s.span_type === 'tool' }
  ];

  const rows = groups.map(g => {
    const list = spans.filter(g.filter).filter(s => {
      const st = new Date(s.start_time).getTime();
      const en = new Date(s.end_time).getTime();
      return st < ve && en > vs;
    }).sort((a,b) => new Date(a.start_time) - new Date(b.start_time));
    const bars = list.map(s => {
      const st = new Date(s.start_time).getTime();
      const en = new Date(s.end_time).getTime();
      const left = ((st - vs) / total * 100).toFixed(3);
      const width = Math.max(0.5, ((en - st) / total * 100)).toFixed(3);
      const label = s.name || s.span_type;
      const active = selectedSpan && selectedSpan.span_id === s.span_id;
      return `<div class="chart-bar ${g.color} ${active ? 'selected' : ''}" data-span-id="${escapeHtml(s.span_id)}" style="left:${left}%;width:${width}%" title="${escapeHtml(label)} ${formatMs(en-st)}" onclick="selectSpan('${escapeHtml(s.span_id)}')">
        <span class="bar-label">${escapeHtml(label)}</span>
      </div>`;
    }).join('');
    return `
      <div class="chart-row">
        <div class="chart-label">${g.label}</div>
        <div class="chart-track">${bars}</div>
      </div>`;
  }).join('');

  const ticks = [0, 25, 50, 75, 100].map(p => {
    const ts = new Date(vs + total * p / 100);
    return `<div class="chart-tick" style="left:${p}%">${ts.toLocaleTimeString('zh-CN', {hour:'2-digit',minute:'2-digit',second:'2-digit'})}</div>`;
  }).join('');

  const eventsHtml = renderEventsForChart(t, vs, ve);
  const detailHtml = selectedSpan ? renderSpanDetail(selectedSpan) : '';

  const eventsSectionClass = selectedSpan ? 'events-section with-detail' : 'events-section';

  return `
    <div class="chart-wrap" id="chartWrap">
      <div class="chart-legend">
        <span><i class="dot model"></i> Model</span>
        <span><i class="dot tools"></i> Tools</span>
        <span><i class="dot phase"></i> 阶段</span>
        <span><i class="dot agent"></i> Agent</span>
        <button class="tab" onclick="resetChartZoom()">重置缩放</button>
        <button class="tab" onclick="clearSelection()">清除选区</button>
        <span class="zoom-hint">滚轮缩放 · 拖动平移 · Shift+拖动选择区间 · 点击空白关闭详情</span>
      </div>
      <div class="chart">
        ${rows}
        <div class="chart-selection-overlay" id="selectionOverlay"></div>
        <div class="chart-axis">${ticks}</div>
      </div>
      <div class="${eventsSectionClass}">
        <div class="events-main">
          <div class="panel-title">原始事件</div>
          <div class="event-list chart-event-list full">${eventsHtml}</div>
        </div>
        ${selectedSpan ? `
        <div class="detail-sidebar">
          <div class="detail-head">
            <span>节点详情</span>
            <button class="close-detail" onclick="clearSelection()">×</button>
          </div>
          ${detailHtml}
        </div>` : ''}
      </div>
    </div>
  `;
}


function eventColorClass(ev) {
  const e = ev.event || '';
  if (/^(rewrite|assess|route)$/.test(e)) return 'phase';
  if (['llm_start','llm_end','plan','answer_start','answer_end'].includes(e)) return e.startsWith('answer') ? 'answer' : 'model';
  if (['tool_start','tool_end'].includes(e)) return 'tools';
  return 'other';
}

function computeSpanEventIndices(span, events) {
  const indices = [];
  const st = span.start_time ? new Date(span.start_time).getTime() : 0;
  const en = span.end_time ? new Date(span.end_time).getTime() : 0;
  const inTime = (tsMs) => st && en && tsMs >= st - 300 && tsMs <= en + 300;

  events.forEach((ev, i) => {
    const data = ev.data || {};
    const type = ev.event || '';
    const tsMs = (ev.timestamp || 0) * 1000;
    if (span.span_type === 'tool') {
      if (data.tool_call_id && data.tool_call_id === span.tool_call_id) indices.push(i);
    } else if (span.span_type === 'answer') {
      if (['answer_start','answer_end'].includes(type) && inTime(tsMs)) indices.push(i);
      if (['llm_start','llm_end'].includes(type) && data.role === 'answer' && inTime(tsMs)) indices.push(i);
    } else if (span.span_type === 'llm') {
      const role = span.name === 'plan_agent' ? 'plan' :
                   span.name === 'plan_agent_retry' ? 'plan_retry' :
                   span.name === 'fast_agent' ? 'fast' :
                   (span.name === 'answer_agent' ? 'answer' : null);
      if (role && ['llm_start','llm_end'].includes(type) && data.role === role && inTime(tsMs)) indices.push(i);
      if (role === 'plan' && type === 'plan' && inTime(tsMs)) indices.push(i);
    } else if (span.span_type === 'rewrite' && type === 'rewrite') {
      indices.push(i);
    } else if (span.span_type === 'assess' && type === 'assess') {
      indices.push(i);
    } else if (span.span_type === 'router' && type === 'route') {
      indices.push(i);
    } else if (span.span_type === 'agent') {
      indices.push(i);
    }
  });
  return indices;
}

function renderEventsForChart(t, vs, ve) {
  const events = t.trace_events || [];
  if (!events.length) return '<div class="empty">没有保存原始事件</div>';
  return events.map((ev, idx) => {
    const tsMs = (ev.timestamp || 0) * 1000;
    let inSel = true;
    let selCls = '';
    if (selectedEventIndices) {
      inSel = selectedEventIndices.includes(idx);
      selCls = inSel ? ' highlight' : ' dimmed';
    } else if (chartSelection) {
      inSel = tsMs >= chartSelection[0] && tsMs <= chartSelection[1];
      selCls = inSel ? ' highlight' : ' dimmed';
    }
    const color = eventColorClass(ev);
    const data = ev.data || {};
    return `<div class="event-row event-${color}${selCls}">
      <span class="event-type">${escapeHtml(ev.event)}</span>
      <span class="event-time">${new Date(tsMs).toLocaleTimeString('zh-CN', {hour:'2-digit',minute:'2-digit',second:'2-digit'})}</span>
      <span class="event-data">${escapeHtml(JSON.stringify(data).slice(0, 240))}</span>
    </div>`;
  }).join('');
}

function renderSpanDetail(span) {
  const dur = durationText(span.start_time, span.end_time);
  let html = `<div class="detail-title">${escapeHtml(span.name)}</div>`;
  html += `<div class="detail-meta"><span>${escapeHtml(span.span_type)}</span><span>${escapeHtml(span.status)}</span>${dur ? `<span>${dur}</span>` : ''}</div>`;
  if (span.model) html += `<div class="label">模型</div><pre>${escapeHtml(span.model)}</pre>`;
  if (span.span_type === 'tool') {
    html += `<div class="label">参数</div><pre>${escapeHtml(JSON.stringify(span.tool_args || {}, null, 2))}</pre>`;
    html += `<div class="label">返回 ${span.result_length ?? ''} 字 ${span.meltdown_trigger ? '· 熔断' : ''}</div><pre>${escapeHtml((span.result_preview || '').slice(0, 800))}</pre>`;
  } else if (span.span_type === 'llm') {
    if ((span.tool_calls || []).length) {
      html += `<div class="label">工具调用</div><pre>${escapeHtml(span.tool_calls.map(c => c.name + '(' + JSON.stringify(c.args || {}) + ')').join(', '))}</pre>`;
    }
    if (span.output && span.output.execution_plan) {
      html += `<div class="label">执行报告</div><pre>${escapeHtml(span.output.execution_plan)}</pre>`;
    }
  } else if (span.span_type === 'answer') {
    html += `<div class="label">最终回答</div><pre>${escapeHtml((span.output && span.output.final_response) || '')}</pre>`;
  } else {
    html += `<pre>${escapeHtml(JSON.stringify(span.output || {}, null, 2))}</pre>`;
  }
  return html;
}

function findSpanById(root, spanId) {
  let found = null;
  (function walk(s) {
    if (found) return;
    if (s.span_id === spanId) { found = s; return; }
    (s.children || []).forEach(walk);
  })(root);
  return found;
}

function selectSpan(spanId) {
  if (!currentTrace) return;
  const span = findSpanById(currentTrace.root_span || {}, spanId);
  if (!span) return;
  selectedSpan = span;
  clickGuard = true;
  setTimeout(() => { clickGuard = false; }, 60);
  const events = currentTrace.trace_events || [];
  const matched = computeSpanEventIndices(span, events);
  selectedEventIndices = matched.length ? matched : null;
  if (span.start_time && span.end_time) {
    chartSelection = [new Date(span.start_time).getTime(), new Date(span.end_time).getTime()];
  }
  updateChartView();
}

function clearSelection() {
  chartSelection = null;
  selectedSpan = null;
  selectedEventIndices = null;
  updateChartView();
}

function getFullRange() {
  let allStart = Infinity, allEnd = 0;
  (function walk(s) {
    if (s.start_time && s.end_time) {
      const a = new Date(s.start_time).getTime();
      const b = new Date(s.end_time).getTime();
      if (a < allStart) allStart = a;
      if (b > allEnd) allEnd = b;
    }
    (s.children || []).forEach(walk);
  })(currentTrace.root_span || {});
  if (allEnd <= allStart) { allStart = Date.now(); allEnd = allStart + 1; }
  return [allStart, allEnd];
}

function updateChartView() {
  const host = document.getElementById('tabChart');
  if (host && currentTrace) {
    host.innerHTML=sanitizeHtml(renderChart(currentTrace), true);
    bindChartWheel();
  }
}

function bindChartWheel() {
  const chartEl = document.querySelector('#tabChart .chart');
  if (!chartEl) return;

  const chartWrap = document.getElementById('chartWrap');
  if (chartWrap) {
    chartWrap.onclick = function(e) {
      if (clickGuard) return;
      if (e.target.closest('.chart-bar')) return;
      if (e.target.closest('.detail-sidebar')) return;
      if (e.target.closest('button')) return;
      clearSelection();
    };
  }

  chartEl.onwheel = function(e) {
    e.preventDefault();
    zoomChart(e);
  };

  chartEl.onmousedown = function(e) {
    if (e.button !== 0) return;
    e.preventDefault();
    const rect = chartEl.getBoundingClientRect();
    if (e.shiftKey) {
      selectingRange = true;
      selectionStartX = e.clientX;
      chartEl.classList.add('selecting');
      const overlay = document.getElementById('selectionOverlay');
      if (overlay) {
        const ratio = (e.clientX - rect.left) / rect.width;
        overlay.style.left = (ratio * 100) + '%';
        overlay.style.width = '0%';
        overlay.style.display = 'block';
      }
      return;
    }
    const [allStart, allEnd] = getFullRange();
    if (!chartZoom) chartZoom = [allStart, allEnd];
    chartDrag = { active: true, startX: e.clientX, startVs: chartZoom[0] };
    chartEl.classList.add('dragging');
  };

  chartEl.onmousemove = function(e) {
    const rect = chartEl.getBoundingClientRect();
    if (rect.width <= 0) return;
    if (selectingRange) {
      const overlay = document.getElementById('selectionOverlay');
      if (overlay) {
        const startRatio = (selectionStartX - rect.left) / rect.width;
        const curRatio = (e.clientX - rect.left) / rect.width;
        const left = Math.min(startRatio, curRatio) * 100;
        const width = Math.abs(curRatio - startRatio) * 100;
        overlay.style.left = left + '%';
        overlay.style.width = width + '%';
      }
      return;
    }
    if (!chartDrag.active || !chartZoom) return;
    const [allStart, allEnd] = getFullRange();
    const deltaPx = e.clientX - chartDrag.startX;
    const range = chartZoom[1] - chartZoom[0];
    const timeDelta = (deltaPx / rect.width) * range;
    let newVs = chartDrag.startVs - timeDelta;
    let newVe = newVs + range;
    if (newVs < allStart) { newVs = allStart; newVe = newVs + range; }
    if (newVe > allEnd) { newVe = allEnd; newVs = newVe - range; }
    chartZoom = [newVs, newVe];
    updateChartView();
  };

  chartEl.onmouseup = function(e) {
    if (selectingRange) {
      const rect = chartEl.getBoundingClientRect();
      const startRatio = (selectionStartX - rect.left) / rect.width;
      const endRatio = (e.clientX - rect.left) / rect.width;
      const [allStart, allEnd] = getFullRange();
      if (!chartZoom) chartZoom = [allStart, allEnd];
      const range = chartZoom[1] - chartZoom[0];
      const t0 = chartZoom[0] + Math.min(startRatio, endRatio) * range;
      const t1 = chartZoom[0] + Math.max(startRatio, endRatio) * range;
      chartSelection = [Math.max(allStart, t0), Math.min(allEnd, t1)];
      selectedSpan = null;
      selectedEventIndices = null;
      selectingRange = false;
      chartEl.classList.remove('selecting');
      updateChartView();
      return;
    }
    chartDrag.active = false;
    chartEl.classList.remove('dragging');
  };
  chartEl.onmouseleave = function() {
    if (selectingRange) {
      selectingRange = false;
      chartEl.classList.remove('selecting');
      const overlay = document.getElementById('selectionOverlay');
      if (overlay) overlay.style.display = 'none';
    }
    chartDrag.active = false;
    chartEl.classList.remove('dragging');
  };
}

function zoomChart(e) {
  if (!currentTrace) return;
  const [allStart, allEnd] = getFullRange();
  if (!chartZoom) chartZoom = [allStart, allEnd];
  let [vs, ve] = chartZoom;
  const range = (ve - vs) || 1;
  const chartEl = document.querySelector('#tabChart .chart');
  const rect = chartEl ? chartEl.getBoundingClientRect() : null;
  let ratio = 0.5;
  if (rect && rect.width > 0) {
    ratio = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
  }
  const factor = e.deltaY < 0 ? 0.7 : 1.4;
  let newRange = range * factor;
  if (newRange < 500) newRange = 500;
  if (newRange > (allEnd - allStart)) newRange = allEnd - allStart;
  const timeAtCursor = vs + ratio * range;
  let newVs = timeAtCursor - newRange * ratio;
  let newVe = newVs + newRange;
  if (newVs < allStart) { newVs = allStart; newVe = newVs + newRange; }
  if (newVe > allEnd) { newVe = allEnd; newVs = newVe - newRange; }
  chartZoom = [newVs, newVe];
  updateChartView();
}

function resetChartZoom() {
  chartZoom = null;
  updateChartView();
}

function locateSpan(spanId) {
  currentTab = 'timeline';
  renderTrace(currentTrace);
  setTimeout(() => {
    const el = document.querySelector(`[data-span-id="${spanId}"]`);
    if (!el) return;
    document.querySelectorAll('.span-body').forEach(b => b.style.display = '');
    el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    el.classList.add('highlight');
  }, 60);
}

function renderAnswersTab(t) {
  const answer = (t.root_span && t.root_span.output && t.root_span.output.final_response) || '';
  const caseInfo = testCases.find(c => c.question === t.question) || null;
  const must = caseInfo && caseInfo.must_contain || [];
  const mustNot = caseInfo && caseInfo.must_not_contain || [];
  const kwRows = must.map(kw => {
    const hit = answer.includes(kw);
    return `<tr><td>${hit ? '✅' : '❌'}</td><td>必须包含</td><td>${escapeHtml(kw)}</td></tr>`;
  }).join('');
  const notRows = mustNot.map(kw => {
    const bad = answer.includes(kw);
    return `<tr><td>${bad ? '❌' : '✅'}</td><td>禁止包含</td><td>${escapeHtml(kw)}</td></tr>`;
  }).join('');

  return `
    <div class="answer-tab">
      <div class="answer-block">
        <div class="panel-title">AI 最终答案</div>
        <pre class="answer-pre">${escapeHtml(answer || '（无回答）')}</pre>
      </div>
      <div class="answer-block">
        <div class="panel-title">Golden Test 参考答案</div>
        ${caseInfo ? `<pre class="answer-pre">${escapeHtml(caseInfo.expected_answer || '（无标准答案）')}</pre>`
                   : `<div class="empty">未找到对应测试用例</div>`}
      </div>
      ${must.length || mustNot.length ? `
      <div class="answer-block">
        <div class="panel-title">关键词检查</div>
        <table>
          <tr><th>结果</th><th>类型</th><th>关键词</th></tr>
          ${kwRows}${notRows}
        </table>
      </div>` : ''}
      ${caseInfo ? `
      <div class="answer-block">
        <div class="diagnosis-head">
          <div class="panel-title">AI 诊断</div>
          <button class="tab" id="diagBtn" onclick="triggerDiagnosis('${escapeHtml(t.trace_id)}', '${escapeHtml(caseInfo.case_id)}')">🔍 诊断</button>
        </div>
        <div id="diagResult" class="diag-result">点击“诊断”后，AI 会分析这条 Trace 并解释为什么没通过。</div>
      </div>` : ''}
    </div>
  `;
}

async function triggerDiagnosis(traceId, caseId) {
  if (diagnosisBusy) return;
  diagnosisBusy = true;
  const btn = document.getElementById('diagBtn');
  const box = document.getElementById('diagResult');
  if (btn) btn.disabled = true;
  if (box) box.textContent = '正在生成运行分析报告...';
  try {
    const runsRes = await fetch('/api/runs');
    const runs = await runsRes.json();
    const run = runs.find(r => (r.results || []).some(x => x.trace_id === traceId && x.case_id === caseId));
    if (!run) {
      box.textContent = '没有找到对应 Run，无法诊断。';
      return;
    }
    let res = await fetch(`/api/runs/${run.run_id}/report/${caseId}`);
    let d;
    if (res.status === 200) {
      d = await res.json();
    } else {
      res = await fetch(`/api/runs/${run.run_id}/report/${caseId}`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'}
      });
      d = await res.json();
    }
    if (d.error) {
      box.textContent = d.error;
    } else if (d.report_html) {
      box.innerHTML=sanitizeHtml(`<div class="report-html">${d.report_html}</div>`, false);
    } else if (d.report) {
      box.innerHTML=sanitizeHtml(`<pre class="report-pre">${escapeHtml(d.report)}</pre>`, true);
    } else {
      box.textContent = '生成报告失败：' + JSON.stringify(d);
    }
  } catch (e) {
    if (box) box.textContent = '报告请求失败：' + e;
  } finally {
    diagnosisBusy = false;
    if (btn) btn.disabled = false;
  }
}


function renderMetricsTab(m) {
  if (!m) return '<div class="empty">暂无指标</div>';
  const phase = Object.entries(m.phase_latency_ms || {}).map(([k, v]) =>
    `<tr><td>${escapeHtml(k)}</td><td>${formatMs(v)}</td></tr>`).join('');
  const tools = Object.entries(m.tool_metrics || {}).map(([name, v]) => `
    <tr>
      <td>${escapeHtml(name)}</td>
      <td>${v.count}</td>
      <td>${v.success}</td>
      <td>${v.not_found}</td>
      <td>${v.intercepted}</td>
      <td>${v.error}</td>
      <td>${formatMs(v.total_ms)}</td>
    </tr>`).join('');
  return `
    <div class="metrics-grid">
      <div class="metric-card"><div class="metric-value">${m.span_count ?? 0}</div><div class="metric-label">总节点</div></div>
      <div class="metric-card"><div class="metric-value">${m.tool_count ?? 0}</div><div class="metric-label">工具调用</div></div>
      <div class="metric-card"><div class="metric-value">${formatMs(m.total_tool_latency_ms)}</div><div class="metric-label">工具总耗时</div></div>
      <div class="metric-card"><div class="metric-value">${formatMs(m.avg_tool_latency_ms)}</div><div class="metric-label">工具平均耗时</div></div>
      <div class="metric-card"><div class="metric-value">${formatMs(m.unattributed_ms)}</div><div class="metric-label">未归属耗时</div></div>
      <div class="metric-card"><div class="metric-value">${m.events_count ?? 0}</div><div class="metric-label">原始事件</div></div>
    </div>
    <h3>阶段耗时</h3>
    <table><tr><th>阶段</th><th>耗时</th></tr>${phase || '<tr><td colspan="2">暂无</td></tr>'}</table>
    <h3>工具级统计</h3>
    <table>
      <tr><th>工具</th><th>调用</th><th>成功</th><th>未找到</th><th>拦截</th><th>错误</th><th>总耗时</th></tr>
      ${tools || '<tr><td colspan="7">无工具</td></tr>'}
    </table>
  `;
}

function renderSpan(span, depth) {
  const dur = durationText(span.start_time, span.end_time);
  const durMs = span.start_time && span.end_time ? (new Date(span.end_time) - new Date(span.start_time)) : 0;
  const maxDur = currentMetrics && currentMetrics.total_tool_latency_ms || 1;
  const barWidth = maxDur > 0 && durMs > 0 ? Math.max(2, Math.round(durMs / maxDur * 100)) : 0;
  const icon = iconFor(span.span_type);
  const status = span.status || 'success';
  let body = '';
  if (span.span_type === 'tool') {
    body = toolBody(span);
  } else if (span.span_type === 'llm') {
    body = llmBody(span);
  } else {
    body = genericBody(span);
  }
  const children = (span.children || []).map(c => renderSpan(c, depth + 1)).join('');
  return `
    <div class="span-card" data-span-id="${span.span_id}">
      <div class="span-header" onclick="toggleBody(this)">
        <div class="span-icon ${span.span_type}">${icon}</div>
        <div class="span-name">${escapeHtml(span.name)}</div>
        <div class="waterfall"><div class="bar" style="width:${Math.min(100, barWidth)}%"></div></div>
        <div class="span-detail">
          ${dur ? `<span>${dur}</span>` : ''}
          <span class="status-dot status-${status}"></span>
        </div>
      </div>
      <div class="span-body" style="display:none">
        ${body}
        ${children ? `<div class="tree">${children}</div>` : ''}
      </div>
    </div>
  `;
}

function toolBody(span) {
  const args = span.tool_args ? JSON.stringify(span.tool_args, null, 2) : '';
  return `
    <div class="label">工具参数</div>
    <pre>${escapeHtml(args || '{}')}</pre>
    <div class="label">返回 ${span.result_length != null ? span.result_length + ' 字' : ''} ${span.meltdown_trigger ? '· 熔断触发' : ''} ${span.result_full ? `<button class="tab" onclick="openFullToolResult('${escapeHtml(span.span_id)}')">查看完整返回</button>` : ''}</div>
    <pre>${escapeHtml((span.result_preview || '').slice(0, 1000))}</pre>
  `;
}

function findSpanByIdForDrawer(spanId) {
  let found = null;
  (function walk(s) {
    if (found) return;
    if (s.span_id === spanId) { found = s; return; }
    (s.children || []).forEach(walk);
  })(currentTrace.root_span || {});
  return found;
}

function openFullToolResult(spanId) {
  const span = findSpanByIdForDrawer(spanId);
  if (!span || !span.result_full) return;
  let drawer = document.getElementById('toolResultDrawer');
  if (!drawer) {
    drawer = document.createElement('div');
    drawer.id = 'toolResultDrawer';
    drawer.className = 'tool-drawer';
    document.body.appendChild(drawer);
  }
  drawer.innerHTML=sanitizeHtml(`
    <div class="tool-drawer-head"><strong>${escapeHtml(span.name || '工具')} · 完整返回</strong><button class="close-detail" onclick="closeToolDrawer()">×</button></div>
    <pre class="tool-drawer-body">${escapeHtml(span.result_full || '')}</pre>
  `, true);
  drawer.classList.add('open');
}

function closeToolDrawer() {
  const drawer = document.getElementById('toolResultDrawer');
  if (drawer) drawer.classList.remove('open');
}

function llmBody(span) {
  const calls = (span.tool_calls || []).map(c => `${c.name}(${escapeHtml(JSON.stringify(c.args || {}))})`).join(', ');
  const plan = (span.output && span.output.execution_plan) || '';
  return `
    <div class="label">模型 ${span.model || '未记录'}</div>
    ${calls ? `<div class="label">工具调用</div><pre>${escapeHtml(calls)}</pre>` : ''}
    ${plan ? `<div class="label">执行报告</div><pre>${escapeHtml(plan)}</pre>` : ''}
  `;
}

function genericBody(span) {
  const modelLine = span.model ? `<div class="label">模型 ${escapeHtml(span.model)}</div>` : '';
  return modelLine + `<pre>${escapeHtml(JSON.stringify(span.output || {}, null, 2))}</pre>`;
}

function renderEvents(t) {
  if (!t.trace_events || !t.trace_events.length) {
    return '<div class="empty">这条 Trace 没有保存原始事件。请使用新版导出器重新导出。</div>';
  }
  return `<div class="event-list">${t.trace_events.map(ev => {
    const d = ev.data || {};
    return `<div class="event-row"><span class="event-type">${escapeHtml(ev.event)}</span><span>${escapeHtml(JSON.stringify(d).slice(0, 300))}</span></div>`;
  }).join('')}</div>`;
}

function setTab(tab) {
  currentTab = tab;
  if (currentTrace) renderTrace(currentTrace);
}

function toggleBody(header) {
  const body = header.nextElementSibling;
  body.style.display = body.style.display === 'none' ? '' : 'none';
}

function bindToggle() {
  document.querySelectorAll('.span-header').forEach(el => {
    const body = el.nextElementSibling;
    if (body && body.style.display === '') {
      body.style.display = 'none';
    }
  });
}

function iconFor(type) {
  switch(type) {
    case 'tool': return '🔧';
    case 'llm': return '✨';
    case 'router': return '🧭';
    case 'answer': return '💬';
    case 'assess': return '⚖️';
    case 'rewrite': return '✏️';
    default: return '📄';
  }
}

function durationText(start, end) {
  if (!start || !end) return '';
  const s = new Date(start).getTime();
  const e = new Date(end).getTime();
  return formatMs(e - s);
}

function formatMs(ms) {
  if (ms == null || isNaN(ms)) return '';
  if (ms < 1000) return ms + 'ms';
  return (ms / 1000).toFixed(1) + 's';
}

function escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

async function loadSample() {
  if (!traces.length) await loadTestCases();
refreshList();
  if (traces.length) selectTrace(traces[0].trace_id);
}

loadTestCases();
refreshList();
