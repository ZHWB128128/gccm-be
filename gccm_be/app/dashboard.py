"""Static dashboard HTML (zero external dependencies).

The page is DATA-DRIVEN: on load it fetches /introspection and renders all model
switches, weights, modes, physics models and diagnostics dynamically. That is
what makes "the page changes when the model changes" true without editing HTML —
you update gccm_be/app/introspection.py and this page reflects it automatically.

Charts are drawn on a plain <canvas> with hand-rolled line plotting, so there is
no CDN / npm dependency.
"""
from __future__ import annotations

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GCCM-BE 控制台</title>
<style>
  :root { --bg:#0f172a; --panel:#1e293b; --ink:#e2e8f0; --muted:#94a3b8;
          --accent:#38bdf8; --ok:#34d399; --warn:#fbbf24; --bad:#f87171;
          --line:#334155; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:system-ui,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
         background:var(--bg); color:var(--ink); }
  header { padding:14px 20px; background:var(--panel); border-bottom:1px solid var(--line);
           display:flex; align-items:center; gap:14px; }
  header h1 { font-size:18px; margin:0; }
  header .ver { color:var(--muted); font-size:12px; }
  header .health { margin-left:auto; font-size:13px; }
  .grid { display:grid; grid-template-columns:340px 1fr; gap:16px; padding:16px; }
  .col { display:flex; flex-direction:column; gap:16px; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px; }
  .card h2 { font-size:14px; margin:0 0 10px; color:var(--accent); letter-spacing:.5px; }
  .row { display:flex; align-items:center; justify-content:space-between; gap:10px;
         padding:6px 0; border-bottom:1px dashed var(--line); }
  .row:last-child { border-bottom:none; }
  .row label { font-size:13px; }
  .row .help { color:var(--muted); font-size:11px; display:block; margin-top:2px; }
  input[type=number] { width:82px; background:#0b1220; color:var(--ink);
         border:1px solid var(--line); border-radius:6px; padding:4px 6px; }
  input[type=range] { width:120px; }
  select, button { background:#0b1220; color:var(--ink); border:1px solid var(--line);
         border-radius:6px; padding:6px 10px; font-size:13px; }
  button.primary { background:var(--accent); color:#08131f; border:none; font-weight:600; cursor:pointer; }
  button.primary:hover { filter:brightness(1.1); }
  button.ghost { cursor:pointer; }
  .switch { position:relative; display:inline-block; width:42px; height:22px; }
  .switch input { display:none; }
  .slider { position:absolute; inset:0; background:#475569; border-radius:22px; transition:.2s; cursor:pointer; }
  .slider:before { content:""; position:absolute; width:16px; height:16px; left:3px; top:3px;
         background:#fff; border-radius:50%; transition:.2s; }
  .switch input:checked + .slider { background:var(--ok); }
  .switch input:checked + .slider:before { transform:translateX(20px); }
  canvas { width:100%; height:220px; background:#0b1220; border-radius:8px; border:1px solid var(--line); }
  .kpi { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; }
  .kpi div { background:#0b1220; border:1px solid var(--line); border-radius:8px; padding:8px; }
  .kpi .k { font-size:11px; color:var(--muted); }
  .kpi .v { font-size:18px; font-weight:600; }
  .badge { padding:2px 8px; border-radius:12px; font-size:12px; }
  .badge.ok { background:rgba(52,211,153,.15); color:var(--ok); }
  .badge.bad { background:rgba(248,113,113,.15); color:var(--bad); }
  .badge.warn { background:rgba(251,191,36,.15); color:var(--warn); }
  pre { background:#0b1220; border:1px solid var(--line); border-radius:8px; padding:10px;
        font-size:12px; max-height:220px; overflow:auto; margin:0; }
  .muted { color:var(--muted); font-size:12px; }
  .inline { display:flex; align-items:center; gap:8px; }
  ul.models { margin:0; padding-left:16px; font-size:13px; }
  ul.models li { margin:3px 0; }
  /* 页签导航 */
  .tabs { display:flex; gap:6px; padding:10px 20px 0; background:var(--panel);
          border-bottom:1px solid var(--line); }
  .tabs button { background:transparent; border:none; border-bottom:2px solid transparent;
                 color:var(--muted); padding:6px 14px; cursor:pointer; font-size:14px; border-radius:0; }
  .tabs button.active { color:var(--accent); border-bottom-color:var(--accent); }
  .tabpanel { display:none; }
  .tabpanel.active { display:block; }
  /* 架构图视图 */
  .arch-card { background:var(--panel); border:1px solid var(--line); border-radius:10px;
               padding:14px; margin:16px; }
  .arch-card h2 { font-size:14px; margin:0 0 10px; color:var(--accent); letter-spacing:.5px; }
  .layers { display:flex; flex-direction:column; gap:8px; }
  .layer { background:#0b1220; border:1px solid var(--line); border-left:3px solid var(--accent);
           border-radius:6px; padding:8px 10px; }
  .layer .lname { font-weight:600; color:var(--accent); }
  .layer .lmod { color:var(--muted); font-size:11px; font-family:monospace; }
  .layer .lhelp { font-size:13px; margin-top:2px; color:var(--ink); }
  .pstep { display:flex; gap:10px; align-items:flex-start; padding:6px 0;
           border-bottom:1px dashed var(--line); }
  .pstep:last-child { border-bottom:none; }
  .pstep .n { background:var(--accent); color:#08131f; border-radius:50%; min-width:22px; height:22px;
              display:inline-flex; align-items:center; justify-content:center;
              font-size:11px; font-weight:700; flex:none; }
  .pstep .pm { font-family:monospace; font-size:11px; color:var(--muted); }
  .pstep .ph { font-size:13px; }
  .pstep.fallback .n { background:var(--bad); }
  .theory table { width:100%; border-collapse:collapse; font-size:12px; }
  .theory th, .theory td { border:1px solid var(--line); padding:5px 8px; text-align:left; vertical-align:top; }
  .theory th { background:#0b1220; color:var(--accent); }
</style>
</head>
<body>
<header>
  <h1>GCCM-BE 控制台</h1>
  <span class="ver" id="ver">v?</span>
  <span class="health" id="health">连接中…</span>
</header>

<nav class="tabs">
  <button id="tab-console" class="active">控制台</button>
  <button id="tab-arch">架构图</button>
</nav>

<div class="grid tabpanel active" id="panel-console">
  <!-- LEFT: operation & management -->
  <div class="col">
    <div class="card">
      <h2>黎曼几何开关</h2>
      <div id="riemannian"></div>
    </div>
    <div class="card">
      <h2>目标权重</h2>
      <div id="weights"></div>
    </div>
    <div class="card">
      <h2>控制器开关</h2>
      <div id="toggles"></div>
    </div>
    <div class="card">
      <div class="inline" style="justify-content:space-between">
        <button class="primary" id="apply">应用配置</button>
        <button class="ghost" id="reload">重新载入</button>
      </div>
      <div class="muted" id="applymsg" style="margin-top:8px"></div>
    </div>
    <div class="card">
      <h2>物理模型</h2>
      <ul class="models" id="physics"></ul>
    </div>
  </div>

  <!-- RIGHT: monitoring -->
  <div class="col">
    <div class="card">
      <h2>运行控制</h2>
      <div class="inline" style="flex-wrap:wrap; gap:12px">
        <label class="inline">模式
          <select id="mode"></select>
        </label>
        <label class="inline">初始温度(°C)
          <input type="number" id="t0" value="28" step="0.5">
        </label>
        <label class="inline">步数
          <input type="number" id="steps" value="48" step="1">
        </label>
        <button class="primary" id="run">运行闭环仿真</button>
        <button class="ghost" id="step1">单步决策</button>
      </div>
      <div class="kpi" style="margin-top:12px" id="kpi"></div>
    </div>
    <div class="card">
      <h2>温度轨迹</h2>
      <canvas id="tempChart"></canvas>
    </div>
    <div class="card">
      <h2>控制量 / 电价</h2>
      <canvas id="ctrlChart"></canvas>
    </div>
    <div class="card">
      <h2>诊断监控</h2>
      <div id="diagBadges" class="inline" style="flex-wrap:wrap; gap:8px; margin-bottom:8px"></div>
      <pre id="diag">—</pre>
    </div>
  </div>
</div>

<!-- 架构图页签（数据来自 /introspection 的 architecture 块，随模型自动同步） -->
<div class="tabpanel" id="panel-arch">
  <div class="arch-card">
    <h2>分层架构</h2>
    <div class="layers" id="arch-layers"></div>
  </div>
  <div class="arch-card">
    <h2>一次 optimize 决策管线</h2>
    <div id="arch-pipeline"></div>
  </div>
  <div class="arch-card">
    <h2>安全降级点（任何一步失败都不失控）</h2>
    <div id="arch-fallbacks"></div>
  </div>
  <div class="arch-card">
    <h2>三个理论层的定位（诚实标注：均为正则子/自适应，不是玄学）</h2>
    <div id="arch-theory"></div>
  </div>
</div>

<script>
const $ = s => document.querySelector(s);
let SPEC = null;

async function jget(u){ const r=await fetch(u); if(!r.ok) throw new Error(await r.text()); return r.json(); }
async function jpost(u,b){ const r=await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});
  if(!r.ok) throw new Error(await r.text()); return r.json(); }

function toggleRow(e){
  const checked = e.value ? 'checked' : '';
  const strength = e.strength_field ?
    `<input type="number" step="0.1" id="str_${e.strength_field}" value="${e.strength_value ?? ''}" title="${e.strength_field}">` : '';
  return `<div class="row"><div><label>${e.label}</label><span class="help">${e.help}</span></div>
    <div class="inline">${strength}
    <label class="switch"><input type="checkbox" id="fld_${e.field}" ${checked}><span class="slider"></span></label></div></div>`;
}
function floatRow(e){
  return `<div class="row"><div><label>${e.label}</label><span class="help">${e.help}</span></div>
    <input type="number" step="0.1" id="fld_${e.field}" value="${e.value ?? ''}"></div>`;
}

function render(){
  $('#ver').textContent = 'v'+SPEC.version;
  $('#riemannian').innerHTML = SPEC.riemannian_switches.map(e =>
      e.type==='bool' ? toggleRow(e) : floatRow(e)).join('');
  $('#weights').innerHTML = SPEC.weights.map(floatRow).join('');
  $('#toggles').innerHTML = SPEC.controller_toggles.map(toggleRow).join('');
  $('#physics').innerHTML = SPEC.physics_models.map(m => `<li><b>${m.label}</b> — ${m.help}</li>`).join('');
  $('#mode').innerHTML = SPEC.modes.map(m => `<option value="${m.id}">${m.label}</option>`).join('');
}

// ---- 页签切换（控制台 / 架构图） ----
function switchTab(tab){
  document.querySelectorAll('.tabpanel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tabs button').forEach(b => b.classList.remove('active'));
  const panelId = tab === 'arch' ? 'panel-arch' : 'panel-console';
  const btnId = tab === 'arch' ? 'tab-arch' : 'tab-console';
  document.getElementById(panelId).classList.add('active');
  document.getElementById(btnId).classList.add('active');
  if(tab === 'arch') renderArch();
}

// ---- 架构图渲染（数据驱动：全部来自 SPEC.architecture，随模型自动同步） ----
function renderArch(){
  const spec = SPEC.architecture || {};
  const esc = s => String(s ?? '').replace(/[<>&]/g, c => ({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]));
  $('#arch-layers').innerHTML = (spec.layers || []).map(l =>
    `<div class="layer"><span class="lname">${esc(l.label)}</span>
     <span class="lmod">${esc(l.module || l.id || '')}</span>
     <div class="lhelp">${esc(l.help || l.id || '')}</div></div>`).join('');
  $('#arch-pipeline').innerHTML = (spec.pipeline || []).map(p =>
    `<div class="pstep"><span class="n">${p.step}</span>
     <div><b>${esc(p.label)}</b> <span class="pm">${esc(p.method || '')}</span>
     <div class="ph">${esc(p.help || '')}</div></div></div>`).join('');
  $('#arch-fallbacks').innerHTML = (spec.fallback_points || []).map(f =>
    `<div class="pstep fallback"><span class="n">!</span>
     <div><b>${esc(f.point || f.label || '')}</b>
     <div class="ph">${esc(f.path || f.help || '')}</div></div></div>`).join('');
  $('#arch-theory').innerHTML = `<table><tr><th>理论层</th><th>数学本质</th><th>在 MPC 中的作用</th><th>何时真正有用</th><th>诚实 KPI</th></tr>
    ${(spec.theory_layers || []).map(t => `<tr><td><b>${esc(t.label || t.id)}</b></td><td>${esc(t.math || '')}</td><td>${esc(t.role || '')}</td><td>${esc(t.when || '')}</td><td>${esc(t.kpi || '')}</td></tr>`).join('')}</table>`;
}

// ---- 页签切换（控制台 / 架构图） ----
function switchTab(name){
  const map = {'console':'panel-console','arch':'panel-arch'};
  document.querySelectorAll('.tabpanel').forEach(p => p.classList.remove('active'));
  document.getElementById(map[name]).classList.add('active');
  document.querySelectorAll('.tabs button').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-'+name).classList.add('active');
  if(name==='arch') renderArch();
}
document.getElementById('tab-console').onclick = () => switchTab('console');
document.getElementById('tab-arch').onclick = () => switchTab('arch');

// ---- 架构图渲染（数据驱动，来自 SPEC.architecture） ----
function renderArch(){
  if(!SPEC || !SPEC.architecture) return;
  const a = SPEC.architecture;
  // 分层
  $('#arch-layers').innerHTML = (a.layers||[]).map(l =>
    `<div class="layer"><div><span class="lname">${l.label}</span>
     <span class="lmod">${l.module||l.id||''}</span></div>
     <div class="lhelp">${l.help||''}</div></div>`).join('');
  // 一次决策管线
  $('#arch-pipeline').innerHTML = (a.pipeline||[]).map(s =>
    `<div class="pstep"><span class="n">${s.step||'•'}</span>
     <div><b>${s.label||''}</b> <span class="pm">${s.method||''}</span>
     <div class="ph">${s.help||''}</div></div></div>`).join('');
  // 安全降级点
  $('#arch-fallbacks').innerHTML = (a.fallback_points||[]).map(f =>
    `<div class="pstep fallback"><span class="n">!</span>
     <div><b>${f.point||f.label||''}</b> — <span class="ph">${f.path||f.help||''}</span></div></div>`).join('');
  // 理论层定位
  $('#arch-theory').innerHTML = `<table><tr><th>理论层</th><th>数学本质</th><th>在 MPC 中的作用</th><th>何时真正有用</th><th>诚实 KPI</th></tr>
    ${(a.theory_layers||[]).map(t => `<tr><td><b>${t.label||t.id||''}</b></td><td>${t.math||''}</td>
      <td>${t.role||''}</td><td>${t.when||''}</td><td>${t.kpi||''}</td></tr>`).join('')}</table>`;
}

function collectConfig(){
  const cfg = {};
  const put = (fld, isBool) => {
    const el = $('#fld_'+fld); if(!el) return;
    cfg[fld] = isBool ? el.checked : parseFloat(el.value);
  };
  SPEC.riemannian_switches.forEach(e => { put(e.field, e.type==='bool');
    if(e.strength_field){ const s=$('#str_'+e.strength_field); if(s&&s.value!=='') cfg[e.strength_field]=parseFloat(s.value);} });
  SPEC.weights.forEach(e => put(e.field, false));
  SPEC.controller_toggles.forEach(e => { put(e.field, true);
    if(e.strength_field){ const s=$('#str_'+e.strength_field); if(s&&s.value!=='') cfg[e.strength_field]=parseFloat(s.value);} });
  // drop NaNs
  Object.keys(cfg).forEach(k => { if(typeof cfg[k]==='number' && isNaN(cfg[k])) delete cfg[k]; });
  return cfg;
}

// ---- minimal canvas line chart ----
function drawChart(canvas, series, opts){
  const dpr = window.devicePixelRatio||1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w*dpr; canvas.height = h*dpr;
  const ctx = canvas.getContext('2d'); ctx.scale(dpr,dpr);
  ctx.clearRect(0,0,w,h);
  const pad = {l:38,r:10,t:10,b:20};
  const xs = series[0].data.map((_,i)=>i);
  let ymin=Infinity, ymax=-Infinity;
  series.forEach(s=>s.data.forEach(v=>{ if(v<ymin)ymin=v; if(v>ymax)ymax=v; }));
  if(!isFinite(ymin)){ymin=0;ymax=1;}
  if(ymin===ymax){ymin-=1;ymax+=1;}
  const X = i => pad.l + (w-pad.l-pad.r)*(i/Math.max(1,xs.length-1));
  const Y = v => pad.t + (h-pad.t-pad.b)*(1-(v-ymin)/(ymax-ymin));
  // axes
  ctx.strokeStyle='#334155'; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(pad.l,pad.t); ctx.lineTo(pad.l,h-pad.b); ctx.lineTo(w-pad.r,h-pad.b); ctx.stroke();
  ctx.fillStyle='#94a3b8'; ctx.font='10px sans-serif';
  ctx.fillText(ymax.toFixed(1), 2, pad.t+8); ctx.fillText(ymin.toFixed(1), 2, h-pad.b);
  // comfort band
  if(opts && opts.band){
    ctx.fillStyle='rgba(52,211,153,.08)';
    const yTop=Y(opts.band[1]), yBot=Y(opts.band[0]);
    ctx.fillRect(pad.l,yTop,w-pad.l-pad.r,yBot-yTop);
  }
  series.forEach(s=>{
    ctx.strokeStyle=s.color; ctx.lineWidth=1.8; ctx.beginPath();
    s.data.forEach((v,i)=>{ const x=X(i),y=Y(v); i?ctx.lineTo(x,y):ctx.moveTo(x,y); });
    ctx.stroke();
  });
  // legend
  let lx=pad.l+6;
  series.forEach(s=>{ ctx.fillStyle=s.color; ctx.fillRect(lx,pad.t,10,3);
    ctx.fillStyle='#e2e8f0'; ctx.fillText(s.name,lx+14,pad.t+6); lx+=14+ctx.measureText(s.name).width+16; });
}

function kpi(items){
  $('#kpi').innerHTML = items.map(i=>`<div><div class="k">${i.k}</div><div class="v">${i.v}</div></div>`).join('');
}

function showDiag(d){
  $('#diag').textContent = JSON.stringify(d, null, 2);
  const b = [];
  b.push(`<span class="badge ${d.solver_success?'ok':'bad'}">求解${d.solver_success?'成功':'失败'}</span>`);
  b.push(`<span class="badge ${d.undecidable?'warn':'ok'}">${d.undecidable?'不可判定':'可判定'}</span>`);
  b.push(`<span class="badge ok">模式 ${d.mode}</span>`);
  const trg = (d.diagnosis&&d.diagnosis.triggers)||[];
  trg.forEach(t=>b.push(`<span class="badge warn">${t}</span>`));
  $('#diagBadges').innerHTML = b.join('');
}

async function boot(){
  try{
    const hp = await jget('/health'); $('#health').innerHTML = '<span class="badge ok">在线</span>';
  }catch(e){ $('#health').innerHTML = '<span class="badge bad">离线</span>'; }
  SPEC = await jget('/introspection');
  render();
}

$('#apply').onclick = async () => {
  try{
    const cfg = collectConfig();
    const res = await jpost('/config', cfg);
    $('#applymsg').textContent = '已应用: ' + Object.keys(res.applied).join(', ');
    SPEC = await jget('/introspection'); render();
  }catch(e){ $('#applymsg').textContent = '错误: '+e.message; }
};
$('#reload').onclick = boot;

$('#step1').onclick = async () => {
  const t0 = parseFloat($('#t0').value);
  const labels = SPEC.state_labels;
  const state = labels.map(()=>t0);
  const res = await jpost('/control', {state, labels, forced_mode:$('#mode').value, time_h:8.0});
  showDiag(res);
  kpi([{k:'控制量[0]', v:Object.values(res.control)[0].toFixed(3)},
       {k:'模式', v:res.mode},{k:'置信', v:(res.confidence*100).toFixed(0)+'%'},
       {k:'总代价', v:res.total_cost.toFixed(2)}]);
};

$('#run').onclick = async () => {
  const t0 = parseFloat($('#t0').value);
  const steps = parseInt($('#steps').value);
  $('#run').textContent='仿真中…'; $('#run').disabled=true;
  try{
    const res = await jpost('/simulate', {t0, steps, forced_mode:$('#mode').value});
    const temp = res.temps, ctrl = res.controls, price = res.prices;
    // 多区模型：每区一条室温曲线；单区模型保持原有单一曲线
    const zoneColors = ['#38bdf8','#a78bfa','#f472b6','#4ade80'];
    const tempSeries = (res.zones && res.zones.length > 1)
      ? res.zones.map((z,i)=>({name:'室温·'+z.label, color:zoneColors[i%zoneColors.length], data:z.temps}))
      : [{name:'室温',color:'#38bdf8',data:temp}];
    drawChart($('#tempChart'), tempSeries, {band:[res.comfort_min, res.comfort_max]});
    // 多设备模型：每台设备一条控制曲线
    const unitSeries = (res.unit_controls && res.unit_controls.length > 1)
      ? res.unit_controls.map((u,i)=>({name:u.label,color:zoneColors[i%zoneColors.length],data:u.data}))
      : [{name:'控制量',color:'#34d399',data:ctrl}];
    unitSeries.push({name:'电价',color:'#fbbf24',data:price});
    drawChart($('#ctrlChart'), unitSeries);
    kpi([{k:'峰值温度', v:Math.max(...temp).toFixed(2)},
         {k:'均温', v:(temp.reduce((a,b)=>a+b,0)/temp.length).toFixed(2)},
         {k:'越界步数', v:res.violations},
         {k:'总电费', v:res.total_cost.toFixed(2)}]);
    showDiag(res.last_decision);
  }catch(e){ $('#applymsg').textContent='仿真错误: '+e.message; }
  finally{ $('#run').textContent='运行闭环仿真'; $('#run').disabled=false; }
};

boot();
</script>
</body>
</html>
"""
