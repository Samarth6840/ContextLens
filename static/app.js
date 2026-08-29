/* ADSCENE — frontend app logic (vanilla JS, hash router) */

const $ = (sel, root) => (root || document).querySelector(sel);
const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));

const el = {
  landing: $('#view-landing'),
  analyse: $('#view-analyse'),
  pipeline: $('#view-pipeline'),
  outreach: $('#view-outreach'),
};

function escapeHtml(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

function fmtDuration(sec) {
  if (!sec || sec <= 0) return '—';
  sec = Math.round(sec);
  if (sec < 60) return `${sec} SEC`;
  return `${(sec / 60).toFixed(1)} MIN`;
}

function fmtTime(sec) {
  sec = Math.round(sec || 0);
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

function fmtScene(n) {
  return 'SCENE ' + String(n).padStart(3, '0');
}

function chip(text, extra) {
  return `<span class="chip${extra ? ' ' + extra : ''}">${escapeHtml(text)}</span>`;
}

async function api(path, opts) {
  const res = await fetch(path, opts);
  let data = null;
  try { data = await res.json(); } catch (e) { /* empty */ }
  if (!res.ok) throw new Error((data && data.error) || `Request failed (${res.status})`);
  return data;
}

/* ── Router ───────────────────────────────────────────────── */

function parseRoute() {
  const hash = location.hash || '#/';
  const [path, query = ''] = hash.split('?');
  const parts = path.replace(/^#\/?/, '').split('/').filter(Boolean);
  const [page, id] = parts;
  const params = new URLSearchParams(query);
  return { page: page || 'landing', id: id ? decodeURIComponent(id) : null, params };
}

function showView(name) {
  Object.values(el).forEach((v) => v.classList.remove('is-active'));
  if (el[name]) el[name].classList.add('is-active');
  window.scrollTo({ top: 0 });
  $$('.nav-link').forEach((a) => {
    a.classList.toggle('is-active', a.getAttribute('href') === `#/${name}`);
  });
}

function router() {
  const { page, id, params } = parseRoute();
  if (page === 'landing' || page === '') showView('landing');
  else if (page === 'analyse') showView('analyse');
  else if (page === 'pipeline') { showView('pipeline'); renderPipeline(id, params); }
  else if (page === 'outreach') { showView('outreach'); renderOutreach(id, params); }
  else { location.hash = '#/'; }
}

window.addEventListener('hashchange', router);

/* ── Analyse page ─────────────────────────────────────────── */

const dropzone = $('#dropzone');
const fileInput = $('#video-file');
let selectedFile = null;

function updateDropzone() {
  const label = $('#file-status');
  if (selectedFile) {
    dropzone.classList.add('has-file');
    dropzone.querySelector('span').textContent = selectedFile.name;
    label.textContent = `${(selectedFile.size / (1024 * 1024)).toFixed(1)} MB — READY`;
    label.classList.add('has-file');
  } else {
    dropzone.classList.remove('has-file');
    dropzone.querySelector('span').textContent = 'DRAG & DROP OR CLICK TO SELECT';
    label.textContent = '';
    label.classList.remove('has-file');
  }
}

dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
});
fileInput.addEventListener('change', () => { selectedFile = fileInput.files[0] || null; updateDropzone(); });
['dragover', 'dragenter'].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add('is-dragover'); })
);
['dragleave', 'drop'].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove('is-dragover'); })
);
dropzone.addEventListener('drop', (e) => {
  selectedFile = e.dataTransfer.files[0] || null;
  updateDropzone();
});

const analyseStatus = $('#analyse-status');

function setStatus(text, kind) {
  analyseStatus.hidden = false;
  analyseStatus.className = 'status-line' + (kind ? ' ' + kind : '');
  analyseStatus.innerHTML = text;
}

$('#analyse-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const btn = $('#btn-extract');
  if (!selectedFile) { setStatus('<span class="dot">▮</span> SELECT A VIDEO FILE FIRST'); return; }
  if (selectedFile.size > 200 * 1024 * 1024) {
    setStatus('FILE EXCEEDS 200 MB LIMIT', 'error');
    return;
  }

  const fd = new FormData();
  fd.append('video', selectedFile);
  fd.append('title', $('#video-title').value.trim() || selectedFile.name.replace(/\.[^.]+$/, ''));
  fd.append('creator', $('#video-creator').value.trim() || 'UNKNOWN');
  fd.append('duration', $('#video-duration').value.trim());

  btn.disabled = true;
  btn.textContent = 'QUEUING…';
  setStatus('<span class="dot">▮</span> UPLOADING VIDEO…');

  try {
    const { job_id } = await api('/api/analyse', { method: 'POST', body: fd });
    selectedFile = null;
    fileInput.value = '';
    updateDropzone();
    pollJob(job_id, btn);
  } catch (err) {
    btn.disabled = false;
    btn.textContent = 'START EXTRACTION';
    setStatus(`UPLOAD FAILED — ${escapeHtml(err.message)}`, 'error');
  }
});

function pollJob(jobId, btn) {
  let attempts = 0;
  const tick = async () => {
    attempts += 1;
    let job;
    try { job = await api(`/api/analyse/${encodeURIComponent(jobId)}`); }
    catch (err) {
      setStatus(`STATUS CHECK FAILED — ${escapeHtml(err.message)}`, 'error');
      btn.disabled = false; btn.textContent = 'START EXTRACTION';
      return;
    }
    if (job.status === 'done') {
      btn.disabled = false;
      btn.textContent = 'START EXTRACTION';
      location.hash = `#/pipeline/${encodeURIComponent(jobId)}`;
      return;
    }
    if (job.status === 'error') {
      btn.disabled = false;
      btn.textContent = 'START EXTRACTION';
      setStatus(`EXTRACTION FAILED — ${escapeHtml(job.error || 'UNKNOWN ERROR')}`, 'error');
      return;
    }
    setStatus(`<span class="dot blink">▮</span> ${escapeHtml(job.stage || 'PROCESSING')}…`);
    if (attempts < 600) setTimeout(tick, 1500);
  };
  tick();
}

/* ── Pipeline page ────────────────────────────────────────── */

let pipeState = { id: null, data: null, tab: 'scenes' };

async function renderPipeline(id, params) {
  const root = $('#pipeline-root');
  if (!id) return renderJobIndex(root, params);

  root.innerHTML = loaderHtml('LOADING INTELLIGENCE');
  pipeState.id = id;
  pipeState.tab = params.get('tab') || 'scenes';

  try {
    const data = await api(`/api/pipeline/${encodeURIComponent(id)}`);
    pipeState.data = data;
    renderDashboard(root, data, pipeState.tab);
  } catch (err) {
    root.innerHTML = errorHtml(err.message);
  }
}

function loaderHtml(stage) {
  return `<div class="loader"><span class="stage">${escapeHtml(stage)}…</span></div>`;
}

function errorHtml(msg) {
  return `<div class="error-box"><div class="error-title">ERROR</div>${escapeHtml(msg)}</div>`;
}

async function renderJobIndex(root) {
  root.innerHTML = loaderHtml('LOADING JOBS');
  try {
    const { jobs } = await api('/api/jobs');
    if (!jobs.length) {
      root.innerHTML =
        `<h2 class="page-title" style="margin-bottom:24px">PIPELINE</h2>` +
        `<div class="empty">NO JOBS YET.<br><br><a href="#/analyse">ANALYSE A VIDEO</a> TO START.</div>`;
      return;
    }
    const rows = jobs.map((j) => `
      <a class="job-row" href="#/pipeline/${encodeURIComponent(j.job_id)}">
        <span class="job-title">${escapeHtml(j.title || 'UNTITLED')}</span>
        <span class="job-meta">
          ${escapeHtml(j.job_id)} — ${escapeHtml((j.status || '').toUpperCase())}${j.stage && j.status === 'running' ? ' · ' + escapeHtml(j.stage) : ''}
        </span>
      </a>`).join('');
    root.innerHTML = `<h2 class="page-title" style="margin-bottom:24px">PIPELINE</h2>` + rows;
  } catch (err) {
    root.innerHTML = errorHtml(err.message);
  }
}

function renderDashboard(root, d, activeTab) {
  // Part A containment flag — outreach (DRAFT EMAIL) controls are gated on
  // the server-provided flag, which defaults OFF until the brand-collaboration
  // data path passes the integrity review.
  outreachEnabled = d.outreach_enabled === true;
  outreachDisabledReason = d.outreach_reason || 'DRAFT EMAIL DISABLED — PENDING DATA-INTEGRITY REVIEW';

  const tabs = ['scenes', 'products', 'openset', 'recommend', 'ads', 'outreach']
    .map((t) => `<button class="tab${t === activeTab ? ' is-active' : ''}" data-tab="${t}">${t}</button>`)
    .join('');

  const panels = {
    scenes: scenesPanel(d),
    products: productsPanel(d),
    openset: openSetPanel(d),
    recommend: recommendPanel(d),
    ads: adsPanel(d),
    outreach: outreachPanel(d),
  };

  root.innerHTML = `
    <h2 class="page-title page-title-huge">${escapeHtml(d.title)}</h2>
    <div class="meta-row">
      <span class="tag"><span class="tag-label">CREATOR</span> ${escapeHtml(d.creator || '—')}</span>
      <span class="tag tag-accent"><span class="tag-label">DURATION</span> ${escapeHtml(fmtDuration(d.duration_sec))}</span>
      <span class="tag"><span class="tag-label">JOB ID</span> ${escapeHtml(d.job_id)}</span>
      <span class="tag"><span class="tag-label">CONFIDENCE</span> ${d.confidence != null ? (d.confidence * 100).toFixed(0) + '%' : '—'}</span>
    </div>
    <div class="tabs">${tabs}</div>
    <div class="tab-panel is-active" data-panel="scenes">${panels.scenes}</div>
    <div class="tab-panel" data-panel="products">${panels.products}</div>
    <div class="tab-panel" data-panel="openset">${panels.openset}</div>
    <div class="tab-panel" data-panel="recommend">${panels.recommend}</div>
    <div class="tab-panel" data-panel="ads">${panels.ads}</div>
    <div class="tab-panel" data-panel="outreach">${panels.outreach}</div>
  `;

  $$('#pipeline-root .tab').forEach((btn) => {
    btn.addEventListener('click', () => {
      $$('#pipeline-root .tab').forEach((b) => b.classList.remove('is-active'));
      $$('#pipeline-root .tab-panel').forEach((p) => p.classList.remove('is-active'));
      btn.classList.add('is-active');
      $('#pipeline-root [data-panel="' + btn.dataset.tab + '"]').classList.add('is-active');
      history.replaceState(null, '', `#/pipeline/${encodeURIComponent(d.job_id)}?tab=${btn.dataset.tab}`);
    });
  });
}

function sceneThumbUrl(jobId, frameIndex) {
  return `/api/scene/${encodeURIComponent(jobId)}/${frameIndex}`;
}

function scenesPanel(d) {
  if (!d.scenes || !d.scenes.length) {
    return `<div class="empty">NO SCENES DETECTED</div>`;
  }
  const head = `<div class="card-head" style="margin-bottom:16px"><span class="card-sub">${d.scenes.length} SCENES · ${d.num_frames} FRAMES · ${(d.video_fps || 0).toFixed(1)} FPS — CLICK A FRAME TO OPEN THE SOURCE IMAGE</span></div>`;
  const rows = d.scenes.map((s) => `
    <div class="card">
      <div class="card-head">
        <div class="card-title">${fmtScene(s.n)}</div>
        <div class="tag"><span class="tag-label">TIME</span> ${fmtTime(s.timestamp)}</div>
      </div>
      <a class="scene-thumb-link" href="${sceneThumbUrl(d.job_id, s.frame_index)}" target="_blank">
        <img class="scene-thumb" src="${sceneThumbUrl(d.job_id, s.frame_index)}" alt="${fmtScene(s.n)} — ANNOTATED FRAME" loading="lazy">
      </a>
      <div class="scene-objects">
        ${s.objects.slice(0, 12).map((o) => chip(`${o.class_name} ${(o.confidence * 100).toFixed(0)}%`)).join('') || chip('NO OBJECTS', 'chip-ghost')}
        ${s.logos && s.logos.length ? s.logos.map((o) => chip(`${o.class_name} ${(o.confidence * 100).toFixed(0)}%`, 'chip-accent')).join('') : ''}
      </div>
    </div>`).join('');
  return head + rows;
}

function productsPanel(d) {
  if (!d.products || !d.products.length) {
    const reason = d.products_status_reason
      || 'BRAND ATTRIBUTION IS NOT YET PRODUCTION-VALIDATED (see MAJOR_REMEDIATION_REPORT.md Part B).';
    const status = d.products_status || 'NOT_AVAILABLE';
    return `
      <div class="empty" style="text-align:left">
        <div class="error-title">${status} — NO BRAND CAN BE ASSERTED AS AN ON-SCREEN APPEARANCE</div>
        ${escapeHtml(reason)}
      </div>`;
  }
  const integrityBanner = `
    <div class="error-box" style="margin-bottom:16px">
      <div class="error-title">UNVERIFIED DETECTION CANDIDATES — NOT CONFIRMED APPEARANCES</div>
      ${escapeHtml(d.products_status_reason || 'BRAND ATTRIBUTION IS NOT YET PRODUCTION-VALIDATED (see MAJOR_REMEDIATION_REPORT.md Part B).')}
    </div>`;
  const rows = d.products.map((p) => `
    <tr>
      <td>
        <div class="brand-cell">
          ${p.first_frame != null ? `<img class="mini-thumb" src="${sceneThumbUrl(d.job_id, p.first_frame)}" loading="lazy" alt="">` : ''}
          <div>
            <span class="cell-brand">${escapeHtml(p.brand)}</span>
            <span class="cell-cat">${escapeHtml(p.category || 'UNKNOWN')}</span>
          </div>
        </div>
      </td>
      <td>${escapeHtml(p.product || p.brand)}</td>
      <td>
        <div class="chips">
          ${p.appearances.slice(0, 8).map((sc) => chip(sc)).join('')}
          ${p.appearances.length > 8 ? chip('+' + (p.appearances.length - 8), 'chip-ghost') : ''}
        </div>
      </td>
      <td>
        <div class="cell-contact">
          ${p.contact_email ? escapeHtml(p.contact_email) : '<span class="muted">NO KNOWN CONTACT</span>'}
          ${p.contact_verified ? '' : '<span class="muted">(UNVERIFIED — EDIT BEFORE SENDING)</span>'}
        </div>
      </td>
      <td>
        ${outreachEnabled
          ? `<button class="btn-mini" data-brand="${escapeHtml(p.brand)}" data-job="${escapeHtml(d.job_id)}">DRAFT EMAIL</button>`
          : '<span class="muted">OUTREACH DISABLED</span>'}
      </td>
    </tr>`).join('');
  const html = `
    ${integrityBanner}
    <div class="table-wrap">
      <table class="table">
        <thead><tr><th>BRAND</th><th>PRODUCT</th><th>APPEARANCES</th><th>CONTACT</th><th>ACTION</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
  const wrap = document.createElement('div');
  wrap.innerHTML = html;
  $$('.btn-mini', wrap).forEach((b) =>
    b.addEventListener('click', () => {
      location.hash = `#/outreach/${encodeURIComponent(b.dataset.job)}?brand=${encodeURIComponent(b.dataset.brand)}`;
    })
  );
  return wrap.innerHTML;
}

function openSetPanel(d) {
  const os = d.open_set || {};
  const cands = os.candidates || [];
  if (!os.available) {
    return `
      <div class="empty" style="text-align:left">
        <div class="error-title">OPEN-SET IDENTIFICATION UNAVAILABLE</div>
        ${escapeHtml(os.reason || 'No runnable reverse-image-search backend.')}
      </div>`;
  }
  if (!cands.length) {
    return `
      <div class="empty">NO UNKNOWN-BRAND CANDIDATES ABOVE THE CONFIDENCE GATE (min ${escapeHtml(String(os.min_confidence))}).</div>`;
  }
  const head = `<div class="card-head" style="margin-bottom:16px"><span class="card-sub">UNKNOWN-BRAND CANDIDATES — REAL REVERSE-IMAGE SEARCH (${escapeHtml(String(os.backend))}) + logo.dev VALIDATION. CANDIDATES ARE LOWER-TRUST EVIDENCE, NEVER CONFIRMED APPEARANCES.</span></div>`;
  const rows = cands.map((c) => {
    const results = c.search_results || [];
    const trail = results
      .filter((r) => r.url)
      .slice(0, 6)
      .map((r) => `<div class="chip chip-ghost"><a href="${escapeHtml(r.url)}" target="_blank" rel="noopener">${escapeHtml(r.url)}</a></div>`)
      .join('');
    const tags = results
      .filter((r) => !r.url)
      .slice(0, 8)
      .map((r) => chip(escapeHtml(r.title), 'chip-accent'))
      .join('');
    const validation = c.logo_dev_validation || {};
    return `
      <div class="card">
        <div class="card-head">
          <div class="card-title">${escapeHtml(c.candidate_name || 'UNRESOLVED CROP')}</div>
          <div class="tag ${c.status === 'candidate_verified' ? 'tag-accent' : ''}"><span class="tag-label">STATUS</span> ${escapeHtml(c.status)}</div>
        </div>
        <div class="card-body">
          <div class="scene-objects">
            <span class="chip chip-ghost">FRAME ${c.frame_index}</span>
            <span class="chip chip-ghost">DET CONF ${(c.confidence * 100).toFixed(0)}%</span>
            <span class="chip chip-ghost">logo.dev ${escapeHtml(String(validation.status || 'n/a'))}</span>
            ${validation.domain ? `<span class="chip chip-ghost">${escapeHtml(validation.domain)}</span>` : ''}
          </div>
          ${tags ? `<div class="scene-objects">ENGINE TAGS: ${tags}</div>` : ''}
          ${trail ? `<div class="scene-objects">SOURCE LINKS: ${trail}</div>` : ''}
          ${c.crop_url ? `<div class="scene-objects"><a href="${escapeHtml(c.crop_url)}" target="_blank" rel="noopener">VIEW CROP</a></div>` : ''}
        </div>
      </div>`;
  }).join('');
  return head + rows;
}

function recommendPanel(d) {
  const recs = d.recommendations || [];
  if (!recs.length) {
    return `<div class="empty">NO RECOMMENDATIONS YET.<br><br>RUN ANALYSIS ON A VIDEO WITH BRAND EVIDENCE.<br><a href="#/analyse">ANALYSE A VIDEO</a></div>`;
  }
  const head = `<div class="card-head" style="margin-bottom:16px"><span class="card-sub">RANKED COLLABORATION OPPORTUNITIES — ${recs.length} BRANDS · DIRECT (EVIDENCE) + SUGGESTED (KNOWLEDGE GRAPH)</span></div>`;
  const cards = recs.map((r, i) => `
    <div class="card rec-card">
      <div class="rec-rank">${String(i + 1).padStart(2, '0')}</div>
      <div class="rec-main">
        <div class="card-head">
          <div class="card-title">${escapeHtml(r.brand)}</div>
          <div class="chips">
            <span class="chip ${r.type === 'DIRECT' ? 'chip-accent' : ''}">${escapeHtml(r.type)}</span>
            <span class="chip chip-ghost">${(r.score * 100).toFixed(0)}% FIT</span>
          </div>
        </div>
        <div class="card-body">
          ${escapeHtml(r.product || r.brand)} — ${escapeHtml(r.category || 'GENERAL')}${r.appearances ? ' · ' + r.appearances + ' APPEARANCES' : ' · NEVER ON SCREEN'}
        </div>
        <div class="rec-reasons">
          ${r.reasons.map((reason) => chip(reason)).join('')}
        </div>
      </div>
    </div>`).join('');
  const wrap = document.createElement('div');
  wrap.innerHTML = head + cards;
  return wrap.innerHTML;
}

function adsPanel(d) {
  if (!d.ads || !d.ads.length) {
    return `<div class="empty">NO AD OPPORTUNITIES DETECTED</div>`;
  }
  const head = `<div class="card-head" style="margin-bottom:16px"><span class="card-sub">${d.ads.length} OPPORTUNITIES · UNVERIFIED DETECTION CANDIDATES — NOT CONFIRMED ON-SCREEN APPEARANCES (brand attribution pending Part B validation)</span></div>`;
  const rows = d.ads.map((a) => `
    <div class="card">
      <div class="card-head">
        <div class="card-title">${escapeHtml(a.brand)} × ${escapeHtml(d.creator || 'CHANNEL')}</div>
        <span class="chip chip-accent">${escapeHtml(a.type)}</span>
      </div>
      <div class="card-body">
        PRODUCT: ${escapeHtml(a.product || a.brand)} — ${escapeHtml(a.category || 'GENERAL')}
      </div>
      <div class="scene-objects">
        ${a.scenes.slice(0, 12).map((sc) => chip(sc)).join('')}
      </div>
    </div>`).join('');
  return head + rows;
}

function outreachPanel(d) {
  if (!outreachEnabled) {
    return `<div class="error-box"><div class="error-title">OUTREACH DISABLED</div>${escapeHtml(outreachDisabledReason)}</div>`;
  }
  if (!d.products || !d.products.length) {
    return `<div class="empty">NO BRANDS TO CONTACT</div>`;
  }
  return `<div class="empty" style="padding:var(--space-lg)">${d.products.length} BRAND OPPORTUNITIES READY.<br><br><a href="#/outreach/${encodeURIComponent(d.job_id)}">OPEN OUTREACH EDITOR</a></div>`;
}

/* ── Outreach page ────────────────────────────────────────── */

let outreachState = { id: null, data: null, brand: null };
let outreachEnabled = false;
let outreachDisabledReason = 'DRAFT EMAIL DISABLED — PENDING DATA-INTEGRITY REVIEW';

async function renderOutreach(id, params) {
  const root = $('#outreach-root');
  if (!id) { root.innerHTML = errorHtml('NO JOB REFERENCED'); return; }

  root.innerHTML = loaderHtml('LOADING BRANDS');
  outreachState.id = id;
  outreachState.brand = params.get('brand');

  try {
    const data = await api(`/api/pipeline/${encodeURIComponent(id)}`);
    outreachState.data = data;
    outreachEnabled = data.outreach_enabled === true;
    outreachDisabledReason = data.outreach_reason || outreachDisabledReason;
    if (!outreachEnabled) {
      root.innerHTML = `<div class="error-box"><div class="error-title">OUTREACH DISABLED</div>${escapeHtml(outreachDisabledReason)}<br><br><a href="#/pipeline/${encodeURIComponent(id)}">BACK TO DASHBOARD</a></div>`;
      return;
    }
    renderOutreachEditor(root, data);
  } catch (err) {
    root.innerHTML = errorHtml(err.message);
  }
}

function renderOutreachEditor(root, d) {
  const products = d.products || [];
  const active = outreachState.brand || (products[0] && products[0].brand);

  const brandsList = products.length
    ? products.map((p) => `
        <div class="brand-row${p.brand === active ? ' is-active' : ''}" data-brand="${escapeHtml(p.brand)}">
          <div class="brand-name">${escapeHtml(p.brand)}</div>
          <div class="brand-product">${escapeHtml(p.product || p.brand)} · ${escapeHtml(p.category || 'GENERAL')} · ${p.appearances.length} SCENES</div>
          <div class="brand-contact">${p.contact_email ? escapeHtml(p.contact_email) : 'NO KNOWN CONTACT — ADD MANUALLY'}</div>
        </div>`).join('')
    : `<div class="empty">NO BRANDS</div>`;

  root.innerHTML = `
    <div class="outreach-brands">${brandsList}</div>
    <div class="outreach-editor">
      <div class="editor-toolbar">
        <div class="editor-target">
          <span class="editor-label">TARGET EMAIL</span>
          <input class="editor-target-name" id="target-name" value="" spellcheck="false" placeholder="AUTO-FILLED FROM BRAND CATALOG — EDITABLE">
          <span class="editor-hint" id="target-hint"></span>
        </div>
        <div class="editor-actions">
          <button class="btn btn-sm" id="btn-generate">GENERATE DRAFT</button>
          <button class="btn btn-sm" id="btn-forward">FORWARD</button>
        </div>
      </div>
      <div class="editor-subject">
        <span class="editor-label">SUBJECT</span>
        <div class="editor-subject-text" id="editor-subject"></div>
      </div>
      <div class="editor-body" id="editor-body"></div>
      <div class="editor-foot" id="editor-foot"></div>
    </div>
  `;

  const selectBrand = (brand) => {
    outreachState.brand = brand;
    $$('.brand-row', root).forEach((r) =>
      r.classList.toggle('is-active', r.dataset.brand === brand)
    );
    const p = products.find((x) => x.brand === brand);
    const hint = $('#target-hint');
    if (p && p.contact_email) {
      $('#target-name').value = p.contact_email;
      hint.textContent = `CONTACT AUTO-FILLED FROM CATALOG${p.contact_verified ? '' : ' (UNVERIFIED — CONFIRM BEFORE SENDING)'}${p.contact_website ? ' · ' + p.contact_website : ''}`;
    } else {
      $('#target-name').value = '';
      hint.textContent = 'NO KNOWN CONTACT — ENTER THE BRAND EMAIL MANUALLY';
    }
    history.replaceState(null, '', `#/outreach/${encodeURIComponent(d.job_id)}?brand=${encodeURIComponent(brand)}`);
  };

  $$('.brand-row', root).forEach((r) =>
    r.addEventListener('click', () => { selectBrand(r.dataset.brand); })
  );

  const subject = $('#editor-subject');
  const body = $('#editor-body');
  const foot = $('#editor-foot');
  const targetInput = $('#target-name');
  const generateBtn = $('#btn-generate');
  const forwardBtn = $('#btn-forward');

  // Prefill the active brand's contact on load
  selectBrand(active);

  generateBtn.addEventListener('click', async () => {
    if (!outreachState.brand) return;
    const target = targetInput.value.trim();
    if (!target) {
      foot.textContent = 'ENTER A TARGET EMAIL BEFORE GENERATING';
      return;
    }
    generateBtn.disabled = true;
    generateBtn.textContent = 'GENERATING…';
    foot.textContent = '';
    try {
      const res = await api('/api/outreach/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          job_id: d.job_id,
          brand: outreachState.brand,
          target,
        }),
      });
      subject.textContent = res.subject;
      body.textContent = res.body;
      foot.textContent = `DRAFT GENERATED · TO ${escapeHtml(res.target)} · ${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
    } catch (err) {
      foot.textContent = `GENERATE FAILED — ${err.message}`;
    } finally {
      generateBtn.disabled = false;
      generateBtn.textContent = 'GENERATE DRAFT';
    }
  });

  forwardBtn.addEventListener('click', async () => {
    if (!outreachState.brand) return;
    if (!body.textContent.trim()) {
      foot.textContent = 'GENERATE A DRAFT BEFORE FORWARDING';
      return;
    }
    forwardBtn.disabled = true;
    forwardBtn.textContent = 'FORWARDING…';
    try {
      const res = await api('/api/outreach/forward', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_id: d.job_id, brand: outreachState.brand }),
      });
      foot.textContent = `FORWARDED TO ${escapeHtml(res.target)} · ${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
      foot.classList.add('forwarded');
    } catch (err) {
      foot.textContent = `FORWARD FAILED — ${err.message}`;
    } finally {
      forwardBtn.disabled = false;
      forwardBtn.textContent = 'FORWARD';
    }
  });
}

/* ── Boot ─────────────────────────────────────────────────── */

router();
