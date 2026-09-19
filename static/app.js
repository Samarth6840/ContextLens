/* ADSCENE — frontend app logic (vanilla JS, hash router) */

const $ = (sel, root) => (root || document).querySelector(sel);
const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));

const el = {
  landing: $('#view-landing'),
  analyse: $('#view-analyse'),
  pipeline: $('#view-pipeline'),
  outreach: $('#view-outreach'),
  insights: $('#view-insights'),
  login: $('#view-login'),
};

/* ── Auth token (localStorage-backed) ─────────────────────── */

const TOKEN_KEY = 'adscene_token';
const getToken = () => localStorage.getItem(TOKEN_KEY) || null;
const setToken = (t) => { t ? localStorage.setItem(TOKEN_KEY, t) : localStorage.removeItem(TOKEN_KEY); };

function escapeHtml(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

function fmtDuration(sec) {
  if (!sec || sec <= 0) return '—';
  sec = Math.round(sec);
  if (sec < 60) return `${sec}s`;
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return s ? `${m}m ${s}s` : `${m}m`;
}

function fmtTime(sec) {
  sec = Math.round(sec || 0);
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

function fmtScene(n) {
  return 'Scene ' + String(n).padStart(3, '0');
}

function chip(text, extra) {
  return `<span class="chip${extra ? ' ' + extra : ''}">${escapeHtml(text)}</span>`;
}

// Design rule: every confidence number is a labeled chip — the metric that a
// number measures is never implied by position alone. A chip carrying a number
// MUST carry title/aria-label naming the metric (resolution_quality,
// detector_box_confidence, recommendation_confidence, …) and render the number
// in the mono .chip-confidence span so it reads as a measurement, not prose.
function pct(v) {
  const n = Number(v == null ? NaN : v);
  return Number.isFinite(n) ? `${(n * 100).toFixed(0)}%` : null;
}

function metaChip(label, pctVal, metric, variant) {
  const conf = pctVal != null ? `<span class="chip-confidence">${escapeHtml(pctVal)}</span>` : '';
  const tip = metric ? ` title="${escapeHtml(metric)}" aria-label="${escapeHtml(label)} ${escapeHtml(pctVal || '')} ${escapeHtml(metric)}"` : '';
  return `<span class="chip${variant ? ' ' + variant : ''}"${tip}>${escapeHtml(label)}${conf ? ' ' + conf : ''}</span>`;
}

function fieldLabel(field) {
  const map = {
    screen_size: 'Screen',
    thickness: 'Thickness',
    battery: 'Battery',
    weight: 'Weight',
    storage: 'Storage',
    refresh_rate: 'Refresh rate',
    camera: 'Camera',
    material: 'Material',
    processor: 'Chip',
  };
  const known = map[field];
  if (known) return known;
  const s = String(field || '').replace(/_/g, ' ');
  return s ? s.charAt(0).toUpperCase() + s.slice(1) : '';
}

// Friendly presentation label for recommendation/opportunity types.
function friendlyRecType(t) {
  if (t === 'DIRECT') return 'Direct match';
  if (t === 'SUGGESTED') return 'Suggested';
  return String(t || 'Match').toLowerCase();
}

function fmtSpec(s) {
  const val = (s.value == null ? '' : String(s.value));
  return (s.unit ? `${val} ${s.unit}` : val) || (s.raw || '');
}

async function api(path, opts) {
  opts = opts || {};
  opts.headers = Object.assign({}, opts.headers);
  const token = getToken();
  if (token) opts.headers['Authorization'] = `Bearer ${token}`;
  const res = await fetch(path, opts);
  let data = null;
  try { data = await res.json(); } catch (e) { /* empty */ }
  if (res.status === 401 && (data && data.error === 'AUTH REQUIRED')) {
    setToken(null);
    if (location.hash.indexOf('#/login') !== 0) location.hash = '#/login';
  }
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
  // Top-level nav + sidebar nav highlight
  $$('.nav-link').forEach((a) => {
    a.classList.toggle('is-active', a.getAttribute('href') === `#/${name}`);
  });
  $$('.nav-item').forEach((a) => {
    a.classList.toggle('is-active', a.getAttribute('href') === `#/${name}`);
  });
  // Back buttons: show only on detail sub-pages
  const hasId = !!(name !== 'landing' && (parseRoute().id));
  $('#btn-back-home') && ($('#btn-back-home').hidden = !hasId);
  $('#btn-back-outreach') && ($('#btn-back-outreach').hidden = !hasId);
  $('#btn-back-insights') && ($('#btn-back-insights').hidden = !hasId);
}

function router() {
  const { page, id, params } = parseRoute();
  refreshNavAuth();
  if (page === 'landing' || page === '') showView('landing');
  else if (page === 'analyse') showView('analyse');
  else if (page === 'pipeline') { showView('pipeline'); renderPipeline(id, params); }
  else if (page === 'outreach') { showView('outreach'); renderOutreach(id, params); }
  else if (page === 'insights') { showView('insights'); renderInsights(id, params); }
  else if (page === 'login') { showView('login'); renderLogin(); }
  else { location.hash = '#/'; }
}

// Sidebar / back-button wiring
['home', 'outreach', 'insights'].forEach((key) => {
  const btn = $(`#btn-back-${key === 'home' ? 'home' : key}`);
  if (btn) btn.addEventListener('click', () => { location.hash = key === 'home' ? '#/pipeline' : `#/${key}`; });
});
if (!window.__navBound) {
  window.__navBound = true;
  document.addEventListener('click', (e) => {
    const navAuth = e.target.closest('#nav-auth');
    if (!navAuth || navAuth.textContent !== 'Log out') return;
    e.preventDefault();
    api('/api/logout', { method: 'POST' }).catch(() => {});
    setToken(null);
    location.hash = '#/login';
  });
}

async function refreshNavAuth() {
  const link = $('#nav-auth');
  if (!link) return;
  try {
    const me = await api('/api/me');
    const authed = me.authenticated === true;
    link.textContent = authed ? 'Log out' : 'Log in';
    link.classList.toggle('is-authed', authed);
    link.setAttribute('href', authed ? '#/' : '#/login');
    if (authed) {
      link.setAttribute('data-user', me.user || '');
    }
  } catch (e) { /* keep default label */ }
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
    label.textContent = `${(selectedFile.size / (1024 * 1024)).toFixed(1)} MB — ready`;
    label.classList.add('has-file');
  } else {
    dropzone.classList.remove('has-file');
    dropzone.querySelector('span').textContent = 'Choose a video';
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
const procPanel = $('#processing-panel');
const procStageEl = $('#proc-stage');
const procPercentEl = $('#proc-percent');
const procBarEl = $('#proc-bar');
const procStepsEl = $('#proc-steps');
const procFeedEl = $('#proc-feed');

function procShow() {
  if (procPanel) procPanel.hidden = false;
}
function procReset() {
  if (!procStepsEl) return;
  $$('.step', procStepsEl).forEach((s) => {
    s.classList.remove('is-active', 'is-done');
    s.classList.toggle('is-done', s.dataset.step === 'upload');
  });
  if (procFeedEl) procFeedEl.innerHTML = '';
  if (procProgress) procProgress(0);
}

function procFeed(feed) {
  if (!procFeedEl || !feed || !feed.length) return;
  procFeedEl.innerHTML = feed
    .slice(-8)
    .map((m) => `<div class="feed-item"><span class="feed-dot">&nbsp;</span>${escapeHtml(m)}</div>`)
    .join('');
  procFeedEl.scrollTop = procFeedEl.scrollHeight;
}

// Stage -> progress step mapping. Multiple stage strings map to one step.
const PROC_STEPS = {
  models: ['LOADING MODELS'],
  signals: ['EXTRACTING AUDIO & VISUAL SIGNALS'],
  compile: ['COMPILING INTELLIGENCE'],
};
function procProgress(percent) {
  if (procPercentEl) procPercentEl.textContent = Math.round(percent) + '%';
  if (procBarEl) procBarEl.style.width = Math.min(100, Math.max(0, percent)) + '%';
}
function procStep(stage) {
  if (!procStepsEl || !stage) return;
  const stageUp = String(stage).toUpperCase();
  let matched = null;
  if (stageUp === 'COMPLETE') matched = 'compile';
  for (const [step, tags] of Object.entries(PROC_STEPS)) {
    if (tags.some((t) => stageUp.includes(t))) { matched = step; break; }
  }
  if (!matched) return;
  $$('.step', procStepsEl).forEach((s) => {
    const idx = s.dataset.step;
    if (idx === matched) { s.classList.add('is-active'); s.classList.remove('is-done'); }
    else if (['upload', 'models', 'signals', 'compile'].indexOf(idx) < ['upload', 'models', 'signals', 'compile'].indexOf(matched)) {
      s.classList.add('is-done'); s.classList.remove('is-active');
    }
  });
}




function setStatus(text, kind) {
  analyseStatus.hidden = false;
  analyseStatus.className = 'status-line' + (kind ? ' ' + kind : '');
  analyseStatus.innerHTML = text;
}

function procError(message) {
  if (procPanel) procPanel.hidden = true;
  setStatus(message, 'error');
}
$('#analyse-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const btn = $('#btn-extract');
  if (!selectedFile) { setStatus('<span class="dot">▮</span> Please choose a video file first'); return; }
  if (selectedFile.size > 200 * 1024 * 1024) {
    setStatus('This file is over the 200 MB limit — try a smaller one.', 'error');
    return;
  }

  const fd = new FormData();
  fd.append('video', selectedFile);
  fd.append('title', $('#video-title').value.trim() || selectedFile.name.replace(/\.[^.]+$/, ''));
  fd.append('creator', $('#video-creator').value.trim() || 'UNKNOWN');
  fd.append('duration', $('#video-duration').value.trim());

  btn.disabled = true;
  btn.textContent = 'Uploading…';
  setStatus('<span class="dot">▮</span> Uploading your video…');
  procReset();

  try {
    const { job_id } = await api('/api/analyse', { method: 'POST', body: fd });
    selectedFile = null;
    fileInput.value = '';
    updateDropzone();
    setStatus('<span class="dot" style="color:var(--success)">▮</span> Uploaded — analysing your video…');
    procShow();
    pollJob(job_id, btn);
  } catch (err) {
    btn.disabled = false;
    btn.textContent = 'Analyse video';
    procError(`Upload failed — ${escapeHtml(err.message)}`);
  }
});

function pollJob(jobId, btn) {
  let attempts = 0;
  const tick = async () => {
    attempts += 1;
    let job;
    try { job = await api(`/api/analyse/${encodeURIComponent(jobId)}`); }
    catch (err) {
      btn.disabled = false; btn.textContent = 'Analyse video';
      procError(`Status check failed — ${escapeHtml(err.message)}`);
      return;
    }
    if (job.status === 'done') {
      btn.disabled = false;
      btn.textContent = 'Analyse video';
      procStageEl && (procStageEl.textContent = 'Complete — loading your results');
      procStep('COMPLETE');
      procProgress(100);
      location.hash = `#/pipeline/${encodeURIComponent(jobId)}`;
      return;
    }
    if (job.status === 'error') {
      btn.disabled = false;
      btn.textContent = 'Analyse video';
      procError(`Analysis failed — ${escapeHtml(job.error || 'something went wrong')}`);
      return;
    }
    // Animate progress bar toward a perceived ceiling while running
    const base = attempts;
    procProgress(base >= 3 ? 92 + Math.min(6, Math.floor((attempts - 3) / 2)) : 15 + attempts * 22);
    procStageEl && (procStageEl.textContent = String(job.stage || 'Working…'));
    procStep(job.stage || 'PROCESSING');
    procFeed(job.feed);
    setStatus(`<span class="dot blink">▮</span> ${escapeHtml(job.stage || 'Working…')}…`);
    if (attempts < 600) setTimeout(tick, 1500);
  };
  tick();
}

/* ── Pipeline page ────────────────────────────────────────── */

let pipeState = { id: null, data: null, tab: 'scenes' };

// Client-side search/sort/filter state for the pipeline "Scenes" panel, plus a
// pending clip target (arriving via ?clip=<sec>) that seeks + highlights on
// render. Kept at module scope so re-renders preserve the user's filters.
const sceneView = { q: '', brand: 'ALL', object: 'ALL', sort: 'time' };
let pendingClip = null;

async function renderPipeline(id, params) {
  const root = $('#pipeline-root');
  if (!id) return renderJobIndex(root, params);

  root.innerHTML = loaderHtml('Loading your analysis');
  pipeState.id = id;
  pipeState.tab = params.get('tab') || 'scenes';
  const clipParam = params.get('clip');
  pendingClip = (clipParam != null && !Number.isNaN(parseFloat(clipParam)))
    ? parseFloat(clipParam)
    : null;

  try {
    const data = await api(`/api/pipeline/${encodeURIComponent(id)}`);
    pipeState.data = data;
    renderDashboard(root, data, pipeState.tab);
  } catch (err) {
    root.innerHTML = errorHtml(err.message);
  }
}

function loaderHtml(stage) {
  return `<div class="loader"><div class="spinner"></div><span class="stage">${escapeHtml(stage)}…</span></div>`;
}

function errorHtml(msg) {
  return `<div class="error-box"><div class="error-title">Something went wrong</div>${escapeHtml(msg)}</div>`;
}

async function renderJobIndex(root) {
  root.innerHTML = loaderHtml('Loading your projects');
  try {
    const { jobs } = await api('/api/jobs');
    if (!jobs.length) {
      root.innerHTML =
        `<h2 class="page-title" style="margin-bottom:24px">Your projects</h2>` +
        `<div class="empty">No projects yet — start by analysing a video to find your first opportunities.<br><br><a href="#/analyse">Analyse a video</a></div>`;
      return;
    }
    const rows = jobs.map((j, i) => `
      <a class="job-row" href="#/pipeline/${encodeURIComponent(j.job_id)}">
        <span class="job-meta" style="min-width:28px;text-align:right;font-family:var(--font-mono)">${String(i + 1).padStart(2, '0')}</span>
        <span class="job-title">${escapeHtml(j.title || 'Untitled video')}</span>
        <span class="job-meta">${escapeHtml(j.creator || '')}</span>
        <span class="job-meta">${escapeHtml(j.job_id)}</span>
        <span class="job-meta">${escapeHtml(j.status || '')}${j.stage && j.status === 'running' ? ' &middot; ' + escapeHtml(j.stage) : ''}</span>
      </a>`).join('');
    root.innerHTML = `<h2 class="page-title" style="margin-bottom:24px">Your projects</h2><div style="display:flex;gap:var(--space-3);margin-bottom:12px"><span class="k-label">Index</span><span class="k-label" style="flex:1">Title</span><span class="k-label">Creator</span><span class="k-label">Project</span><span class="k-label">Status</span></div>` + rows;
  } catch (err) {
    root.innerHTML = errorHtml(err.message);
  }
}

function summaryStrip(d) {
  // Real "small things" summary — actual counts from the completed analysis.
  const scenes = (d.scenes || []).length;
  const frames = d.num_frames != null ? d.num_frames : '—';

  // Uniquely resolved brands (dedup across scene appearances + recommendations)
  const brands = new Set();
  (d.products || []).forEach((p) => brands.add(p.brand));
  (d.recommendations || []).forEach((r) => brands.add(r.brand));
  (d.brands || []).forEach((b) => brands.add(typeof b === 'string' ? b : b.name));
  const brandCount = brands.size || (d.recommendations || []).length;

  // Logo boxes across all scenes
  let logos = 0;
  (d.scenes || []).forEach((s) => { logos += (s.logos || []).length; });

  const stat = (label, val, accent) => `
    <div class="stat-box${accent ? ' is-accent' : ''}">
      <div class="stat-box-val">${val}</div>
      <div class="stat-box-label">${escapeHtml(label)}</div>
    </div>`;

  return `
    <div class="summary-strip">
      ${stat('Scenes', scenes)}
      ${stat('Frames', frames)}
      ${stat('Logo boxes', logos)}
      ${stat('Brands found', brandCount, true)}
    </div>`;
}

// Plain-language "what we found" summary shown at the top of the dashboard.
function whatFoundPanel(d) {
  const scenes = (d.scenes || []).length;
  const brands = new Set();
  (d.products || []).forEach((p) => brands.add(p.brand));
  (d.recommendations || []).forEach((r) => brands.add(r.brand));
  (d.brands || []).forEach((b) => brands.add(typeof b === 'string' ? b : b.name));
  const brandCount = brands.size;
  const opps = (d.recommendations || []).length;
  const conf = d.confidence != null ? (d.confidence * 100).toFixed(0) : null;

  const brandWord = brandCount === 1 ? '1 brand' : `${brandCount || 'no'} brands`;
  const oppWord = opps === 1 ? '1 opportunity' : opps ? `${opps} opportunities` : 'no opportunities';
  const confWord = conf != null
    ? (d.is_confident ? `The read feels confident (${conf}%).` : `We're reasonably sure — overall confidence ${conf}%.`)
    : '';
  const plural = scenes === 1 ? 'scene' : 'scenes';
  return `
    <div class="what-found">
      <div class="what-found-icon">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="9"/><path d="M12 8v4l3 2"/></svg>
      </div>
      <div>
        <span class="k-label" style="display:block;margin-bottom:4px">Summary</span>
        In <strong>${escapeHtml(d.title || 'this video')}</strong> we found <em>${brandWord}</em>,
        ${oppWord} to act on, across ${scenes} ${plural}. ${confWord}
      </div>
    </div>`;
}

function renderDashboard(root, d, activeTab) {
  // Part A containment flag — outreach (DRAFT EMAIL) controls are gated on
  // the server-provided flag, which defaults OFF until the brand-collaboration
  // data path passes the integrity review.
  outreachEnabled = d.outreach_enabled === true;
  outreachDisabledReason = d.outreach_reason || 'Draft email is off — the outreach data path hasn\u2019t passed its integrity review yet.';

  const tabDefs = [
    ['scenes', 'Scenes'],
    ['products', 'Brands'],
    ['openset', 'Unidentified brands'],
    ['recommend', 'Opportunities'],
    ['ads', 'Audios'],
    ['outreach', 'Outreach'],
  ];
  const tabs = tabDefs
    .map(([key, label]) => `<button class="tab${key === activeTab ? ' is-active' : ''}" data-tab="${key}">${label}</button>`)
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
      <span class="tag"><span class="tag-label">Creator</span> ${escapeHtml(d.creator || '—')}</span>
      ${d.creator_card && d.creator_card.handle ? `<span class="tag"><span class="tag-label">@</span>${escapeHtml(d.creator_card.handle)} ${d.creator_card.followers_label ? escapeHtml(d.creator_card.followers_label) : ''}</span>` : ''}
      <span class="tag tag-accent"><span class="tag-label">Duration</span> ${escapeHtml(fmtDuration(d.duration_sec))}</span>
      <span class="tag"><span class="tag-label">Project</span> ${escapeHtml(d.job_id)}</span>
      ${d.confidence != null ? `<span class="tag"><span class="tag-label">Confidence</span> ${(d.confidence * 100).toFixed(0)}%</span>` : ''}
    </div>
    ${confidenceBar(d)}
    ${whatFoundPanel(d)}
    ${summaryStrip(d)}
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
      if (btn.dataset.tab === 'scenes' && !window.__sceneFilterBox) wireScenePanel(root, d);
    });
  });

  // Scene search/sort/filter controls + clip seek (Part C/D). The video element
  // 404s gracefully (no clip-player when no <video> can load) and the scenes
  // list still jumps + highlights on clip links.
  if (activeTab === 'scenes') wireScenePanel(root, d);
}

function sceneThumbUrl(jobId, frameIndex) {
  return `/api/scene/${encodeURIComponent(jobId)}/${frameIndex}`;
}

/* ── Confidence bar (design: evidence strength is shown, not one opaque %) ── */

function confidenceBar(d) {
  const total = d.confidence;
  if (total == null) return '';
  const evids = Object.entries(d.evidence_breakdown || {});
  if (!evids.length) return '';
  const band = (v) => (v >= 0.75 ? 'high' : (v >= 0.40 ? 'mid' : 'low'));
  const sumW = evids.reduce((a, [, e]) => a + (e.weight || 0), 0) || 0;
  const fills = evids.map(([src, e]) => {
    const strength = e.strength || 0;
    const w = e.weight || 0;
    const width = sumW > 0 ? (w / sumW) * 100 : 0;
    return `<span class="confidence-bar-fill" data-band="${band(strength)}" style="width:${width.toFixed(1)}%" title="${escapeHtml(src)} · ${(strength * 100).toFixed(0)}% strength · weight ${w.toFixed(3)}"></span>`;
  }).join('');
  const remainder = sumW > 0 && sumW < 1
    ? `<span class="confidence-bar-unweighted" style="flex:1 1 0%" title="Unweighted — evidence weights sum to ${(sumW * 100).toFixed(0)}% of the score"></span>`
    : '';
  const status = d.is_confident ? 'confident' : (String(d.confidence_status || 'caution').toLowerCase());
  const keys = evids.map(([src, e]) => {
    const strength = e.strength || 0;
    return `
      <span class="confidence-key"><span class="confidence-key-dot" data-band="${band(strength)}"></span>${escapeHtml(src)} ${(strength * 100).toFixed(0)}%</span>`;
  }).join('');
  return `
    <div class="confidence-block">
      <div class="confidence-block-head">
        <span class="k-label">Confidence &middot; ${escapeHtml(status)}</span>
        <span class="mono">${(total * 100).toFixed(0)}%</span>
      </div>
      <div class="confidence-bar">${fills}${remainder}</div>
      <div class="confidence-legend">${keys}</div>
    </div>`;
}

/* ── Scenes panel: search / sort / filter + clip seek ─────── */

let lastSceneData = null;

// Brand filter source = RESOLVED brand logos only (never the object table, so
// brand and object filters stay two distinct groups — a resolver-named brand is
// a different signal than a generic COCO object class).
function sceneBrands(s) {
  const set = new Set();
  (s.logos || []).forEach((o) => {
    if (o.class_name && o.class_name !== 'UNKNOWN BRAND') set.add(o.class_name);
  });
  return Array.from(set);
}

function sceneObjects(s) {
  const set = new Set();
  (s.objects || []).forEach((o) => {
    if (o.class_name) set.add(o.class_name);
  });
  return Array.from(set);
}

// Filter dropdown lists: BRANDS sorted by appearance count desc, OBJECT TYPES
// alphabetical — never merged into one flat list.
function sceneFilterCounts(d) {
  const brands = {};
  const objects = {};
  (d.scenes || []).forEach((s) => {
    sceneBrands(s).forEach((b) => { brands[b] = (brands[b] || 0) + 1; });
    sceneObjects(s).forEach((o) => { objects[o] = (objects[o] || 0) + 1; });
  });
  return {
    brands: Object.entries(brands).sort((a, b) => b[1] - a[1]),
    objects: Object.entries(objects).sort((a, b) => a[0].localeCompare(b[0])),
  };
}

function sceneChips(s) {
  const parts = [];
  if (s.objects && s.objects.length) {
    parts.push(s.objects.slice(0, 12).map((o) => metaChip(
      o.class_name,
      pct(o.confidence),
      'DETECTOR BOX CONFIDENCE',
      'chip--object',
    )).join(''));
  }
  if (s.logos && s.logos.length) {
    parts.push(s.logos.map((o) => {
      const unknown = o.class_name === 'UNKNOWN BRAND';
      const metric = (o.confidence_metric || (unknown ? 'detector_box_confidence' : 'resolution_quality')).toUpperCase();
      return metaChip(o.class_name, pct(o.confidence), metric, unknown ? 'chip--unknown' : 'chip--brand');
    }).join(''));
  }
  const joined = parts.join('');
  return joined || chip('No objects detected', 'chip-ghost');
}

function scenePassesFilter(s, v) {
  const q = (v.q || '').trim().toLowerCase();
  const hay = [fmtScene(s.n), String(s.timestamp), sceneBrands(s).join(' '), sceneObjects(s).join(' '), s.caption || ''].join(' ').toLowerCase();
  if (q && !hay.includes(q)) return false;
  if (v.brand && v.brand !== 'ALL' && !sceneBrands(s).includes(v.brand)) return false;
  if (v.object && v.object !== 'ALL' && !sceneObjects(s).includes(v.object)) return false;
  return true;
}

function sceneSortValue(s) {
  if (sceneView.sort === 'objects') {
    return (s.objects || []).length + (s.logos || []).length;
  }
  if (sceneView.sort === 'brands') {
    return (s.objects || []).concat(s.logos || []).filter((o) => o.class_name && o.class_name !== 'UNKNOWN BRAND').length;
  }
  return s.timestamp;
}

function renderSceneRows(d) {
  const src = d || lastSceneData;
  if (!src || !src.scenes) return `<div class="empty">No scenes found.</div>`;
  const matched = src.scenes.filter((s) => scenePassesFilter(s, sceneView));
  const desc = sceneView.sort === 'time-desc';
  const sorted = matched.slice().sort((a, b) => {
    const va = sceneSortValue(a);
    const vb = sceneSortValue(b);
    return desc ? (vb - va) : (va - vb);
  });
  if (!sorted.length) {
    const filterTip = sceneView.brand !== 'ALL' || sceneView.object !== 'ALL'
      ? ' Try a different filter.'
      : ' Try a different search.';
    return `<div class="empty">No scenes match "${escapeHtml(sceneView.q)}".${filterTip}</div>`;
  }
  const hl = pendingClip;
  return sorted.map((s) => {
    const near = hl != null && Math.abs(s.timestamp - hl) < 2.0;
    const hasUnresolved = (s.logos || []).some((o) => o.class_name === 'UNKNOWN BRAND');
    return `
      <div class="scene-row ${near ? 'scene-highlight' : ''} ${hasUnresolved ? 'scene-row--has-unresolved' : ''}" data-ts="${Number(s.timestamp).toFixed(2)}">
        <a class="scene-thumb-link" href="${sceneThumbUrl(src.job_id, s.frame_index)}" target="_blank">
          <img class="scene-thumb" src="${sceneThumbUrl(src.job_id, s.frame_index)}" alt="${fmtScene(s.n)} — annotated frame" loading="lazy">
        </a>
        <div class="scene-body">
          <div class="scene-meta">
            <span class="scene-time">${fmtScene(s.n)} &middot; ${fmtTime(s.timestamp)}</span>
            ${s.caption ? `<div class="scene-caption">${escapeHtml(s.caption)}</div>` : ''}
          </div>
          <div class="scene-chips">
            ${sceneChips(s)}
            <a class="chip chip-ghost clip-chip" role="button" data-ts="${Number(s.timestamp).toFixed(2)}">&blacktriangleright; Play @ ${fmtTime(s.timestamp)}</a>
          </div>
        </div>
      </div>`;
  }).join('');
}

function scenesPanel(d) {
  if (!d.scenes || !d.scenes.length) {
    return `<div class="empty">No scenes were detected in this video.</div>`;
  }
  lastSceneData = d;
  const unknown = d.unknown_brand_regions || [];
  // Header count = UNIQUE unresolved spatial regions, deduplicated across
  // frames (the same on-screen logo box appears in many frames but is one
  // region). The per-scene UNKNOWN chips below, by contrast, count raw
  // per-frame detection instances, so they can legitimately sum higher than
  // this header. Labels are kept distinct so the two numbers aren't confused.
  const unknownNote = unknown.length
    ? `<span class="tag" style="margin-left:8px"><span class="tag-label">Unidentified brands</span> ${unknown.length} unique spot${unknown.length === 1 ? '' : 's'}<span class="tag-label" style="margin-left:8px">across all frames</span></span>`
    : '';
  const sortOpts = [
    ['time', 'Sort by time'],
    ['time-desc', 'Newest first'],
    ['objects', 'Most objects'],
    ['brands', 'Most brands'],
  ].map(([val, label]) => `<option value="${val}"${val === sceneView.sort ? ' selected' : ''}>${label}</option>`).join('');
  const filterActive = sceneView.brand !== 'ALL' || sceneView.object !== 'ALL';
  const audioBlock = (d.audio_events || []).length ? `
    <div class="audio-strip">
      <span class="card-sub">Audio events</span>
      <div class="scene-chips">
        ${d.audio_events.slice(0, 12).map((ev) => metaChip(
          `${ev.event}${ev.start_time != null ? ' @ ' + fmtTime(ev.start_time) : ''}`,
          pct(ev.confidence),
          ev.mode === 'fallback' ? (ev.fallback_reason || 'AUDIO ENERGY HEURISTIC') : 'BEATs CLASS CONFIDENCE',
          ev.mode === 'fallback' ? 'chip--audio-fallback' : 'chip--audio-real',
        )).join('')}
        ${d.audio_events.length > 12 ? chip('+' + (d.audio_events.length - 12) + ' more', 'chip-ghost') : ''}
      </div>
    </div>` : '';
  const head = `
    <div class="card-head" style="margin-bottom:16px"><span class="card-sub">${d.scenes.length} scenes &middot; ${d.num_frames} frames &middot; ${(d.video_fps || 0).toFixed(1)} fps &mdash; click a frame to open the source image</span>${unknownNote}</div>
    <div class="scene-toolbar">
      <input id="scene-search" class="input scene-search" type="search" placeholder="Search scenes (brand, object, time)…" value="${escapeHtml(sceneView.q)}" autocomplete="off">
      <button id="scene-filter-btn" class="btn" type="button" aria-haspopup="true" aria-expanded="false">Filter${filterActive ? ' \u2713' : ''}</button>
      ${sceneFilterDropdown(d, sceneView)}
      <select id="scene-sort" class="input scene-select" aria-label="Sort scenes">${sortOpts}</select>
      <button id="scene-filter-reset" class="btn" type="button">Reset</button>
    </div>
    ${audioBlock}
    <video id="clip-player" class="clip-player" controls preload="metadata" src="/api/video/${encodeURIComponent(d.job_id)}"></video>
    <div id="scene-meta" class="muted" style="margin:8px 0"></div>
    <div id="scene-rows">${renderSceneRows(d)}</div>`;
  return head;
}

function sceneFilterDropdown(d, state) {
  const counts = sceneFilterCounts(d);
  const q = (state.filterQ || '').toLowerCase();
  const brandItems = counts.brands
    .filter(([b]) => !q || b.toLowerCase().includes(q))
    .map(([b, n]) => `
      <button class="filter-item filter-item--brand ${state.brand === b ? 'is-checked' : ''}" data-kind="brand" data-value="${escapeHtml(b)}" type="button">
        <span>${escapeHtml(b)}</span><span class="filter-count">${n}</span>
      </button>`).join('');
  const objectItems = counts.objects
    .filter(([o]) => !q || o.toLowerCase().includes(q))
    .map(([o, n]) => `
      <button class="filter-item filter-item--object ${state.object === o ? 'is-checked' : ''}" data-kind="object" data-value="${escapeHtml(o)}" type="button">
        <span>${escapeHtml(o)}</span><span class="filter-count">${n}</span>
      </button>`).join('');
  const brandSel = state.brand !== 'ALL' ? escapeHtml(state.brand) : 'ALL';
  const objSel = state.object !== 'ALL' ? escapeHtml(state.object) : 'ALL';
  return `
    <div class="filter-dropdown" id="scene-filter" hidden>
      <input class="filter-search" id="scene-filter-search" type="search" placeholder="Filter brands and objects…" autocomplete="off" value="${escapeHtml(state.filterQ || '')}">
      <div class="filter-group-label">Brands · <span class="filter-group-value">${brandSel}</span></div>
      ${brandItems || `<div class="filter-item-empty">No brands match</div>`}
      <div class="filter-group-label">Object types · <span class="filter-group-value">${objSel}</span></div>
      ${objectItems || `<div class="filter-item-empty">No objects match</div>`}
    </div>`;
}

function wireScenePanel(root, d) {
  const rows = root.querySelector('#scene-rows');
  if (!rows) return;
  const search = root.querySelector('#scene-search');
  const sortSel = root.querySelector('#scene-sort');
  const resetBtn = root.querySelector('#scene-filter-reset');
  const meta = root.querySelector('#scene-meta');
  const filterBtn = root.querySelector('#scene-filter-btn');
  const filterBox = root.querySelector('#scene-filter');
  const filterSearch = root.querySelector('#scene-filter-search');

  const syncFilterUI = () => {
    const active = sceneView.brand !== 'ALL' || sceneView.object !== 'ALL';
    if (filterBtn) {
      filterBtn.textContent = 'Filter' + (active ? ' ✓' : '');
      filterBtn.setAttribute('aria-expanded', String(filterBox && !filterBox.hidden));
    }
    if (filterBox) {
      $$('.filter-item', filterBox).forEach((it) => {
        const target = it.dataset.kind === 'brand' ? sceneView.brand : sceneView.object;
        it.classList.toggle('is-checked', it.dataset.value === target);
      });
      const brandGroup = filterBox.querySelector('.filter-group-value');
      const objGroup = filterBox.querySelectorAll('.filter-group-value')[1];
      if (brandGroup) brandGroup.textContent = sceneView.brand !== 'ALL' ? sceneView.brand : 'ALL';
      if (objGroup) objGroup.textContent = sceneView.object !== 'ALL' ? sceneView.object : 'ALL';
    }
  };

  const rerender = () => {
    rows.innerHTML = renderSceneRows(d);
    const shown = rows.querySelectorAll('.scene-row').length;
    if (meta) meta.textContent = `Showing ${shown} of ${(d.scenes || []).length} scenes`;
    syncFilterUI();
  };
  if (search) search.addEventListener('input', () => { sceneView.q = search.value; rerender(); });
  if (sortSel) sortSel.addEventListener('change', () => { sceneView.sort = sortSel.value; rerender(); });
  if (filterBtn && filterBox) {
    const open = () => { filterBox.hidden = false; filterBtn.setAttribute('aria-expanded', 'true'); };
    const close = () => { filterBox.hidden = true; filterBtn.setAttribute('aria-expanded', 'false'); };
    filterBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      if (filterBox.hidden) { open(); if (filterSearch) filterSearch.focus(); }
      else close();
    });
    if (filterSearch) {
      filterSearch.addEventListener('input', () => {
        const q = filterSearch.value.trim().toLowerCase();
        sceneView.filterQ = filterSearch.value;
        $$('.filter-item', filterBox).forEach((it) => {
          const name = (it.firstElementChild ? it.firstElementChild.textContent : it.textContent) || '';
          it.style.display = (!q || name.toLowerCase().includes(q)) ? '' : 'none';
        });
      });
    }
    $$('.filter-item', filterBox).forEach((it) => {
      it.addEventListener('click', (e) => {
        e.stopPropagation();
        const val = it.dataset.value;
        if (it.dataset.kind === 'brand') sceneView.brand = (sceneView.brand === val ? 'ALL' : val);
        else sceneView.object = (sceneView.object === val ? 'ALL' : val);
        if (filterSearch) {
          sceneView.filterQ = '';
          filterSearch.value = '';
          $$('.filter-item', filterBox).forEach((i) => { i.style.display = ''; });
        }
        rerender();
      });
    });
    window.__sceneFilterBox = filterBox;
    document.addEventListener('click', (e) => {
      const box = window.__sceneFilterBox;
      if (!box || box.hidden) return;
      if (!e.target.closest('#scene-filter') && !e.target.closest('#scene-filter-btn')) {
        box.hidden = true;
        const btn = document.querySelector('#scene-filter-btn');
        if (btn) btn.setAttribute('aria-expanded', 'false');
      }
    });
  }
  if (resetBtn) resetBtn.addEventListener('click', () => {
    sceneView.q = ''; sceneView.brand = 'ALL'; sceneView.object = 'ALL'; sceneView.sort = 'time';
    if (search) search.value = '';
    if (sortSel) sortSel.value = 'time';
    if (filterSearch) { sceneView.filterQ = ''; filterSearch.value = ''; $$('.filter-item', filterBox).forEach((i) => { i.style.display = ''; }); }
    rerender();
  });
  if (meta) meta.textContent = `Showing ${(d.scenes || []).length} of ${(d.scenes || []).length} scenes`;

  const video = root.querySelector('#clip-player');
  const flashScene = (ts) => {
    const el = rows.querySelector(`[data-ts="${Number(ts).toFixed(2)}"]`);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'center' });
      el.classList.add('scene-flash');
      setTimeout(() => el.classList.remove('scene-flash'), 2600);
    }
  };
  const seekClip = (ts) => {
    if (video) {
      const target = Math.max(0, ts - 1.5);
      try {
        if (video.readyState >= 2) {
          video.currentTime = target;
        } else {
          video.addEventListener('loadedmetadata', () => { video.currentTime = target; }, { once: true });
        }
        video.play().catch(() => {});
      } catch (e) { /* video may be absent */ }
    }
    flashScene(ts);
  };
  rows.addEventListener('click', (e) => {
    const chipEl = e.target.closest('.clip-chip');
    if (chipEl && chipEl.dataset.ts != null) {
      e.preventDefault();
      seekClip(parseFloat(chipEl.dataset.ts));
      if (history.replaceState) {
        history.replaceState(null, '', `#/pipeline/${encodeURIComponent(d.job_id)}?tab=scenes&clip=${chipEl.dataset.ts}`);
      }
    }
  });
  if (pendingClip != null) {
    seekClip(pendingClip);
    pendingClip = null;
  }
}

function productResolutionsBlock(d) {
  const pr = d.product_resolutions || [];
  if (!pr.length) return '';
  const rows = pr.map((r) => `
    <tr>
      <td>
        <div class="brand-cell">
          <div>
            <span class="cell-brand">${escapeHtml(r.product)}</span>
            <span class="cell-cat">→ ${escapeHtml(r.brand)}</span>
          </div>
        </div>
      </td>
      <td>
        <div class="chips">
          <span class="chip chip-ghost">${escapeHtml(String(r.resolution_tier || 'n/a'))}</span>
          <span class="chip chip-ghost">${escapeHtml(String(r.source || 'n/a'))}</span>
          <span class="chip ${r.mode === 'ocr' ? 'chip--object' : 'chip--brand'}">${r.mode === 'ocr' ? 'On-screen text' : 'Spoken'}</span>
          ${r.confidence != null ? metaChip('Resolution quality', pct(r.confidence), (r.confidence_metric || 'RESOLUTION_QUALITY').toUpperCase(), 'chip-ghost') : ''}
        </div>
      </td>
      <td>${r.start_time != null ? fmtTime(r.start_time) : '<span class="muted">No timestamp</span>'}</td>
      <td>
        ${r.start_time != null
          ? `<a class="chip chip-accent clip-chip" data-ts="${Number(r.start_time).toFixed(2)}" href="#/pipeline/${encodeURIComponent(d.job_id)}?tab=scenes&clip=${Number(r.start_time).toFixed(2)}">&blacktriangleright; Play @ ${fmtTime(r.start_time)}</a>`
          : '<span class="muted">—</span>'}
      </td>
    </tr>`).join('');
  return `
    <div class="section-head">Product mentions (spoken and on-screen)</div>
    <div class="table-wrap">
      <table class="table">
        <thead><tr><th>Product → brand</th><th>Evidence</th><th>Time</th><th>Clip</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
}

function productsPanel(d) {
  const specCards = (d.specs && d.specs.length) ? `
    <div class="section-head">Spec details</div>
    <div class="chips">
      ${d.specs.map((s) => chip(`${escapeHtml(fieldLabel(s.field))}: ${escapeHtml(fmtSpec(s))}`, 'chip-accent')).join('')}
    </div>` : '';
  if (!d.products || !d.products.length) {
    const reason = d.products_status_reason
      || 'No brand was resolved on screen.';
    const status = d.products_status || 'NO_PRODUCTS';
    return `
      <div class="empty" style="text-align:left">
        <div class="note-title">No on-screen brand identified</div>
        ${escapeHtml(reason)}
        ${specCards}
        ${productResolutionsBlock(d)}
      </div>`;
  }
  const integrityBanner = `
    <div class="note-box" style="margin-bottom:16px">
      <div class="note-title">Resolved on-screen brands</div>
      ${escapeHtml(d.products_status_reason || 'Brand appearances are aggregated from the logo pipeline.')}
      <div class="muted" style="margin-top:4px">Contact emails are placeholders — verify before sending any outreach.</div>
    </div>`;
  const rows = d.products.map((p) => `
    <tr>
      <td>
        <div class="brand-cell">
          ${p.first_frame != null ? `<img class="mini-thumb" src="${sceneThumbUrl(d.job_id, p.first_frame)}" loading="lazy" alt="">` : ''}
          <div>
            <span class="cell-brand">${escapeHtml(p.brand)}</span>
            <span class="cell-cat">${escapeHtml(p.category || 'Uncategorised')}</span>
          </div>
        </div>
      </td>
      <td>${escapeHtml(p.product || p.brand)}</td>
      <td>
        <div class="chips">
          ${p.confidence != null ? metaChip('Resolution quality', pct(p.confidence), (p.confidence_metric || 'RESOLUTION_QUALITY').toUpperCase(), 'chip--brand') : ''}
          ${p.appearances.slice(0, 8).map((sc) => chip(sc)).join('')}
          ${p.appearances.length > 8 ? chip('+' + (p.appearances.length - 8), 'chip-ghost') : ''}
        </div>
      </td>
      <td>
        <div class="cell-contact">
          ${p.contact_email ? escapeHtml(p.contact_email) : '<span class="muted">No known contact</span>'}
          ${p.contact_email_source === 'gemini_grounding'
            ? '<span class="chip chip-ghost">Found via Gemini</span><span class="muted">Double-check before sending</span>'
            : (p.contact_verified ? '' : '<span class="muted">Double-check before sending</span>')}
        </div>
      </td>
      <td>
        ${outreachEnabled
          ? `<button class="btn-mini" data-brand="${escapeHtml(p.brand)}" data-job="${escapeHtml(d.job_id)}">Draft email</button>`
          : '<span class="muted">Outreach off</span>'}
      </td>
    </tr>`).join('');
  const html = `
    ${integrityBanner}
    ${specCards}
    <div class="table-wrap">
      <table class="table">
        <thead><tr><th>Brand</th><th>Product</th><th>Appearances</th><th>Contact</th><th>Action</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
    ${productResolutionsBlock(d)}`;
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
  const rejected = os.rejected || [];
  const skipped = os.skipped_counts || {};
  const skipNote = (skipped.below_min_confidence || skipped.duplicate_hash)
    ? `Skipped: ${skipped.below_min_confidence || 0} below the minimum confidence · ${skipped.duplicate_hash || 0} duplicate crops`
    : '';
  const gateInfo = `Minimum confidence ${Math.round((os.min_confidence || 0) * 100)}% · minimum crop ${escapeHtml(String(os.min_crop_area))} px² · max aspect ${escapeHtml(String(os.max_crop_aspect))}:1`;

  if (!os.available) {
    return `
      <div class="empty" style="text-align:left">
        <div class="note-title">Unidentified brands aren't searchable here yet</div>
        ${escapeHtml(os.reason || 'No reverse-image-search backend is available on this deployment.')}
        <div class="muted" style="margin-top:4px">You can still review unidentified logos scene by scene in the Scenes tab.</div>
      </div>`;
  }

  const rejectedRows = rejected.length ? `
    <div class="card">
      <div class="card-head"><div class="card-title">Skipped areas (${rejected.length})</div><span class="card-sub" style="text-transform:none;letter-spacing:0.02em">${gateInfo}</span></div>
      <div class="card-body">
        ${rejected.map((r) => `
          <div class="openset-state ${r.reason === 'too_small' ? 'openset-state--too-small' : 'openset-state--rejected-shape'}">
            <span>Frame ${r.frame_index} — ${r.reason === 'too_small' ? 'too small to search' : 'banner-shaped, skipped'}</span>
            <span class="openset-state-dims">${r.width}×${r.height} px · aspect ${r.aspect}:1 · confidence ${pct(r.confidence)}</span>
          </div>`).join('')}
      </div>
    </div>` : '';

  const skipRow = skipNote ? `<div class="muted" style="margin-top:8px">${escapeHtml(skipNote)}</div>` : '';

  if (!cands.length) {
    return `
      <div class="empty" style="text-align:left">
        <div class="note-title">No unidentified brands matched a known logo</div>
        The search ran and nothing cleared the gate (${gateInfo}). Any unresolved logos are still listed scene by scene.
      </div>
      ${skipRow}
      ${rejectedRows}`;
  }

  const head = `<div class="card-head" style="margin-bottom:16px"><span class="card-sub">Unidentified brand candidates — matched with a real reverse-image search (${escapeHtml(String(os.backend))}) plus logo.dev validation · ${os.resolved} verified. Candidates are lower-trust evidence, never confirmed appearances.</span></div>`;
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
      .map((r) => chip(r.title, 'chip--object'))
      .join('');
    const validation = c.logo_dev_validation || {};
    const named = !!c.candidate_name;
    return `
      <div class="card">
        <div class="card-head">
          <div class="card-title">${named ? escapeHtml(c.candidate_name) : '<span class="openset-state openset-state--unresolved" style="padding:2px 10px">Unidentified crop</span>'}</div>
          <div class="tag ${c.status === 'candidate_verified' ? 'tag-accent' : ''}"><span class="tag-label">Status</span> ${escapeHtml(c.status)}</div>
        </div>
        <div class="card-body">
          <div class="scene-chips">
            <span class="chip chip-ghost">Frame ${c.frame_index}</span>
            ${metaChip('Detection confidence', pct(c.confidence), 'DETECTOR BOX CONFIDENCE', 'chip-ghost')}
            <span class="chip chip-ghost">logo.dev ${escapeHtml(String(validation.status || 'n/a'))}</span>
            ${validation.domain ? `<span class="chip chip-ghost">${escapeHtml(validation.domain)}</span>` : ''}
          </div>
          ${c.search_error ? `<div class="openset-state openset-state--too-small" style="margin-top:8px">${escapeHtml(c.search_error)}</div>` : ''}
          ${tags ? `<div class="scene-objects">Engine tags: ${tags}</div>` : ''}
          ${trail ? `<div class="scene-objects">Source links: ${trail}</div>` : ''}
          ${c.crop_url ? `<div class="scene-objects"><a class="chip chip-ghost" href="${escapeHtml(c.crop_url)}" target="_blank" rel="noopener">View crop</a></div>` : ''}
        </div>
      </div>`;
  }).join('');
  return head + rows + rejectedRows + skipRow;
}

function recommendPanel(d) {
  const recs = d.recommendations || [];
  if (!recs.length) {
    return `<div class="empty">No opportunities yet.<br><br>Try analysing a video with on-screen brand evidence.<br><a href="#/analyse">Analyse a video</a></div>`;
  }
  const head = `<div class="card-head" style="margin-bottom:16px"><span class="card-sub">Ranked collaboration opportunities &middot; ${recs.length} brands &middot; direct matches (on-screen evidence) first, then suggested matches (brand knowledge)</span></div>`;
  const cards = recs.map((r, i) => `
    <div class="card rec-card">
      <div class="rec-rank">${String(i + 1).padStart(2, '0')}</div>
      <div class="rec-main">
        <div class="card-head">
          <div class="card-title">${escapeHtml(r.brand)}</div>
          <div class="chips">
            <span class="chip ${r.type === 'DIRECT' ? 'chip-accent' : ''}">${escapeHtml(friendlyRecType(r.type))}</span>
            ${metaChip('Fit', pct(r.score), 'RECOMMENDATION CONFIDENCE', 'chip-ghost')}
          </div>
        </div>
        <div class="card-body" style="font-size:var(--text-sm)">
          ${escapeHtml(r.product || r.brand)} — ${escapeHtml(r.category || 'General')}${r.appearances ? ' · ' + r.appearances + ' appearances' : ' · never on screen'}
        </div>
        <div class="rec-reasons">
          ${r.reasons.map((reason) => chip(reason)).join('')}
        </div>
        <div class="rec-contact" style="margin-top:var(--space-2)">
          ${r.hr_emails && r.hr_emails.length
            ? `<span class="rec-contact-label">Contact:</span> ${r.hr_emails.slice(0, 3).map((e) => metaChip('HR', e.replace(/@/g, ' @ '), 'GEMINI-GROUNDED EMAIL · UNVERIFIED', 'chip-ghost')).join('')}${r.hr_emails.length > 3 ? `<span class="chip chip-ghost">+${r.hr_emails.length - 3} more HR</span>` : ''}`
            : (r.gemini_emails && r.gemini_emails.length
              ? `<span class="rec-contact-label">Contact:</span> ${r.gemini_emails.slice(0, 4).map((e) => metaChip(e.type === 'hr' ? 'HR' : 'Contact', e.email.replace(/@/g, ' @ '), 'GEMINI-GROUNDED EMAIL · UNVERIFIED', 'chip-ghost')).join('')}${r.gemini_emails.length > 4 ? `<span class="chip chip-ghost">+${r.gemini_emails.length - 4} more</span>` : ''}`
              : (r.contact_email
                ? `<span class="rec-contact-label">Contact:</span> <span class="chip chip-ghost">${escapeHtml(r.contact_email)}${r.contact_email_source === 'gemini_grounding' ? ' · Gemini' : ''}</span>`
                : `<span class="rec-contact-label">Contact:</span> <span class="chip chip-ghost">No known contact</span>`))}
        </div>
      </div>
    </div>`).join('');
  const wrap = document.createElement('div');
  wrap.innerHTML = head + cards;
  return wrap.innerHTML;
}

function adsPanel(d) {
  if (!d.ads || !d.ads.length) {
    return `<div class="empty">No spoken brand mentions were found in the audio.</div>`;
  }
  const head = `<div class="card-head" style="margin-bottom:16px"><span class="card-sub">${d.ads.length} spoken brand mentions, transcribed from the audio</span></div>`;
  const rows = d.ads.map((a) => `
    <div class="card">
      <div class="card-head">
        <div class="card-title">${escapeHtml(a.brand)} × ${escapeHtml(d.creator || 'this channel')}</div>
        <span class="chip chip-accent">${escapeHtml(a.type)}</span>
      </div>
      <div class="card-body" style="font-size:var(--text-sm)">
        ${escapeHtml(a.product || a.brand)} — ${escapeHtml(a.category || 'General')}
      </div>
      <div class="scene-objects">
        ${a.start_time != null ? chip(`Mentioned at ${fmtTime(a.start_time)}`, 'chip-ghost') : ''}
        ${a.start_time != null ? `<a class="chip chip-accent clip-chip" data-ts="${Number(a.start_time).toFixed(2)}" href="#/pipeline/${encodeURIComponent(d.job_id)}?tab=scenes&clip=${Number(a.start_time).toFixed(2)}">&blacktriangleright; Jump to clip</a>` : ''}
        ${a.scenes.slice(0, 12).map((sc) => chip(sc)).join('')}
      </div>
    </div>`).join('');
  return head + rows;
}

function outreachPanel(d) {
  if (!outreachEnabled) {
    return `<div class="error-box"><div class="error-title">Outreach is off</div>${escapeHtml(outreachDisabledReason)}</div>`;
  }
  const brands = outreachBrands(d);
  if (!brands.length) {
    return `<div class="empty">No brands ready to contact yet.<br><br>Try analysing a video with on-screen brand evidence.</div>`;
  }
  return `<div class="empty" style="padding:var(--space-lg)">${brands.length} brand opportunities are ready.<br><br><a href="#/outreach/${encodeURIComponent(d.job_id)}">Open the outreach editor</a></div>`;
}

// Outreach is driven by ranked Layer 3 recommendations, not the (deliberately
// empty) dashboard `products` table. Enriched server-side with catalog contact.
const outreachBrands = (d) => (d.recommendations || []);

/* ── Outreach page ────────────────────────────────────────── */

let outreachState = { id: null, data: null, brand: null };
let outreachEnabled = false;
let outreachDisabledReason = 'Draft email is off — the outreach data path hasn\u2019t passed its integrity review yet.';

async function renderOutreach(id, params) {
  const root = $('#outreach-root');
  if (!id) { root.innerHTML = errorHtml('No project was referenced'); return; }

  root.innerHTML = loaderHtml('Loading your brands');
  outreachState.id = id;
  outreachState.brand = params.get('brand');

  try {
    const data = await api(`/api/pipeline/${encodeURIComponent(id)}`);
    outreachState.data = data;
    outreachEnabled = data.outreach_enabled === true;
    outreachDisabledReason = data.outreach_reason || outreachDisabledReason;
    if (!outreachEnabled) {
      root.innerHTML = `<div class="error-box"><div class="error-title">Outreach is off</div>${escapeHtml(outreachDisabledReason)}<br><br><a href="#/pipeline/${encodeURIComponent(id)}">Back to the dashboard</a></div>`;
      return;
    }
    renderOutreachEditor(root, data);
  } catch (err) {
    root.innerHTML = errorHtml(err.message);
  }
}

function renderOutreachEditor(root, d) {
  const products = outreachBrands(d);
  const active = outreachState.brand || (products[0] && products[0].brand);

  const brandsList = products.length
    ? products.map((p) => `
        <div class="brand-row${p.brand === active ? ' is-active' : ''}" data-brand="${escapeHtml(p.brand)}">
          <div class="brand-name">${escapeHtml(p.brand)}<span class="brand-type"> ${escapeHtml(friendlyRecType(p.type || ''))}</span></div>
          <div class="brand-product">${escapeHtml(p.product || p.brand)} · ${escapeHtml(p.category || 'General')} · ${p.appearances ? p.appearances + ' appearances' : 'never on screen'}</div>
          <div class="brand-contact">${p.contact_email
            ? escapeHtml(p.contact_email) + (p.contact_email_source === 'gemini_grounding'
              ? ' <span class="chip chip-ghost">Found via Gemini · unverified</span>' : '')
            : 'No known contact — add one manually'}</div>
          ${(p.hr_emails || []).length
            ? `<div class="brand-product">HR: ${p.hr_emails.map((e) => escapeHtml(e)).join(' · ')}</div>`
            : ''}
        </div>`).join('')
    : `<div class="empty">No recommended brands</div>`;

  root.innerHTML = `
    <div class="outreach-brands">${brandsList}</div>
    <div class="outreach-editor">
      <div class="editor-toolbar">
        <div class="editor-target">
          <span class="editor-label">Target email</span>
          <input class="editor-target-name" id="target-name" value="" spellcheck="false" placeholder="Auto-filled from the brand catalog — editable">
          <span class="editor-hint" id="target-hint"></span>
        </div>
        <div class="editor-actions">
          <button class="btn btn-sm" id="btn-generate">Generate draft</button>
          <button class="btn btn-sm" id="btn-forward">Forward</button>
        </div>
      </div>
      <div class="editor-subject">
        <span class="editor-label">Subject</span>
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
      hint.textContent = p.contact_email_source === 'gemini_grounding'
        ? `Found via Gemini search — not verified, confirm before sending${p.contact_website ? ' · ' + p.contact_website : ''}`
        : `Auto-filled from the brand catalog${p.contact_verified ? '' : ' · unverified, confirm before sending'}${p.contact_website ? ' · ' + p.contact_website : ''}`;
    } else {
      $('#target-name').value = '';
      hint.textContent = 'No known contact — enter the brand email manually';
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
      foot.textContent = 'Enter a target email before generating';
      return;
    }
    generateBtn.disabled = true;
    generateBtn.textContent = 'Generating…';
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
      foot.textContent = `Draft generated · to ${escapeHtml(res.target)} · ${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
    } catch (err) {
      foot.textContent = `Generate failed — ${err.message}`;
    } finally {
      generateBtn.disabled = false;
      generateBtn.textContent = 'Generate draft';
    }
  });

  forwardBtn.addEventListener('click', async () => {
    if (!outreachState.brand) return;
    if (!body.textContent.trim()) {
      foot.textContent = 'Generate a draft before forwarding';
      return;
    }
    forwardBtn.disabled = true;
    forwardBtn.textContent = 'Forwarding…';
    try {
      const res = await api('/api/outreach/forward', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_id: d.job_id, brand: outreachState.brand }),
      });
      foot.textContent = `Forwarded to ${escapeHtml(res.target)} · ${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
      foot.classList.add('forwarded');
    } catch (err) {
      foot.textContent = `Forward failed — ${err.message}`;
    } finally {
      forwardBtn.disabled = false;
      forwardBtn.textContent = 'Forward';
    }
  });
}

/* ── Login page ───────────────────────────────────────────── */

async function renderLogin() {
  const root = $('#login-root');
  try {
    const me = await api('/api/me');
    if (me.authenticated === true) {
      root.innerHTML = `
        <div class="auth-card" style="max-width:480px">
          <h2 class="page-title">Signed in</h2>
          <p class="page-copy">You're signed in as <strong>${escapeHtml(me.user)}</strong>.</p>
          <div class="auth-actions">
            <a class="btn btn-primary" href="#/pipeline">Open your projects</a>
            <a class="btn" id="btn-logout" href="#/">Log out</a>
          </div>
        </div>`;
      $('#btn-logout').addEventListener('click', async (e) => {
        e.preventDefault();
        try { await api('/api/logout', { method: 'POST' }); } catch (err) { /* noop */ }
        setToken(null);
        location.hash = '#/login';
      });
      return;
    }
    if (me.auth_enabled !== true) {
      root.innerHTML = `<div class="auth-card" style="max-width:480px">
        <h2 class="page-title">Open access</h2>
        <p class="page-copy">No login is required in this deployment.<br><a href="#/pipeline">Open your projects</a></p>
      </div>`;
      return;
    }
    root.innerHTML = `
      <div class="auth-card" style="max-width:480px">
        <h2 class="page-title">Sign in</h2>
        <p class="page-copy">Admin-only access for this deployment.</p>
        <form id="login-form" class="analyse-form">
          <div class="field">
            <label class="field-label" for="login-user">Username</label>
            <input id="login-user" class="input" type="text" autocomplete="username" spellcheck="false">
          </div>
          <div class="field">
            <label class="field-label" for="login-pass">Password</label>
            <input id="login-pass" class="input" type="password" autocomplete="current-password">
          </div>
          <button class="btn btn-primary btn-block" type="submit">Sign in</button>
        </form>
        <div id="login-status" class="status-line" hidden></div>
      </div>`;
    $('#login-form').addEventListener('submit', async (e) => {
      e.preventDefault();
      const status = $('#login-status');
      status.hidden = false;
      status.textContent = 'Signing you in…';
      try {
        const res = await api('/api/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            username: $('#login-user').value.trim(),
            password: $('#login-pass').value,
          }),
        });
        setToken(res.token);
        location.hash = '#/pipeline';
      } catch (err) {
        status.textContent = `Sign-in failed — ${escapeHtml(err.message)}`;
        status.className = 'status-line error';
      }
    });
  } catch (err) {
    root.innerHTML = errorHtml(err.message);
  }
}

/* ── Insights page ────────────────────────────────────────── */

async function renderInsights(id, params) {
  const root = $('#insights-root');
  if (!id) return renderInsightsIndex(root);

  root.innerHTML = loaderHtml('Loading insights');
  try {
    const data = await api(`/api/insights/${encodeURIComponent(id)}`);
    renderInsightsDetail(root, data);
  } catch (err) {
    root.innerHTML = errorHtml(err.message);
  }
}

async function renderInsightsIndex(root) {
  root.innerHTML = loaderHtml('Loading your projects');
  try {
    // Prefer the persisted archive (survives restarts + seed data); fall back
    // to the live job registry when the archive is empty or unreachable.
    let jobs = [];
    try {
      const archive = await api('/api/jobs/archive');
      jobs = archive.jobs || [];
    } catch (e) { /* fall through to live jobs */ }
    if (!jobs.length) {
      try {
        const live = await api('/api/jobs');
        jobs = live.jobs || [];
      } catch (e) { /* empty */ }
    }
    if (!jobs.length) {
      root.innerHTML =
        `<h2 class="page-title" style="margin-bottom:24px">Insights</h2>` +
        `<div class="empty">No projects yet — analyse a video to start building brand insights.<br><br><a href="#/analyse">Analyse a video</a></div>`;
      return;
    }
    const rows = jobs.map((j, i) => `
      <a class="job-row" href="#/insights/${encodeURIComponent(j.job_id)}">
        <span class="job-meta" style="min-width:28px;text-align:right;font-family:var(--font-mono)">${String(i + 1).padStart(2, '0')}</span>
        <span class="job-title">${escapeHtml(j.title || 'Untitled video')}</span>
        <span class="job-meta">${escapeHtml(j.creator || '')}</span>
        <span class="job-meta">${escapeHtml(j.job_id)}</span>
        <span class="job-meta">${escapeHtml(j.status || '')}</span>
      </a>`).join('');
    root.innerHTML = `<h2 class="page-title" style="margin-bottom:24px">Insights</h2><div style="display:flex;gap:var(--space-3);margin-bottom:12px"><span class="k-label">Index</span><span class="k-label" style="flex:1">Title</span><span class="k-label">Creator</span><span class="k-label">Project</span><span class="k-label">Status</span></div>` + rows;
  } catch (err) {
    root.innerHTML = errorHtml(err.message);
  }
}

function renderInsightsDetail(root, d) {
  const profile = d.creator_profile || {};
  const tallies = profile.brand_tallies || {};
  const tallyRows = Object.entries(tallies)
    .sort((a, b) => b[1] - a[1])
    .map(([brand, n]) => `
      <div class="insight-row">
        <span class="chip chip-accent">${escapeHtml(brand)}</span>
        <span class="chip chip-ghost">${n}×</span>
      </div>`).join('') || `<div class="empty">No brand tallies</div>`;
  const catRows = Object.entries(profile.categories || {})
    .sort((a, b) => b[1] - a[1])
    .map(([cat, n]) => `${escapeHtml(cat)} <span class="muted">×${n}</span>`).join(' · ') || '—';
  const mem = d.brand_memory || {};
  const memoryBrands = (mem.brands || []).map((b) => chip(escapeHtml(b), 'chip-ghost')).join('') || '<span class="muted">Empty</span>';
  const resolutions = (mem.indirect_resolutions || []).map((r) => `
    <div class="insight-row">
      <span class="chip chip-accent">${escapeHtml(r.brand || '?')}</span>
      <span class="chip chip-ghost">"${escapeHtml(r.reference || '')}"</span>
      ${r.match_score != null ? metaChip('Match', pct(r.match_score), 'MATCH SCORE', 'chip-ghost') : ''}
      ${r.reason ? `<span class="muted">${escapeHtml(r.reason)}</span>` : ''}
    </div>`).join('') || '<div class="empty">No indirect resolutions</div>';
  const recs = d.recommendations || [];
  const recRows = recs.map((r, i) => `
    <div class="card rec-card">
      <div class="rec-rank">${String(i + 1).padStart(2, '0')}</div>
      <div class="rec-main">
        <div class="card-head">
          <div class="card-title">${escapeHtml(r.brand)}</div>
          <div class="chips">
            <span class="chip ${r.type === 'DIRECT' ? 'chip-accent' : ''}">${escapeHtml(friendlyRecType(r.type))}</span>
            ${metaChip('Fit', pct(r.score), 'RECOMMENDATION CONFIDENCE', 'chip-ghost')}
          </div>
        </div>
        <div class="rec-reasons">${(r.reasons || []).map((reason) => chip(escapeHtml(reason))).join('')}</div>
      </div>
    </div>`).join('') || '<div class="empty">No recommendations</div>';

  root.innerHTML = `
    <h2 class="page-title page-title-huge">${escapeHtml(d.job_id)}</h2>
    <div class="meta-row">
      <span class="tag"><span class="tag-label">Creator</span> ${escapeHtml(profile.handle || profile.creator_id || '—')}</span>
      ${profile.followers != null ? `<span class="tag"><span class="tag-label">Followers</span> ${escapeHtml(profile.followers.toLocaleString())}</span>` : ''}
      ${profile.engagement_rate != null ? `<span class="tag"><span class="tag-label">Engagement</span> ${(profile.engagement_rate * 100).toFixed(2)}%</span>` : ''}
      ${profile.dominant_category ? `<span class="tag"><span class="tag-label">Dominant niche</span> ${escapeHtml(profile.dominant_category)}</span>` : ''}
    </div>

    <div class="insight-grid">
      <div class="card">
        <div class="card-head"><div class="card-title">Content niche</div></div>
        <div class="card-body">${catRows}</div>
      </div>
      <div class="card">
        <div class="card-head"><div class="card-title">Brand tallies</div></div>
        <div class="card-body">${tallyRows}</div>
      </div>
    </div>

    <div class="insight-grid">
      <div class="card">
        <div class="card-head"><div class="card-title">Brands seen across videos <span class="muted">(${mem.size != null ? escapeHtml(String(mem.size)) : '—'} brands)</span></div></div>
        <div class="card-body insight-chips">${memoryBrands}</div>
      </div>
      <div class="card">
        <div class="card-head"><div class="card-title">Indirect references</div></div>
        <div class="card-body">${resolutions}</div>
      </div>
    </div>

    <h2 class="page-title" style="margin-bottom:16px">Opportunities</h2>
    <div class="insight-rec-list">${recRows}</div>

    <div class="meta-row" style="margin-top:24px">
      <a class="btn" href="#/pipeline/${encodeURIComponent(d.job_id)}">Open the full dashboard</a>
    </div>`;
}

/* ── Boot ─────────────────────────────────────────────────── */

router();
