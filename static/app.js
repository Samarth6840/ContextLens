/* ContextLens — frontend app logic (vanilla JS, hash router) */

const $ = (sel, root) => (root || document).querySelector(sel);
const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));

const TOKEN_KEY = 'adscene_token';
const getToken = () => localStorage.getItem(TOKEN_KEY) || null;
const setToken = (t) => { t ? localStorage.setItem(TOKEN_KEY, t) : localStorage.removeItem(TOKEN_KEY); };

const el = {
  landing: $('#view-landing'),
  analyse: $('#view-analyse'),
  pipeline: $('#view-pipeline'),
  outreach: $('#view-outreach'),
  insights: $('#view-insights'),
  login: $('#view-login'),
};

function escapeHtml(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

// A number that is already numeric needs no escaping; this is here so the
// diagnostics renderer can call it on any field without a type check at
// every use site. escapeHtml() alone is correct for strings; this only
// decides whether a value is worth the round trip.
function escapeHtmlNum(n) {
  return Number.isFinite(Number(n)) ? String(n) : escapeHtml(n);
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
  const tip = metric ? ` title="${escapeHtml(metric)}"` : '';
  return `<span class="chip${variant ? ' ' + variant : ''}"${tip}>${escapeHtml(label)}${conf ? ' ' + conf : ''}</span>`;
}

// Reason strings come from the server shouted in caps. De-shout those, but
// leave normal prose alone so proper nouns survive ("NVIDIA GeForce"). Judged
// by the ratio of upper-case letters rather than an equality test, which misses
// mixed strings like "LOGO DETECTED — 1 appearance(s)".
function sentence(text) {
  const t = String(text == null ? '' : text).trim();
  if (!t) return t;
  const letters = t.replace(/[^a-z]/gi, '');
  if (!letters || letters.length < 3) return t;
  const upper = (t.match(/[A-Z]/g) || []).length;
  if (upper / letters.length <= 0.6) return t;
  return t.charAt(0) + t.slice(1).toLowerCase();
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
  if (page === 'landing' || page === '') showView('landing');
  else if (page === 'analyse') showView('analyse');
  else if (page === 'pipeline') { showView('pipeline'); renderPipeline(id, params); }
  else if (page === 'outreach') { showView('outreach'); renderOutreach(id, params); }
  else if (page === 'insights') { showView('insights'); renderInsights(id, params); }
  else if (page === 'login') { showView('login'); renderLogin(); }
  else { location.hash = '#/'; }
  refreshNavAuth();
}

// Sidebar / back-button wiring
['home', 'outreach', 'insights'].forEach((key) => {
  const btn = $(`#btn-back-${key === 'home' ? 'home' : key}`);
  if (btn) btn.addEventListener('click', () => { location.hash = key === 'home' ? '#/pipeline' : `#/${key}`; });
});
window.addEventListener('hashchange', router);
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
    if (authed) link.setAttribute('data-user', me.user || '');
  } catch (e) { /* keep default label */ }
}

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
  // Long videos legitimately take 30-90+ minutes (uncapped Whisper + BEATs on
  // MPS), and the server keeps working after the tab gives up — but a page
  // that freezes at a fixed ceiling reads as a hang. So: poll indefinitely
  // while the job runs; give up only after a STREAK of consecutive network
  // failures (server down/restarted), never on elapsed time. A single 502 or
  // laptop sleep no longer kills the wait.
  const POLL_MS = 1500;
  const MAX_CONSECUTIVE_FAILURES = 12; // ~18s of unreachable server before bailing
  const startedAt = Date.now();
  let timerId = null;
  let consecutiveFailures = 0;

  const fmtElapsed = (ms) => {
    const total = Math.max(0, Math.floor(ms / 1000));
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const s = total % 60;
    return h > 0 ? `${h}h ${String(m).padStart(2, '0')}m ${String(s).padStart(2, '0')}s`
      : m > 0 ? `${m}m ${String(s).padStart(2, '0')}s`
      : `${s}s`;
  };

  // Elapsed-time line between the progress bar and the live feed. Injected so
  // the markup stays server-rendered and other pages are untouched.
  let elapsedEl = document.getElementById('proc-elapsed');
  if (!elapsedEl && procFeedEl && procFeedEl.parentElement) {
    elapsedEl = document.createElement('div');
    elapsedEl.id = 'proc-elapsed';
    elapsedEl.className = 'processing-stage';
    elapsedEl.style.marginTop = '4px';
    procFeedEl.parentElement.insertBefore(elapsedEl, procFeedEl);
  }

  const stopPolling = () => { if (timerId !== null) { clearInterval(timerId); timerId = null; } };
  const finish = () => {
    stopPolling();
    btn.disabled = false;
    btn.textContent = 'Analyse video';
  };

  const renderStage = (stageText, pct) => {
    if (procStageEl) procStageEl.textContent = stageText;
    procStep(stageText);
    procProgress(pct);
    if (elapsedEl) elapsedEl.textContent = `Elapsed: ${fmtElapsed(Date.now() - startedAt)} — long videos can take a while`;
  };

  const fail = (message) => {
    finish();
    if (elapsedEl) elapsedEl.remove();
    procError(message);
  };

  const tick = async () => {
    // A throw inside tick must never leave the interval spinning forever with
    // an unhandled rejection — every exit path either reschedules implicitly
    // (interval) or tears the poll down deliberately.
    try {
      await tickInner();
    } catch (err) {
      stopPolling();
      fail(`Status poll failed — ${escapeHtml(err && err.message || 'unknown error')}`);
    }
  };

  const tickInner = async () => {
    let job;
    try {
      job = await api(`/api/analyse/${encodeURIComponent(jobId)}`);
      consecutiveFailures = 0;
    } catch (err) {
      // Transient errors (laptop sleep, server busy mid-GPU-work, brief 502)
      // must not abandon a job that is still running server-side. Only a
      // sustained streak ends the poll — and then we say why.
      consecutiveFailures += 1;
      if (consecutiveFailures >= MAX_CONSECUTIVE_FAILURES) {
        fail(`Lost contact with the server (${err.message}). The analysis may still be running — reload the page to check.`);
      }
      return;
    }
    if (job.status === 'done') {
      finish();
      renderStage('Complete — loading your results', 100);
      procStep('COMPLETE');
      if (elapsedEl) elapsedEl.textContent = `Done in ${fmtElapsed(Date.now() - startedAt)}`;
      location.hash = `#/pipeline/${encodeURIComponent(jobId)}`;
      return;
    }
    if (job.status === 'error') {
      fail(`Analysis failed — ${escapeHtml(job.error || 'something went wrong')}`);
      return;
    }
    // Animate the bar toward a perceived ceiling while running; the stage
    // name + feed + elapsed line carry the real progress signal.
    const mins = (Date.now() - startedAt) / 60000;
    const pct = mins < 1 ? 15 + mins * 60 : Math.min(95, 75 + Math.log2(1 + mins));
    renderStage(String(job.stage || 'Working…'), pct);
    procFeed(job.feed);
    setStatus(`<span class="dot blink">▮</span> ${escapeHtml(job.stage || 'Working…')}… (${fmtElapsed(Date.now() - startedAt)})`);
  };

  tick();
  timerId = setInterval(tick, POLL_MS);
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

function jobRow(j, i, hrefFor) {
  return `
      <a class="job-row" href="${hrefFor(j.job_id)}">
        <span class="job-meta" style="min-width:28px;text-align:right;font-family:var(--font-mono)">${String(i + 1).padStart(2, '0')}</span>
        <span class="job-title">${escapeHtml(j.title || 'Untitled video')}</span>
        <span class="job-meta">${escapeHtml(j.creator || '')}</span>
        <span class="job-meta">${escapeHtml(j.job_id)}</span>
        <span class="job-meta">${escapeHtml(j.status || '')}${j.stage && j.status === 'running' ? ' &middot; ' + escapeHtml(j.stage) : ''}</span>
      </a>`;
}

const JOB_LIST_HEADERS = `<div style="display:flex;gap:var(--space-3);margin-bottom:12px"><span class="k-label">Index</span><span class="k-label" style="flex:1">Title</span><span class="k-label">Creator</span><span class="k-label">Project</span><span class="k-label">Status</span></div>`;

async function renderJobIndex(root) {
  root.innerHTML = loaderHtml('Loading your projects');
  try {
    const { jobs } = await api('/api/jobs');
    if (!jobs.length) {
      root.innerHTML =
        `<div class="empty">No projects yet — start by analysing a video to find your first opportunities.<br><br><a href="#/analyse">Analyse a video</a></div>`;
      return;
    }
    const rows = jobs.map((j, i) => jobRow(j, i, (id) => `#/pipeline/${encodeURIComponent(id)}`)).join('');
    root.innerHTML = JOB_LIST_HEADERS + rows;
  } catch (err) {
    root.innerHTML = errorHtml(err.message);
  }
}

function uniqueBrands(d) {
  const brands = new Set();
  (d.products || []).forEach((p) => brands.add(p.brand));
  (d.recommendations || []).forEach((r) => brands.add(r.brand));
  (d.brands || []).forEach((b) => brands.add(typeof b === 'string' ? b : b.name));
  return brands;
}

function summaryStrip(d) {
  // The figures, printed once. Everything else on this screen refers back to
  // these numbers instead of repeating them.
  const brands = uniqueBrands(d).size;
  const opps = (d.recommendations || []).length;
  const scenes = (d.scenes || []).length;
  const logos = (d.scenes || []).reduce((a, s) => a + (s.logos || []).length, 0);
  const cell = (val, label) => `<div class="figure"><span class="figure-val">${val}</span><span class="figure-label">${escapeHtml(label)}</span></div>`;
  return `
    <div class="figure-line">
      ${cell(brands, brands === 1 ? 'brand' : 'brands')}
      ${cell(opps, opps === 1 ? 'opportunity' : 'opportunities')}
      ${cell(scenes, scenes === 1 ? 'scene' : 'scenes')}
      <div class="figure figure-quiet">
        <span class="figure-val">${d.num_frames != null ? d.num_frames : '—'}</span>
        <span class="figure-label">frames at ${(d.video_fps || 0).toFixed(1)} fps, ${logos} logo ${logos === 1 ? 'box' : 'boxes'}</span>
      </div>
    </div>`;
}

// Plain-language "what we found" summary. It states the verdict and the single
// strongest signal — never a number, which lives once in the figure line above.
const EVIDENCE_LABELS = {
  audio_event: 'Audio events',
  logo_detected: 'Logo detection',
  ocr_hit: 'On-screen text',
  product_retrieval: 'Product catalog',
  scene_context: 'Scene context',
  speech_mention: 'Spoken mentions',
  visual_product_match: 'Visual product match',
};
const evidenceLabel = (k) => EVIDENCE_LABELS[k] || String(k).replace(/_/g, ' ').replace(/^./, (c) => c.toUpperCase());

// Sources that carry the read: a source with no strength or no weight is dead
// weight in both the bar and the legend, so it is left out of both.
function liveEvidence(d) {
  return Object.entries(d.evidence_breakdown || {})
    .filter(([, e]) => (e.strength || 0) > 0 && (e.weight || 0) > 0);
}

function whatFoundPanel(d) {
  const evids = liveEvidence(d);
  const strongest = evids.slice().sort((a, b) => b[1].strength - a[1].strength)[0];
  const verdict = d.is_confident
    ? 'This read is confident'
    : 'Read this with care';
  const because = strongest
    ? `, carried mostly by ${evidenceLabel(strongest[0]).toLowerCase()}`
    : '';
  return `
    <div class="what-found">
      <div class="what-found-icon">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="9"/><path d="M12 8v4l3 2"/></svg>
      </div>
      <p>${verdict}${because}. The breakdown below shows what each signal contributed.</p>
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
    <h1 class="page-title page-title-huge">${escapeHtml(d.title)}</h1>
    <p class="project-line">
      <span>${escapeHtml(d.job_id)}</span>
      <span>${escapeHtml(fmtDuration(d.duration_sec))}</span>
      ${d.creator && d.creator !== 'UNKNOWN' ? `<span>${escapeHtml(d.creator)}</span>` : ''}
      ${d.creator_card && d.creator_card.handle ? `<span>@${escapeHtml(d.creator_card.handle)}</span>` : ''}
    </p>
    ${whatFoundPanel(d)}
    ${summaryStrip(d)}
    ${confidenceBar(d)}
    <div class="tabs">${tabs}</div>
    ${Object.keys(panels).map((key) => `<div class="tab-panel${key === activeTab ? ' is-active' : ''}" data-panel="${key}">${panels[key]}</div>`).join('')}
  `;

  // Bound here, after innerHTML: a listener attached to a detached node is lost
  // the moment its markup is serialised into the page.
  $$('.btn-mini', root).forEach((b) => b.addEventListener('click', () => {
    location.hash = `#/outreach/${encodeURIComponent(b.dataset.job)}?brand=${encodeURIComponent(b.dataset.brand)}`;
  }));

  $$('#pipeline-root .tab').forEach((btn) => {
    btn.addEventListener('click', () => {
      $$('#pipeline-root .tab').forEach((b) => b.classList.remove('is-active'));
      $$('#pipeline-root .tab-panel').forEach((p) => p.classList.remove('is-active'));
      btn.classList.add('is-active');
      $('#pipeline-root [data-panel="' + btn.dataset.tab + '"]').classList.add('is-active');
      history.replaceState(null, '', `#/pipeline/${encodeURIComponent(d.job_id)}?tab=${btn.dataset.tab}`);
      if (btn.dataset.tab === 'scenes' && !window.__sceneFilterBox) wireScenePanel(root, d);
      if (btn.dataset.tab === 'ads') wireAdsPanel(root);
    });
  });

  // Scene search/sort/filter controls + clip seek (Part C/D). The video element
  // 404s gracefully (no clip-player when no <video> can load) and the scenes
  // list still jumps + highlights on clip links.
  if (activeTab === 'scenes') wireScenePanel(root, d);
  wireAdsPanel(root);
}

function sceneThumbUrl(jobId, frameIndex) {
  return `/api/scene/${encodeURIComponent(jobId)}/${frameIndex}`;
}

/* ── Confidence bar (design: evidence strength is shown, not one opaque %) ── */

function confidenceBar(d) {
  const total = d.confidence;
  if (total == null) return '';
  const evids = liveEvidence(d);
  if (!evids.length) return '';
  const band = (v) => (v >= 0.75 ? 'high' : (v >= 0.40 ? 'mid' : 'low'));
  const sumW = evids.reduce((a, [, e]) => a + (e.weight || 0), 0) || 0;
  const fills = evids.map(([src, e]) => {
    const width = (e.weight / sumW) * 100;
    return `<span class="confidence-bar-fill" data-band="${band(e.strength)}" style="width:${width.toFixed(1)}%" title="${escapeHtml(evidenceLabel(src))}, ${(e.strength * 100).toFixed(0)}% strength, weight ${e.weight.toFixed(3)}"></span>`;
  }).join('');
  const remainder = sumW < 1
    ? `<span class="confidence-bar-unweighted" style="flex:1 1 0%" title="Unweighted — evidence weights sum to ${(sumW * 100).toFixed(0)}% of the score"></span>`
    : '';
  const status = d.is_confident ? 'confident' : (String(d.confidence_status || 'caution').toLowerCase());
  const keys = evids
    .slice()
    .sort((a, b) => b[1].strength - a[1].strength)
    .map(([src, e]) => `
      <span class="confidence-key"><span class="confidence-key-dot" data-band="${band(e.strength)}"></span>${escapeHtml(evidenceLabel(src))} <b>${(e.strength * 100).toFixed(0)}%</b></span>`)
    .join('');
  return `
    <div class="confidence-block">
      <div class="confidence-block-head">
        <span class="k-label">What the score is made of</span>
        <span class="mono">${(total * 100).toFixed(0)}%</span>
      </div>
      <div class="confidence-bar">${fills}${remainder}</div>
      <div class="confidence-legend" title="Bar width is each signal's weight in the score. The percentage is how strong that signal is.">${keys}</div>
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

const OBJECT_CHIP_CAP = 4;
const LOGO_CHIP_CAP = 6;
// Above this many logo boxes on one frame the detector is emitting a swarm, not
// a scene. OpenLogo measures 0.9 boxes/frame; LogoDet measured 73. The UI says so
// rather than letting a tidy chip row imply the frame is under control.
const LOGO_OVERLOAD = 20;

function groupLogos(logos) {
  /* Collapse repeats of one label into a single chip carrying its count.

    A frame the detector could not resolve shows the identical string
    "UNKNOWN BRAND" once per box. Printing 131 of them buries the scene under
    noise and answers nothing — the count is the information, not the repetition.
    Confidence is kept at the strongest instance, so the chip never claims more
    certainty than the best box on the frame supports.
  */
  const by = new Map();
  logos.forEach((o) => {
    const key = o.class_name || 'UNKNOWN BRAND';
    const prev = by.get(key);
    if (!prev) {
      by.set(key, { label: key, count: 1, confidence: o.confidence, metric: o.confidence_metric });
      return;
    }
    prev.count += 1;
    if ((o.confidence ?? 0) > (prev.confidence ?? 0)) {
      prev.confidence = o.confidence;
      prev.metric = o.confidence_metric;
    }
  });
  return Array.from(by.values()).sort((a, b) => b.count - a.count
    || (b.confidence ?? 0) - (a.confidence ?? 0));
}

function sceneChips(s) {
  const parts = [];
  const objects = s.objects || [];
  const logos = s.logos || [];
  const shown = objects.slice(0, OBJECT_CHIP_CAP);
  if (shown.length) {
    parts.push(shown.map((o) => metaChip(
      o.class_name,
      pct(o.confidence),
      'DETECTOR BOX CONFIDENCE',
      'chip--object',
    )).join(''));
  }
  if (logos.length) {
    const grouped = groupLogos(logos);
    if (logos.length > LOGO_OVERLOAD) {
      parts.push(metaChip(
        `${logos.length} logo boxes on this frame`,
        null,
        'The detector emitted far more boxes than a single frame should contain. '
        + 'Treat this frame as detector noise until a labelled audit says otherwise.',
        'chip--overload',
      ));
    }
    const shownLogos = grouped.slice(0, LOGO_CHIP_CAP);
    parts.push(shownLogos.map((g) => {
      const unknown = g.label === 'UNKNOWN BRAND';
      const metric = (g.metric || (unknown ? 'detector_box_confidence' : 'resolution_quality')).toUpperCase();
      const text = g.count > 1 ? `${g.label} ×${g.count}` : g.label;
      const tip = g.count > 1 ? `${metric} · ${g.count} BOXES OF THIS LABEL ON THIS FRAME` : metric;
      return metaChip(text, pct(g.confidence), tip, unknown ? 'chip--unknown' : 'chip--brand');
    }).join(''));
    const hiddenLogos = logos.length - shownLogos.reduce((n, g) => n + g.count, 0);
    if (hiddenLogos > 0) parts.push(chip(`+${hiddenLogos} more`, 'chip-ghost'));
  }
  const hidden = objects.length - shown.length;
  if (hidden > 0) parts.push(chip(`+${hidden} more`, 'chip-ghost'));
  return parts.join('') || chip('No objects detected', 'chip-ghost');
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
    ? `, ${unknown.length} unidentified logo spot${unknown.length === 1 ? '' : 's'} across every frame`
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
    <div class="scenes-bay">
      <div class="scenes-source">
        <video id="clip-player" class="clip-player" controls preload="metadata" src="/api/video/${encodeURIComponent(d.job_id)}"></video>
        ${audioBlock}
      </div>
      <div class="scenes-bin">
        <div class="card-head bin-head"><span class="card-sub">${d.scenes.length} scenes${unknownNote}. Click a frame to open the source image.</span></div>
        <div class="scene-toolbar">
          <input id="scene-search" class="input scene-search" type="search" placeholder="Search scenes (brand, object, time)…" value="${escapeHtml(sceneView.q)}" autocomplete="off">
          <button id="scene-filter-btn" class="btn" type="button" aria-haspopup="true" aria-expanded="false">Filter${filterActive ? ' \u2713' : ''}</button>
          ${sceneFilterDropdown(d, sceneView)}
          <select id="scene-sort" class="input scene-select" aria-label="Sort scenes">${sortOpts}</select>
          <button id="scene-filter-reset" class="btn" type="button">Reset</button>
        </div>
        <div id="scene-meta" class="muted bin-count"></div>
        <div id="scene-rows" class="bin-rows">${renderSceneRows(d)}</div>
      </div>
    </div>`;
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
      <div class="filter-group-label">Brands<span class="filter-group-value">${brandSel}</span></div>
      ${brandItems || `<div class="filter-item-empty">No brands match</div>`}
      <div class="filter-group-label">Object types<span class="filter-group-value">${objSel}</span></div>
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
  return html;
}

// "TSMC (TAIWAN SEMICONDUCTOR MANUFACTURING COMPANY LIMITED)" -> short + legal tail
function splitCandidateName(name) {
  const raw = String(name == null ? '' : name).trim();
  const m = raw.match(/^([^(]+?)\s*\((.+)\)$/);
  if (!m) return { short: raw, legal: '' };
  const legal = m[2].trim();
  // Registries return legal names in caps; sentence case reads better.
  return { short: m[1].trim(), legal: legal === legal.toUpperCase() ? titleCase(legal) : legal };
}

// Grounding redirects are 200-char blobs; the title is already the readable host.
function titleCase(text) {
  return text.toLowerCase().replace(/(^|\s)([a-z])/g, (m, sp, ch) => sp + ch.toUpperCase());
}

function sourceLabel(result) {
  const t = String(result.title == null ? '' : result.title).trim();
  if (t && t.length <= 60) return t;
  try {
    return new URL(result.url).hostname.replace(/^www\./, '');
  } catch (_) {
    return t ? t.slice(0, 60) + '…' : 'source';
  }
}

function openSetPanel(d) {
  const os = d.open_set || {};
  const cands = os.candidates || [];
  const rejected = os.rejected || [];
  const skipped = os.skipped_counts || {};
  const gate = `gate: ${Math.round((os.min_confidence || 0) * 100)}% confidence, ${escapeHtml(String(os.min_crop_area))} px&sup2;, ${escapeHtml(String(os.max_crop_aspect))}:1 max aspect`;

  if (!os.available) {
    return `
      <div class="os-empty">
        <div class="note-title">No reverse-image backend on this deployment</div>
        <p>${escapeHtml(os.reason || 'Open-set identification is unavailable, so unidentified logos cannot be searched.')}</p>
        <p class="muted">Unidentified logos are still listed crop by crop in the Scenes tab.</p>
      </div>`;
  }

  const skippedCount = (skipped.below_min_confidence || 0) + (skipped.duplicate_hash || 0);
  const examined = cands.length + rejected.length + skippedCount;

  // Saturation is upstream of every gate, so it decides how the funnel above
  // should be read. Without it, a detector proposing 112 boxes per frame looks
  // identical to a resolver failing to find 57 brands.
  const dd = os.detector_diagnostics || null;
  const saturatedFrames = (dd && dd.saturated_frames) || [];
  const saturation = dd && dd.saturated_frame_count
    ? `<div class="os-warn">
         <div class="note-title">⚠ Detector saturated on ${escapeHtmlNum(dd.saturated_frame_count)} frame${dd.saturated_frame_count === 1 ? '' : 's'}</div>
         <p>${escapeHtmlNum(dd.max_proposals_in_a_frame)} logo proposals in the busiest frame (median ${escapeHtmlNum(dd.median_proposals_in_a_frame)}, ${escapeHtmlNum(dd.proposals_total)} total).
         High proposal density means the detector is not separating marks from background, so identity work on those frames is unreliable —
         the funnel above is a detector reading, not a verdict on the brands in this video.</p>
         <p class="muted">${saturatedFrames.length ? `First saturated frames: ${saturatedFrames.slice(0, 12).map(escapeHtml).join(', ')}${dd.saturated_frame_count > saturatedFrames.length ? ' …' : ''}` : ''}</p>
       </div>`
    : '';

  // The funnel is the honest read of this stage: most crops are noise, and the
  // UI should say so rather than presenting three cards as if three were found.
  const funnel = `
    <div class="figure-line figure-line--sm">
      <div class="figure"><span class="figure-val">${examined}</span><span class="figure-label">logo boxes examined</span></div>
      <div class="figure"><span class="figure-val">${skippedCount + rejected.length}</span><span class="figure-label">dropped by the gate</span></div>
      <div class="figure"><span class="figure-val">${cands.length}</span><span class="figure-label">searched as candidates</span></div>
      <div class="figure"><span class="figure-val">${escapeHtml(String(os.resolved == null ? 0 : os.resolved))}</span><span class="figure-label">matched a real brand</span></div>
      <div class="figure figure-quiet"><span class="figure-label">${gate}</span></div>
    </div>`;

  const cards = cands.map((c) => {
    const v = c.logo_dev_validation || {};
    const verified = v.status === 'verified';
    const { short, legal } = splitCandidateName(c.candidate_name);
    const frameHref = `/api/scene/${encodeURIComponent(d.job_id)}/${encodeURIComponent(String(c.frame_index))}`;

    // Only real citable pages get links. The model's own prose is not a source.
    const sources = (c.search_results || [])
      .filter((r) => r.url && !/gemini_grounded_desc/.test(r.source || ''))
      .slice(0, 5)
      .map((r) => `<li><a href="${escapeHtml(r.url)}" target="_blank" rel="noopener">${escapeHtml(sourceLabel(r))}</a></li>`)
      .join('');

    return `
      <article class="cand${verified ? ' is-verified' : ''}">
        <a class="cand-crop" href="${escapeHtml(c.crop_url || frameHref)}" target="_blank" rel="noopener">
          ${c.crop_url ? `<img src="${escapeHtml(c.crop_url)}" alt="Logo crop from frame ${escapeHtml(String(c.frame_index))}" loading="lazy">` : '<span class="cand-crop-empty">no crop</span>'}
        </a>
        <div class="cand-body">
          <div class="cand-head">
            <span class="cand-name">${escapeHtml(short || 'Unidentified crop')}</span>
            <span class="cand-state">${
              verified ? 'matched a real brand'
              : c.status === 'candidate_void' ? 'the search could not identify it'
              : c.status === 'candidate_no_name' ? 'searched, no name found'
              : 'unverified'
            }</span>
          </div>
          ${legal ? `<p class="cand-legal">${escapeHtml(legal)}</p>` : ''}
          ${verified && v.domain
            ? `<p class="cand-match">Matched <b>${escapeHtml(v.brand || short)}</b> on <a href="https://${escapeHtml(v.domain)}" target="_blank" rel="noopener">${escapeHtml(v.domain)}</a></p>`
            : ''}
          <div class="cand-meta">
            <span>${pct(c.confidence)} confidence</span>
            <a href="${frameHref}" target="_blank" rel="noopener">frame ${escapeHtml(String(c.frame_index))}</a>
          </div>
          ${c.search_error ? `<p class="cand-error">${escapeHtml(c.search_error)}</p>` : ''}
          ${sources ? `<ul class="cand-sources">${sources}</ul>` : ''}
        </div>
      </article>`;
  }).join('');

  const rejectRows = rejected.map((r) => `
    <li>
      <span class="mono">frame ${escapeHtml(String(r.frame_index))}</span>
      <span>${escapeHtml(String(r.width))}&times;${escapeHtml(String(r.height))} px, ${escapeHtml(String(r.aspect))}:1</span>
      <span class="muted">${r.reason === 'too_small' ? 'too small to search' : 'banner-shaped, skipped'}</span>
    </li>`).join('');

  const rejects = rejectRows ? `
    <details class="os-rejects">
      <summary>${rejected.length} areas skipped on shape</summary>
      <ul>${rejectRows}</ul>
    </details>` : '';

  if (!cands.length) {
    return `${funnel}
      ${saturation}
      <div class="os-empty">
        <div class="note-title">Nothing cleared the gate</div>
        <p>${examined} logo boxes were examined and none were distinctive enough to search. Unresolved logos stay listed crop by crop in the Scenes tab.</p>
        ${saturatedFrames.length ? '<p class="muted">The detector was saturated on this video, so a zero here is a detector reading, not a statement about the brands in it.</p>' : ''}
      </div>
      ${rejects}`;
  }

  return `${funnel}
    ${saturation}
    <div class="cand-grid">${cards}</div>
    <p class="os-note">Candidates are lower-trust than on-screen detections. A match here is a lead, not an appearance — confirm against the crop before you count it.</p>
    ${rejects}`;
}

function recommendPanel(d) {
  const recs = d.recommendations || [];
  if (!recs.length) {
    return `<div class="empty">No opportunities yet.<br><br>Try analysing a video with on-screen brand evidence.<br><a href="#/analyse">Analyse a video</a></div>`;
  }
  const head = `<div class="card-head" style="margin-bottom:16px"><span class="card-sub">${recs.length} brands, ranked. Direct matches (on-screen evidence) come first, then suggested matches from brand knowledge</span></div>`;
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
        <div class="card-body" style="font-size:var(--t-sm)">
          ${escapeHtml(r.product || r.brand)}, ${escapeHtml(r.category || 'General')}${r.appearances ? ', seen ' + r.appearances + ' times' : ', never on screen'}
        </div>
        <div class="rec-reasons">
          ${r.reasons.map((reason) => chip(sentence(reason))).join('')}
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
  return head + cards;
}

function adsPanel(d) {
  if (!d.ads || !d.ads.length) {
    return `<div class="empty">No spoken brand mentions were found in the audio.</div>`;
  }
  const rows = d.ads.map((a) => `
    <div class="ads-row">
      <div class="ads-row-head">
        <div class="card-title">${escapeHtml(a.brand)} <span class="ads-channel">${escapeHtml(d.creator || 'this channel')}</span></div>
        <span class="chip chip-ghost">${escapeHtml(a.product || a.brand)}</span>
      </div>
      <p class="ads-where">${a.start_time != null ? `Mentioned at ${fmtTime(a.start_time)}` : 'No timestamp for this mention'}</p>
      <div class="ads-row-actions">
        ${a.start_time != null ? `<button type="button" class="btn btn-sm ads-play" data-ts="${Number(a.start_time).toFixed(2)}">&blacktriangleright; Play @ ${fmtTime(a.start_time)}</button>` : ''}
        ${a.start_time != null ? `<a class="ads-scene-link" href="#/pipeline/${encodeURIComponent(d.job_id)}?tab=scenes&clip=${Number(a.start_time).toFixed(2)}">See it in the scene &rarr;</a>` : ''}
      </div>
    </div>`).join('');

  return `
    <p class="card-sub" style="max-width:66ch;margin-bottom:var(--space-4)">${d.ads.length} brand${d.ads.length === 1 ? '' : 's'} named out loud in the audio. Play the moment, or open the scene it landed in.</p>
    <div class="ads-grid">
      <div class="ads-player">
        <video id="ads-player" class="clip-player" controls preload="metadata" src="/api/video/${encodeURIComponent(d.job_id)}"></video>
        <span class="ads-player-hint" id="ads-hint">Pick a mention to play the moment it was said.</span>
      </div>
      <div class="ads-list">${rows}</div>
    </div>`;
}

// The ads tab used to have no player at all: "jump to clip" navigated to the
// scenes tab, so the only way to hear a mention was to leave the page you were
// reading. This binds a player that lives on the tab.
function wireAdsPanel(root) {
  const player = $('#ads-player', root);
  const hint = $('#ads-hint', root);
  if (!player || player.dataset.adsWired) return;
  player.dataset.adsWired = '1';
  $$('.ads-play', root).forEach((btn) => {
    btn.addEventListener('click', () => {
      const target = Math.max(0, parseFloat(btn.dataset.ts) - 1.5);
      const play = () => {
        player.currentTime = target;
        player.play().catch(() => { /* needs a gesture the browser gave us */ });
      };
      if (player.readyState >= 2) play();
      else player.addEventListener('loadedmetadata', play, { once: true });
      $$('.ads-play', root).forEach((b) => b.classList.toggle('is-on', b === btn));
      if (hint) hint.textContent = `Playing from ${btn.textContent.replace(/^▸\s*/, '')}`;
    });
  });
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

let outreachState = { id: null, data: null, brand: null, facts: null };
let outreachEnabled = false;
let outreachDisabledReason = 'Draft email is off — the outreach data path hasn\u2019t passed its integrity review yet.';

// Outreach only exists per project, so the bare route asks which one rather
// than erroring. It is also where the editor's "back to projects" button lands.
async function renderOutreachPicker(root) {
  root.innerHTML = loaderHtml('Loading your projects');
  try {
    const { jobs } = await api('/api/jobs');
    if (!jobs.length) {
      root.innerHTML =
        `<h2 class="page-title" style="margin-bottom:24px">Write to a brand</h2>` +
        `<div class="empty">No projects yet \u2014 analyse a video to find brands worth writing to.<br><br><a href="#/analyse">Analyse a video</a></div>`;
      return;
    }
    root.innerHTML =
      `<h2 class="page-title" style="margin-bottom:24px">Write to a brand</h2>` +
      `<p class="muted" style="max-width:60ch;margin-bottom:20px">Pick the project whose brands you want to draft to.</p>` +
      JOB_LIST_HEADERS +
      jobs.map((j, i) => jobRow(j, i, (jobId) => `#/outreach/${encodeURIComponent(jobId)}`)).join('');
  } catch (err) {
    root.innerHTML = errorHtml(err.message);
  }
}

async function renderOutreach(id, params) {
  const root = $('#outreach-root');
  if (!id) return renderOutreachPicker(root);

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

function brandDetailHtml(p) {
  if (!p) return '';
  const type = friendlyRecType(p.type || '');
  const reasons = (p.reasons || []).map((r) => `<li>${escapeHtml(sentence(r))}</li>`).join('');
  const hr = (p.hr_emails || []).filter(Boolean);
  return `
    <div class="brand-detail">
      <h2 class="page-title">${escapeHtml(p.brand)}</h2>
      <p class="project-line">
        <span>${escapeHtml(p.product || p.brand)}</span>
        <span>${escapeHtml(p.category || 'General')}</span>
        <span>${p.appearances ? p.appearances + (p.appearances === 1 ? ' appearance' : ' appearances') : 'never on screen'}</span>
        <span>${type}</span>
      </p>
    </div>`;
}

function evidenceHtml(p, facts) {
  if (!p) return '';
  const items = (facts && facts.length ? facts : (p.reasons || [])).map((f) => `<li>${escapeHtml(sentence(f))}</li>`).join('');
  if (!items) return '';
  return `
    <div class="why">
      <span class="editor-label">Why this brand</span>
      <ul class="why-list">${items}</ul>
    </div>`;
}

function renderOutreachEditor(root, d) {
  const products = outreachBrands(d);
  const active = outreachState.brand || (products[0] && products[0].brand);

  // The list is for choosing, not for reading: one brand per row, its detail
  // lives in the editor. Twelve rows of four lines each was a wall of prose.
  const brandsList = products.length
    ? products.map((p) => `
        <button type="button" class="brand-row${p.brand === active ? ' is-active' : ''}" data-brand="${escapeHtml(p.brand)}">
          <span class="brand-name">${escapeHtml(p.brand)}</span>
          <span class="brand-meta">${escapeHtml(friendlyRecType(p.type || ''))}${p.appearances ? ` · ${p.appearances} on screen` : ' · never on screen'}</span>
        </button>`).join('')
    : `<div class="empty">No recommended brands yet</div>`;

  root.innerHTML = `
    <div class="outreach-brands" role="group" aria-label="Brands to contact">${brandsList}</div>
    <div class="outreach-editor">
      <div id="brand-detail"></div>
      <div class="editor-toolbar">
        <div class="editor-target">
          <span class="editor-label">Target email</span>
          <input class="editor-target-name" id="target-name" value="" spellcheck="false" placeholder="Auto-filled from the brand catalog — editable">
          <span class="editor-hint" id="target-hint"></span>
        </div>
        <div class="editor-actions">
          <button class="btn btn-sm" id="btn-generate">Generate draft</button>
          <button class="btn btn-sm" id="btn-forward">Mark reviewed</button>
        </div>
      </div>
      <div id="brand-evidence"></div>
      <div class="editor-subject">
        <span class="editor-label">Subject</span>
        <div class="editor-subject-text" id="editor-subject"></div>
      </div>
      <div class="editor-body" id="editor-body"></div>
      <div class="editor-foot" id="editor-foot"></div>
    </div>
  `;

  const subject = $('#editor-subject', root);
  const body = $('#editor-body', root);
  const foot = $('#editor-foot', root);
  const targetInput = $('#target-name', root);
  const detail = $('#brand-detail', root);
  const evidence = $('#brand-evidence', root);

  const selectBrand = (brand) => {
    outreachState.brand = brand;
    outreachState.facts = null;
    $$('.brand-row', root).forEach((r) =>
      r.classList.toggle('is-active', r.dataset.brand === brand)
    );
    const p = products.find((x) => x.brand === brand);
    detail.innerHTML = brandDetailHtml(p);
    evidence.innerHTML = evidenceHtml(p, null);
    subject.textContent = '';
    body.textContent = '';
    foot.textContent = '';
    foot.classList.remove('forwarded');
    const hint = $('#target-hint', root);
    if (p && p.contact_email) {
      targetInput.value = p.contact_email;
      hint.textContent = p.contact_email_source === 'gemini_grounding'
        ? `Found via Gemini search, not verified. Confirm before sending.${p.contact_website ? ' ' + p.contact_website : ''}`
        : `Auto-filled from the brand catalog${p.contact_verified ? '' : ' Unverified, confirm before sending.'}${p.contact_website ? ' ' + p.contact_website : ''}`;
    } else {
      targetInput.value = '';
      hint.textContent = 'No known contact — enter the brand email manually';
    }
    const hr = (p && p.hr_emails || []).filter(Boolean);
    if (hr.length) {
      const wrap = document.createElement('div');
      wrap.className = 'hr-chips';
      wrap.innerHTML = hr.map((e) => `<button type="button" class="chip chip-ghost" data-email="${escapeHtml(e)}">${escapeHtml(e)}</button>`).join('');
      $$('.chip', wrap).forEach((c) => c.addEventListener('click', () => {
        targetInput.value = c.dataset.email;
        hint.textContent = 'Copied from the brand record. Confirm before sending.';
      }));
      hint.after(wrap);
    }
    history.replaceState(null, '', `#/outreach/${encodeURIComponent(d.job_id)}?brand=${encodeURIComponent(brand)}`);
  };

  $$('.brand-row', root).forEach((r) =>
    r.addEventListener('click', () => { selectBrand(r.dataset.brand); })
  );

  selectBrand(active);

  $('#btn-generate', root).addEventListener('click', async () => {
    if (!outreachState.brand) return;
    const target = targetInput.value.trim();
    if (!target) {
      foot.textContent = 'Enter a target email before generating';
      return;
    }
    const btn = $('#btn-generate', root);
    btn.disabled = true;
    btn.textContent = 'Generating…';
    foot.textContent = '';
    foot.classList.remove('forwarded');
    try {
      const res = await api('/api/outreach/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_id: d.job_id, brand: outreachState.brand, target }),
      });
      subject.textContent = res.subject;
      body.textContent = res.body;
      // The server already worked out why it wrote those claims; show them
      // rather than making the operator take the letter on faith.
      const facts = (res.rationale && res.rationale.evidence_facts) || [];
      outreachState.facts = facts;
      evidence.innerHTML = evidenceHtml(products.find((x) => x.brand === outreachState.brand), facts);
      foot.textContent = `Draft generated · to ${res.target} · ${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
    } catch (err) {
      foot.textContent = `Generate failed — ${err.message}`;
    } finally {
      btn.disabled = false;
      btn.textContent = 'Generate draft';
    }
  });

  $('#btn-forward', root).addEventListener('click', async () => {
    if (!outreachState.brand) return;
    if (!body.textContent.trim()) {
      foot.textContent = 'Generate a draft before marking it reviewed';
      return;
    }
    const btn = $('#btn-forward', root);
    btn.disabled = true;
    btn.textContent = 'Saving…';
    try {
      const res = await api('/api/outreach/approve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_id: d.job_id, brand: outreachState.brand }),
      });
      // Never say "Forwarded" — nothing was sent. The server says so too.
      foot.textContent = `Marked reviewed · nothing was sent · ${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
      foot.classList.add('forwarded');
    } catch (err) {
      foot.textContent = `Could not save — ${err.message}`;
    } finally {
      btn.disabled = false;
      btn.textContent = 'Mark reviewed';
    }
  });
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
        <div class="rec-reasons">${(r.reasons || []).map((reason) => chip(sentence(reason))).join('')}</div>
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

/* ── Login ────────────────────────────────────────────────── */

async function renderLogin() {
  const root = $('#login-root');
  try {
    const me = await api('/api/me');
    if (me.authenticated === true) {
      root.innerHTML = `
        <div class="auth-card">
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
      root.innerHTML = `
        <div class="auth-card">
          <h2 class="page-title">Open access</h2>
          <p class="page-copy">No login is required in this deployment.<br><a href="#/pipeline">Open your projects</a></p>
        </div>`;
      return;
    }
    root.innerHTML = `
      <div class="auth-card">
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

/* ── Boot ─────────────────────────────────────────────────── */

router();
