/* =========================================================
   ClaimCheck — Frontend State & UI Controller
   ========================================================= */

const API = 'http://localhost:8000/api';

// Maintains central app state
const state = {
  paragraphs: [],          // [{paragraph_id, text}]
  activeParagraphId: null,
  priorArtResults: [],     // latest results from /query-prior-art
  allPriorArtHits: [],     // accumulated for report generation
  sectionFilter: '',
  reportGenerated: false, // dead state with no effect for now
};

// DOM refs
const $ = id => document.getElementById(id);
const leftEmpty    = $('left-empty');
const leftContent  = $('left-content');
const midEmpty     = $('middle-empty');
const midContent   = $('middle-content');
const rightEmpty   = $('right-empty');
const rightContent = $('right-content');
const paraCount    = $('para-count');
const statusDot    = $('status-dot');
const statusText   = $('status-text');
const searchSpinner = $('search-spinner');
const btnReport    = $('btn-report');
const fileInput    = $('file-upload');

// ── Status helpers ─────────────────────────────────────────
function setStatus(label, color) {
  statusText.textContent = label;
  statusDot.className = `w-2 h-2 rounded-full ${color}`;
}
const idle    = () => setStatus('Ready', 'bg-gray-600');
const loading = msg => setStatus(msg, 'bg-yellow-400 animate-pulse');
const done    = msg => setStatus(msg, 'bg-emerald-400');
const err     = msg => setStatus(msg, 'bg-red-400');

function showSpinner(show) {
  searchSpinner.classList.toggle('hidden', !show);
}

// ── Show / hide column sections ───────────────────────────
function showColumn(emptyEl, contentEl, show) {
  emptyEl.classList.toggle('hidden', show);
  contentEl.classList.toggle('hidden', !show);
}

// ── File upload ────────────────────────────────────────────
fileInput.addEventListener('change', async e => {
  const file = e.target.files[0];
  if (!file) return;

  loading('Parsing document…');
  const fd = new FormData();
  fd.append('file', file);

  try {
    const res = await fetch(`${API}/analyze-document`, { method: 'POST', body: fd });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const data = await res.json();
    state.paragraphs = data.paragraphs;
    state.activeParagraphId = null;
    state.allPriorArtHits = [];
    renderParagraphs();
    done(`${data.total_paragraphs} paragraphs loaded`);
    btnReport.disabled = false;
    // Reset right column
    showColumn(rightEmpty, rightContent, false);
  } catch (ex) {
    err('Parse failed');
    console.error(ex);
  } finally {
    fileInput.value = '';
  }
});

// ── Render left column ─────────────────────────────────────
function renderParagraphs() {
  leftContent.innerHTML = '';
  state.paragraphs.forEach((para, idx) => {
    const div = document.createElement('div');
    div.className = 'para-block';
    div.dataset.id = para.paragraph_id;
    div.innerHTML = `
      <div class="para-num">¶ ${idx + 1}</div>
      <div class="para-text">${escapeHtml(para.text)}</div>`;
    div.addEventListener('click', () => onParagraphClick(para, div));
    leftContent.appendChild(div);
  });
  paraCount.textContent = `${state.paragraphs.length} paragraph${state.paragraphs.length !== 1 ? 's' : ''}`;
  showColumn(leftEmpty, leftContent, true);
}

// ── Paragraph click ────────────────────────────────────────
async function onParagraphClick(para, el, isFilterRefresh = false) {
  // Toggle: if already active and not a filter refresh, deactivate
  if (!isFilterRefresh && state.activeParagraphId === para.paragraph_id) {
    el.classList.remove('active');
    state.activeParagraphId = null;
    state.priorArtResults = [];
    showColumn(midEmpty, midContent, false);
    idle();
    return;
  }

  // Deactivate previous (only if switching to a different paragraph)
  if (!isFilterRefresh) {
    document.querySelectorAll('.para-block.active').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.para-block.loading').forEach(b => b.classList.remove('loading'));
    el.classList.add('active');
  }

  el.classList.add('loading');
  state.activeParagraphId = para.paragraph_id;
  showSpinner(true);
  loading('Searching prior art…');

  try {
    const res = await fetch(`${API}/query-prior-art`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: para.text,
        paragraph_id: para.paragraph_id,
        section_filter: state.sectionFilter || null,
      }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const data = await res.json();
    state.priorArtResults = data.results;

    // Accumulate unique hits for the report
    data.results.forEach(hit => {
      const key = `${hit.patent_id}-${hit.text.slice(0, 40)}`;
      if (!state.allPriorArtHits.find(h => `${h.patent_id}-${h.text.slice(0, 40)}` === key)) {
        state.allPriorArtHits.push(hit);
      }
    });

    renderPriorArt(data.results, para.text);
    done(`${data.results.length} matches found`);
  } catch (ex) {
    err('Search failed');
    console.error(ex);
  } finally {
    el.classList.remove('loading');
    showSpinner(false);
  }
}

// ── Render middle column ───────────────────────────────────
function renderPriorArt(results, queryText) {
  midContent.innerHTML = '';

  if (!results.length) {
    showColumn(midEmpty, midContent, false);
    midEmpty.querySelector('p').textContent = 'No matches found for this paragraph.';
    return;
  }

  showColumn(midEmpty, midContent, true);

  results.forEach((hit, idx) => {
    const score = hit.similarity_score;
    const scoreClass = score >= 0.75 ? 'high' : score >= 0.50 ? 'mid' : 'low';
    const scorePct = Math.round(score * 100);

    const card = document.createElement('div');
    card.className = `patent-card${idx === 0 ? ' highlight' : ''}`;
    card.dataset.patentId = hit.patent_id;

    const highlightedText = highlightMatch(hit.text, queryText);

    card.innerHTML = `
      <div class="card-header">
        <div class="card-title">${escapeHtml(hit.title)}</div>
        <span class="score-badge ${scoreClass}">${scorePct}%</span>
      </div>
      <div class="card-meta">
        <span class="meta-chip">${escapeHtml(hit.patent_id)}</span>
        <span class="meta-chip">${escapeHtml(hit.section)}${hit.claim_number ? ` · Claim ${hit.claim_number}` : ''}</span>
        ${hit.grant_date ? `<span class="meta-chip">${escapeHtml(hit.grant_date)}</span>` : ''}
        ${hit.inventor ? `<span class="meta-chip">Inv: ${escapeHtml(hit.inventor)}</span>` : ''}
      </div>
      <div class="card-text">${highlightedText}</div>`;

    midContent.appendChild(card);
  });

  // Smooth scroll highest-ranked card into view
  const first = midContent.querySelector('.patent-card');
  if (first) first.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ── Highlight shared words between query and result ────────
function highlightMatch(text, query) {
  const safe = escapeHtml(text);
  const words = query
    .toLowerCase()
    .split(/\W+/)
    .filter(w => w.length > 4);

  if (!words.length) return safe;

  const pattern = words.map(w => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|');
  return safe.replace(new RegExp(`(${pattern})`, 'gi'),
    '<span class="highlight-match">$1</span>');
}

// ── Filter toggles ─────────────────────────────────────────
document.querySelectorAll('.filter-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    state.sectionFilter = btn.dataset.filter;

    // Re-run search with new filter if a paragraph is active
    if (state.activeParagraphId) {
      const para = state.paragraphs.find(p => p.paragraph_id === state.activeParagraphId);
      const el   = document.querySelector(`.para-block[data-id="${state.activeParagraphId}"]`);
      if (para && el) onParagraphClick(para, el, true);
    }
  });
});

// ── Generate Report ────────────────────────────────────────
btnReport.addEventListener('click', async () => {
  if (!state.paragraphs.length) return;

  loading('Generating AI report…');
  showColumn(rightEmpty, rightContent, true);
  rightContent.innerHTML = reportSkeleton();

  const fullText = state.paragraphs.map(p => p.text).join('\n\n');
  const hits = state.allPriorArtHits.slice(0, 10);

  // If no hits yet, fetch some
  if (!hits.length) {
    try {
      const res = await fetch(`${API}/query-prior-art`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: fullText.slice(0, 2000) }),
      });
      if (res.ok) {
        const d = await res.json();
        hits.push(...d.results);
      }
    } catch { /* continue without */ }
  }

  try {
    const res = await fetch(`${API}/generate-report`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_text: fullText, prior_art_hits: hits }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const report = await res.json();
    console.log('Report received:', report);
    renderReport(report);
    done('Report ready');
  } catch (ex) {
    rightContent.innerHTML = `<p class="text-red-400 text-sm p-4">Failed to generate report: ${escapeHtml(ex.message)}</p>`;
    err('Report failed');
    console.error(ex);
  }
});

// ── Render right column ────────────────────────────────────
function renderReport(report) {
  const score = Math.min(100, Math.max(1, report.overall_risk_score ?? 50));
  const color = score >= 70 ? '#ef4444' : score >= 40 ? '#f97316' : '#22c55e';
  const label = score >= 70 ? 'High Risk' : score >= 40 ? 'Moderate Risk' : 'Low Risk';

  rightContent.innerHTML = `
    <!-- Gauge -->
    <div class="text-center">
      ${buildGauge(score, color)}
      <div class="mt-1 text-lg font-bold" style="color:${color}">${label}</div>
      <div class="text-xs text-gray-500">Infringement Risk Score</div>
    </div>

    <!-- Summary -->
    ${report.summary ? `
    <div class="bg-gray-800 rounded-lg p-4">
      <h3 class="text-xs font-semibold text-gray-400 uppercase mb-2">Executive Summary</h3>
      <p class="text-sm text-gray-300 leading-relaxed">${escapeHtml(report.summary)}</p>
    </div>` : ''}

    <!-- Critical Conflicts -->
    ${(report.critical_conflicts?.length) ? `
    <div>
      <h3 class="text-xs font-semibold text-gray-400 uppercase mb-2">
        Critical Conflicts (${report.critical_conflicts.length})
      </h3>
      <div class="space-y-3">
        ${report.critical_conflicts.map(c => `
          <div class="conflict-card severity-${c.severity}">
            <div class="flex justify-between items-start mb-2">
              <span class="text-xs font-bold text-gray-200">${escapeHtml(c.patent_id)}</span>
              <span class="text-xs px-2 py-0.5 rounded-full ${severityBadge(c.severity)}">${escapeHtml(c.severity)}</span>
            </div>
            <div class="text-xs text-gray-400 mb-1 font-medium">${escapeHtml(c.patent_title || '')}</div>
            <div class="text-xs text-gray-300 mb-1"><span class="text-gray-500">Patent: </span>${escapeHtml(c.conflicting_claim || '')}</div>
            <div class="text-xs text-gray-300 mb-2"><span class="text-gray-500">Your text: </span>${escapeHtml(c.user_text_excerpt || '')}</div>
            <p class="text-xs text-gray-400 italic">${escapeHtml(c.conflict_explanation || '')}</p>
          </div>`).join('')}
      </div>
    </div>` : ''}

    <!-- Workarounds -->
    ${(report.design_workaround_suggestions?.length) ? `
    <div>
      <h3 class="text-xs font-semibold text-gray-400 uppercase mb-2">Design Workarounds</h3>
      <ul class="space-y-2">
        ${report.design_workaround_suggestions.map((s, i) => `
          <li class="flex gap-2 text-sm text-gray-300">
            <span class="flex-none w-5 h-5 rounded-full bg-brand-600 text-white text-xs flex items-center justify-center font-bold mt-0.5">${i + 1}</span>
            <span>${escapeHtml(s)}</span>
          </li>`).join('')}
      </ul>
    </div>` : ''}`;
}

// ── SVG circular gauge ─────────────────────────────────────
function buildGauge(score, color) {
  const r = 54, cx = 70, cy = 70;
  const circ = 2 * Math.PI * r;
  // Half-circle gauge: arc from 180° to 0° (bottom to top)
  const pct = score / 100;
  const dashArray = circ * 0.5;          // half circle total
  const dashOffset = dashArray * (1 - pct); // how much to trim

  return `
  <svg id="risk-gauge-svg" width="140" height="90" viewBox="0 0 140 90">
    <!-- Track -->
    <path d="M 16,70 A 54,54 0 0,1 124,70"
      fill="none" stroke="#374151" stroke-width="10" stroke-linecap="round"/>
    <!-- Fill -->
    <path d="M 16,70 A 54,54 0 0,1 124,70"
      fill="none" stroke="${color}" stroke-width="10" stroke-linecap="round"
      stroke-dasharray="${dashArray}"
      stroke-dashoffset="${dashOffset}"
      transform="rotate(0,${cx},${cy})"
      style="transition: stroke-dashoffset 0.6s ease"/>
    <!-- Score text -->
    <text x="${cx}" y="${cy - 4}" text-anchor="middle"
      font-size="28" font-weight="700" fill="${color}" font-family="sans-serif">${score}</text>
    <text x="${cx}" y="${cy + 16}" text-anchor="middle"
      font-size="10" fill="#6b7280" font-family="sans-serif">/ 100</text>
    <text x="16" y="85" text-anchor="middle" font-size="9" fill="#4b5563" font-family="sans-serif">Low</text>
    <text x="124" y="85" text-anchor="middle" font-size="9" fill="#4b5563" font-family="sans-serif">High</text>
  </svg>`;
}

function severityBadge(sev) {
  if (sev === 'High')   return 'bg-red-900 text-red-300';
  if (sev === 'Medium') return 'bg-orange-900 text-orange-300';
  return 'bg-green-900 text-green-300';
}

// ── Skeleton loader ────────────────────────────────────────
function reportSkeleton() {
  return `
  <div class="space-y-4 p-2">
    <div class="skeleton w-32 h-32 rounded-full mx-auto"></div>
    <div class="skeleton h-4 w-3/4 mx-auto"></div>
    <div class="space-y-2">
      ${Array(4).fill('<div class="skeleton"></div>').join('')}
    </div>
    <div class="space-y-2">
      ${Array(3).fill('<div class="skeleton"></div>').join('')}
    </div>
  </div>`;
}

// ── Utility ────────────────────────────────────────────────
function escapeHtml(str) {
  return String(str ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
