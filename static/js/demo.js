/* Live demo page (static/demo.html, served at /demo by rag_eval/demo_web.py).
 *
 * One question -> POST /eval/api/run/demo, which streams one JSON event per
 * line while the selected systems run in parallel. Every event updates the
 * page right away:
 *   step    -> the system's architecture strip + its step-by-step process log
 *   answer  -> the answer text and sources
 *   metric  -> one of the three score tiles
 *   done    -> the comparison scoreboard (when more than one system ran)
 */
'use strict';

const API = { config: '/demo/api/config', run: '/eval/api/run/demo' };
const METRIC_ORDER = ['groundedness', 'context_relevance', 'answer_relevance'];
const SYS_ORDER = ['sagot', 'c0', 'c1', 'c2'];
const SYS_COLOR = { sagot: 'var(--sagot)', c0: 'var(--c0)', c1: 'var(--c1)', c2: 'var(--c2)' };
const MODES = [
  { id: 'none', label: 'Sagot AI only', systems: ['sagot'] },
  { id: 'c0', label: 'vs C0 · LLM only', systems: ['sagot', 'c0'] },
  { id: 'c1', label: 'vs C1 · Standard RAG', systems: ['sagot', 'c1'] },
  { id: 'c2', label: 'vs C2 · REVIE', systems: ['sagot', 'c2'] },
  { id: 'all', label: 'vs All (C0, C1, C2)', systems: ['sagot', 'c0', 'c1', 'c2'] },
];

// Architecture strip per system: stage id, label, icon.
const PIPES = {
  sagot: [
    ['input', 'Input check', 'rule'], ['language', 'Language trigger', 'translate'], ['intent', 'Intent & routing', 'alt_route'],
    ['retrieval', 'Hybrid retrieval', 'manage_search'], ['fusion', 'RRF fusion', 'merge'], ['rerank', 'Reranker', 'filter_alt'],
    ['generate', 'Generator', 'edit_note'], ['verify', 'Verifier', 'verified_user'], ['answer', 'Answer', 'task_alt'],
  ],
  c0: [['question', 'Question', 'help'], ['llm', 'LLM only', 'psychology'], ['answer', 'Answer', 'task_alt']],
  c1: [['question', 'Question', 'help'], ['dense', 'Dense search', 'search'], ['llm', 'LLM + standard prompt', 'psychology'], ['answer', 'Answer', 'task_alt']],
  c2: [['question', 'Question', 'help'], ['revie', 'REVIE (external)', 'support_agent'], ['answer', 'Answer', 'task_alt']],
};
const STEP_STAGE = {
  sagot: { sanitize: 'input', language: 'language', intent: 'intent', route: 'intent', year_filter: 'intent',
           index: 'retrieval', cache: 'retrieval', semantic: 'retrieval', bm25: 'retrieval', issuance_id: 'retrieval',
           fusion: 'fusion', rerank: 'rerank', evidence: 'rerank', computation: 'generate',
           generate: 'generate', verify: 'verify', final: 'answer' },
  c0: { question: 'question', llm_only: 'llm', final: 'answer' },
  c1: { question: 'question', dense: 'dense', llm_rag: 'llm', final: 'answer' },
  c2: { question: 'question', external: 'revie', final: 'answer' },
};

// Plain-language status line shown while a system works (the detailed,
// technical step log is only shown when "Show the process" is clicked).
function liveText(sys, ev) {
  const d = ev.data || {};
  switch (ev.step) {
    case 'sanitize': case 'language': case 'intent': case 'route': case 'year_filter':
      return 'Understanding the question';
    case 'index': case 'cache': case 'semantic': case 'bm25': case 'issuance_id': case 'fusion': case 'dense':
      return 'Searching the BIR documents';
    case 'rerank': case 'evidence':
      return 'Picking the most relevant excerpts';
    case 'computation':
      return 'Computing with the official tax tables';
    case 'generate':
      return d.attempt > 1 ? `Rewriting the answer to fix what the checker flagged (attempt ${d.attempt})` : 'Writing the answer from the excerpts';
    case 'verify':
      return 'Double-checking every claim against the documents';
    case 'llm_only':
      return 'Answering from the model’s memory';
    case 'llm_rag':
      return 'Writing the answer';
    case 'question':
      return sys === 'c2' ? 'Loading REVIE’s answer' : 'Reading the question';
    case 'external':
      return 'Loading REVIE’s answer';
    default:
      return 'Finishing';
  }
}

const OUTCOME_TEXT = {
  verified: ['Verified answer', 'good'],
  generator_no_answer: ['Declined: documents don’t answer it', 'warn'],
  verification_failed: ['Declined: no draft passed verification', 'warn'],
  no_retrieval: ['Declined: nothing relevant retrieved', 'warn'],
  no_index: ['No knowledge base loaded', 'bad'],
  computation_done: ['Computed by the tax calculator', 'good'],
  computation_needs_input: ['Calculator needs more information', 'info'],
  greeting: ['Greeting', 'neutral'], chat: ['Chat reply', 'neutral'], clarification: ['Asked to clarify', 'info'],
  rejected_input: ['Input rejected', 'warn'], answered: ['Answered', 'neutral'],
};

const state = { config: null, compare: 'none', running: false, computation: null };
const $ = (id) => document.getElementById(id);

// ── Small helpers ───────────────────────────────────────────────────────────
function esc(v) {
  return String(v ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
function md(text) {
  const src = String(text ?? '');
  try {
    if (window.marked && window.DOMPurify) return DOMPurify.sanitize(marked.parse(src, { breaks: true }));
  } catch (e) { /* fall through to plain text */ }
  return esc(src).replace(/\n/g, '<br>');
}
function el(tag, cls, html) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html != null) n.innerHTML = html;
  return n;
}
const pct = (s) => `${Math.round(s * 100)}%`;
const icon = (name) => `<span class="material-symbols-outlined">${name}</span>`;
const pill = (text, kind) => `<span class="pill ${kind}">${esc(text)}</span>`;
const shortModel = (m) => String(m || '').split('/').pop();
function hitsList(hits, fmt) {
  if (!hits || !hits.length) return '';
  return `<ul class="hits">${hits.map((h) => `<li><span class="lbl" title="${esc(h.preview || '')}">${esc(h.label)}</span>${
    h.score != null ? `<span class="val">${esc(fmt ? fmt(h.score) : h.score)}</span>` : ''}</li>`).join('')}</ul>`;
}
function textBox(summary, text) {
  return `<details class="draft"><summary>${esc(summary)}</summary><div class="box">${esc(text)}</div></details>`;
}

// ── Step descriptions (plain language for the panel) ───────────────────────
// Returns {title, body, kind} for one event. kind: running | done | failed | warn
function describe(sys, ev) {
  const d = ev.data || {};
  const run = ev.status === 'running';
  const k = (ok) => (ev.status === 'failed' ? 'failed' : run ? 'running' : ok === false ? 'warn' : 'done');
  switch (ev.step) {
    case 'sanitize':
      return ev.status === 'failed'
        ? { title: 'Input check', body: `Rejected (${esc(d.reason)}): the question is too long or asks too many things at once.`, kind: 'failed' }
        : { title: 'Input check', body: `The question passed the input rules (length, number of questions, citations).${d.changed ? ' Extra spaces/characters were cleaned.' : ''}`, kind: 'done' };
    case 'language':
      return { title: 'Language trigger', kind: 'done',
        body: `Detected <b>${esc(d.display_name || d.label)}</b> (Filipino markers: ${d.filipino_hits}, English markers: ${d.english_hits}). The answer will be written in this language.` };
    case 'intent':
      return { title: 'Intent classification', kind: 'done',
        body: `Intent: ${pill(d.label, 'info')} · topic: ${esc(d.topic)}${d.vague ? ' · <span class="why">too short to search on</span>' : ''}` };
    case 'route': {
      const r = {
        rag: 'Tax information question → search the documents and answer only from them (RAG).',
        computation: 'Tax computation → deterministic calculator over verified rule files (no LLM writes the numbers).',
        chat: 'General chat → fixed reply about what Sagot AI can do (no model call).',
        greeting: 'Greeting → fixed reply (no model call).',
        clarify: 'Too vague to search → asks a clarifying question (no model call).',
      }[d.target] || esc(d.target);
      return { title: 'Routing', body: r, kind: 'done' };
    }
    case 'computation':
      return { title: 'Tax calculator (deterministic)', kind: 'done',
        body: `Outcome: ${pill(d.outcome, d.outcome === 'computation_done' ? 'good' : 'info')}${
          d.awaiting && d.awaiting.length ? ` · still needs: ${esc(d.awaiting.join(', '))}` : ''}` };
    case 'year_filter':
      return { title: 'Year routing', kind: 'done',
        body: d.year ? `The question names <b>${esc(d.year)}</b>, so only documents from that year are searched.`
          : 'No specific year in the question, so the whole knowledge base is searched.' };
    case 'index':
      return ev.status === 'failed'
        ? { title: 'Knowledge base', body: 'The document index is empty or could not be opened.', kind: 'failed' }
        : { title: 'Knowledge base', body: `${esc(d.chunks)} indexed document chunks available.`, kind: 'done' };
    case 'cache':
      return { title: 'Answer cache', body: 'Bypassed: the demo always runs every step live.', kind: 'done' };
    case 'semantic':
      return run ? { title: 'Semantic search (meaning)', body: 'Searching the documents by meaning with Jina embeddings…', kind: 'running' }
        : { title: 'Semantic search (meaning)', kind: 'done',
            body: `Found the ${esc(d.n)} closest chunks by meaning (smaller distance = closer).${hitsList(d.hits && d.hits.slice(0, 5), (s) => `dist ${s.toFixed(3)}`)}` };
    case 'bm25':
      return { title: 'Keyword search (BM25)', kind: 'done',
        body: `Found ${esc(d.n)} chunks sharing the question's keywords.${hitsList(d.hits && d.hits.slice(0, 5), (s) => `score ${s.toFixed(2)}`)}` };
    case 'issuance_id':
      return { title: 'Issuance-number match', kind: 'done',
        body: `The question names ${esc((d.ids || []).join(', '))}: ${esc(d.n)} chunks from that issuance were added.` };
    case 'fusion':
      return { title: 'Reciprocal Rank Fusion', kind: 'done',
        body: `Merged the searches into ${esc(d.n)} candidates; ${esc(d.both)} were found by more than one search.${
          hitsList(d.hits && d.hits.slice(0, 5), (s) => `RRF ${s.toFixed(4)}`)}` };
    case 'rerank':
      if (run) return { title: 'Reranker (cross-encoder)', body: `Scoring how well each of the ${esc(d.n_candidates)} candidates answers the question…`, kind: 'running' };
      if (ev.status === 'failed') return { title: 'Reranker (cross-encoder)', body: `Reranker unavailable; used the top ${esc(d.fallback_top_k)} fused candidates instead.`, kind: 'warn' };
      return { title: 'Reranker (cross-encoder)', kind: 'done',
        body: `Kept <b>${esc(d.kept)}</b> of ${esc(d.n_candidates)} candidates (score ≥ ${esc(d.min_score)} and ≥ ${Math.round(d.relative * 100)}% of the best, at most ${esc(d.max_docs)}).${
          hitsList(d.hits, (s) => `score ${s.toFixed(2)}`)}` };
    case 'evidence':
      if (ev.status === 'failed') return { title: 'Evidence', body: 'No document cleared the relevance bar, so Sagot AI will not try to answer (it never guesses).', kind: 'warn' };
      return { title: 'Evidence given to the generator', kind: 'done',
        body: `${(d.excerpts || []).length} numbered excerpt(s); the answer may use only these.${
          (d.excerpts || []).map((x) => textBox(x.label, x.text)).join('')}` };
    case 'generate':
      if (ev.status === 'failed') return { title: 'Generator', body: `The model call failed: ${esc(d.error)}`, kind: 'failed' };
      if (run) return { title: `Generator · ${esc(shortModel(d.model))} — attempt ${d.attempt} of ${d.max_attempts}`, kind: 'running',
        body: d.attempt > 1 ? 'Rewriting the draft to fix the claims the verifier flagged…'
          : `Writing an answer in ${esc(d.language)} using only the excerpts, citing them as [n]…` };
      if (d.no_answer) return { title: `Generator — attempt ${d.attempt}`, kind: 'warn',
        body: `${pill('NO_ANSWER', 'warn')} The excerpts don’t answer this question, so the generator declined instead of guessing.` };
      if (d.truncated) return { title: `Generator — attempt ${d.attempt}`, kind: 'warn', body: 'The draft was cut off by the length limit, so it is discarded and retried.' };
      return { title: `Generator · ${esc(shortModel(d.model))} — attempt ${d.attempt}`, kind: 'done',
        body: `Draft written (not shown to the user yet — it must pass the verifier first).${textBox('Show the draft', d.draft)}` };
    case 'verify':
      if (run) return { title: `Verifier · ${esc(shortModel(d.model))} — checking attempt ${d.attempt}`, kind: 'running',
        body: 'A different model, from a different vendor, checks every claim in the draft against the excerpts…' };
      if (d.passed) return { title: `Verifier · ${esc(shortModel(d.model))} — attempt ${d.attempt}`, kind: 'done',
        body: `${pill('SUPPORTED', 'good')} Every claim is backed by the excerpts, so the answer is released.` };
      return { title: `Verifier · ${esc(shortModel(d.model))} — attempt ${d.attempt}`, kind: 'warn',
        body: `${pill(d.verdict === 'UNPARSEABLE' ? 'NO VERDICT' : 'UNSUPPORTED', 'bad')} ${
          d.will_retry ? 'Sent back to the generator to fix.' : 'No attempts left, so Sagot AI declines instead of showing an unverified answer.'}${
          d.issues && d.issues.length ? `<ul class="issues">${d.issues.map((i) => `<li>${esc(i)}</li>`).join('')}</ul>` : ''}` };
    case 'final': {
      const [text, kind] = OUTCOME_TEXT[d.outcome] || [d.outcome, 'neutral'];
      return { title: 'Result', kind: kind === 'warn' ? 'warn' : 'done',
        body: `${pill(text, kind)}${d.note ? ` <span class="why">${esc(d.note)}</span>` : ''}` };
    }
    // Baselines
    case 'question':
      return { title: 'Question received', kind: 'done',
        body: sys === 'c0' ? 'Sent to the model exactly as typed: no documents, no grounding rules, no language instruction.'
          : sys === 'c1' ? 'Used as-is for a dense search: no language trigger, no intent routing, no year routing.'
          : 'The same question was asked to REVIE on the BIR website.' };
    case 'llm_only':
      if (run) return { title: `LLM only · ${esc(shortModel(d.model))}`, body: 'Answering from the model’s own memory…', kind: 'running' };
      if (ev.status === 'failed') return { title: 'LLM only', body: esc(d.error), kind: 'failed' };
      return { title: `LLM only · ${esc(shortModel(d.model))}`, kind: 'done',
        body: `Answered from memory. Nothing was looked up and nothing was checked.${textBox('Show the raw output', d.draft)}` };
    case 'dense':
      if (run) return { title: `Dense search · top ${esc(d.k)}`, body: 'Searching by meaning only (no keyword search, no reranker)…', kind: 'running' };
      if (ev.status === 'failed') return { title: 'Dense search', body: esc(d.error), kind: 'failed' };
      return { title: `Dense search · top ${esc(d.k)}`, kind: 'done',
        body: `Took the ${esc(d.n)} closest chunks as they are, relevant or not.${
          (d.excerpts || []).map((x) => textBox(`${x.label} · dist ${Number(x.score).toFixed(3)}`, x.text)).join('')}` };
    case 'llm_rag':
      if (run) return { title: `LLM + standard RAG prompt · ${esc(shortModel(d.model))}`, body: `Answering from the ${esc(d.n_excerpts)} chunks with LangChain’s standard QA prompt…`, kind: 'running' };
      if (ev.status === 'failed') return { title: 'LLM + standard RAG prompt', body: esc(d.error), kind: 'failed' };
      return { title: `LLM + standard RAG prompt · ${esc(shortModel(d.model))}`, kind: 'done',
        body: `Answer written in one pass. No verifier checks it.${textBox('Show the raw output', d.draft)}` };
    case 'external':
      return { title: 'REVIE (BIR chatbot)', kind: 'done', body: 'Answer pasted by the operator. REVIE’s own process is not visible from outside.' };
    default:
      return { title: esc(ev.step), body: '', kind: run ? 'running' : 'done' };
  }
}

// ── Setup ───────────────────────────────────────────────────────────────────
function banner(kind, html) {
  const b = el('div', `banner ${kind}`, `${icon(kind === 'error' ? 'error' : 'warning')}<div>${html}</div>`);
  $('banners').appendChild(b);
}

function buildGuide(cfg) {
  const sys = cfg.systems || {};
  $('sys-legend').innerHTML = SYS_ORDER.filter((s) => sys[s]).map((s) => `
    <div class="sys-card" style="--sys:${SYS_COLOR[s]}">
      <div class="name"><span class="sys-dot"></span>${esc(sys[s].name)} <span class="muted small">${esc(sys[s].tag)}</span></div>
      <p>${esc(sys[s].description)}</p>
    </div>`).join('');
  const mets = cfg.metrics || {};
  $('metric-legend').innerHTML = METRIC_ORDER.filter((m) => mets[m]).map((m) => `
    <div class="metric-card">
      <div class="name">${esc(mets[m].name)}</div>
      <div class="q">${esc(mets[m].question)}</div>
      <div class="formula"><b>How it is computed:</b> ${esc(mets[m].formula)}. Needs ${esc(mets[m].needs)}.</div>
    </div>`).join('');
}

function setGuide(open) {
  $('guide').hidden = !open;
  $('btn-guide').setAttribute('aria-expanded', String(open));
  if (open) window.scrollTo({ top: 0, behavior: 'smooth' });
}

function setOptions(open) {
  $('options-panel').hidden = !open;
  $('btn-options').setAttribute('aria-expanded', String(open));
}

// What will be sent besides the question, as small removable chips above the
// question box (so it is visible even while the Options panel is closed).
function renderChips() {
  const chips = [];
  if (state.compare !== 'none') {
    chips.push(`<span class="mchip">${esc(MODES.find((m) => m.id === state.compare).label)}<button type="button" data-clear="compare" title="Back to Sagot AI only">×</button></span>`);
  }
  if ($('ref-input').value.trim()) {
    chips.push(`<span class="mchip">Reference answer added<button type="button" data-clear="ref" title="Remove the reference answer">×</button></span>`);
  }
  if (state.computation) chips.push('<span class="mchip info">Tax computation in progress</span>');
  $('mode-chips').innerHTML = chips.join('');
  $('mode-chips').querySelectorAll('button[data-clear]').forEach((b) => {
    b.onclick = () => {
      if (b.dataset.clear === 'compare') setCompare('none');
      else { $('ref-input').value = ''; syncDots(); }
      renderChips();
    };
  });
}

// The Evaluation button is hidden for the defense. Visiting any page once with
// ?eval=1 shows it again (remembered in this browser); ?eval=0 hides it.
function evalLinkVisibility() {
  try {
    const q = new URLSearchParams(location.search).get('eval');
    if (q === '1') localStorage.setItem('show-eval', '1');
    if (q === '0') localStorage.removeItem('show-eval');
    if (localStorage.getItem('show-eval') === '1') $('eval-link').style.display = '';
  } catch (e) { /* storage unavailable: keep it hidden */ }
}

function buildModes() {
  const box = $('compare-modes');
  box.innerHTML = '';
  MODES.forEach((m) => {
    const b = el('button', '', `${m.systems.slice(1).map((s) => `<span class="sw" style="background:${SYS_COLOR[s]}"></span>`).join('')}${esc(m.label)}`);
    b.type = 'button';
    b.setAttribute('role', 'radio');
    b.dataset.mode = m.id;
    b.onclick = () => setCompare(m.id);
    box.appendChild(b);
  });
  setCompare('none');  // every visit starts with Sagot AI only
}

function setCompare(mode) {
  if (!MODES.some((m) => m.id === mode)) mode = 'none';
  state.compare = mode;
  document.querySelectorAll('#compare-modes button').forEach((b) => b.setAttribute('aria-checked', String(b.dataset.mode === mode)));
  const needsRevie = mode === 'c2' || mode === 'all';
  $('revie-box').hidden = !needsRevie;
  updateNote();
  if ($('mode-chips')) renderChips();
}

function updateNote() {
  $('composer-note').textContent = state.config && !state.config.can_run
    ? 'Running questions is only available on the host computer.'
    : '';
  if ($('mode-chips')) renderChips();
}

async function loadConfig() {
  try {
    const res = await fetch(API.config);
    state.config = await res.json();
  } catch (e) {
    banner('error', 'Could not reach the server. Make sure it is running (uvicorn main:app --port 8000), then refresh.');
    return;
  }
  const cfg = state.config;
  buildGuide(cfg);
  if (!cfg.can_run) {
    banner('warn', esc(cfg.lock_reason));
    $('ask-input').disabled = true;
    $('ask-send').disabled = true;
  }
  if (cfg.judge_error) {
    banner('warn', `<b>Scores are off:</b> ${esc(cfg.judge_error)} Answers and the process still work.`);
  }
  updateNote();
}

// ── One turn ────────────────────────────────────────────────────────────────
function makeSystemCard(sys, info) {
  const card = el('article', 'sys');
  card.style.setProperty('--sys', SYS_COLOR[sys]);
  const pipe = PIPES[sys].map(([id, label, ic], i) =>
    `${i ? '<span class="arrow">›</span>' : ''}<span class="stage" data-stage="${id}">${icon(ic)}<span class="slabel">${esc(label)}</span></span>`).join('');
  // Answer first, then the three scores. The process (architecture strip +
  // step log) is recorded live but stays hidden until someone clicks
  // "Show the process"; while running, one short line says what is happening.
  card.innerHTML = `
    <div class="sys-head"><span class="sys-dot"></span><span class="sys-name">${esc(info.name)}</span>
      <span class="sys-tag">${esc(info.tag)}</span><span class="sys-status"><span class="spinner"></span><span class="st">starting…</span></span></div>
    <div class="live-line"><span class="spinner"></span><span class="live-text">Starting…</span></div>
    <div class="answer"><div class="answer-label">Answer <span class="opill"></span></div>
      <div class="answer-body"><span class="answer-wait">Waiting for the answer…</span></div><div class="sources"></div></div>
    <div class="metrics">${METRIC_ORDER.map((m) => metricTile(m, null, 'wait')).join('')}</div>
    <button type="button" class="how-toggle" aria-expanded="false">${icon('account_tree')}<span class="how-label">Show the process</span>
      <span class="muted small pcount"></span>${icon('expand_more').replace('outlined', 'outlined chev')}</button>
    <div class="how" hidden>
      <div class="pipe">${pipe}</div>
      <div class="proc"><ol class="steps"></ol></div>
    </div>`;
  const toggle = card.querySelector('.how-toggle');
  toggle.onclick = () => {
    const how = card.querySelector('.how');
    how.hidden = !how.hidden;
    toggle.setAttribute('aria-expanded', String(!how.hidden));
    card.querySelector('.how-label').textContent = how.hidden ? 'Show the process' : 'Hide the process';
  };
  return card;
}

function metricTile(metric, result, status) {
  const def = (state.config && state.config.metrics && state.config.metrics[metric]) || { name: metric, formula: '' };
  if (status === 'wait' || status === 'running') {
    return `<div class="mtile wait" data-metric="${metric}"><div class="mname">${esc(def.name)}${status === 'running' ? '<span class="spinner"></span>' : ''}</div>
      <div class="mscore">—</div><div class="bar"><i></i></div><div class="mnote">${status === 'running' ? 'Judging…' : 'After the answer'}</div></div>`;
  }
  if (!result || result.score == null) {
    return `<div class="mtile" data-metric="${metric}"><div class="mname">${esc(def.name)}</div>
      <div class="mscore na">N/A</div><div class="bar"><i></i></div><div class="mnote">${esc(result ? result.note : '')}</div>
      <details><summary>How it is computed</summary><div class="formula">${esc(def.formula)}</div></details></div>`;
  }
  const items = (result.items || []).map((it) => {
    if (metric === 'context_relevance') {
      return `<li>${icon('quiz')}<span>${esc(it.text)} <span class="muted tabular">(${Number(it.similarity).toFixed(2)})</span></span></li>`;
    }
    const v = it.verdict;
    const [ic, cls, word] = v === 'supported' || v === 'correct' ? ['check_circle', 'ok', v]
      : v === 'missing' ? ['remove_circle', 'miss', 'missing'] : ['cancel', 'no', v];
    return `<li><span class="material-symbols-outlined ${cls}" title="${esc(word)}">${ic}</span><span>${esc(it.text)}${
      it.evidence ? ` <span class="muted">— “${esc(it.evidence)}”</span>` : ''}</span></li>`;
  }).join('');
  return `<div class="mtile" data-metric="${metric}"><div class="mname">${esc(def.name)}</div>
    <div class="mscore">${pct(result.score)}</div><div class="bar"><i style="width:${Math.max(2, result.score * 100)}%"></i></div>
    <div class="msum">${esc(result.summary)}</div>
    <details><summary>Details &amp; formula</summary><div class="formula"><b>Formula:</b> ${esc(def.formula)}</div><ul class="mitems">${items}</ul></details></div>`;
}

class Turn {
  constructor(query, compare, hasRef) {
    this.query = query;
    this.mode = MODES.find((m) => m.id === compare);
    this.cards = {};
    this.steps = {};
    this.metrics = {};
    this.started = Date.now();
    this.root = el('div', 'turn');
    this.root.appendChild(el('div', 'user-q', esc(query)));
    const chips = `${compare !== 'none' ? pill(this.mode.label, 'info') : ''}${hasRef ? pill('Reference answer given', 'good') : ''}`;
    if (chips) this.root.appendChild(el('div', 'user-meta', chips));
    const n = this.mode.systems.length;
    this.grid = el('div', `systems n${n}`);
    const infos = (state.config && state.config.systems) || {};
    this.mode.systems.forEach((s) => {
      const card = makeSystemCard(s, infos[s] || { name: s, tag: '' });
      this.cards[s] = { el: card, done: false, t0: Date.now(), stages: {}, attempts: 0 };
      this.metrics[s] = {};
      this.grid.appendChild(card);
    });
    this.root.appendChild(this.grid);
    this.timer = setInterval(() => this.tick(), 1000);
    $('thread').appendChild(this.root);
    $('empty-state').hidden = true;
    this.root.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  tick() {
    Object.values(this.cards).forEach((c) => {
      if (!c.done) c.el.querySelector('.st').textContent = `running · ${Math.round((Date.now() - c.t0) / 1000)}s`;
    });
  }

  stage(sys, id, cls, badge) {
    const s = this.cards[sys].el.querySelector(`.stage[data-stage="${id}"]`);
    if (!s) return;
    s.classList.remove('running', 'done', 'failed', 'retry', 'skipped');
    s.classList.add(cls);
    const old = s.querySelector('.badge-n');
    if (old) old.remove();
    if (badge) s.insertAdjacentHTML('beforeend', `<span class="badge-n">${esc(badge)}</span>`);
  }

  skipStages(sys, ids) { ids.forEach((id) => this.stage(sys, id, 'skipped')); }

  onStep(ev) {
    const sys = ev.system;
    const card = this.cards[sys];
    if (!card) return;
    const d = ev.data || {};
    // Architecture strip
    const stageId = (STEP_STAGE[sys] || {})[ev.step];
    if (stageId) {
      let cls = ev.status === 'running' ? 'running' : ev.status === 'failed' ? 'failed' : 'done';
      let badge = null;
      if (ev.step === 'generate' && d.attempt > 1) badge = `×${d.attempt}`;
      if (ev.step === 'verify' && ev.status === 'done' && !d.passed) cls = 'retry';
      if (ev.step === 'verify' && d.attempt > 1) badge = `×${d.attempt}`;
      if (ev.step === 'generate' && ev.status === 'done' && (d.no_answer || d.truncated)) cls = 'retry';
      if (ev.step === 'evidence' && ev.status === 'failed') cls = 'retry';
      if (ev.step === 'final' && /declin|no_|failed/.test(d.outcome || '') && d.outcome !== 'verified') cls = 'retry';
      this.stage(sys, stageId, cls, badge);
      if (ev.step === 'computation') {
        const g = card.el.querySelector('.stage[data-stage="generate"] .slabel');
        if (g) g.textContent = 'Tax calculator';
      }
    }
    if (sys === 'sagot' && ev.step === 'route' && d.target !== 'rag') {
      this.skipStages(sys, d.target === 'computation' ? ['retrieval', 'fusion', 'rerank', 'verify'] : ['retrieval', 'fusion', 'rerank', 'generate', 'verify']);
      if (d.target === 'greeting') this.stage(sys, 'intent', 'skipped');
    }
    if (sys === 'sagot' && ev.step === 'evidence' && ev.status === 'failed') this.skipStages(sys, ['generate', 'verify']);
    if (sys === 'sagot' && ev.step === 'generate' && d.no_answer) this.skipStages(sys, ['verify']);

    // Process log: a "running" item is replaced by its "done" event.
    const key = `${sys}:${ev.step}:${d.attempt || ''}`;
    const info = describe(sys, ev);
    const time = ev.t_ms != null ? `${(ev.t_ms / 1000).toFixed(1)}s` : '';
    const iconName = { running: '', done: 'check', failed: 'close', warn: 'priority_high' }[info.kind];
    const html = `<span class="step-icon">${info.kind === 'running' ? '<span class="spinner"></span>' : icon(iconName)}</span>
      <div class="step-title">${info.title}<span class="step-time">${time}</span></div><div class="step-body">${info.body || ''}</div>`;
    let li = this.steps[key];
    if (!li) {
      li = el('li', 'step');
      this.steps[key] = li;
      card.el.querySelector('.steps').appendChild(li);
    }
    li.className = `step ${info.kind}`;
    li.innerHTML = html;
    const list = card.el.querySelector('.steps');
    list.scrollTop = list.scrollHeight;
    card.el.querySelector('.pcount').textContent = `· ${list.children.length} steps`;
    const live = card.el.querySelector('.live-text');
    if (live && !card.done) live.textContent = `${liveText(sys, ev)}…`;
  }

  onAnswer(ev) {
    const card = this.cards[ev.system];
    if (!card) return;
    const [text, kind] = OUTCOME_TEXT[ev.outcome] || [ev.outcome, 'neutral'];
    card.el.querySelector('.opill').innerHTML = pill(text, kind);
    card.el.querySelector('.answer-body').innerHTML = md(ev.answer || '');
    card.el.querySelector('.sources').innerHTML = (ev.sources || []).map((s) => `<span>${esc(s)}</span>`).join('');
    card.el.querySelector('.st').textContent = `answered in ${ev.answer_elapsed_s}s · scoring…`;
    card.el.querySelector('.live-text').textContent = 'Scoring the answer…';
    if (ev.system === 'sagot') {
      state.computation = ev.computation || null;
      updateNote();
    }
  }

  onMetric(ev) {
    const card = this.cards[ev.system];
    if (!card) return;
    const tile = card.el.querySelector(`.mtile[data-metric="${ev.metric}"]`);
    if (ev.status === 'done') this.metrics[ev.system][ev.metric] = ev.result;
    if (tile) tile.outerHTML = metricTile(ev.metric, ev.result, ev.status === 'running' ? 'running' : 'done');
  }

  onMetricsSkipped(ev) {
    const card = this.cards[ev.system];
    if (!card) return;
    card.el.querySelector('.metrics').outerHTML = `<div class="metrics-note">${icon('info')} ${esc(ev.reason)}</div>`;
    this.metrics[ev.system] = null;
  }

  onError(ev) {
    if (!ev.system || !this.cards[ev.system]) {
      this.root.appendChild(el('div', 'turn-error', `${icon('error')} ${esc(ev.message)}`));
      return;
    }
    const card = this.cards[ev.system];
    card.el.querySelector('.answer-body').innerHTML = `<div class="turn-error">${icon('error')} ${esc(ev.message)}</div>`;
    card.el.querySelectorAll('.stage.running').forEach((s) => { s.classList.remove('running'); s.classList.add('failed'); });
  }

  onSystemDone(ev) {
    const card = this.cards[ev.system];
    if (!card) return;
    card.done = true;
    card.el.querySelector('.live-line').hidden = true;
    card.el.querySelector('.sys-status').innerHTML = `${icon('done_all')}<span class="st">done in ${ev.elapsed_s}s</span>`;
    card.el.querySelectorAll('.mtile.wait').forEach((t) => {
      t.outerHTML = metricTile(t.dataset.metric, { score: null, note: 'Not computed.' }, 'done');
    });
  }

  finish() {
    clearInterval(this.timer);
    Object.keys(this.cards).forEach((s) => { if (!this.cards[s].done) this.onSystemDone({ system: s, elapsed_s: '—' }); });
    if (this.mode.systems.length > 1) this.root.appendChild(this.scoreboard());
  }

  scoreboard() {
    const systems = this.mode.systems;
    const infos = (state.config && state.config.systems) || {};
    const defs = (state.config && state.config.metrics) || {};
    const rows = METRIC_ORDER.map((m) => {
      const vals = systems.map((s) => (this.metrics[s] && this.metrics[s][m] ? this.metrics[s][m].score : null));
      const real = vals.filter((v) => v != null);
      // Star the best score only when it actually stands out (not when every system ties).
      const best = real.length > 1 && Math.max(...real) !== Math.min(...real) ? Math.max(...real) : null;
      return `<tr><th scope="row">${esc((defs[m] || {}).name || m)}</th>${vals.map((v, i) => v == null
        ? '<td class="cell na">N/A</td>'
        : `<td class="cell ${best != null && v === best ? 'best' : ''}" style="--sys:${SYS_COLOR[systems[i]]}"><span class="val">${pct(v)}</span><div class="bar"><i style="width:${Math.max(2, v * 100)}%"></i></div></td>`).join('')}</tr>`;
    }).join('');
    const board = el('div', 'board', `<h3>Side by side</h3><div style="overflow-x:auto"><table>
      <thead><tr><th>Score</th>${systems.map((s) => `<th><span class="sysh"><span class="sys-dot" style="--sys:${SYS_COLOR[s]}"></span>${
        esc((infos[s] || {}).name || s)} <span class="muted">${esc((infos[s] || {}).tag || '')}</span></span></th>`).join('')}</tr></thead>
      <tbody>${rows}</tbody></table></div>
      <p class="muted small" style="margin:8px 0 0">${rows.includes('class="cell best"') ? '★ = highest score for that row. ' : ''}N/A means the score does not apply to that system (for example, no retrieved documents to check against) — it is not a zero.</p>`);
    return board;
  }

  handle(ev) {
    switch (ev.type) {
      case 'step': this.onStep(ev); break;
      case 'answer': this.onAnswer(ev); break;
      case 'metric': this.onMetric(ev); break;
      case 'metrics_skipped': this.onMetricsSkipped(ev); break;
      case 'error': this.onError(ev); break;
      case 'system_done': this.onSystemDone(ev); break;
      case 'done': this.finish(); break;
      default: break; // start, ping
    }
  }
}

// ── Sending ─────────────────────────────────────────────────────────────────
function setRunning(on) {
  state.running = on;
  $('ask-send').disabled = on || (state.config && !state.config.can_run);
  $('ask-input').placeholder = on ? 'Working on the answer…' : 'Magtanong ka... Ask a BIR tax question';
}

async function ask(query) {
  const reference = $('ref-input').value.trim();
  const revie = $('revie-input').value.trim();
  const needsRevie = state.compare === 'c2' || state.compare === 'all';
  if (needsRevie && !revie) {
    setOptions(true);
    $('revie-box').open = true;
    $('revie-input').focus();
    $('composer-note').textContent = 'C2 needs REVIE’s answer: ask REVIE the same question, paste its answer, then send.';
    return;
  }
  const turn = new Turn(query, state.compare, !!reference);
  setOptions(false);
  setRunning(true);
  $('ask-input').value = '';
  $('ref-input').value = '';
  $('revie-input').value = '';
  syncDots();
  renderChips();
  try {
    const res = await fetch(API.run, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, compare: state.compare, reference, revie_answer: revie, computation_state: state.computation }),
    });
    if (!res.ok) {
      let msg = `${res.status} ${res.statusText}`;
      try {
        const j = await res.json();
        msg = typeof j.detail === 'string' ? j.detail : (j.detail && j.detail.message) || JSON.stringify(j.detail);
      } catch (e) { /* keep the status text */ }
      turn.handle({ type: 'error', system: null, message: msg });
      turn.handle({ type: 'done' });
      return;
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    let finished = false;
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, i).trim();
        buf = buf.slice(i + 1);
        if (!line) continue;
        const ev = JSON.parse(line);
        if (ev.type === 'done') finished = true;
        turn.handle(ev);
      }
    }
    if (!finished) {
      turn.handle({ type: 'error', system: null, message: 'The connection closed before the run finished.' });
      turn.handle({ type: 'done' });
    }
  } catch (e) {
    turn.handle({ type: 'error', system: null, message: `Could not reach the server: ${e.message}` });
    turn.handle({ type: 'done' });
  } finally {
    setRunning(false);
    $('ask-input').focus();
  }
}

function syncDots() {
  $('ref-dot').hidden = !$('ref-input').value.trim();
  $('revie-dot').hidden = !$('revie-input').value.trim();
}

function clearAll() {
  if (state.running) return;
  $('thread').innerHTML = '';
  $('empty-state').hidden = false;
  state.computation = null;
  updateNote();
}

document.addEventListener('DOMContentLoaded', () => {
  buildModes();
  setGuide(false);  // guide stays hidden until "Guide" is clicked
  setOptions(false);
  evalLinkVisibility();
  $('btn-guide').onclick = () => setGuide($('guide').hidden);
  $('btn-options').onclick = () => setOptions($('options-panel').hidden);
  $('btn-guide-close').onclick = () => setGuide(false);
  $('btn-clear').onclick = clearAll;
  $('ref-input').addEventListener('input', () => { syncDots(); renderChips(); });
  $('revie-input').addEventListener('input', syncDots);
  $('ask-form').addEventListener('submit', (e) => {
    e.preventDefault();
    const q = $('ask-input').value.trim();
    if (q && !state.running) ask(q);
  });
  loadConfig();
  $('ask-input').focus();
});
