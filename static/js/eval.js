/* SagotAI · Evaluation tool (static/eval.html).
 *
 * Talks only to rag_eval/web.py (/eval/api/...). Every string that came from
 * a user, a chatbot or the judge is escaped (esc) or, for chatbot answers
 * (markdown), rendered through marked + DOMPurify — never inserted raw.
 */
'use strict';

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const S = {
  config: null,
  dataset: null,
  mode: { single: 'compare', batch: 'compare' },
  batchRows: null,
  batchSource: '',
  job: null,          // header of the package run being shown
  jobItems: [],
  pollTimer: null,
  thesis: null,
};

const SERIES = ['var(--series-1)', 'var(--series-2)'];
const SERIES_TRACK = ['var(--series-1-track)', 'var(--series-2-track)'];
const RQ1_COLORS = ['#2563eb', '#1baf7a'];

// ── Definitions (one source for the guide AND the results) ───────────────────

const METRICS = {
  groundedness: {
    name: 'Groundedness', short: 'Grounded', icon: 'verified', group: 'sop',
    question: 'Is every fact in the answer backed by the context?',
    how: 'The judge splits the answer into individual factual claims and checks each one against the context. Score = supported claims ÷ all claims.',
    why: 'A claim the documents do not support is a hallucination. This is the main score for hallucination mitigation.',
    needs: ['Context', 'Answer'],
    bands: [[1, 'All claims supported'], [0.8, 'Mostly supported'], [0.5, 'Partly supported'], [0, 'Mostly unsupported']],
    count: c => c.n_claims ? `${c.n_claims_supported} of ${c.n_claims} claims supported` : '',
  },
  context_relevance: {
    name: 'Context Relevance', short: 'Context', icon: 'plagiarism', group: 'sop',
    question: 'Did the chatbot find the right document text?',
    how: 'The judge goes through the context sentence by sentence and marks which ones help answer this question. Score = relevant sentences ÷ all sentences.',
    why: 'This rates the search step, not the answer. Scores of 0.2–0.5 are normal: each excerpt includes surrounding text from the regulation.',
    needs: ['Question', 'Context'],
    bands: [[0.5, 'Focused retrieval'], [0.2, 'Typical: excerpts include surrounding text'], [0, 'Mostly off-topic excerpts']],
    count: c => c.n_sentences ? `${c.n_sentences_relevant} of ${c.n_sentences} context sentences relevant` : '',
  },
  answer_relevance: {
    name: 'Answer Relevance', short: 'Relevant', icon: 'target', group: 'sop',
    question: 'Does the answer address the question that was asked?',
    how: 'The judge reads only the answer and writes 3 questions it would answer. Each is compared with the real question by meaning (embedding similarity), and the scores are averaged.',
    why: 'Best read by comparison between chatbots: the similarity rarely reaches 1.0 and rarely falls below about 0.4. A refusal scores low, since it answers nothing.',
    needs: ['Question', 'Answer'],
    bands: [[0.8, 'Directly on topic'], [0.6, 'Mostly on topic'], [0, 'Drifts from the question']],
    count: c => c.n_reverse_questions ? `average over ${c.n_reverse_questions} regenerated questions` : '',
  },
  answer_correctness: {
    name: 'Answer Correctness', short: 'Correct', icon: 'fact_check', group: 'supp',
    question: 'Does the answer state the correct facts?',
    how: 'The judge splits the ground truth into its key facts and marks each one as stated correctly, contradicted, or missing in the answer. Score = correct facts ÷ all facts.',
    why: 'A contradicted fact means the chatbot said something wrong: a hallucination measured against the official answer. Unlike Groundedness it needs no context, so it compares any two chatbots fairly.',
    needs: ['Answer', 'Ground truth'],
    bands: [[1, 'Matches the correct answer'], [0.5, 'Partly correct'], [0, 'Mostly missing or wrong']],
    count: c => c.n_facts ? `${c.n_facts_correct} of ${c.n_facts} key facts correct` : '',
  },
};
const METRIC_KEYS = Object.keys(METRICS);

const TRIGGER = [
  { key: 'precision', name: 'Trigger Precision', icon: 'my_location', formula: 'TP ÷ (TP + FP)',
    question: 'When Sagot AI flags a question as Taglish, how often is it right?',
    target: null, read: 'Higher is better. Low precision means English questions are being treated as Taglish.' },
  { key: 'recall', name: 'Trigger Recall', icon: 'radar', formula: 'TP ÷ (TP + FN)',
    question: 'Of all the Taglish questions, how many did Sagot AI catch?',
    target: { op: '>=', value: 0.90, text: 'Target ≥ 0.90 (SOP): no more than 1 in 10 Taglish questions missed.' },
    read: 'Higher is better.' },
  { key: 'fpr', name: 'False Positive Rate', icon: 'warning', formula: 'FP ÷ (FP + TN)', lowerBetter: true,
    question: 'Of all the English questions, how many were wrongly flagged as Taglish?',
    target: { op: '<', value: 0.10, text: 'Target < 0.10 (SOP).' },
    read: 'Lower is better.' },
];

const DATAPOINTS = [
  { name: 'Question', icon: 'help', what: 'What the user asks.',
    example: 'Is offshore gaming or POGO now completely banned and illegal in the Philippines?',
    sagot: ['edit', 'You type it'], other: ['edit', 'You type it (the same question you asked that chatbot)'] },
  { name: 'Context', icon: 'description', what: 'The document excerpts the chatbot read before answering.',
    example: '[1] RMC No. 13-2026 … Republic Act No. 12312 bans and declares illegal all offshore gaming operations …',
    sagot: ['bolt', 'Automatic: the excerpts Sagot AI retrieved'], other: ['content_paste', 'Paste it if the chatbot shows its sources. REVIE does not, so leave it blank'] },
  { name: 'Answer', icon: 'chat', what: 'What the chatbot replied.',
    example: 'REVIE: “POGOs remain fully legal and operational in the Philippines. Under PAGCOR Memorandum Circular No. 4-2023 …”',
    sagot: ['bolt', 'Automatic: Sagot AI is asked live'], other: ['content_paste', 'You paste its reply exactly'] },
  { name: 'Ground truth', icon: 'task_alt', what: 'The correct answer, written by you from the official BIR document.',
    example: 'Yes. Republic Act No. 12312 officially bans and declares illegal all offshore gaming operations in the Philippines.',
    sagot: ['edit', 'You write it (recommended)'], other: ['edit', 'You write it (recommended)'] },
];

const MODES = [
  { key: 'sagot', icon: 'smart_toy', name: 'Sagot AI', desc: 'Asked live. Its answer and context are filled in automatically.' },
  { key: 'external', icon: 'forum', name: 'Another chatbot', desc: 'You paste its answer, for example from REVIE.' },
  { key: 'compare', icon: 'compare', name: 'Compare both', desc: 'Sagot AI and the other chatbot, side by side, on the same question.' },
];

const COLUMNS = [
  ['id', 'No', 'A label for the row, so you can find it in the results.', 'A2-EN'],
  ['question', 'Yes', 'The question, exactly as asked.', 'Is POGO now banned in the Philippines?'],
  ['ground_truth', 'Recommended', 'The correct answer from the official document. Needed for Answer Correctness.', 'Yes. RA 12312 bans all offshore gaming.'],
  ['expected_language', 'Optional', '"english" or "taglish" (Filipino counts as taglish). Needed for the trigger scores.', 'english'],
  ['other_answer', 'For another chatbot', 'That chatbot\'s reply, pasted exactly.', 'POGOs remain fully legal…'],
  ['other_context', 'Optional', 'The document text that chatbot showed, if any. Leave blank for REVIE.', ''],
];

const OUTCOMES = {
  verified: ['good', 'check_circle', a => `Verified on attempt ${a || 1}`],
  verification_failed: ['neutral', 'block', () => 'Declined: failed verification'],
  generator_no_answer: ['neutral', 'block', () => 'Declined: not in the documents'],
  no_retrieval: ['neutral', 'block', () => 'Declined: no relevant documents'],
  no_index: ['critical', 'error', () => 'No documents indexed'],
  rejected_input: ['neutral', 'block', () => 'Question rejected by input rules'],
  cache_hit: ['good', 'check_circle', () => 'Verified (cached)'],
  greeting: ['neutral', 'waving_hand', () => 'Treated as a greeting'],
};

// ── Small helpers ────────────────────────────────────────────────────────────

function esc(v) {
  return String(v ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
function md(text) {
  const src = String(text || '');
  if (window.marked && window.DOMPurify) return window.DOMPurify.sanitize(window.marked.parse(src));
  return esc(src).replace(/\n/g, '<br>');
}
const isNum = v => typeof v === 'number' && Number.isFinite(v);
const fmt = (v, d = 2) => isNum(v) ? v.toFixed(d) : '—';
function fmtP(p) { if (!isNum(p)) return '—'; return p < 0.001 ? '< 0.001' : p.toFixed(3); }
function pText(p) { return !isNum(p) ? 'p not available' : (p < 0.001 ? 'p < 0.001' : `p = ${p.toFixed(3)}`); }
function band(key, v) {
  if (!isNum(v)) return '';
  for (const [t, label] of METRICS[key].bands) if (v >= t - 1e-9) return label;
  return '';
}
function duration(sec) {
  if (sec < 90) return `about ${Math.max(5, Math.round(sec / 5) * 5)} seconds`;
  const m = Math.round(sec / 60);
  return m < 90 ? `about ${m} minutes` : `about ${(sec / 3600).toFixed(1)} hours`;
}
function icon(name, cls = 'text-[18px]') { return `<span class="material-symbols-outlined ${cls}" aria-hidden="true">${name}</span>`; }
function statusChip(kind, iconName, text) {
  const styles = {
    good: 'background:#e7f6e7;color:var(--good-text)', critical: 'background:#fbe9e9;color:#9b1c1c',
    warning: 'background:#fff4dc;color:#7a5200', neutral: 'background:#f1f5f9;color:#334155',
  };
  return `<span class="chip" style="${styles[kind] || styles.neutral}">${icon(iconName, 'text-[14px]')}${esc(text)}</span>`;
}
function alertBox(kind, iconName, html) {
  const styles = {
    info: 'background:#eff6ff;border-color:#bfdbfe', warning: 'background:#fffbeb;border-color:#fde68a',
    critical: 'background:#fef2f2;border-color:#fecaca', good: 'background:#f0fdf4;border-color:#bbf7d0',
  };
  return `<div class="rounded-xl border p-4 flex gap-3 items-start text-body-sm" style="${styles[kind]}">${icon(iconName, 'text-[20px] shrink-0')}<div class="min-w-0">${html}</div></div>`;
}
function showError(el, err) {
  el.classList.remove('hidden');
  const problems = err.problems ? `<ul class="list-disc pl-5 mt-1">${err.problems.map(p => `<li>${esc(p)}</li>`).join('')}</ul>` : '';
  el.innerHTML = alertBox('critical', 'error', `<b>${esc(err.message)}</b>${problems}`);
}
function hideError(el) { el.classList.add('hidden'); el.innerHTML = ''; }
function otherName(scope) { return ($(`#${scope}-other-name`).value || '').trim() || 'Other chatbot'; }

async function api(path, options = {}) {
  const opts = { ...options, headers: { 'Content-Type': 'application/json', ...(options.headers || {}) } };
  if (opts.body && typeof opts.body !== 'string') opts.body = JSON.stringify(opts.body);
  let resp;
  try { resp = await fetch(path, opts); } catch (e) {
    throw Object.assign(new Error('Could not reach the server. Is it still running?'), {});
  }
  let data = null;
  try { data = await resp.json(); } catch (e) { /* non-JSON */ }
  if (!resp.ok) {
    const detail = data && data.detail;
    let message = `Request failed (HTTP ${resp.status}).`, problems = null;
    if (typeof detail === 'string') message = detail;
    else if (detail && detail.message) { message = detail.message; problems = detail.problems || null; }
    else if (Array.isArray(detail)) message = detail.map(d => d.msg).join('; ');
    throw Object.assign(new Error(message), { status: resp.status, problems, detail });
  }
  return data;
}

// Tooltip for any element with data-tip (charts, info icons).
document.addEventListener('mouseover', e => {
  const el = e.target.closest('[data-tip]');
  const tip = $('#tooltip');
  if (!el) { tip.classList.add('hidden'); return; }
  tip.textContent = el.getAttribute('data-tip');
  tip.classList.remove('hidden');
});
document.addEventListener('mousemove', e => {
  const tip = $('#tooltip');
  if (tip.classList.contains('hidden')) return;
  const x = Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
  const y = e.clientY + 18 + tip.offsetHeight > window.innerHeight ? e.clientY - tip.offsetHeight - 10 : e.clientY + 18;
  tip.style.left = `${x}px`; tip.style.top = `${y}px`;
});

// ── Tabs ─────────────────────────────────────────────────────────────────────

function showTab(name) {
  if (!['guide', 'single', 'batch', 'thesis'].includes(name)) name = 'guide';
  $$('.tab-btn').forEach(b => b.setAttribute('aria-selected', String(b.dataset.tab === name)));
  $$('[role="tabpanel"]').forEach(p => p.classList.toggle('hidden', p.id !== `tab-${name}`));
  if (location.hash !== `#${name}`) history.replaceState(null, '', `#${name}`);
  if (name === 'thesis' && !S.thesis) loadThesis();
  if (name === 'batch') loadHistory();
  window.scrollTo({ top: 0 });
}

// ── Guide ────────────────────────────────────────────────────────────────────

function renderGuide() {
  $('#datapoints').innerHTML = DATAPOINTS.map(d => `
    <div class="card flex flex-col gap-3">
      <div class="flex items-center gap-2">
        <span class="w-9 h-9 rounded-lg bg-primary-fixed text-primary flex items-center justify-center">${icon(d.icon, 'text-[20px]')}</span>
        <div><div class="text-headline-sm">${esc(d.name)}</div><div class="text-body-sm text-on-surface-variant">${esc(d.what)}</div></div>
      </div>
      <div class="text-body-sm bg-surface-container-low rounded-lg px-3 py-2 italic text-on-surface-variant">${esc(d.example)}</div>
      <div class="grid grid-cols-2 gap-2 text-body-sm">
        <div><div class="text-label-sm text-on-surface-variant uppercase">Sagot AI</div><div class="flex gap-1 mt-0.5">${icon(d.sagot[0], 'text-[16px] text-primary mt-[1px]')}<span>${esc(d.sagot[1])}</span></div></div>
        <div><div class="text-label-sm text-on-surface-variant uppercase">Other chatbot</div><div class="flex gap-1 mt-0.5">${icon(d.other[0], 'text-[16px] text-on-surface-variant mt-[1px]')}<span>${esc(d.other[1])}</span></div></div>
      </div>
    </div>`).join('');

  const metricCard = key => {
    const m = METRICS[key];
    return `
    <div class="card flex flex-col gap-2">
      <div class="flex items-center gap-2">${icon(m.icon, 'text-[22px] text-primary')}<div class="text-headline-sm">${m.name}</div></div>
      <div class="text-body-lg font-semibold">“${esc(m.question)}”</div>
      <p class="text-body-sm text-on-surface-variant"><b class="text-on-surface">How:</b> ${esc(m.how)}</p>
      <p class="text-body-sm text-on-surface-variant"><b class="text-on-surface">Why it matters:</b> ${esc(m.why)}</p>
      <div class="flex flex-wrap items-center gap-1.5 mt-1">
        <span class="text-label-sm text-on-surface-variant uppercase mr-1">Needs</span>
        ${m.needs.map(n => `<span class="chip bg-surface-container-low text-on-surface">${esc(n)}</span>`).join('')}
      </div>
      <div class="mt-1 text-body-sm">
        <div class="text-label-sm text-on-surface-variant uppercase mb-1">How to read it</div>
        <ul class="flex flex-col gap-0.5">${m.bands.map(([t, l], i) => {
          const upper = i === 0 ? '' : ` – ${m.bands[i - 1][0] === 1 ? '0.99' : (m.bands[i - 1][0] - 0.01).toFixed(2)}`;
          const range = t === 1 ? '1.00' : (i === 0 ? `${t.toFixed(2)} and up` : `${t.toFixed(2)}${upper}`);
          return `<li class="flex gap-2"><span class="tabular text-on-surface-variant w-24 shrink-0">${range}</span><span>${esc(l)}</span></li>`;
        }).join('')}</ul>
      </div>
    </div>`;
  };

  const trig = TRIGGER.map(t => `
    <div class="card flex flex-col gap-2">
      <div class="flex items-center gap-2">${icon(t.icon, 'text-[22px] text-primary')}<div class="text-headline-sm">${t.name}</div></div>
      <div class="text-body-lg font-semibold">“${esc(t.question)}”</div>
      <p class="text-body-sm text-on-surface-variant"><b class="text-on-surface">Formula:</b> <span class="tabular">${t.formula}</span>. ${esc(t.read)}</p>
      ${t.target ? `<p class="text-body-sm flex gap-1.5 items-start rounded-lg bg-surface-container-low px-3 py-2">${icon('flag', 'text-[16px] mt-[1px] shrink-0')}<span>${esc(t.target.text)}</span></p>` : ''}
    </div>`).join('');

  $('#metric-guide').innerHTML = `
    <div>
      <div class="flex items-baseline gap-2 flex-wrap"><h3 class="text-headline-sm">Answer quality</h3>
        <span class="text-body-sm text-on-surface-variant">The three RAGAS metrics in the SOP</span></div>
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-4 mt-3">${['groundedness', 'context_relevance', 'answer_relevance'].map(metricCard).join('')}</div>
    </div>
    <div>
      <div class="flex items-baseline gap-2 flex-wrap"><h3 class="text-headline-sm">Correctness</h3>
        <span class="text-body-sm text-on-surface-variant">Supplementary: the only score that uses the ground truth</span></div>
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-4 mt-3">${metricCard('answer_correctness')}
        <div class="card lg:col-span-2 flex flex-col gap-2 text-body-sm text-on-surface-variant">
          <div class="text-headline-sm text-on-surface">Why a fourth score?</div>
          <p>Groundedness checks an answer against the excerpts <i>the chatbot itself</i> retrieved. An answer faithfully built from an outdated excerpt still scores 1.00,
             and a chatbot that shows no excerpts, like REVIE, gets no Groundedness score at all.</p>
          <p>Answer Correctness checks every chatbot against the <b class="text-on-surface">same human-written ground truth</b>, so it is the fair way to answer
             “did it get the facts right?” for Sagot AI and REVIE alike. Extra correct detail is not penalized, so longer answers are not disadvantaged.</p>
          <p>It is not one of the SOP's three RAGAS metrics. The RAGAS framework has its own versions (answer correctness / factual correctness);
             present it as a supplementary measure.</p>
        </div>
      </div>
    </div>
    <div>
      <div class="flex items-baseline gap-2 flex-wrap"><h3 class="text-headline-sm">Language trigger</h3>
        <span class="text-body-sm text-on-surface-variant">Sagot AI only: does it notice when a question is Taglish? Measured over a package of questions.</span></div>
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-4 mt-3">${trig}</div>
      <div class="card mt-4 grid grid-cols-1 md:grid-cols-[auto,1fr] gap-5 items-center">
        ${confusionTable({ tp: 'TP', fn: 'FN', fp: 'FP', tn: 'TN' }, true)}
        <div class="text-body-sm text-on-surface-variant flex flex-col gap-1.5">
          <p><b class="text-on-surface">TP</b> (true positive): a Taglish question Sagot AI correctly flagged as Taglish.</p>
          <p><b class="text-on-surface">FN</b> (false negative): a Taglish question it missed and treated as English.</p>
          <p><b class="text-on-surface">FP</b> (false positive): an English question it wrongly flagged as Taglish.</p>
          <p><b class="text-on-surface">TN</b> (true negative): an English question it correctly left as English.</p>
          <p class="mt-1">“Taglish” here means any non-English question: Sagot AI's detector labels a question English, Filipino or Taglish, and both of the last two count as flagged.</p>
        </div>
      </div>
    </div>`;
}

function confusionTable(cm, legend = false) {
  const cell = (v, label, good) => `<td class="w-28 h-16 text-center rounded-lg" style="background:${good ? '#eef6ee' : '#fbefef'}">
      <div class="text-headline-md tabular">${esc(v)}</div>${legend ? '' : `<div class="text-label-sm text-on-surface-variant">${label}</div>`}</td>`;
  return `<table class="border-separate border-spacing-1.5 text-body-sm" aria-label="Confusion matrix">
    <thead><tr><td></td><th class="text-label-md text-on-surface-variant px-2" colspan="2">Sagot AI said</th></tr>
      <tr><td></td><th class="text-label-md text-on-surface-variant px-2">Taglish</th><th class="text-label-md text-on-surface-variant px-2">English</th></tr></thead>
    <tbody>
      <tr><th class="text-label-md text-on-surface-variant text-right pr-2">Actually<br>Taglish</th>${cell(cm.tp, 'TP', true)}${cell(cm.fn, 'FN', false)}</tr>
      <tr><th class="text-label-md text-on-surface-variant text-right pr-2">Actually<br>English</th>${cell(cm.fp, 'FP', false)}${cell(cm.tn, 'TN', true)}</tr>
    </tbody></table>`;
}

// ── Mode choice (shared by single + package) ─────────────────────────────────

function renderModeChoices() {
  $$('.mode-choices').forEach(box => {
    const scope = box.dataset.scope;
    box.innerHTML = MODES.map(m => `
      <label class="choice cursor-pointer">
        <input type="radio" name="mode-${scope}" value="${m.key}" class="sr-only" ${S.mode[scope] === m.key ? 'checked' : ''}/>
        <div class="h-full rounded-xl border border-outline-variant p-4 transition-all hover:border-primary/60">
          <div class="flex items-center gap-2 font-semibold">${icon(m.icon, 'text-[20px] text-primary')}${esc(m.name)}</div>
          <div class="text-body-sm text-on-surface-variant mt-1">${esc(m.desc)}</div>
        </div>
      </label>`).join('');
    box.addEventListener('change', e => {
      if (e.target.name !== `mode-${scope}`) return;
      S.mode[scope] = e.target.value;
      applyMode(scope);
    });
  });
  ['single', 'batch'].forEach(scope => {
    $(`#${scope}-other-name`).addEventListener('input', () => {
      $$(`#tab-${scope} .other-name`).forEach(el => { el.textContent = otherName(scope); });
      if (scope === 'batch') renderBatchPreview();
      updateStale(scope);
    });
    applyMode(scope);
  });
}

function applyMode(scope) {
  const mode = S.mode[scope];
  const root = $(`#tab-${scope}`);
  const hasSagot = mode !== 'external', hasOther = mode !== 'sagot';
  $$('.sagot-only', root).forEach(el => el.classList.toggle('hidden', !hasSagot));
  $$('.sagot-auto', root).forEach(el => el.classList.toggle('hidden', !hasSagot));
  $$('.other-only', root).forEach(el => el.classList.toggle('hidden', !hasOther));
  $('.other-name-row', root).classList.toggle('hidden', !hasOther);
  if (scope === 'single') updateSingleEstimate();
  else renderBatchPreview();
  updateStale(scope);
}

// Every result on screen carries a "Tested: …" bar stating the chatbot choice
// it was produced with. If the user then changes the choice, the bar says the
// result belongs to the earlier choice — the page never presents a result as
// if it came from a choice that wasn't the one tested.
function resultModeBar(scope, mode, name) {
  return `<div class="result-mode flex flex-wrap items-center gap-2 text-body-sm" data-mode="${esc(mode)}" data-name="${esc(name || '')}">
    ${icon('fact_check', 'text-[18px] text-primary')}<span>Tested: <b>${esc(modeLabel(mode, name))}</b></span>
    <span class="stale-note"></span></div>`;
}

function updateStale(scope) {
  $$(`#tab-${scope} .result-mode`).forEach(bar => {
    const mode = bar.dataset.mode, name = bar.dataset.name;
    const current = S.mode[scope], currentName = otherName(scope);
    const changed = current !== mode || (mode !== 'sagot' && name && currentName !== name);
    const what = scope === 'single' ? 'press Evaluate to test it' : 'start a new package test to test it';
    bar.querySelector('.stale-note').innerHTML = changed
      ? statusChip('warning', 'info', `Earlier choice. Your current choice is ${modeLabel(current, currentName)}: ${what}.`) : '';
  });
}

function secondsFor(mode, n) {
  const c = S.config || { seconds_per_sagot: 25, seconds_per_external: 10 };
  return n * ((mode !== 'external' ? c.seconds_per_sagot : 0) + (mode !== 'sagot' ? c.seconds_per_external : 0));
}

// ── Single question ──────────────────────────────────────────────────────────

function updateSingleEstimate() {
  const c = S.config;
  let text = `Takes ${duration(secondsFor(S.mode.single, 1))}.`;
  if (c && !c.can_run) text = 'Running tests only works on the host computer.';
  else if (c && c.judge_error) text = 'Set up the judge model first (see above).';
  $('#single-estimate').textContent = text;
}

function fillExample(id) {
  const row = (S.dataset || []).find(r => r.id === id);
  if (!row) return;
  $('#f-question').value = row.question;
  $('#f-ground-truth').value = row.ground_truth;
  $('#f-language').value = row.expected_language;
  $('#f-other-answer').value = row.other_answer;
  $('#f-other-context').value = '';
  $('#single-other-name').value = 'REVIE';

  $$('#tab-single .other-name').forEach(el => { el.textContent = 'REVIE'; });
}

function renderExamplePicker() {
  const groups = {};
  (S.dataset || []).forEach(r => { (groups[r.subset] = groups[r.subset] || []).push(r); });
  $('#example-picker').innerHTML = '<option value="">Choose…</option>' + Object.entries(groups).map(([subset, rows]) =>
    `<optgroup label="${esc(subset)}">${rows.map(r =>
      `<option value="${esc(r.id)}">${esc(r.id)} · ${esc(r.question.length > 60 ? r.question.slice(0, 57) + '…' : r.question)}</option>`).join('')}</optgroup>`).join('');
}

async function runSingle(e) {
  e.preventDefault();
  const err = $('#single-error');
  hideError(err);
  const mode = S.mode.single;
  const item = {
    question: $('#f-question').value.trim(),
    ground_truth: $('#f-ground-truth').value.trim(),
    expected_language: mode === 'external' ? '' : $('#f-language').value,
    other_answer: mode === 'sagot' ? '' : $('#f-other-answer').value.trim(),
    other_context: mode === 'sagot' ? '' : $('#f-other-context').value.trim(),
  };
  const problems = [];
  if (!item.question) problems.push('Enter the question.');
  if (mode !== 'sagot' && !item.other_answer) problems.push(`Paste ${otherName('single')}'s answer.`);
  if (problems.length) { showError(err, { message: 'Almost there:', problems }); return; }

  // Lock the form (mode, fields, button) while this question is evaluated, so
  // what is on screen can't drift from what was sent.
  const lock = $('#single-lock');
  const name = otherName('single');
  lock.disabled = true;
  const note = $('#single-lock-note');
  note.innerHTML = alertBox('info', 'lock', `<b>Evaluating: ${esc(modeLabel(mode, name))}.</b>
    The chatbot choice and the fields are locked until the result appears.`);
  note.classList.remove('hidden');
  const started = Date.now();
  const out = $('#single-result');
  const tick = () => {
    const s = Math.round((Date.now() - started) / 1000);
    out.innerHTML = `<div class="card flex items-center gap-3">${icon('progress_activity', 'spin text-primary text-[24px]')}
      <div><div class="font-semibold flex flex-wrap items-center gap-2">Evaluating… ${s}s
        <span class="chip bg-primary-fixed text-on-primary-fixed-variant">${esc(modeLabel(mode, name))}</span></div>
      <div class="text-body-sm text-on-surface-variant">${mode !== 'external' ? 'Sagot AI is answering, then ' : ''}the judge checks every claim, sentence and fact. This usually takes ${duration(secondsFor(mode, 1))}.
        The form above is locked until the result appears.</div></div></div>`;
  };
  tick();
  const timer = setInterval(tick, 1000);
  try {
    const result = await api('/eval/api/run/single', { method: 'POST', body: { mode, other_name: name, items: [item] } });
    out.innerHTML = `<div class="card !py-3 mb-4">${resultModeBar('single', mode, name)}</div>${renderItem(result)}`;
    out.scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (ex) {
    out.innerHTML = '';
    showError(err, ex);
  } finally {
    clearInterval(timer);
    lock.disabled = false;
    note.classList.add('hidden');
    note.innerHTML = '';
    updateStale('single');
  }
}

// ── Rendering one evaluated question ─────────────────────────────────────────

function languageLine(lang) {
  if (!lang) return '';
  const detected = `Sagot AI detected <b>${esc(lang.detected_label)}</b> → treated as <b>${lang.predicted_non_english ? 'Taglish' : 'English'}</b>.`;
  let verdict = '<span class="text-on-surface-variant">No expected language was given, so this is not scored.</span>';
  if (lang.correct === true) verdict = statusChip('good', 'check_circle', `Correct: expected ${lang.expected_non_english ? 'Taglish' : 'English'}`);
  if (lang.correct === false) verdict = statusChip('critical', 'cancel', `Wrong: expected ${lang.expected_non_english ? 'Taglish' : 'English'}`);
  return `<div class="card !py-3 flex flex-wrap items-center gap-x-3 gap-y-1 text-body-sm">${icon('translate', 'text-[20px] text-primary')}
    <span class="font-semibold">Language trigger</span><span>${detected}</span>${verdict}</div>`;
}

function outcomeChip(side) {
  if (side.system !== 'sagot') return statusChip('neutral', 'content_paste', 'Answer provided by you');
  const o = OUTCOMES[side.outcome];
  if (!o) return side.outcome ? statusChip('neutral', 'info', side.outcome) : '';
  return statusChip(o[0], o[1], o[2](side.meta && side.meta.attempts));
}

function metricRow(key, side, idx) {
  const m = METRICS[key];
  const v = side.metrics[key];
  const note = side.notes && side.notes[key];
  const count = isNum(v) ? m.count(side.counts || {}) : '';
  const bar = isNum(v)
    ? `<div class="meter mt-1.5" style="background:${SERIES_TRACK[idx]}" role="img" aria-label="${m.name} ${fmt(v)} out of 1">
         <div class="fill bar" style="width:${Math.max(v * 100, 1.5)}%;background:${SERIES[idx]}"></div></div>`
    : '';
  const contradicted = key === 'answer_correctness' && (side.counts || {}).n_facts_contradicted > 0
    ? `<div class="mt-1.5">${statusChip('critical', 'cancel', `Contradicts the ground truth on ${side.counts.n_facts_contradicted} fact${side.counts.n_facts_contradicted > 1 ? 's' : ''}`)}</div>` : '';
  return `
    <div class="py-3 border-b border-surface-container-low last:border-0">
      <div class="flex items-baseline justify-between gap-3">
        <div class="flex items-center gap-1.5 font-semibold text-body-sm">${m.name}
          <span class="material-symbols-outlined text-[15px] text-outline cursor-help" data-tip="${esc(m.question)}" tabindex="0" aria-label="${esc(m.question)}">info</span>
          ${m.group === 'supp' ? '<span class="chip bg-surface-container-low text-on-surface-variant !text-label-sm">supplementary</span>' : ''}</div>
        <div class="text-headline-sm">${fmt(v)}</div>
      </div>
      ${bar}
      <div class="text-body-sm mt-1 ${isNum(v) ? '' : 'text-on-surface-variant'}">${isNum(v)
        ? `<span class="font-medium">${esc(band(key, v))}</span>${count ? ` <span class="text-on-surface-variant">· ${esc(count)}</span>` : ''}`
        : esc(note || 'Not available.')}</div>
      ${contradicted}
    </div>`;
}

function detailsBlock(side) {
  const d = side.details || {};
  const list = (title, items, render) => items && items.length ? `
    <div><div class="text-label-md text-on-surface-variant uppercase mb-1.5">${title}</div>
      <ul class="flex flex-col gap-1.5">${items.map(render).join('')}</ul></div>` : '';
  const mark = (ok, yes, no) => ok
    ? `<span class="shrink-0" style="color:var(--good-text)" title="${yes}">${icon('check_circle', 'text-[17px]')}</span>`
    : `<span class="shrink-0" style="color:var(--critical)" title="${no}">${icon('cancel', 'text-[17px]')}</span>`;
  const verdictChip = v => v === 'correct' ? statusChip('good', 'check_circle', 'correct')
    : v === 'contradicted' ? statusChip('critical', 'cancel', 'contradicted') : statusChip('neutral', 'remove', 'missing');
  const body = [
    list('Claims in the answer (Groundedness)', d.claims, c => `<li class="flex gap-2">${mark(c.supported, 'supported', 'not supported')}<span>${esc(c.claim)}</span></li>`),
    list('Ground-truth facts (Answer Correctness)', d.facts, f => `<li class="flex flex-wrap items-start gap-2">${verdictChip(f.verdict)}<span class="flex-1 min-w-[12rem]">${esc(f.fact)}${f.evidence ? `<span class="block text-on-surface-variant italic">Answer: “${esc(f.evidence)}”</span>` : ''}</span></li>`),
    list('Questions the judge wrote from the answer (Answer Relevance)', d.reverse_questions, q => `<li class="flex gap-3"><span class="tabular text-on-surface-variant w-10 shrink-0">${fmt(q.similarity)}</span><span>${esc(q.question)}</span></li>`),
    list('Context sentences (Context Relevance)', d.sentences, s => `<li class="flex gap-2">${s.relevant ? mark(true, 'relevant', '') : `<span class="shrink-0 text-outline" title="not relevant">${icon('radio_button_unchecked', 'text-[17px]')}</span>`}<span class="${s.relevant ? '' : 'text-on-surface-variant'}">${esc(s.text)}</span></li>`),
  ].filter(Boolean).join('');
  if (!body) return '';
  return `<details class="mt-3 rounded-lg bg-surface-container-low">
    <summary class="px-4 py-2.5 flex items-center gap-1.5 text-body-sm font-semibold">${icon('chevron_right', 'chev text-[18px] transition-transform')}How each score was computed</summary>
    <div class="px-4 pb-4 flex flex-col gap-4 text-body-sm">${body}</div></details>`;
}

function renderSide(side, idx) {
  if (side.error) {
    return `<div class="card">${sideHeader(side, idx)}${alertBox('critical', 'error', `<b>This side could not be evaluated.</b><div class="mt-1 break-words">${esc(side.error)}</div>`)}</div>`;
  }
  const ctx = side.context && side.context.trim();
  const meta = [];
  if (isNum(side.elapsed_s)) meta.push(`${side.elapsed_s}s`);
  if (side.meta && isNum(side.meta.cost)) meta.push(`Sagot AI pipeline cost $${side.meta.cost.toFixed(5)}`);
  return `<div class="card flex flex-col">
    ${sideHeader(side, idx)}
    <div class="mt-3">
      <div class="text-label-md text-on-surface-variant uppercase mb-1">Answer</div>
      <div class="md text-body-md bg-surface-container-low rounded-lg px-4 py-3 break-words">${md(side.answer)}</div>
    </div>
    <div class="mt-2">${METRIC_KEYS.map(k => metricRow(k, side, idx)).join('')}</div>
    ${detailsBlock(side)}
    ${ctx ? `<details class="mt-2 rounded-lg bg-surface-container-low">
      <summary class="px-4 py-2.5 flex items-center gap-1.5 text-body-sm font-semibold">${icon('chevron_right', 'chev text-[18px] transition-transform')}Context the chatbot used</summary>
      <pre class="px-4 pb-4 text-body-sm whitespace-pre-wrap break-words font-sans text-on-surface-variant">${esc(side.context)}</pre></details>` : ''}
    ${meta.length ? `<div class="text-label-md text-on-surface-variant mt-3 font-normal">${esc(meta.join(' · '))}</div>` : ''}
  </div>`;
}

function sideHeader(side, idx) {
  return `<div class="flex flex-wrap items-center justify-between gap-2">
    <div class="flex items-center gap-2"><span class="w-3 h-3 rounded-full" style="background:${SERIES[idx]}" aria-hidden="true"></span>
      <span class="text-headline-sm">${esc(side.system_name)}</span></div>${side.error ? '' : outcomeChip(side)}</div>`;
}

function renderItem(item) {
  const gt = item.ground_truth
    ? `<div class="mt-2"><span class="text-label-md text-on-surface-variant uppercase">Ground truth</span><div class="text-body-md">${esc(item.ground_truth)}</div></div>`
    : `<div class="mt-2 text-body-sm text-on-surface-variant">No ground truth given, so Answer Correctness is not scored.</div>`;
  const sides = item.sides || [];
  return `<div class="flex flex-col gap-4">
    <div class="card">
      <div class="text-label-md text-on-surface-variant uppercase">Question${item.id ? ` · ${esc(item.id)}` : ''}</div>
      <div class="text-body-lg font-semibold">${esc(item.question)}</div>${gt}
    </div>
    ${languageLine(item.language)}
    <div class="grid grid-cols-1 ${sides.length > 1 ? 'lg:grid-cols-2' : ''} gap-4 items-start">${sides.map((s, i) => renderSide(s, sideIndex(s, sides, i))).join('')}</div>
  </div>`;
}
// Sagot AI is always series 1 (blue); another chatbot always series 2 (orange),
// so a color means the same system on every screen.
function sideIndex(side) { return side.system === 'sagot' ? 0 : 1; }

// ── Package: setup ───────────────────────────────────────────────────────────

function renderColumnGuide() {
  $('#column-guide').innerHTML = `<thead><tr><th class="th">Column</th><th class="th">Required?</th><th class="th">What to put</th><th class="th">Example</th></tr></thead>
    <tbody>${COLUMNS.map(([c, r, w, ex]) => `<tr><td class="td font-mono text-[12px]">${c}</td><td class="td whitespace-nowrap">${r}</td><td class="td">${esc(w)}</td><td class="td text-on-surface-variant">${esc(ex)}</td></tr>`).join('')}</tbody>`;
}

async function onFileChosen(e) {
  const file = e.target.files[0];
  e.target.value = '';
  if (!file) return;
  const err = $('#batch-error');
  hideError(err);
  try {
    const text = await file.text();
    const parsed = await api('/eval/api/parse', { method: 'POST', body: { text, filename: file.name } });
    setBatchRows(parsed.rows, file.name, parsed.warnings);
  } catch (ex) { showError(err, ex); }
}

async function loadTed() {
  hideError($('#batch-error'));
  const subset = $('#ted-subset').value;
  const rows = (S.dataset || []).filter(r => subset === 'all' || r.subset.startsWith(`${subset}:`));
  if (S.mode.batch !== 'sagot') $('#batch-other-name').value = 'REVIE';
  $$('#tab-batch .other-name').forEach(el => { el.textContent = otherName('batch'); });
  setBatchRows(rows.map(({ id, question, ground_truth, expected_language, other_answer, other_context }) =>
    ({ id, question, ground_truth, expected_language, other_answer, other_context })), `T-TED dataset (${$('#ted-subset').selectedOptions[0].text})`, []);
}

function setBatchRows(rows, source, warnings) {
  S.batchRows = rows;
  S.batchSource = source;
  S.batchWarnings = warnings || [];
  renderBatchPreview();
}

function renderBatchPreview() {
  const box = $('#batch-preview');
  const run = $('#batch-run');
  const est = $('#batch-estimate');
  const rows = S.batchRows;
  if (!rows || !rows.length) {
    box.innerHTML = '';
    run.disabled = true;
    est.textContent = (S.job && S.job.status === 'running') ? 'Locked while the current package test runs.' : 'Add questions first.';
    return;
  }
  const mode = S.mode.batch, name = otherName('batch');
  const n = rows.length;
  const nGt = rows.filter(r => r.ground_truth).length;
  const nLang = rows.filter(r => r.expected_language).length;
  const nOther = rows.filter(r => r.other_answer).length;
  const missingOther = mode !== 'sagot' ? n - nOther : 0;
  const warn = [...(S.batchWarnings || [])];
  if (missingOther) warn.unshift(`${missingOther} question${missingOther > 1 ? 's have' : ' has'} no answer from ${name}. Add an other_answer column, or choose “Sagot AI” only.`);
  if (mode !== 'external' && nLang && nLang < n) warn.push(`${n - nLang} question(s) have no expected language and are left out of the trigger scores.`);
  const stat = (v, label) => `<div class="rounded-lg bg-surface-container-low px-3 py-2"><div class="text-headline-sm tabular">${v}</div><div class="text-label-md text-on-surface-variant font-normal">${label}</div></div>`;
  const show = rows.slice(0, 6);
  box.innerHTML = `
    <div class="flex flex-wrap items-center justify-between gap-2">
      <div class="text-body-sm"><b>${esc(S.batchSource)}</b></div>
      <button type="button" class="btn-ghost !py-1" id="batch-clear">${icon('close', 'text-[16px]')}Remove</button>
    </div>
    <div class="grid grid-cols-2 md:grid-cols-4 gap-2 mt-2">
      ${stat(n, 'questions')}${stat(nGt, 'with ground truth')}${stat(nLang, 'with expected language')}${stat(nOther, `with ${esc(name)}'s answer`)}
    </div>
    ${warn.length ? `<div class="mt-3">${alertBox(missingOther ? 'critical' : 'warning', missingOther ? 'error' : 'warning', `<ul class="list-disc pl-4">${warn.map(w => `<li>${esc(w)}</li>`).join('')}</ul>`)}</div>` : ''}
    <div class="overflow-x-auto mt-3 rounded-lg border border-surface-container">
      <table class="w-full text-body-sm"><thead><tr><th class="th">id</th><th class="th">Question</th><th class="th">Ground truth</th><th class="th">Language</th>${mode !== 'sagot' ? `<th class="th">${esc(name)}'s answer</th>` : ''}</tr></thead>
      <tbody>${show.map(r => `<tr><td class="td whitespace-nowrap">${esc(r.id)}</td><td class="td min-w-[14rem]">${esc(r.question)}</td>
        <td class="td min-w-[12rem] text-on-surface-variant">${esc(r.ground_truth) || '<i>none</i>'}</td><td class="td">${esc(r.expected_language) || '—'}</td>
        ${mode !== 'sagot' ? `<td class="td min-w-[12rem] text-on-surface-variant">${esc(r.other_answer) || '<span class="text-error">missing</span>'}</td>` : ''}</tr>`).join('')}</tbody></table>
      ${n > show.length ? `<div class="px-3 py-2 text-body-sm text-on-surface-variant">…and ${n - show.length} more.</div>` : ''}
    </div>`;
  $('#batch-clear').onclick = () => setBatchRows(null, '', []);
  const blocked = !!missingOther || !(S.config && S.config.can_run && !S.config.judge_error) || !!(S.job && S.job.status === 'running');
  run.disabled = blocked;
  est.textContent = (S.job && S.job.status === 'running') ? 'Locked while the current package test runs.'
    : missingOther ? 'Fix the missing answers first.'
    : `${n} question${n > 1 ? 's' : ''}; takes ${duration(secondsFor(mode, n))}. Every question makes paid model calls.`;
}

async function startBatch() {
  const err = $('#batch-error');
  hideError(err);
  const mode = S.mode.batch;
  const items = S.batchRows.map(r => ({
    id: r.id || '', question: r.question, ground_truth: r.ground_truth || '',
    expected_language: mode === 'external' ? '' : (r.expected_language || ''),
    other_answer: mode === 'sagot' ? '' : (r.other_answer || ''), other_context: mode === 'sagot' ? '' : (r.other_context || ''),
  }));
  const n = items.length;
  if (!confirm(`Start a package test of ${n} question${n > 1 ? 's' : ''}?\n\nIt takes ${duration(secondsFor(mode, n))} and makes paid model calls for every question. You can cancel at any time; finished questions are kept.`)) return;
  $('#batch-run').disabled = true;
  try {
    const header = await api('/eval/api/run/batch', { method: 'POST', body: { mode, other_name: otherName('batch'), items } });
    attachJob(header);
  } catch (ex) {
    showError(err, ex);
    if (ex.status === 409 && ex.detail && ex.detail.job_id) attachJob({ id: ex.detail.job_id, status: 'running' });
    renderBatchPreview();
  }
}

// ── Package: progress + results ──────────────────────────────────────────────

function modeLabel(mode, name) {
  const other = name || 'Other chatbot';
  return { sagot: 'Sagot AI only', external: `Another chatbot (${other})`,
           compare: `Compare both (Sagot AI vs ${other})` }[mode] || mode || 'package test';
}

// A running package test keeps the mode it was started with (the server fixes
// it at start). So while one runs, its setup is locked: the chatbot choice and
// the questions can't be changed, and the selected card is set to the mode
// actually running, so the screen never shows one mode while another runs.
// Also applies after a page reload, when the page reattaches to a running test.
function syncBatchLock() {
  const job = S.job;
  const running = !!(job && job.status === 'running');
  const note = $('#batch-lock-note');
  if (running && job.mode && S.mode.batch !== job.mode) {
    S.mode.batch = job.mode;
    const radio = $(`input[name="mode-batch"][value="${job.mode}"]`);
    if (radio) radio.checked = true;
  }
  if (running && job.other_name && job.mode !== 'sagot') {
    $('#batch-other-name').value = job.other_name;
    $$('#tab-batch .other-name').forEach(el => { el.textContent = job.other_name; });
  }
  $('#batch-setup').disabled = running;
  note.classList.toggle('hidden', !running);
  note.innerHTML = running ? alertBox('info', 'lock',
    `<b>A package test is running: ${esc(job.mode ? modeLabel(job.mode, job.other_name) : 'loading…')}.</b>
     The chatbot choice and the questions are locked until it finishes or you cancel it.`) : '';
  applyMode('batch');  // show/hide the fields for the (possibly synced) mode
}

function attachJob(header) {
  S.job = header;
  S.jobItems = [];
  clearTimeout(S.pollTimer);
  syncBatchLock();
  poll();
}

async function poll() {
  if (!S.job) return;
  try {
    const st = await api(`/eval/api/jobs/${encodeURIComponent(S.job.id)}?since=${S.jobItems.length}`);
    S.jobItems.push(...st.items);
    S.job = st;
    syncBatchLock();  // also re-renders the preview for the current mode
    renderProgress(st);
    renderBatchResults(st.summary, S.jobItems, st);
    if (st.status === 'running') S.pollTimer = setTimeout(poll, 1500);
    else loadHistory();
  } catch (ex) {
    $('#batch-progress').innerHTML = alertBox('critical', 'error', `<b>Lost track of the package test.</b> ${esc(ex.message)} Reload the page to reattach.`);
  }
}

function renderProgress(st) {
  const box = $('#batch-progress');
  if (st.status !== 'running') { box.innerHTML = ''; return; }
  const pct = st.total ? Math.round(100 * st.done / st.total) : 0;
  const remaining = secondsFor(st.mode, st.total - st.done);
  box.innerHTML = `<div class="card">
    <div class="flex flex-wrap items-center justify-between gap-3">
      <div class="flex items-center gap-2 font-semibold">${icon('progress_activity', 'spin text-primary text-[22px]')}Running: ${st.done} of ${st.total} questions done
        <span class="chip bg-primary-fixed text-on-primary-fixed-variant">${esc(modeLabel(st.mode, st.other_name))}</span></div>
      <button type="button" id="batch-cancel" class="btn-ghost">${icon('stop_circle', 'text-[18px]')}Cancel</button>
    </div>
    <div class="meter mt-3" style="background:var(--series-1-track)" role="progressbar" aria-valuenow="${pct}" aria-valuemin="0" aria-valuemax="100">
      <div class="fill bar" style="width:${Math.max(pct, 1)}%;background:var(--series-1)"></div></div>
    <div class="text-body-sm text-on-surface-variant mt-2">${st.current ? `Now: “${esc(st.current.length > 110 ? st.current.slice(0, 107) + '…' : st.current)}”. ` : ''}About ${duration(remaining).replace('about ', '')} left. You can leave this page open; results appear below as they finish.</div>
  </div>`;
  $('#batch-cancel').onclick = async () => {
    try { await api(`/eval/api/run/cancel/${encodeURIComponent(st.id)}`, { method: 'POST' }); } catch (ex) { /* already finished */ }
  };
}

function groupedBars(rows, series, opts = {}) {
  // rows: [{label, values:[v|null...], notes:[...], n:[...]}]; series: [{name, color}]
  const legend = series.length > 1 ? `<div class="flex flex-wrap gap-4 text-body-sm mb-3">${series.map(s =>
    `<span class="flex items-center gap-1.5"><span class="w-3 h-3 rounded-sm" style="background:${s.color}"></span>${esc(s.name)}</span>`).join('')}</div>` : '';
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  const body = rows.map(r => `
    <div class="grid grid-cols-[8.5rem,1fr] sm:grid-cols-[10rem,1fr] items-center gap-3 py-2">
      <div class="text-body-sm font-semibold leading-tight">${esc(r.label)}${r.sub ? `<div class="text-label-sm text-on-surface-variant font-normal">${esc(r.sub)}</div>` : ''}</div>
      <div class="relative flex flex-col gap-[2px]">
        ${ticks.map(t => `<div class="absolute top-0 bottom-0 w-px" style="left:${t * 100}%;background:var(--grid)"></div>`).join('')}
        ${series.map((s, i) => {
          const v = r.values[i];
          if (!isNum(v)) return `<div class="relative h-[18px] flex items-center text-label-md text-on-surface-variant font-normal pl-1" data-tip="${esc(`${s.name} · ${r.label}: ${r.notes && r.notes[i] || 'not measurable'}`)}">— ${esc(r.notes && r.notes[i] ? r.notes[i] : 'not measurable')}</div>`;
          const tip = `${s.name} · ${r.label}: ${fmt(v)}${r.n && isNum(r.n[i]) ? ` (${r.n[i]} scored)` : ''}`;
          return `<div class="relative h-[18px] flex items-center" data-tip="${esc(tip)}">
            <div class="h-full bar" style="width:${Math.max(v * 100, 0.8)}%;background:${s.color};border-radius:0 4px 4px 0"></div>
            <span class="text-label-md tabular pl-1.5 text-on-surface">${fmt(v)}</span></div>`;
        }).join('')}
      </div>
    </div>`).join('');
  const axis = `<div class="grid grid-cols-[8.5rem,1fr] sm:grid-cols-[10rem,1fr] gap-3"><div></div>
    <div class="relative h-4 text-label-sm text-on-surface-variant font-normal tabular">${ticks.map(t =>
      `<span class="absolute -translate-x-1/2" style="left:${t * 100}%">${t}</span>`).join('')}</div></div>`;
  return `${opts.title ? `<div class="text-headline-sm">${esc(opts.title)}</div>` : ''}
    ${opts.subtitle ? `<div class="text-body-sm text-on-surface-variant mb-3">${esc(opts.subtitle)}</div>` : ''}${legend}
    <div class="pr-10">${body}${axis}</div>`;
}

function interpret(c, nameA, nameB, context = '') {
  if (!c) return '';
  if (c.n_pairs === 0) return context || 'Not comparable: no question has this score on both sides.';
  if (c.n_pairs < 3) return `Only ${c.n_pairs} question${c.n_pairs > 1 ? 's have' : ' has'} this score on both sides, too few to test (at least 3 needed).`;
  if (!c.test_used || c.test_used.startsWith('none')) return c.note || 'No test could be run.';
  const higher = c.mean_diff > 0 ? nameA : nameB;
  const diff = Math.abs(c.mean_diff);
  const eff = c.effect_size_label ? `${c.effect_size_label} effect size (${c.effect_size_type === 'cohens_d' ? 'd' : 'r'} = ${fmt(Math.abs(c.effect_size))})` : 'effect size not available';
  if (c.significant) return `${higher} scored higher by ${fmt(diff)} on average. The difference is statistically significant (${pText(c.p_value)}, ${c.test_used}) with a ${eff}.`;
  return `No statistically significant difference (${pText(c.p_value)}, ${c.test_used}); ${eff}. ${diff > 0.0005 ? `${higher} was ${fmt(diff)} higher on average, which may be chance.` : ''}`;
}

function statsTable(comparison, nameA, nameB, reasons = {}) {
  if (!comparison) return '';
  // The plain-English reading gets its own full-width line under each metric:
  // it is the part a panel member reads, so it must never be cut off by width.
  const rows = METRIC_KEYS.filter(k => comparison[k]).map(k => {
    const c = comparison[k];
    return `<tr>
      <td class="td !border-0 !pb-1 font-semibold whitespace-nowrap">${METRICS[k].name}</td>
      <td class="td !border-0 !pb-1 tabular text-right">${fmt(c.mean_a)}</td><td class="td !border-0 !pb-1 tabular text-right">${fmt(c.mean_b)}</td>
      <td class="td !border-0 !pb-1 tabular text-right">${isNum(c.mean_diff) ? (c.mean_diff > 0 ? '+' : '') + c.mean_diff.toFixed(2) : '—'}</td>
      <td class="td !border-0 !pb-1 tabular text-right">${c.n_pairs}</td>
      <td class="td !border-0 !pb-1">${esc(c.test_used || '—')}</td>
      <td class="td !border-0 !pb-1 tabular text-right whitespace-nowrap">${fmtP(c.p_value)}</td>
      <td class="td !border-0 !pb-1 whitespace-nowrap">${c.significant ? statusChip('good', 'check', 'yes') : (isNum(c.p_value) ? '<span class="text-on-surface-variant">no</span>' : '—')}</td>
      <td class="td !border-0 !pb-1 whitespace-nowrap">${isNum(c.effect_size) ? `${fmt(Math.abs(c.effect_size))} <span class="text-on-surface-variant">${esc(c.effect_size_label || '')}</span>` : '—'}</td>
    </tr>
    <tr><td class="td !pt-0 text-on-surface-variant" colspan="9">
      <span class="inline-flex gap-1.5 items-start">${icon('subdirectory_arrow_right', 'text-[16px] mt-[2px] shrink-0')}<span>${esc(interpret(c, nameA, nameB, reasons[k]))}</span></span></td></tr>`;
  }).join('');
  return `<div class="overflow-x-auto rounded-lg border border-surface-container">
    <table class="w-full text-body-sm"><thead><tr>
      <th class="th">Metric</th><th class="th text-right">${esc(nameA)}</th><th class="th text-right">${esc(nameB)}</th><th class="th text-right">Difference</th>
      <th class="th text-right">Pairs</th><th class="th">Test</th><th class="th text-right">p-value</th><th class="th">Significant?</th><th class="th">Effect size</th>
    </tr></thead><tbody>${rows}</tbody></table></div>`;
}

function triggerSection(t, title = 'Language trigger (Sagot AI)') {
  if (!t) return '';
  const tile = def => {
    const v = t[def.key];
    let status = '';
    if (def.target && isNum(v)) {
      const ok = def.target.op === '>=' ? v >= def.target.value : v < def.target.value;
      status = ok ? statusChip('good', 'check_circle', `Meets target (${def.target.op === '>=' ? '≥' : '<'} ${def.target.value.toFixed(2)})`)
        : statusChip('critical', 'cancel', `Misses target (${def.target.op === '>=' ? '≥' : '<'} ${def.target.value.toFixed(2)})`);
    }
    const undefinedWhy = def.key === 'fpr' ? 'Needs at least one English question.' : def.key === 'recall' ? 'Needs at least one Taglish question.' : 'Nothing was flagged as Taglish.';
    return `<div class="rounded-xl border border-surface-container p-4">
      <div class="flex items-center gap-1.5 text-body-sm font-semibold">${def.name}
        <span class="material-symbols-outlined text-[15px] text-outline cursor-help" data-tip="${esc(def.question + ' ' + def.formula)}" tabindex="0">info</span></div>
      <div class="text-display-lg mt-1">${fmt(v)}</div>
      ${isNum(v) ? `<div class="meter mt-2" style="background:var(--series-1-track)">
        <div class="fill bar" style="width:${Math.max(v * 100, 1)}%;background:var(--series-1)"></div>
        ${def.target ? `<div class="mark" style="left:calc(${def.target.value * 100}% - 1px)" data-tip="Target ${def.target.op === '>=' ? '≥' : '<'} ${def.target.value}"></div>` : ''}</div>` : ''}
      <div class="mt-2 text-body-sm">${isNum(v) ? status || `<span class="text-on-surface-variant">${esc(def.read)}</span>` : `<span class="text-on-surface-variant">${undefinedWhy}</span>`}</div>
    </div>`;
  };
  return `<div class="card">
    <div class="text-headline-sm">${esc(title)}</div>
    <div class="text-body-sm text-on-surface-variant">${t.n} question${t.n === 1 ? '' : 's'} with an expected language${t.n_unlabeled ? `; ${t.n_unlabeled} without one left out` : ''}.</div>
    <div class="grid grid-cols-1 md:grid-cols-3 gap-3 mt-4">${TRIGGER.map(tile).join('')}</div>
    <div class="mt-4 flex flex-wrap items-center gap-6">${confusionTable(t)}
      <p class="text-body-sm text-on-surface-variant max-w-md">Green cells are correct calls, red cells are mistakes. Precision, recall and the false positive rate are all computed from these four counts.</p></div>
  </div>`;
}

function headlineTiles(summary) {
  const tiles = [];
  Object.values(summary.systems).forEach(sys => {
    const idx = sys.system_name === 'Sagot AI' ? 0 : 1;
    const c = sys.contradictions;
    if (c && c.judged) {
      tiles.push(`<div class="card"><div class="flex items-center gap-2 text-body-sm font-semibold"><span class="w-2.5 h-2.5 rounded-full" style="background:${SERIES[idx]}"></span>${esc(sys.system_name)}</div>
        <div class="text-display-lg mt-1">${c.with_contradiction} <span class="text-headline-sm text-on-surface-variant">of ${c.judged}</span></div>
        <div class="text-body-sm text-on-surface-variant">answers contradict the ground truth (state a fact wrongly)</div></div>`);
    }
    if (sys.refusals && sys.refusals.n) {
      tiles.push(`<div class="card"><div class="flex items-center gap-2 text-body-sm font-semibold"><span class="w-2.5 h-2.5 rounded-full" style="background:${SERIES[idx]}"></span>${esc(sys.system_name)}</div>
        <div class="text-display-lg mt-1">${sys.refusals.refused} <span class="text-headline-sm text-on-surface-variant">of ${sys.refusals.n}</span></div>
        <div class="text-body-sm text-on-surface-variant">questions declined (not enough support in the documents)</div></div>`);
    }
    if (sys.n_errors) {
      tiles.push(`<div class="card"><div class="text-body-sm font-semibold">${esc(sys.system_name)}</div>
        <div class="text-display-lg mt-1" style="color:var(--critical)">${sys.n_errors}</div><div class="text-body-sm text-on-surface-variant">questions could not be evaluated (see the table)</div></div>`);
    }
  });
  return tiles.length ? `<div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">${tiles.join('')}</div>` : '';
}

function averagesChart(summary) {
  const systems = Object.entries(summary.systems);
  const series = systems.map(([key, sys]) => ({ name: sys.system_name, color: SERIES[key === 'sagot' ? 0 : 1] }));
  const rows = METRIC_KEYS.map(k => ({
    label: METRICS[k].name, sub: METRICS[k].group === 'supp' ? 'supplementary' : '',
    values: systems.map(([, sys]) => sys.metrics[k].mean),
    n: systems.map(([, sys]) => sys.metrics[k].n_scored),
    notes: systems.map(([key, sys]) => sys.metrics[k].n_scored ? '' :
      (k === 'answer_correctness' ? 'no ground truths' : (key !== 'sagot' && ['groundedness', 'context_relevance'].includes(k) ? 'no context shown' : 'not measurable'))),
  }));
  return `<div class="card">${groupedBars(rows, series, {
    title: 'Average score by metric',
    subtitle: series.length > 1 ? 'Higher is better. Questions a score could not be computed for are left out of its average.' : `${series[0].name}. Higher is better.`,
  })}</div>`;
}

function renderBatchResults(summary, items, header) {
  const box = $('#batch-results');
  if (!items.length) { box.innerHTML = ''; return; }
  const systems = Object.values(summary.systems);
  const nameA = systems[0] ? systems[0].system_name : 'A';
  const nameB = systems[1] ? systems[1].system_name : 'B';
  const statusText = { running: 'In progress', done: 'Finished', cancelled: 'Cancelled', error: 'Stopped by an error', interrupted: 'Interrupted (server restarted)' }[header.status] || header.status;
  box.innerHTML = `<div class="flex flex-col gap-4">
    <div class="flex flex-wrap items-end justify-between gap-3">
      <div><h2 class="text-headline-md">Results</h2>
        <div class="text-body-sm text-on-surface-variant">${items.length} of ${header.total} questions · ${esc(statusText)}${header.error ? ` · ${esc(header.error)}` : ''}</div>
        <div class="mt-1.5">${resultModeBar('batch', header.mode, header.other_name)}</div></div>
      <div class="flex gap-2 no-print">
        <button type="button" class="btn-secondary" id="batch-download">${icon('download', 'text-[18px]')}Download CSV</button>
        <button type="button" class="btn-ghost" onclick="window.print()">${icon('print', 'text-[18px]')}Print / PDF</button>
      </div>
    </div>
    ${headlineTiles(summary)}
    ${averagesChart(summary)}
    ${summary.comparison ? `<div class="card"><div class="text-headline-sm">Is the difference real?</div>
      <div class="text-body-sm text-on-surface-variant mb-3">Paired comparison: the same questions answered by both chatbots.</div>
      ${statsTable(summary.comparison, nameA, nameB, { groundedness: `${nameB} shows no context, so this score exists only for ${nameA}.`, context_relevance: `${nameB} shows no context, so this score exists only for ${nameA}.` })}</div>` : ''}
    ${triggerSection(summary.trigger)}
    <div class="card">
      <div class="text-headline-sm">Every question</div>
      <div class="text-body-sm text-on-surface-variant mb-3">Click a row to see the answers and how each score was computed.</div>
      ${questionTable(items)}
    </div>
  </div>`;
  $('#batch-download').onclick = () => downloadCsv(items, header);
  updateStale('batch');
  $$('#batch-results tr[data-idx]').forEach(tr => {
    const open = () => openDetail(`${tr.dataset.label}`, renderItem(items[Number(tr.dataset.idx)]));
    tr.onclick = open;
    tr.onkeydown = e => { if (e.key === 'Enter') open(); };
  });
}

function cellScore(v, contradicted) {
  return `<td class="td tabular text-right whitespace-nowrap">${fmt(v)}${contradicted ? ` <span style="color:var(--critical)" data-tip="Contradicts the ground truth">${icon('cancel', 'text-[14px] align-[-2px]')}</span>` : ''}</td>`;
}

function questionTable(items) {
  const first = items[0] || { sides: [] };
  const heads = first.sides.map(s => `<th class="th text-center" colspan="4"><span class="inline-block w-2.5 h-2.5 rounded-full mr-1 align-middle" style="background:${SERIES[sideIndex(s)]}"></span>${esc(s.system_name)}</th>`).join('');
  const sub = first.sides.map(() => METRIC_KEYS.map(k => `<th class="th text-right" data-tip="${esc(METRICS[k].name)}">${METRICS[k].short}</th>`).join('')).join('');
  const rows = items.map((it, i) => {
    const lang = it.language && it.language.correct !== null ? (it.language.correct ? statusChip('good', 'check', it.language.detected_label) : statusChip('critical', 'close', it.language.detected_label)) : '';
    const cells = it.sides.map(s => s.error
      ? `<td class="td" colspan="4" style="color:var(--critical)">${icon('error', 'text-[14px] align-[-2px]')} error</td>`
      : METRIC_KEYS.map(k => cellScore(s.metrics[k], k === 'answer_correctness' && (s.counts || {}).n_facts_contradicted > 0)).join('')).join('');
    return `<tr data-idx="${i}" data-label="${esc(it.id || `#${i + 1}`)}" tabindex="0" class="cursor-pointer hover:bg-surface-container-low focus:bg-surface-container-low focus:outline-none">
      <td class="td whitespace-nowrap font-semibold">${esc(it.id || i + 1)}</td>
      <td class="td min-w-[16rem] max-w-md"><div class="line-clamp-2">${esc(it.question)}</div>${lang ? `<div class="mt-1">${lang}</div>` : ''}</td>${cells}</tr>`;
  }).join('');
  return `<div class="overflow-x-auto rounded-lg border border-surface-container max-h-[36rem] overflow-y-auto">
    <table class="w-full text-body-sm"><thead class="sticky top-0 bg-surface-container-lowest z-[1]"><tr><th class="th" rowspan="2">id</th><th class="th" rowspan="2">Question</th>${heads}</tr><tr>${sub}</tr></thead>
    <tbody>${rows}</tbody></table></div>`;
}

function openDetail(title, html) {
  $('#detail-title').textContent = title;
  $('#detail-body').innerHTML = html;
  const dlg = $('#detail-dialog');
  if (!dlg.open) dlg.showModal();
  $('#detail-body').scrollTop = 0;
}

function csvCell(v) {
  const s = v === null || v === undefined ? '' : String(v);
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function downloadCsv(items, header) {
  const first = items[0] || { sides: [] };
  const cols = ['id', 'question', 'ground_truth', 'expected_language', 'detected_language', 'trigger_correct'];
  first.sides.forEach(s => {
    const p = s.system_name.replace(/\W+/g, '_');
    cols.push(`${p}_answer`, `${p}_outcome`, ...METRIC_KEYS.map(k => `${p}_${k}`), `${p}_facts_contradicted`, `${p}_error`);
  });
  const lines = [cols.join(',')];
  items.forEach(it => {
    const row = [it.id, it.question, it.ground_truth,
      it.expected_language === null || it.expected_language === undefined ? '' : (it.expected_language ? 'taglish' : 'english'),
      it.language ? it.language.detected_label : '', it.language && it.language.correct !== null ? it.language.correct : ''];
    it.sides.forEach(s => {
      row.push(s.answer || '', s.outcome || '', ...METRIC_KEYS.map(k => s.metrics ? s.metrics[k] : ''),
        s.counts ? s.counts.n_facts_contradicted : '', s.error || '');
    });
    lines.push(row.map(csvCell).join(','));
  });
  const blob = new Blob(['﻿' + lines.join('\r\n')], { type: 'text/csv;charset=utf-8' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `evaluation_${header.id || 'results'}.csv`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

async function loadHistory() {
  const box = $('#batch-history');
  try {
    const runs = await api('/eval/api/runs');
    if (!runs.length) { box.innerHTML = ''; return; }
    const modeName = m => ({ sagot: 'Sagot AI', external: 'Other chatbot', compare: 'Compare' }[m] || m);
    box.innerHTML = `<details class="card" ${S.job ? '' : 'open'}>
      <summary class="flex items-center gap-1.5 text-headline-sm">${icon('chevron_right', 'chev text-[20px] transition-transform')}Previous package tests</summary>
      <div class="overflow-x-auto mt-3"><table class="w-full text-body-sm"><thead><tr><th class="th">Started</th><th class="th">Chatbot</th><th class="th">Questions</th><th class="th">Status</th><th class="th"></th></tr></thead>
      <tbody>${runs.map(r => `<tr><td class="td whitespace-nowrap">${esc(new Date(r.created * 1000).toLocaleString())}</td>
        <td class="td">${esc(modeName(r.mode))}${r.mode !== 'sagot' && r.other_name ? ` · ${esc(r.other_name)}` : ''}</td>
        <td class="td tabular">${r.done} / ${r.total}</td><td class="td">${esc(r.status)}</td>
        <td class="td text-right"><button type="button" class="btn-ghost !py-1" data-run="${esc(r.id)}">Open</button></td></tr>`).join('')}</tbody></table></div></details>`;
    $$('#batch-history [data-run]').forEach(b => { b.onclick = () => { attachJob({ id: b.dataset.run }); $('#batch-results').scrollIntoView({ behavior: 'smooth' }); }; });
  } catch (ex) { box.innerHTML = ''; }
}

// ── Thesis results ───────────────────────────────────────────────────────────

async function loadThesis() {
  const body = $('#thesis-body');
  body.innerHTML = `<div class="card flex items-center gap-3">${icon('progress_activity', 'spin text-primary')}Loading results…</div>`;
  try {
    S.thesis = await api('/eval/api/thesis');
    renderThesis();
  } catch (ex) {
    body.innerHTML = alertBox('critical', 'error', `<b>Could not load the thesis results.</b> ${esc(ex.message)}`);
  }
}

function renderThesis() {
  const t = S.thesis;
  const p = t.progress;
  const complete = p.sagot_scored === p.sagot_total && p.revie_scored === p.revie_total;
  const progressMeter = (label, n, total) => `<div class="flex-1 min-w-[14rem]">
      <div class="flex justify-between text-body-sm"><span class="font-semibold">${label}</span><span class="tabular">${n} / ${total}</span></div>
      <div class="meter mt-1.5" style="background:var(--series-1-track)"><div class="fill" style="width:${total ? Math.max(100 * n / total, n ? 1 : 0) : 0}%;background:var(--series-1)"></div></div></div>`;
  const status = `<div class="card">
    <div class="flex flex-wrap items-center justify-between gap-3">
      <div class="text-headline-sm">Run progress</div>
      <button type="button" class="btn-ghost" id="thesis-refresh">${icon('refresh', 'text-[18px]')}Refresh</button>
    </div>
    <div class="flex flex-wrap gap-6 mt-3">${progressMeter('Sagot AI answers scored', p.sagot_scored, p.sagot_total)}${progressMeter('REVIE answers scored (Subset B)', p.revie_scored, p.revie_total)}</div>
    ${!complete ? `<div class="mt-3">${alertBox('warning', 'hourglass_top', `The offline run is not complete yet, so the results below are <b>partial</b>. Finish it with
      <code class="bg-white/70 px-1 rounded">python -m rag_eval.run_evaluation --subset all</code> (it resumes where it stopped), then press Refresh.`)}</div>` : ''}
    ${p.outdated_scoring ? `<div class="mt-3">${alertBox('info', 'update', `${p.outdated_scoring} answer(s) were scored before Answer Correctness was added. Run the same command again:
      they are re-scored automatically <b>without calling the chatbot again</b>.`)}</div>` : ''}
  </div>`;

  const rq1 = t.rq1, rq3 = t.rq3, rq2 = t.rq2.confusion_metrics;
  const rq1Rows = METRIC_KEYS.map(k => ({ label: METRICS[k].name, sub: METRICS[k].group === 'supp' ? 'supplementary' : '',
    values: [rq1.metrics[k].mean_a, rq1.metrics[k].mean_b], n: [rq1.metrics[k].n_pairs, rq1.metrics[k].n_pairs], notes: ['not scored yet', 'not scored yet'] }));
  const rq3Rows = METRIC_KEYS.map(k => ({ label: METRICS[k].name, sub: METRICS[k].group === 'supp' ? 'supplementary' : '',
    values: [rq3.metrics[k].mean_a, rq3.metrics[k].mean_b], n: [rq3.metrics[k].n_pairs, rq3.metrics[k].n_pairs],
    notes: ['not scored yet', ['groundedness', 'context_relevance'].includes(k) ? 'REVIE shows no context' : 'not scored yet'] }));
  // A paired comparison drops unpaired questions, so for G/CR (REVIE has none)
  // show Sagot AI's own average over Subset B instead of a blank.
  const bTable = t.table.filter(r => r.subset.startsWith('B'));
  ['groundedness', 'context_relevance'].forEach((k, i) => {
    const vals = bTable.map(r => r.sagot && r.sagot.metrics ? r.sagot.metrics[k] : null).filter(isNum);
    rq3Rows[i].values[0] = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
    rq3Rows[i].n[0] = vals.length;
  });
  const c3 = rq3.contradictions || {};
  const contradictionTile = (name, idx, c) => c && c.judged ? `<div class="card">
      <div class="flex items-center gap-2 text-body-sm font-semibold"><span class="w-2.5 h-2.5 rounded-full" style="background:${SERIES[idx]}"></span>${name}</div>
      <div class="text-display-lg mt-1">${c.with_contradiction} <span class="text-headline-sm text-on-surface-variant">of ${c.judged}</span></div>
      <div class="text-body-sm text-on-surface-variant">answers contradict the ground truth</div></div>` : '';
  const reasons3 = { groundedness: 'REVIE shows no document context, so Groundedness exists only for Sagot AI and cannot be compared.',
    context_relevance: 'REVIE shows no document context, so Context Relevance exists only for Sagot AI and cannot be compared.' };

  $('#thesis-body').innerHTML = `${status}
    <div class="flex flex-col gap-4">
      <div><div class="text-label-md text-primary uppercase">RQ1</div><h2 class="text-headline-md">English vs Taglish: does Sagot AI perform the same?</h2>
        <p class="text-body-sm text-on-surface-variant">Subset A: 20 English questions and their Taglish translations, answered by Sagot AI (${Math.max(...METRIC_KEYS.map(k => rq1.metrics[k].n_pairs))} of 20 pairs scored).
        The hoped-for result is <b>no meaningful difference</b>.</p></div>
      <div class="card">${groupedBars(rq1Rows, [{ name: 'English', color: RQ1_COLORS[0] }, { name: 'Taglish', color: RQ1_COLORS[1] }], { title: 'Average score by language' })}</div>
      <div class="card"><div class="text-headline-sm mb-3">Statistical test</div>${statsTable(rq1.metrics, 'English', 'Taglish')}
        <p class="text-body-sm text-on-surface-variant mt-3">“No significant difference” with 20 pairs is consistent with equal performance, but it does not prove the two are identical. Report the effect size alongside it.</p></div>
    </div>
    <div class="flex flex-col gap-4">
      <div><div class="text-label-md text-primary uppercase">RQ2</div><h2 class="text-headline-md">Does Sagot AI notice Taglish questions?</h2>
        <p class="text-body-sm text-on-surface-variant">Subset A's 40 questions (20 known English, 20 known Taglish) through Sagot AI's language detector. Needs no model calls, so it is always complete.</p></div>
      ${triggerSection({ ...rq2, n_unlabeled: 0 }, 'Language trigger on Subset A')}
    </div>
    <div class="flex flex-col gap-4">
      <div><div class="text-label-md text-primary uppercase">RQ3</div><h2 class="text-headline-md">Sagot AI vs BIR's REVIE</h2>
        <p class="text-body-sm text-on-surface-variant">Subset B: 60 Taglish questions. Sagot AI answered live; REVIE's answers were collected by hand. Both were graded the same way.</p></div>
      <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">${contradictionTile('Sagot AI', 0, c3.sagot)}${contradictionTile('REVIE', 1, c3.revie)}</div>
      <div class="card">${groupedBars(rq3Rows, [{ name: 'Sagot AI', color: SERIES[0] }, { name: 'REVIE', color: SERIES[1] }],
        { title: 'Average score by chatbot', subtitle: 'Groundedness and Context Relevance need the document context a chatbot used; REVIE shows none, so only Sagot AI has them.' })}</div>
      <div class="card"><div class="text-headline-sm mb-3">Statistical test</div>${statsTable(rq3.metrics, 'Sagot AI', 'REVIE', reasons3)}</div>
    </div>
    <div class="card">
      <div class="flex flex-wrap items-center justify-between gap-3">
        <div><div class="text-headline-sm">Every question</div><div class="text-body-sm text-on-surface-variant">Click a row to see both answers and how each score was computed.</div></div>
        <select id="thesis-filter" class="field !w-auto"><option value="">All subsets</option><option value="A">Subset A</option><option value="B">Subset B</option></select>
      </div>
      <div id="thesis-table" class="mt-3"></div>
    </div>`;
  $('#thesis-refresh').onclick = () => { S.thesis = null; loadThesis(); };
  $('#thesis-filter').onchange = renderThesisTable;
  renderThesisTable();
}

function renderThesisTable() {
  const filter = $('#thesis-filter').value;
  const rows = S.thesis.table.filter(r => !filter || r.subset.startsWith(filter));
  const cells = (u, keys) => {
    if (!u) return `<td class="td text-on-surface-variant" colspan="${keys.length}">not run</td>`;
    if (u.error) return `<td class="td" colspan="${keys.length}" style="color:var(--critical)">error</td>`;
    return keys.map(k => cellScore(u.metrics[k], k === 'answer_correctness' && u.contradicted > 0)).join('');
  };
  const revieKeys = ['answer_relevance', 'answer_correctness'];
  $('#thesis-table').innerHTML = `<div class="overflow-x-auto rounded-lg border border-surface-container max-h-[40rem] overflow-y-auto">
    <table class="w-full text-body-sm"><thead class="sticky top-0 bg-surface-container-lowest z-[1]">
      <tr><th class="th" rowspan="2">id</th><th class="th" rowspan="2">Question</th>
        <th class="th text-center" colspan="4"><span class="inline-block w-2.5 h-2.5 rounded-full mr-1 align-middle" style="background:${SERIES[0]}"></span>Sagot AI</th>
        <th class="th text-center" colspan="2"><span class="inline-block w-2.5 h-2.5 rounded-full mr-1 align-middle" style="background:${SERIES[1]}"></span>REVIE</th></tr>
      <tr>${METRIC_KEYS.map(k => `<th class="th text-right">${METRICS[k].short}</th>`).join('')}${revieKeys.map(k => `<th class="th text-right">${METRICS[k].short}</th>`).join('')}</tr></thead>
    <tbody>${rows.map(r => `<tr data-qid="${esc(r.qid)}" tabindex="0" class="cursor-pointer hover:bg-surface-container-low focus:bg-surface-container-low focus:outline-none">
      <td class="td whitespace-nowrap font-semibold">${esc(r.qid)}</td><td class="td min-w-[16rem] max-w-md"><div class="line-clamp-2">${esc(r.question)}</div></td>
      ${cells(r.sagot, METRIC_KEYS)}${r.subset.startsWith('B') ? cells(r.revie, revieKeys) : `<td class="td text-on-surface-variant text-center" colspan="2" data-tip="REVIE is compared on Subset B">—</td>`}</tr>`).join('')}</tbody></table></div>`;
  $$('#thesis-table tr[data-qid]').forEach(tr => {
    const open = async () => {
      openDetail(tr.dataset.qid, `<div class="flex items-center gap-2">${icon('progress_activity', 'spin text-primary')}Loading…</div>`);
      try {
        const detail = await api(`/eval/api/thesis/${encodeURIComponent(tr.dataset.qid)}`);
        $('#detail-body').innerHTML = detail.sides.length ? renderItem(detail)
          : alertBox('info', 'hourglass_top', 'This question has not been scored yet by the offline run.');
      } catch (ex) { $('#detail-body').innerHTML = alertBox('critical', 'error', esc(ex.message)); }
    };
    tr.onclick = open;
    tr.onkeydown = e => { if (e.key === 'Enter') open(); };
  });
}

// ── Start-up ─────────────────────────────────────────────────────────────────

function renderBanners() {
  const c = S.config;
  const out = [];
  if (!c.can_run) out.push(alertBox('info', 'lock', `<b>View-only.</b> ${esc(c.lock_reason)}`));
  if (c.judge_error) out.push(alertBox('warning', 'gavel', `<b>The judge model is not set up, so tests cannot run yet.</b> ${esc(c.judge_error)}`));
  $('#banners').innerHTML = out.join('');
  const chip = $('#judge-chip');
  if (c.judge_model && !c.judge_error) {
    chip.innerHTML = `${icon('gavel', 'text-[14px]')}Judge: ${esc(c.judge_model)}`;
    chip.classList.add('lg:inline-flex');  // stays hidden on small screens
  }
  $('#single-run').disabled = !(c.can_run && !c.judge_error);
  updateSingleEstimate();
}

async function init() {
  renderGuide();
  renderColumnGuide();
  renderModeChoices();
  $$('.tab-btn').forEach(b => { b.onclick = () => showTab(b.dataset.tab); });
  $$('[data-goto]').forEach(b => { b.onclick = () => showTab(b.dataset.goto); });
  window.addEventListener('hashchange', () => showTab(location.hash.slice(1)));
  $('#single-form').addEventListener('submit', runSingle);
  $('#example-picker').addEventListener('change', e => { if (e.target.value) fillExample(e.target.value); });
  $('#batch-file').addEventListener('change', onFileChosen);
  $('#ted-load').addEventListener('click', loadTed);
  $('#batch-run').addEventListener('click', startBatch);
  $('#detail-dialog').addEventListener('click', e => { if (e.target.id === 'detail-dialog') e.target.close(); });

  try {
    [S.config, S.dataset] = await Promise.all([api('/eval/api/config'), api('/eval/api/dataset')]);
  } catch (ex) {
    $('#banners').innerHTML = alertBox('critical', 'error', `<b>Could not load the evaluation tool.</b> ${esc(ex.message)}`);
    return;
  }
  renderBanners();
  renderExamplePicker();
  renderBatchPreview();
  showTab(location.hash.slice(1) || 'guide');
  if (S.config.running_job) attachJob({ id: S.config.running_job, status: 'running' });
}

init();