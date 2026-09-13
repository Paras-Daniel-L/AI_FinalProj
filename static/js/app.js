/* ═══════════════════════════════════════════════
   State
═══════════════════════════════════════════════ */
let messages = [];        // current chat messages
let currentChatId = null;
let savedChats = {};      // { id: { title, ts, messages[] } }
let isLoading = false;

const API_BASE = '';      // same origin (served by FastAPI)

marked.setOptions({ gfm: true, breaks: true });

/* ═══════════════════════════════════════════════
   Init
═══════════════════════════════════════════════ */
window.addEventListener('DOMContentLoaded', () => {
  loadSavedChats();
  renderChatList();
  checkStatus();
  setInterval(checkStatus, 15000);
  applyTheme(localStorage.getItem('theme') || 'dark');
  newChat();
});

/* ═══════════════════════════════════════════════
   Theme
═══════════════════════════════════════════════ */
function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  document.getElementById('theme-icon').textContent = theme === 'dark' ? '☀️' : '🌙';
  localStorage.setItem('theme', theme);
}
function toggleTheme() {
  const cur = document.documentElement.getAttribute('data-theme');
  applyTheme(cur === 'dark' ? 'light' : 'dark');
}

/* ═══════════════════════════════════════════════
   Sidebar
═══════════════════════════════════════════════ */
function toggleSidebar() {
  document.getElementById('sidebar').classList.toggle('open');
  document.getElementById('sidebar-overlay').classList.toggle('visible');
}
function closeSidebar() {
  document.getElementById('sidebar').classList.remove('open');
  document.getElementById('sidebar-overlay').classList.remove('visible');
}

/* ═══════════════════════════════════════════════
   Input handling
═══════════════════════════════════════════════ */
function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 180) + 'px';
}
function updateSendBtn() {
  const val = document.getElementById('user-input').value.trim();
  const btn = document.getElementById('btn-send');
  btn.disabled = !val || isLoading;
  btn.classList.toggle('ready', !!val && !isLoading);
}
function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    if (!isLoading) sendMessage();
  }
}
function sendSuggestion(text) {
  document.getElementById('user-input').value = text;
  autoResize(document.getElementById('user-input'));
  updateSendBtn();
  sendMessage();
}

/* ═══════════════════════════════════════════════
   Chat management
═══════════════════════════════════════════════ */
function newChat() {
  if (messages.length > 0) saveCurrentChat();
  messages = [];
  currentChatId = 'chat_' + Date.now();
  document.getElementById('chat-title').textContent = 'New Conversation';
  clearMessageArea();
  document.getElementById('welcome').style.display = '';
  updateSendBtn();
  closeSidebar();
  renderChatList();
}

function clearChat() {
  if (!confirm('Clear this conversation?')) return;
  messages = [];
  clearMessageArea();
  document.getElementById('welcome').style.display = '';
  document.getElementById('chat-title').textContent = 'New Conversation';
}

function clearMessageArea() {
  // Remove all .msg-row elements
  const inner = document.getElementById('messages-inner');
  [...inner.querySelectorAll('.msg-row')].forEach(el => el.remove());
}

function saveCurrentChat() {
  if (!messages.length || !currentChatId) return;
  const firstUserMsg = messages.find(m => m.role === 'user');
  const title = firstUserMsg
    ? firstUserMsg.content.substring(0, 42) + (firstUserMsg.content.length > 42 ? '…' : '')
    : 'Conversation';
  savedChats[currentChatId] = { title, ts: Date.now(), messages: [...messages] };
  try { localStorage.setItem('sagot_chats', JSON.stringify(savedChats)); } catch {}
  renderChatList();
}

function loadSavedChats() {
  try {
    const raw = localStorage.getItem('sagot_chats');
    if (raw) savedChats = JSON.parse(raw);
  } catch {}
}

function loadChat(id) {
  if (messages.length > 0) saveCurrentChat();
  const chat = savedChats[id];
  if (!chat) return;
  currentChatId = id;
  messages = [...chat.messages];
  document.getElementById('chat-title').textContent = chat.title;
  clearMessageArea();
  document.getElementById('welcome').style.display = 'none';
  messages.forEach(m => {
    if (m.role === 'user') appendUserBubble(m.content, false);
    else appendBotBubble(m, false);
  });
  scrollToBottom();
  closeSidebar();
  renderChatList();
}

function renderChatList() {
  const list = document.getElementById('chat-list');
  const sorted = Object.entries(savedChats)
    .sort(([, a], [, b]) => b.ts - a.ts)
    .slice(0, 20);

  if (!sorted.length) {
    list.innerHTML = '<div style="padding:12px 16px; font-size:0.78rem; color:var(--text-faint);">No saved chats yet.</div>';
    return;
  }

  list.innerHTML = sorted.map(([id, chat]) => {
    const ago = timeAgo(chat.ts);
    const active = id === currentChatId ? 'active' : '';
    return `
      <div class="chat-item ${active}" onclick="loadChat('${id}')">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
        </svg>
        <span class="chat-item-text">${escHtml(chat.title)}</span>
        <span class="chat-item-time">${ago}</span>
      </div>`;
  }).join('');
}

/* ═══════════════════════════════════════════════
   Send message
═══════════════════════════════════════════════ */
async function sendMessage() {
  const input = document.getElementById('user-input');
  const query = input.value.trim();
  if (!query || isLoading) return;

  // Hide welcome
  document.getElementById('welcome').style.display = 'none';

  // Add user message to state + DOM
  messages.push({ role: 'user', content: query });
  appendUserBubble(query, true);

  // Reset input
  input.value = '';
  input.style.height = 'auto';
  updateSendBtn();

  // Update chat title
  if (messages.filter(m => m.role === 'user').length === 1) {
    const title = query.substring(0, 42) + (query.length > 42 ? '…' : '');
    document.getElementById('chat-title').textContent = title;
  }

  // Show typing
  setLoading(true);

  // Build history (exclude the current user message we just added)
  const history = messages.slice(0, -1).map(m => ({ role: m.role, content: m.content }));

  try {
    const res = await fetch(`${API_BASE}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, history }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    const botMsg = {
      role: 'assistant',
      content: data.answer,
      classification: data.classification,
      predicted_class: data.predicted_class,
      mode: data.mode,
      sources: data.sources || [],
    };
    messages.push(botMsg);
    appendBotBubble(botMsg, true);
    saveCurrentChat();

  } catch (err) {
    const errMsg = {
      role: 'assistant',
      content: `**Error:** ${err.message}\n\nPlease check that the API server is running and try again.`,
      classification: 'Error',
      predicted_class: -1,
      mode: 'error',
      sources: [],
    };
    messages.push(errMsg);
    appendBotBubble(errMsg, true);
  }

  setLoading(false);
}

/* ═══════════════════════════════════════════════
   DOM — Bubble helpers
═══════════════════════════════════════════════ */
function appendUserBubble(text, animate) {
  const row = document.createElement('div');
  row.className = 'msg-row user';
  row.innerHTML = `
    <div class="msg-header">
      <div class="msg-avatar user-av">U</div>
      <span class="msg-role">You</span>
      <span class="msg-time">${timeAgo(Date.now())}</span>
    </div>
    <div class="msg-bubble">${escHtml(text).replace(/\n/g, '<br>')}</div>
  `;
  insertBeforeTyping(row);
  if (animate) scrollToBottom();
}

function appendBotBubble(msg, animate) {
  const row = document.createElement('div');
  row.className = 'msg-row bot';

  // Classification badge class
  let classKey = 'chitchat';
  const pc = msg.predicted_class;
  if (pc === 1) classKey = 'tax-2001';
  else if (pc === 2) classKey = 'tax-2002';
  else if (pc === 3) classKey = 'tax-2003';
  else if (pc === 4) classKey = 'tax-2022';
  else if (pc === 5) classKey = 'general-tax';
  else if (pc === 6) classKey = 'board-games';

  const modeLabel = msg.mode === 'rag' ? '📚 RAG' : msg.mode === 'error' ? '⚠️ Error' : '💬 AI';
  const modeBadgeClass = msg.mode === 'rag' ? 'badge-mode-rag' : msg.mode === 'error' ? '' : 'badge-mode-conv';

  // Sources HTML
  let sourcesHtml = '';
  if (msg.sources && msg.sources.length) {
    const chips = msg.sources.map(s => `<span class="source-chip">${escHtml(s)}</span>`).join('');
    sourcesHtml = `
      <button class="sources-toggle" onclick="toggleSources(this)">
        <svg class="chevron" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>
        ${msg.sources.length} source${msg.sources.length > 1 ? 's' : ''}
      </button>
      <div class="sources-list">${chips}</div>
    `;
  }

  const parsed = marked.parse(msg.content || '');

  row.innerHTML = `
    <div class="msg-header">
      <div class="msg-avatar bot-av">S</div>
      <span class="msg-role">TaxAI</span>
      <span class="msg-time">${timeAgo(Date.now())}</span>
    </div>
    <div class="msg-bubble">
      ${parsed}
      <div class="msg-meta">
        <span class="badge badge-class ${classKey}">${escHtml(msg.classification || 'Unknown')}</span>
        <span class="badge ${modeBadgeClass}">${modeLabel}</span>
      </div>
      ${sourcesHtml}
    </div>
  `;

  // Add copy buttons to code blocks
  row.querySelectorAll('pre').forEach(pre => {
    const btn = document.createElement('button');
    btn.className = 'copy-code-btn';
    btn.textContent = 'copy';
    btn.onclick = () => {
      navigator.clipboard.writeText(pre.querySelector('code')?.textContent || pre.textContent);
      btn.textContent = 'copied!';
      setTimeout(() => { btn.textContent = 'copy'; }, 1800);
    };
    pre.style.position = 'relative';
    pre.appendChild(btn);
  });

  insertBeforeTyping(row);
  if (animate) scrollToBottom();
}

function insertBeforeTyping(el) {
  const inner = document.getElementById('messages-inner');
  const typingRow = document.getElementById('typing-row');
  inner.insertBefore(el, typingRow);
}

function toggleSources(btn) {
  btn.classList.toggle('open');
  btn.nextElementSibling.classList.toggle('visible');
}

/* ═══════════════════════════════════════════════
   Loading state
═══════════════════════════════════════════════ */
function setLoading(state) {
  isLoading = state;
  const typingRow = document.getElementById('typing-row');
  typingRow.style.display = state ? 'flex' : 'none';
  updateSendBtn();
  if (state) scrollToBottom();
}

function scrollToBottom() {
  const wrap = document.getElementById('messages-wrap');
  setTimeout(() => { wrap.scrollTop = wrap.scrollHeight; }, 30);
}

/* ═══════════════════════════════════════════════
   Upload
═══════════════════════════════════════════════ */
async function handleUpload(input) {
  const file = input.files[0];
  if (!file) return;
  if (!file.name.endsWith('.pdf')) { alert('Only PDF files are accepted.'); return; }

  const progress = document.getElementById('upload-progress');
  const bar = document.getElementById('upload-bar');
  const label = document.getElementById('upload-label');

  progress.style.display = 'block';
  bar.style.width = '20%';
  label.textContent = `Uploading ${file.name}…`;

  const fd = new FormData();
  fd.append('file', file);

  try {
    bar.style.width = '60%';
    const res = await fetch(`${API_BASE}/upload`, { method: 'POST', body: fd });
    bar.style.width = '100%';

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      label.textContent = `❌ ${err.detail || 'Upload failed'}`;
    } else {
      const data = await res.json();
      label.textContent = `✅ ${data.message}`;
      checkStatus();
    }
  } catch (e) {
    label.textContent = `❌ Network error`;
  }

  input.value = '';
  setTimeout(() => { progress.style.display = 'none'; bar.style.width = '0%'; }, 3000);
}

/* ═══════════════════════════════════════════════
   Reset DB
═══════════════════════════════════════════════ */
async function resetDB() {
  if (!confirm('This will permanently delete all indexed documents. Continue?')) return;
  try {
    const res = await fetch(`${API_BASE}/reset`, { method: 'DELETE' });
    const data = await res.json();
    alert(data.message);
    checkStatus();
  } catch (e) {
    alert('Reset failed: ' + e.message);
  }
}

/* ═══════════════════════════════════════════════
   Status
═══════════════════════════════════════════════ */
async function checkStatus() {
  const dot   = document.getElementById('status-dot');
  const label = document.getElementById('status-label');
  const count = document.getElementById('kb-count');

  try {
    const res = await fetch(`${API_BASE}/status`);
    if (!res.ok) throw new Error();
    const data = await res.json();
    dot.className = 'status-dot online';
    label.textContent = 'Online';
    count.textContent = `${data.document_count.toLocaleString()} chunks`;
  } catch {
    dot.className = 'status-dot offline';
    label.textContent = 'Offline';
    count.textContent = '—';
  }
}

/* ═══════════════════════════════════════════════
   Utilities
═══════════════════════════════════════════════ */
function timeAgo(ts) {
  const diff = Math.floor((Date.now() - ts) / 1000);
  if (diff < 60)   return 'just now';
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400)return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
