/* ═══════════════════════════════════════════════
   State & Configuration
═══════════════════════════════════════════════ */
let messages = [];
let isLoading = false;
const API_BASE = ''; // Backend server origin (FastAPI)[cite: 3]

if (typeof marked !== 'undefined') {
  marked.setOptions({ gfm: true, breaks: true });
}

/* ═══════════════════════════════════════════════
   View Transitions
═══════════════════════════════════════════════ */
function switchToChatView() {
  document.getElementById('landing-view').classList.add('hidden');
  document.getElementById('chat-view').classList.remove('hidden');
  document.getElementById('chat-footer').classList.remove('hidden');
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function newChat() {
  if (isLoading) return;
  
  // 1. Wipe the backend message history array
  messages = [];
  
  // 2. Clear the UI chat history, but keep the typing indicator
  const chatHistory = document.getElementById('chat-history');
  const typingRow = document.getElementById('typing-row');
  chatHistory.innerHTML = ''; 
  if (typingRow) chatHistory.appendChild(typingRow);
  
  // 3. Reset the view back to the landing page
  document.getElementById('chat-view').classList.add('hidden');
  document.getElementById('chat-footer').classList.add('hidden');
  document.getElementById('landing-view').classList.remove('hidden');
  
  // 4. Clear the input fields
  document.getElementById('landingQueryInput').value = '';
  document.getElementById('activeChatInput').value = '';
  
  // 5. Scroll to top
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

/* ═══════════════════════════════════════════════
   Input handling
═══════════════════════════════════════════════ */
function handleKey(e, context) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    triggerSearch(context);
  }
}

function triggerSearch(context) {
  const inputEl = context === 'landing' ? document.getElementById('landingQueryInput') : document.getElementById('activeChatInput');
  const query = inputEl.value.trim();
  
  if (!query || isLoading) return;
  
  inputEl.value = '';
  sendMessage(query);
}

function sendSuggestion(text) {
  sendMessage(text);
}

/* ═══════════════════════════════════════════════
   Core Send & Receive Logic
═══════════════════════════════════════════════ */
async function sendMessage(query) {
  if (isLoading) return;

  switchToChatView();

  // Push to local history and UI[cite: 3]
  messages.push({ role: 'user', content: query });
  appendUserBubble(query);
  setLoading(true);

  // Extract history for context
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
    appendBotBubble(botMsg);

  } catch (err) {
    const errMsg = {
      role: 'assistant',
      content: `**Error:** ${err.message}\n\nPlease check your connection or backend server.`,
      classification: 'Error',
      predicted_class: -1,
      mode: 'error',
      sources: [],
    };
    messages.push(errMsg);
    appendBotBubble(errMsg);
  }

  setLoading(false);
}

/* ═══════════════════════════════════════════════
   DOM - Bubble Rendering
═══════════════════════════════════════════════ */
function appendUserBubble(text) {
  const row = document.createElement('div');
  row.className = 'flex flex-col items-end gap-space-xs self-end max-w-2xl w-full';
  
  // Basic html escaping
  const escapedText = String(text).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/\n/g, '<br>');

  row.innerHTML = `
    <div class="flex items-center gap-space-sm pr-space-xs">
      <span class="font-code-citation text-code-citation text-on-surface-variant">Just now • You</span>
      <div class="w-8 h-8 rounded-lg bg-primary-container flex items-center justify-center shadow-sm">
        <span class="font-label-md text-label-md text-on-primary font-bold">U</span>
      </div>
    </div>
    <div class="bg-primary-container text-on-primary px-space-lg py-space-md rounded-2xl rounded-tr-none shadow-md w-fit text-right">
      <p class="font-body-md text-body-md text-on-primary text-left">
        ${escapedText}
      </p>
    </div>
  `;
  insertBeforeTyping(row);
  scrollToBottom();
}

function appendBotBubble(msg) {
  const row = document.createElement('div');
  row.className = 'flex flex-col items-start gap-space-xs self-start max-w-3xl w-full';

  // Format mode label (RAG vs Conv vs Error)[cite: 3]
  const modeLabel = msg.mode === 'rag' ? 'Verified RAG Synthesis' : msg.mode === 'error' ? 'Error' : 'Conversational';
  const parsedContent = typeof marked !== 'undefined' ? marked.parse(msg.content || '') : msg.content;

  // Build the sources accordion if sources exist
  let sourcesHtml = '';
  if (msg.sources && msg.sources.length > 0) {
    const sourceItems = msg.sources.map(s => `
      <div class="flex items-center gap-2 bg-surface-container-lowest/60 hover:bg-surface-container-lowest transition-colors px-3 py-1.5 rounded-lg">
        <span class="font-label-sm text-label-sm bg-primary text-on-primary px-1.5 py-0.5 rounded">Source</span>
        <span class="font-body-sm text-body-sm text-on-secondary-fixed font-medium break-all">${String(s).replace(/</g, '&lt;')}</span>
      </div>
    `).join('');

    sourcesHtml = `
      <div class="w-full h-px bg-on-secondary-fixed/90 my-space-xs"></div>
      <div class="flex flex-col gap-space-xs">
        <button class="flex items-center justify-between w-full text-left font-label-md text-label-md text-on-secondary-fixed hover:opacity-80 transition-opacity" type="button" onclick="const content = this.nextElementSibling; const chevron = this.querySelector('.chevron'); content.classList.toggle('hidden'); chevron.style.transform = content.classList.contains('hidden') ? 'rotate(-90deg)' : 'rotate(0deg)';">
          <span class="flex items-center gap-1 font-bold">
            <span class="chevron text-[10px] inline-block transition-transform duration-200" style="transform: rotate(-90deg);">▼</span> ${msg.sources.length} sources
          </span>
          <span class="font-code-citation text-code-citation text-on-secondary-fixed/70">Click to expand index</span>
        </button>
        <div class="flex flex-col gap-2 pt-space-xs hidden">
          ${sourceItems}
        </div>
      </div>
    `;
  }

  row.innerHTML = `
    <div class="flex items-center gap-space-sm pl-space-xs">
      <div class="w-8 h-8 rounded-lg bg-secondary-container flex items-center justify-center shadow-sm">
        <span class="font-label-md text-label-md text-on-secondary-fixed font-bold">T</span>
      </div>
      <span class="font-code-citation text-code-citation text-on-surface">TaxSight PH • Just Now</span>
      <span class="font-label-sm text-label-sm bg-surface-container px-2 py-0.5 rounded-full text-secondary">${modeLabel}</span>
    </div>
    
    <div class="bg-secondary-fixed text-on-secondary-fixed px-space-xl py-space-lg rounded-2xl rounded-tl-none shadow-md w-full flex flex-col gap-space-md">
      <div class="bot-bubble-content font-body-md text-body-md text-on-secondary-fixed">
        ${parsedContent}
      </div>
      ${sourcesHtml}
    </div>
    
    <div class="flex items-center gap-space-xs mt-space-2xs pl-space-xs">
      <span class="font-label-sm text-label-sm px-space-md py-1 rounded-full bg-secondary-fixed text-on-secondary-fixed font-medium shadow-sm">
        ${msg.mode === 'rag' ? 'RAG Processed' : 'Direct LLM'}
      </span>
      <span class="font-label-sm text-label-sm px-space-md py-1 rounded-full bg-secondary-container text-on-secondary-fixed font-medium shadow-sm">
        ${msg.classification || 'General Context'}
      </span>
    </div>
  `;

  insertBeforeTyping(row);
  scrollToBottom();
}

function insertBeforeTyping(el) {
  const inner = document.getElementById('chat-history');
  const typingRow = document.getElementById('typing-row');
  inner.insertBefore(el, typingRow);
}

/* ═══════════════════════════════════════════════
   Utilities
═══════════════════════════════════════════════ */
function setLoading(state) {
  isLoading = state;
  const typingRow = document.getElementById('typing-row');
  if (state) {
    typingRow.classList.remove('hidden');
    typingRow.classList.add('flex');
  } else {
    typingRow.classList.add('hidden');
    typingRow.classList.remove('flex');
  }
  
  if (state) scrollToBottom();
}

function scrollToBottom() {
  window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
}