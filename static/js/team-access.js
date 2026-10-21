/* Team access (v1.5) for the demo (/demo) and the evaluation tool (/eval).
 *
 * When the server has TEAM_ACCESS_CODE set and this page was opened through
 * the public link, running questions needs that code. This script:
 *   1. adds the saved code to every request this page makes to its own server
 *      (the X-Team-Code header), so the page's own scripts need no changes;
 *   2. asks for the code once, in a small card, when the server says it is
 *      needed and this browser doesn't have a valid one yet.
 * Load it BEFORE the page's own script. The code is kept in this browser only
 * (localStorage); "Forget code" removes it.
 */
(function () {
  'use strict';
  var KEY = 'sagot-team-code';
  function getCode() { try { return localStorage.getItem(KEY) || ''; } catch (e) { return ''; } }
  function setCode(v) { try { v ? localStorage.setItem(KEY, v) : localStorage.removeItem(KEY); } catch (e) { /* ignore */ } }

  // 1. Send the code with this page's own requests (same-origin URLs only).
  var origFetch = window.fetch.bind(window);
  window.fetch = function (input, init) {
    var url = typeof input === 'string' ? input : (input && input.url) || '';
    var sameOrigin = url.charAt(0) === '/' || url.indexOf(location.origin) === 0;
    var code = getCode();
    if (sameOrigin && code) {
      init = Object.assign({}, init || {});
      var headers = new Headers(init.headers || (typeof input !== 'string' && input.headers) || {});
      headers.set('X-Team-Code', code);
      init.headers = headers;
    }
    return origFetch(input, init);
  };

  // 2. Ask for the code when it is needed.
  var css = '' +
    '#team-access{position:fixed;left:50%;top:76px;transform:translateX(-50%);z-index:9998;width:min(420px,calc(100vw - 32px));' +
    'background:#fff;border:1px solid #cbd5e1;border-radius:14px;box-shadow:0 12px 32px rgba(15,23,42,.18);padding:16px 18px;' +
    'font:14px/1.5 Inter,system-ui,-apple-system,"Segoe UI",sans-serif;color:#0f172a}' +
    '#team-access h2{margin:0 0 4px;font-size:16px}#team-access p{margin:0 0 10px;color:#475569;font-size:13px}' +
    '#team-access form{display:flex;gap:8px}#team-access input{flex:1;min-width:0;border:1px solid #cbd5e1;border-radius:10px;padding:8px 10px;font:inherit}' +
    '#team-access input:focus{outline:2px solid #93c5fd;outline-offset:0}' +
    '#team-access button{border:0;border-radius:10px;padding:8px 14px;font:600 13px Inter,system-ui,sans-serif;cursor:pointer}' +
    '#team-access .go{background:#2563eb;color:#fff}#team-access .later{background:transparent;color:#475569;margin-top:8px;padding:4px 0}' +
    '#team-access .err{color:#b91c1c;font-size:13px;margin:8px 0 0}' +
    '#team-access-chip{position:fixed;right:12px;top:72px;z-index:39;font:600 11px Inter,system-ui,sans-serif;background:#dcfce7;color:#15803d;' +
    'border-radius:999px;padding:4px 10px;border:0;cursor:pointer;opacity:.85}';

  function showCard(message) {
    if (document.getElementById('team-access')) return;
    var style = document.createElement('style'); style.textContent = css; document.head.appendChild(style);
    var card = document.createElement('div');
    card.id = 'team-access';
    card.setAttribute('role', 'dialog');
    card.setAttribute('aria-label', 'Team access code');
    card.innerHTML = '<h2>Team access code</h2>' +
      '<p>Running questions on this shared link needs the code from the person hosting Sagot AI.</p>' +
      '<form><input id="team-code-input" type="password" autocomplete="off" placeholder="Enter the code" aria-label="Team access code">' +
      '<button class="go" type="submit">Unlock</button></form>' +
      (message ? '<p class="err">' + message + '</p>' : '') +
      '<button class="later" type="button">Not now (read only)</button>';
    document.body.appendChild(card);
    var input = card.querySelector('input');
    input.focus();
    card.querySelector('form').addEventListener('submit', function (e) {
      e.preventDefault();
      var v = input.value.trim();
      if (!v) return;
      setCode(v);
      location.reload();   // reload so the page asks the server again, now with the code
    });
    card.querySelector('.later').addEventListener('click', function () { card.remove(); });
  }

  function showChip() {
    var style = document.createElement('style'); style.textContent = css; document.head.appendChild(style);
    var chip = document.createElement('button');
    chip.id = 'team-access-chip';
    chip.type = 'button';
    chip.title = 'Forget the team code in this browser';
    chip.textContent = 'Team access ✓ · Forget code';
    chip.addEventListener('click', function () { setCode(''); location.reload(); });
    document.body.appendChild(chip);
  }

  function check() {
    fetch('/team/api/status').then(function (r) { return r.json(); }).then(function (s) {
      if (!s.enabled || s.host) return;           // not needed here
      if (s.valid) { showChip(); return; }
      var had = !!getCode();
      if (had) setCode('');                        // a saved code that no longer works
      showCard(had ? 'That code did not work. Check it and try again.' : '');
    }).catch(function () { /* the page itself reports a server problem */ });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', check);
  else check();
})();
