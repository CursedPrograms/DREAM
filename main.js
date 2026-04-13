/**
 * ComCentre v2.7 — main.js
 *
 * Responsibilities:
 *  - SSE connection for live state / transcript / stats / wifi / node events
 *  - Text input  → POST /api/chat  { text, voice: false }  → no TTS
 *  - Mic input   → POST /api/chat  multipart audio upload   → TTS fires
 *  - Audio playback of TTS wav returned by the server
 *  - Peer node list rendering
 *  - WiFi device list rendering
 *  - Resource gauges (CPU / RAM / Disk)
 *  - Status dot polling
 */

'use strict';

/* ── Helpers ──────────────────────────────────────────────────────────────── */
const $ = id => document.getElementById(id);

// Read char name injected via data attribute on <script> tag
const CHAR_NAME = (document.currentScript || { dataset: {} }).dataset.char || 'FRIDAY';

/* ── State ────────────────────────────────────────────────────────────────── */
let recording    = false;
let mediaRec     = null;
let audioChunks  = [];
let currentAudio = null;   // HTMLAudioElement currently playing TTS

/* ════════════════════════════════════════════════════════════════════════════
   STATE / UI
════════════════════════════════════════════════════════════════════════════ */
function applyState(s) {
  const badge = $('state-badge');
  badge.className = s;
  badge.textContent = s.toUpperCase();

  $('avatar-ring').className = 'avatar-ring ' + (
    ['listening','thinking','talking'].includes(s) ? s : ''
  );

  $('waveform').className =
    (s === 'listening' || s === 'talking') ? 'waveform-active' : '';

  const bar = $('thinking-bar');
  if (s === 'thinking') {
    bar.style.display = 'block';
    bar.classList.add('active');
  } else {
    bar.style.display = 'none';
    bar.classList.remove('active');
  }
}

/* ════════════════════════════════════════════════════════════════════════════
   CHAT LOG
════════════════════════════════════════════════════════════════════════════ */
function addMessage(role, text) {
  if (!text || !text.trim()) return;
  const log  = $('chat-log');
  const wrap = document.createElement('div');
  wrap.className = `msg ${role}`;

  const roleLabel = role === 'assistant' ? CHAR_NAME
                  : role === 'user'      ? 'YOU'
                  :                        role.toUpperCase();

  // Escape HTML, preserve newlines
  const escaped = text
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/\n/g,'<br>');

  wrap.innerHTML =
    `<div class="msg-role">${roleLabel}</div>` +
    `<div class="msg-bubble">${escaped}</div>`;

  log.appendChild(wrap);
  log.scrollTop = log.scrollHeight;
}

function clearChat() {
  $('chat-log').innerHTML = '';
}

/* ════════════════════════════════════════════════════════════════════════════
   SSE — server-sent events
════════════════════════════════════════════════════════════════════════════ */
function connectSSE() {
  const sse = new EventSource('/events');

  sse.onopen = () => {
    $('dot-sse').className = 'dot on';
  };

  sse.onerror = () => {
    $('dot-sse').className = 'dot warn';
    setTimeout(connectSSE, 5000);
  };

  sse.onmessage = e => {
    let d;
    try { d = JSON.parse(e.data); } catch { return; }

    switch (d.type) {
      case 'state':
        applyState(d.state);
        break;

      case 'transcript':
        // Don't double-add messages we already showed optimistically
        if (d.role === 'assistant' || d.role === 'system') {
          addMessage(d.role, d.text);
        }
        break;

      case 'stats':
        updateStatsUI(d.data);
        break;

      case 'wifi':
        renderWifiDevices(d.devices);
        break;

      case 'nodes':
        renderNodes(d.nodes);
        break;

      case 'error':
        addMessage('system', `⚠ ${d.msg}`);
        break;

      case 'ping':
        // keepalive — ignore
        break;
    }
  };
}

/* ════════════════════════════════════════════════════════════════════════════
   API — text (typed) input  →  no TTS
════════════════════════════════════════════════════════════════════════════ */
async function sendText(text, voiceMode = false) {
  if (!text) return;
  if (!voiceMode) addMessage('user', text);  // optimistic for typed input
  applyState('thinking');

  try {
    const res = await fetch('/api/chat', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ text, voice: voiceMode }),
    });
    const d = await res.json();

    if (d.error) {
      addMessage('system', `Error: ${d.error}`);
      applyState('idle');
      return;
    }

    // For typed mode, the reply comes back in the response.
    // For voice mode, the SSE transcript event handles the bubble.
    if (!voiceMode && d.reply) {
      addMessage('assistant', d.reply);
    }

    // Handle audio only in voice mode
    if (voiceMode && d.audio_url) {
      playTTS(d.audio_url);
    } else {
      applyState('idle');
    }

    // Extra data
    if (d.devices) renderWifiDevices(d.devices);
    if (d.stats)   updateStatsUI(d.stats);

  } catch (err) {
    addMessage('system', `Network error: ${err.message}`);
    applyState('idle');
  }
}

function quickSend(text, voice = false) {
  sendText(text, voice);
}

/* ════════════════════════════════════════════════════════════════════════════
   AUDIO — TTS playback
════════════════════════════════════════════════════════════════════════════ */
function playTTS(url) {
  // Stop any previous TTS
  if (currentAudio) {
    currentAudio.pause();
    currentAudio = null;
  }
  applyState('talking');
  const a = new Audio(url + '?cb=' + Date.now());
  currentAudio = a;
  a.onended  = () => { currentAudio = null; applyState('idle'); };
  a.onerror  = () => { currentAudio = null; applyState('idle'); };
  a.play().catch(() => applyState('idle'));
}

/* ════════════════════════════════════════════════════════════════════════════
   MIC RECORDING
════════════════════════════════════════════════════════════════════════════ */
$('mic-btn').addEventListener('click', async () => {
  if (!recording) {
    // ── Start recording ────────────────────────────────────────────────────
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      // Pick a supported MIME type
      const mimeType = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg']
        .find(m => MediaRecorder.isTypeSupported(m)) || '';

      const opts = mimeType ? { mimeType } : {};
      mediaRec    = new MediaRecorder(stream, opts);
      audioChunks = [];

      mediaRec.ondataavailable = e => {
        if (e.data.size > 0) audioChunks.push(e.data);
      };

      mediaRec.onstop = async () => {
        // Release mic tracks
        stream.getTracks().forEach(t => t.stop());

        if (!audioChunks.length) {
          addMessage('system', 'No audio captured.');
          applyState('idle');
          return;
        }

        const blob = new Blob(audioChunks, { type: mimeType || 'audio/webm' });
        const fd   = new FormData();
        fd.append('audio', blob, 'recording.webm');

        addMessage('system', '🎙 Processing voice…');
        applyState('thinking');

        try {
          const res = await fetch('/api/chat', { method: 'POST', body: fd });
          const d   = await res.json();

          if (d.error) {
            addMessage('system', `⚠ ${d.error}`);
            applyState('idle');
            return;
          }

          // TTS audio plays automatically (voice_mode=True on server)
          if (d.audio_url) {
            playTTS(d.audio_url);
          } else {
            applyState('idle');
          }

          if (d.devices) renderWifiDevices(d.devices);
          if (d.stats)   updateStatsUI(d.stats);

        } catch (err) {
          addMessage('system', `Network error: ${err.message}`);
          applyState('idle');
        }
      };

      mediaRec.start(250);  // collect chunks every 250 ms
      recording = true;
      $('mic-btn').classList.add('active');
      applyState('listening');

    } catch (err) {
      if (err.name === 'NotAllowedError') {
        addMessage('system', '⚠ Microphone access denied. Allow mic in browser settings.');
      } else {
        addMessage('system', `⚠ Mic error: ${err.message}`);
      }
    }

  } else {
    // ── Stop recording ─────────────────────────────────────────────────────
    mediaRec.stop();
    recording = false;
    $('mic-btn').classList.remove('active');
  }
});

/* ════════════════════════════════════════════════════════════════════════════
   WIFI SCAN — triggered from sidebar button
════════════════════════════════════════════════════════════════════════════ */
function triggerWifiScan() {
  applyState('thinking');
  $('wifi-list').innerHTML = '<div class="dim-note">Scanning…</div>';

  fetch('/api/wifi')
    .then(r => r.json())
    .then(d => {
      renderWifiDevices(d.devices || []);
      applyState('idle');
    })
    .catch(() => {
      $('wifi-list').innerHTML = '<div class="dim-note" style="color:var(--danger)">Scan failed.</div>';
      applyState('idle');
    });
}

function renderWifiDevices(devices) {
  const el = $('wifi-list');
  if (!devices || !devices.length) {
    el.innerHTML = '<div class="dim-note">No devices found.</div>';
    return;
  }

  el.innerHTML = devices.map(d => {
    const cls  = d.me ? 'wifi-device me' : 'wifi-device';
    const self = d.me ? ' ← THIS MACHINE' : '';
    return `
      <div class="${cls}">
        <span class="wifi-ip">${d.ip}${self}</span>
        <span class="wifi-type">${escapeHtml(d.type || 'Unknown')}</span>
        ${d.hostname ? `<span class="wifi-host">${escapeHtml(d.hostname)}</span>` : ''}
      </div>
    `;
  }).join('');
}

/* ════════════════════════════════════════════════════════════════════════════
   PEER NODES
════════════════════════════════════════════════════════════════════════════ */
function renderNodes(nodes) {
  const el  = $('node-list');
  const dot = $('dot-nodes');

  const keys = Object.keys(nodes || {});

  if (!keys.length) {
    dot.className = 'dot';
    el.innerHTML  = '<div class="dim-note">No peer nodes.</div>';
    return;
  }

  dot.className = 'dot purple';

  // Always show self at top
  let html = `<div class="node-self">◉ FRIDAY (self)</div>`;

  html += keys.map(name => {
    const url = nodes[name];
    return `
      <div class="node-item">
        <div>
          <div class="node-name">${escapeHtml(name)}</div>
          <div class="node-status">ONLINE</div>
        </div>
        <a class="node-link" href="${escapeHtml(url)}" target="_blank" rel="noopener">OPEN</a>
      </div>
    `;
  }).join('');

  el.innerHTML = html;
}

/* ════════════════════════════════════════════════════════════════════════════
   RESOURCE STATS
════════════════════════════════════════════════════════════════════════════ */
function updateStatsUI(s) {
  if (!s) return;

  // CPU
  $('sv-cpu').textContent = s.cpu_pct.toFixed(0) + '%';
  setBar('bf-cpu', s.cpu_pct);

  // RAM
  $('sv-ram').textContent = s.ram_used.toFixed(1) + 'GB';
  setBar('bf-ram', s.ram_pct);

  // Disk
  if ($('sv-disk')) {
    $('sv-disk').textContent = s.disk_used.toFixed(0) + 'GB';
    setBar('bf-disk', s.disk_pct);
  }

  // Temp (optional)
  if (s.cpu_temp != null) {
    const row = $('temp-row');
    if (row) {
      row.style.display = 'flex';
      $('sv-temp').textContent = s.cpu_temp.toFixed(0) + '°C';
    }
  }
}

function setBar(id, pct) {
  const el = $(id);
  if (!el) return;
  el.style.width = Math.min(100, pct) + '%';
  el.className = 'bar-fill' +
    (pct > 90 ? ' danger' : pct > 70 ? ' warn' : '');
}

async function pollStats() {
  try {
    const r = await fetch('/api/stats');
    const s = await r.json();
    updateStatsUI(s);
  } catch { /* silent */ }
  setTimeout(pollStats, 5000);
}

/* ════════════════════════════════════════════════════════════════════════════
   STATUS DOTS POLLING
════════════════════════════════════════════════════════════════════════════ */
async function pollStatus() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    $('dot-ollama').className = d.ollama       ? 'dot on'   : 'dot warn';
    $('dot-piper').className  = d.piper        ? 'dot on'   : 'dot warn';
  } catch {
    $('dot-ollama').className = 'dot warn';
  }
  setTimeout(pollStatus, 10000);
}

/* ════════════════════════════════════════════════════════════════════════════
   TEXT INPUT / SEND BUTTON
════════════════════════════════════════════════════════════════════════════ */
$('send-btn').addEventListener('click', () => {
  const v = $('text-input').value.trim();
  if (v) {
    sendText(v, false);   // false = typed, no TTS
    $('text-input').value = '';
  }
});

$('text-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') $('send-btn').click();
});

/* ════════════════════════════════════════════════════════════════════════════
   UTILS
════════════════════════════════════════════════════════════════════════════ */
function escapeHtml(str) {
  return String(str)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

/* ════════════════════════════════════════════════════════════════════════════
   INIT
════════════════════════════════════════════════════════════════════════════ */
window.addEventListener('DOMContentLoaded', () => {
  connectSSE();
  pollStatus();
  pollStats();
});
