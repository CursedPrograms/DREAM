/**
 * ComCentre v2.8 — main.js
 *
 * Fixes applied:
 *  - Assistant messages only rendered via SSE (single source of truth).
 *    The fetch response no longer calls addMessage for assistant replies,
 *    eliminating the duplicate bubble.
 *  - Send button + Enter key are disabled while a request is in-flight.
 *    Re-enabled only when the server returns (success or error).
 *  - 429 "busy" response handled gracefully (button unlocks immediately).
 */

'use strict';

const $ = id => document.getElementById(id);
const CHAR_NAME = (document.currentScript || { dataset: {} }).dataset.char || 'FRIDAY';
const NODE_NAME = (document.currentScript || { dataset: {} }).dataset.node || 'COMCENTRE';

/* ── State ──────────────────────────────────────────────────────────────── */
let recording = false;
let mediaRec = null;
let audioChunks = [];
let currentAudio = null;
let _sending = false;   // true while a /api/chat request is in-flight

/* ════════════════════════════════════════════════════════════════════════
   SEND LOCK — disables input while waiting for a reply
════════════════════════════════════════════════════════════════════════ */
function lockSend() {
  _sending = true;
  $('send-btn').disabled = true;
  $('text-input').disabled = true;
  $('send-btn').textContent = '…';
}

function unlockSend() {
  _sending = false;
  $('send-btn').disabled = false;
  $('text-input').disabled = false;
  $('send-btn').textContent = 'SEND';
  $('text-input').focus();
}

/* ════════════════════════════════════════════════════════════════════════
   STATE / UI
════════════════════════════════════════════════════════════════════════ */
function applyState(s) {
  const badge = $('state-badge');
  badge.className = s;
  badge.textContent = s.toUpperCase();

  $('avatar-ring').className = 'avatar-ring ' + (
    ['listening', 'thinking', 'talking'].includes(s) ? s : ''
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

/* ════════════════════════════════════════════════════════════════════════
   CHAT LOG
════════════════════════════════════════════════════════════════════════ */
function addMessage(role, text) {
  if (!text || !text.trim()) return;
  const log = $('chat-log');
  const wrap = document.createElement('div');
  wrap.className = `msg ${role}`;

  const roleLabel = role === 'assistant' ? CHAR_NAME
    : role === 'user' ? 'YOU'
      : role.toUpperCase();

  const escaped = text
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/\n/g, '<br>');

  wrap.innerHTML =
    `<div class="msg-role">${roleLabel}</div>` +
    `<div class="msg-bubble">${escaped}</div>`;

  log.appendChild(wrap);
  log.scrollTop = log.scrollHeight;
}

function clearChat() {
  $('chat-log').innerHTML = '';
}

/* ════════════════════════════════════════════════════════════════════════
   SSE — server-sent events
   SSE is the ONLY place assistant/system messages are rendered.
   The fetch response is used only for audio_url, devices, stats.
════════════════════════════════════════════════════════════════════════ */
function connectSSE() {
  const sse = new EventSource('/events');

  sse.onopen = () => { $('dot-sse').className = 'dot on'; };

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
        // When server goes back to idle, ensure input is unlocked
        if (d.state === 'idle') unlockSend();
        break;

      case 'transcript':
        // ALL transcript messages (user, assistant, system) rendered here only.
        // The fetch response never calls addMessage for assistant replies.
        addMessage(d.role, d.text);
        break;

      case 'stats': updateStatsUI(d.data); break;
      case 'wifi': renderWifiDevices(d.devices); break;
      case 'nodes': renderNodes(d.nodes); break;
      case 'error': addMessage('system', `⚠ ${d.msg}`); break;
      case 'ping': break; // keepalive
    }
  };
}

/* ════════════════════════════════════════════════════════════════════════
   API — text (typed) input
════════════════════════════════════════════════════════════════════════ */
async function sendText(text, voiceMode = false) {
  if (!text || _sending) return;
  lockSend();

  // Show user bubble immediately for typed input.
  // For voice mode the SSE transcript event handles it.
  if (!voiceMode) addMessage('user', text);

  applyState('thinking');

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, voice: voiceMode }),
    });

    const d = await res.json();

    if (res.status === 429 || d.error === 'busy') {
      // Server was already processing — silently drop, already shown in SSE
      applyState('idle');
      unlockSend();
      return;
    }

    if (d.error) {
      addMessage('system', `Error: ${d.error}`);
      applyState('idle');
      unlockSend();
      return;
    }

    // ── DO NOT call addMessage here for assistant replies ──────────────────
    // SSE already fired a 'transcript' event with the same text.
    // Calling addMessage here too is what caused the duplicate bubble.

    // Handle TTS audio (voice mode only)
    if (voiceMode && d.audio_url) {
      playTTS(d.audio_url);
    } else {
      applyState('idle');
      unlockSend();
    }

    // Side-channel data
    if (d.devices) renderWifiDevices(d.devices);
    if (d.stats) updateStatsUI(d.stats);

  } catch (err) {
    addMessage('system', `Network error: ${err.message}`);
    applyState('idle');
    unlockSend();
  }
}

function quickSend(text, voice = false) {
  sendText(text, voice);
}

/* ════════════════════════════════════════════════════════════════════════
   AUDIO — TTS playback
════════════════════════════════════════════════════════════════════════ */
function playTTS(url) {
  if (currentAudio) { currentAudio.pause(); currentAudio = null; }
  applyState('talking');
  const a = new Audio(url + '?cb=' + Date.now());
  currentAudio = a;
  a.onended = () => { currentAudio = null; applyState('idle'); unlockSend(); };
  a.onerror = () => { currentAudio = null; applyState('idle'); unlockSend(); };
  a.play().catch(() => { applyState('idle'); unlockSend(); });
}

/* ════════════════════════════════════════════════════════════════════════
   MIC RECORDING
════════════════════════════════════════════════════════════════════════ */
$('mic-btn').addEventListener('click', async () => {
  if (!recording) {
    // Check for mediaDevices support (requires HTTPS or localhost)
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      addMessage('system',
        '⚠ Microphone unavailable. Open via <b>http://localhost:5009</b> or serve over HTTPS. ' +
        'Browsers block mic access on plain HTTP from an IP address.'
      );
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      const mimeType = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg']
        .find(m => MediaRecorder.isTypeSupported(m)) || '';

      mediaRec = new MediaRecorder(stream, mimeType ? { mimeType } : {});
      audioChunks = [];

      mediaRec.ondataavailable = e => {
        if (e.data.size > 0) audioChunks.push(e.data);
      };

      mediaRec.onstop = async () => {
        stream.getTracks().forEach(t => t.stop());

        if (!audioChunks.length) {
          addMessage('system', 'No audio captured.');
          applyState('idle');
          return;
        }

        const blob = new Blob(audioChunks, { type: mimeType || 'audio/webm' });
        const fd = new FormData();
        fd.append('audio', blob, 'recording.webm');

        addMessage('system', '🎙 Processing voice…');
        applyState('thinking');
        lockSend();

        try {
          const res = await fetch('/api/chat', { method: 'POST', body: fd });
          const d = await res.json();

          if (d.error) {
            addMessage('system', `⚠ ${d.error}`);
            applyState('idle');
            unlockSend();
            return;
          }

          // SSE handles transcript bubbles — only handle audio here
          if (d.audio_url) {
            playTTS(d.audio_url);
          } else {
            applyState('idle');
            unlockSend();
          }

          if (d.devices) renderWifiDevices(d.devices);
          if (d.stats) updateStatsUI(d.stats);

        } catch (err) {
          addMessage('system', `Network error: ${err.message}`);
          applyState('idle');
          unlockSend();
        }
      };

      mediaRec.start(250);
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
    mediaRec.stop();
    recording = false;
    $('mic-btn').classList.remove('active');
  }
});

/* ════════════════════════════════════════════════════════════════════════
   WIFI SCAN
════════════════════════════════════════════════════════════════════════ */
function triggerWifiScan() {
  applyState('thinking');
  $('wifi-list').innerHTML = '<div class="dim-note">Scanning…</div>';

  fetch('/api/wifi')
    .then(r => r.json())
    .then(d => { renderWifiDevices(d.devices || []); applyState('idle'); })
    .catch(() => {
      $('wifi-list').innerHTML =
        '<div class="dim-note" style="color:var(--danger)">Scan failed.</div>';
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
    const cls = d.me ? 'wifi-device me' : 'wifi-device';
    const self = d.me ? ' ← THIS MACHINE' : '';
    return `
      <div class="${cls}">
        <span class="wifi-ip">${d.ip}${self}</span>
        <span class="wifi-type">${escapeHtml(d.type || 'Unknown')}</span>
        ${d.hostname ? `<span class="wifi-host">${escapeHtml(d.hostname)}</span>` : ''}
      </div>`;
  }).join('');
}

/* ════════════════════════════════════════════════════════════════════════
   PEER NODES
════════════════════════════════════════════════════════════════════════ */
function renderNodes(nodes) {
  const el = $('node-list');
  const dot = $('dot-nodes');
  const keys = Object.keys(nodes || {});

  if (!keys.length) {
    dot.className = 'dot';
    el.innerHTML = '<div class="dim-note">No peer nodes.</div>';
    return;
  }

  dot.className = 'dot purple';
  let html = `<div class="node-self">◉ ${NODE_NAME} (self)</div>`;
  html += keys.map(name => {
    const url = nodes[name];
    return `
      <div class="node-item">
        <div>
          <div class="node-name">${escapeHtml(name)}</div>
          <div class="node-status">ONLINE</div>
        </div>
        <a class="node-link" href="${escapeHtml(url)}" target="_blank" rel="noopener">OPEN</a>
      </div>`;
  }).join('');

  el.innerHTML = html;
}

/* ════════════════════════════════════════════════════════════════════════
   RESOURCE STATS
════════════════════════════════════════════════════════════════════════ */
function updateStatsUI(s) {
  if (!s) return;
  $('sv-cpu').textContent = s.cpu_pct.toFixed(0) + '%';
  setBar('bf-cpu', s.cpu_pct);
  $('sv-ram').textContent = s.ram_used.toFixed(1) + 'GB';
  setBar('bf-ram', s.ram_pct);
  if ($('sv-disk')) {
    $('sv-disk').textContent = s.disk_used.toFixed(0) + 'GB';
    setBar('bf-disk', s.disk_pct);
  }
  if (s.cpu_temp != null) {
    const row = $('temp-row');
    if (row) { row.style.display = 'flex'; $('sv-temp').textContent = s.cpu_temp.toFixed(0) + '°C'; }
  }
}

function setBar(id, pct) {
  const el = $(id);
  if (!el) return;
  el.style.width = Math.min(100, pct) + '%';
  el.className = 'bar-fill' + (pct > 90 ? ' danger' : pct > 70 ? ' warn' : '');
}

async function pollStats() {
  try {
    const r = await fetch('/api/stats');
    const s = await r.json();
    updateStatsUI(s);
  } catch { /* silent */ }
  setTimeout(pollStats, 5000);
}

/* ════════════════════════════════════════════════════════════════════════
   STATUS DOTS
════════════════════════════════════════════════════════════════════════ */
async function pollStatus() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    $('dot-ollama').className = d.ollama ? 'dot on' : 'dot warn';
    $('dot-piper').className = d.piper ? 'dot on' : 'dot warn';
  } catch {
    $('dot-ollama').className = 'dot warn';
  }
  setTimeout(pollStatus, 10000);
}

/* ════════════════════════════════════════════════════════════════════════
   TEXT INPUT / SEND BUTTON
════════════════════════════════════════════════════════════════════════ */
$('send-btn').addEventListener('click', () => {
  if (_sending) return;
  const v = $('text-input').value.trim();
  if (v) {
    $('text-input').value = '';
    sendText(v, false);
  }
});

$('text-input').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !_sending) $('send-btn').click();
});

/* ════════════════════════════════════════════════════════════════════════
   UTILS
════════════════════════════════════════════════════════════════════════ */
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

/* ════════════════════════════════════════════════════════════════════════
   INIT
════════════════════════════════════════════════════════════════════════ */
window.addEventListener('DOMContentLoaded', () => {
  connectSSE();
  pollStatus();
  pollStats();
});
