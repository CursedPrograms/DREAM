/**
 * dream.js — phone/browser client for DREAM's video avatar.
 *
 * Mirrors scripts/dream.py's local video-pool state machine (idle/listening/
 * thinking/talking, one clip picked at random per state, no repeats) but runs
 * against the existing /api/chat + /events Flask endpoints, so it works from
 * any browser on the network — no dedicated hardware required.
 */

'use strict';

const $ = id => document.getElementById(id);

/* ── Video pools & stage (two <video> elements cross-faded) ────────────────── */
let pools = { idle: [], listening: [], thinking: [], talking: [] };
let curState = null;
let curPath  = null;
let currentAudio = null;
let recording = false;
let mediaRec = null;
let audioChunks = [];
let _sending = false;

const stage = $('stage');
const videoA = document.createElement('video');
const videoB = document.createElement('video');
for (const v of [videoA, videoB]) {
  v.muted = true; v.playsInline = true; v.autoplay = true;
  stage.appendChild(v);
}
let front = videoA, back = videoB;

function pickClip(state, avoid) {
  const pool = pools[state] && pools[state].length ? pools[state] : pools.idle;
  if (!pool || !pool.length) return null;
  if (pool.length === 1) return pool[0];
  const choices = pool.filter(p => p !== avoid);
  return choices[Math.floor(Math.random() * choices.length)] || pool[0];
}

function playClip(path, { loop = true, onEnded = null } = {}) {
  if (!path) return;
  back.loop = loop;
  back.src = path;
  back.onended = onEnded ? () => onEnded() : null;
  const swap = () => {
    back.classList.add('showing');
    front.classList.remove('showing');
    [front, back] = [back, front];
  };
  back.oncanplay = () => { back.play().catch(() => {}); swap(); back.oncanplay = null; };
  back.load();
  curPath = path;
}

/* ── Visual state (drives which video pool is playing) ─────────────────────
   Distinct from the server's own /events "state" — talking is timed by the
   TTS <audio> element client-side, exactly like main.js's applyState(). */
function setVisualState(state) {
  if (state === curState) return;
  curState = state;
  $('dream-state').textContent = state.toUpperCase();
  const clip = pickClip(state, curPath);
  playClip(clip, { loop: true });
}

/* ── Wake intro — plays once on load, then falls into the idle loop ────────── */
function playIntro() {
  const introPath = '/static/videos/intro1.mp4';
  curState = 'intro';
  $('dream-state').textContent = 'WAKING UP';
  playClip(introPath, {
    loop: false,
    onEnded: () => { curState = null; setVisualState('idle'); },
  });
}

/* ── Fetch video pools from the server ──────────────────────────────────────*/
async function loadVideoPools() {
  try {
    const r = await fetch('/api/videos');
    pools = await r.json();
  } catch { /* keep empty pools; playClip no-ops safely */ }
  if (pools.idle && pools.idle.length) {
    playIntro();
  } else {
    setVisualState('idle');
  }
}

/* ── Captions ────────────────────────────────────────────────────────────── */
let captionTimer = null;
function showCaption(text) {
  const el = $('caption');
  el.textContent = text;
  clearTimeout(captionTimer);
  captionTimer = setTimeout(() => { el.textContent = ''; }, 8000);
}

/* ── SSE ─────────────────────────────────────────────────────────────────── */
function connectSSE() {
  const sse = new EventSource('/events');

  sse.onopen = () => { $('dream-dot').className = 'on'; };
  sse.onerror = () => { $('dream-dot').className = 'warn'; setTimeout(connectSSE, 5000); };

  sse.onmessage = e => {
    let d;
    try { d = JSON.parse(e.data); } catch { return; }

    switch (d.type) {
      case 'state':
        // Talking is timed locally by TTS playback (see playTTS) — ignore the
        // server's brief "talking" pulse so we don't cut the video short.
        if (d.state !== 'talking') setVisualState(d.state);
        break;
      case 'transcript':
        if (d.role !== 'system') showCaption(d.text);
        break;
      case 'error':
        showCaption('⚠ ' + d.msg);
        break;
    }
  };
}

/* ── Chat send / reply handling ──────────────────────────────────────────── */
function lockSend() { _sending = true; }
function unlockSend() { _sending = false; }

function handleReply(d) {
  if (d.error) { showCaption('⚠ ' + d.error); setVisualState('idle'); unlockSend(); return; }
  if (d.reply) showCaption(d.reply);
  if (d.audio_url) playTTS(d.audio_url); else { setVisualState('idle'); unlockSend(); }
}

function playTTS(url) {
  if (currentAudio) { currentAudio.pause(); currentAudio = null; }
  setVisualState('talking');
  const a = new Audio(url + '?cb=' + Date.now());
  currentAudio = a;
  const done = () => { currentAudio = null; setVisualState('idle'); unlockSend(); };
  a.onended = done;
  a.onerror = done;
  a.play().catch(done);
}

async function sendText(text) {
  if (!text || _sending) return;
  lockSend();
  setVisualState('thinking');
  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, voice: true }),
    });
    const d = await res.json();
    if (res.status === 429) { setVisualState('idle'); unlockSend(); return; }
    handleReply(d);
  } catch (err) {
    showCaption('Network error: ' + err.message);
    setVisualState('idle'); unlockSend();
  }
}

/* ── Mic recording (phone microphone) ───────────────────────────────────────*/
$('mic-btn').addEventListener('click', async () => {
  if (_sending) return;

  if (!recording) {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showCaption('⚠ Mic needs HTTPS or localhost — use the ⌨ button to type instead.');
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg']
        .find(m => MediaRecorder.isTypeSupported(m)) || '';
      mediaRec = new MediaRecorder(stream, mimeType ? { mimeType } : {});
      audioChunks = [];
      mediaRec.ondataavailable = e => { if (e.data.size > 0) audioChunks.push(e.data); };

      mediaRec.onstop = async () => {
        stream.getTracks().forEach(t => t.stop());
        if (!audioChunks.length) { setVisualState('idle'); return; }

        const blob = new Blob(audioChunks, { type: mimeType || 'audio/webm' });
        const fd = new FormData();
        fd.append('audio', blob, 'recording.webm');

        lockSend();
        setVisualState('thinking');
        try {
          const res = await fetch('/api/chat', { method: 'POST', body: fd });
          const d = await res.json();
          handleReply(d);
        } catch (err) {
          showCaption('Network error: ' + err.message);
          setVisualState('idle'); unlockSend();
        }
      };

      mediaRec.start(250);
      recording = true;
      $('mic-btn').classList.add('active');
      $('mic-hint').textContent = 'LISTENING… TAP TO STOP';
      setVisualState('listening');
    } catch (err) {
      showCaption(err.name === 'NotAllowedError'
        ? '⚠ Microphone access denied.'
        : '⚠ Mic error: ' + err.message);
    }
  } else {
    mediaRec.stop();
    recording = false;
    $('mic-btn').classList.remove('active');
    $('mic-hint').textContent = 'TAP TO TALK';
  }
});

/* ── Text fallback (keyboard toggle) ────────────────────────────────────────*/
$('kb-toggle').addEventListener('click', () => {
  $('text-fallback').classList.toggle('show');
  if ($('text-fallback').classList.contains('show')) $('text-input').focus();
});
$('send-btn').addEventListener('click', () => {
  const v = $('text-input').value.trim();
  if (v) { $('text-input').value = ''; sendText(v); }
});
$('text-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') $('send-btn').click();
});

/* ── Init ────────────────────────────────────────────────────────────────── */
window.addEventListener('DOMContentLoaded', () => {
  connectSSE();
  loadVideoPools();
});
