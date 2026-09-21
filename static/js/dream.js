/**
 * dream.js — DREAM in the browser: the web version of scripts/dream.py.
 *
 * Same behaviour as the desktop app, with the browser's mic and speakers:
 *   - video avatar: idle / listening / thinking / talking pools, one clip picked
 *     at random per state with no immediate repeats, plus the intro, the
 *     flirt clips and the sleeping loop
 *   - "Hey DREAM" wake word: listens in short clips, the server transcribes them
 *     (Whisper, like dream.py), then she greets you and listens for a command
 *   - sleeping: falls asleep when idle; wakes on "wake up", on the mmWave
 *     sensor seeing someone, or on any interaction
 *   - the sensor reactions (greeting by distance, farewell) and the alarm/light
 *     voice commands, all decided by app.py and announced over /events
 *
 * Tap anywhere on the start screen first: browsers only allow sound and the mic
 * after a tap.
 */

'use strict';

const $ = id => document.getElementById(id);
const wait = ms => new Promise(r => setTimeout(r, ms));

/* Tuning. Levels are peak amplitude, 0..1. */
const LEVEL_WAKE_CLIP = 0.01;   // a wake clip quieter than this isn't sent to Whisper
const LEVEL_SPEECH    = 0.02;   // above this while recording a command counts as speech
const SILENCE_END_MS  = 1500;   // this long quiet after speech ends the command
const NO_SPEECH_MS    = 6000;   // this long with no speech at all gives up
const MANUAL_MAX_MS   = 60000;  // longest tap-to-talk recording

let cfg = { startup_text: '', wake_seconds: 3, record_seconds: 16 };

/* ── State ───────────────────────────────────────────────────────────────── */
let pools = { idle: [], listening: [], thinking: [], talking: [], flirtytalk: [], sleeping: [], intro: [] };
let curState = null;      // what the stage is showing
let curPath = null;
let sleeping = false;
let pendingSleep = null;  // a sleep/wake that arrived mid-conversation; applied when she's done
let busy = false;         // recording, thinking, speaking or playing a clip — nothing else may start
let started = false;      // the start screen was tapped
let wakeEnabled = true;
let currentAudio = null;

let micStream = null, audioCtx = null, analyser = null, levelBuf = null;
let manualRecording = false, manualStop = false;

/* ── Video stage (two <video> elements cross-faded) ──────────────────────── */
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
  if (!path) { if (onEnded) onEnded(); return; }
  back.loop = loop;
  back.src = path;
  back.onended = onEnded ? () => onEnded() : null;
  back.onerror = onEnded ? () => onEnded() : null;  // a clip that won't load mustn't hang the flow
  back.oncanplay = () => {
    back.play().catch(() => {});
    back.classList.add('showing');
    front.classList.remove('showing');
    [front, back] = [back, front];
    front.oncanplay = null;
  };
  back.load();
  curPath = path;
}

function setLabel(text) { $('dream-state').textContent = text; }

/* The resting visuals. While asleep every state shows the sleeping loop. */
function setVisualState(state) {
  if (sleeping && state !== 'sleeping') state = 'sleeping';
  if (state === curState) return;
  curState = state;
  setLabel(state.toUpperCase());
  playClip(pickClip(state, curPath), { loop: true });
}

/* Play one clip through once (intro, flirt), then resolve. */
function playOnce(path, label) {
  return new Promise(resolve => {
    curState = 'clip';
    setLabel(label);
    playClip(path, { loop: false, onEnded: () => resolve() });
  });
}

/* Back to resting: applies a sleep/wake that arrived mid-conversation. */
function settle() {
  if (pendingSleep !== null) { sleeping = pendingSleep; pendingSleep = null; }
  setVisualState(sleeping ? 'sleeping' : 'idle');
}

function setBusy(b) {
  if (busy === b) return;
  busy = b;
  fetch('/api/dream/activity', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ busy: b }),
  }).catch(() => {});
}

/* Run one flow at a time (wake flow, typed message, tap-to-talk, flirt clip…). */
async function runExclusive(fn) {
  if (busy) return;
  setBusy(true);
  try { await fn(); }
  catch (err) { showCaption('⚠ ' + err.message); }
  finally { setBusy(false); settle(); }
}

/* ── Captions ────────────────────────────────────────────────────────────── */
let captionTimer = null;
function showCaption(text) {
  const el = $('caption');
  el.textContent = text;
  clearTimeout(captionTimer);
  captionTimer = setTimeout(() => { el.textContent = ''; }, 8000);
}

/* ── Speaking ────────────────────────────────────────────────────────────── */
function playAudio(url) {
  return new Promise(resolve => {
    if (currentAudio) currentAudio.pause();
    setVisualState('talking');
    const a = new Audio(url + '?cb=' + Date.now());
    currentAudio = a;
    const done = () => { if (currentAudio === a) currentAudio = null; resolve(); };
    a.onended = done;
    a.onerror = done;
    a.play().catch(done);
  });
}

/* Show a line and play its audio (already generated by the server). */
async function say(text, audioUrl) {
  if (text) showCaption(text);
  if (audioUrl) await playAudio(audioUrl);
}

/* Show a line and have the server voice it. */
async function sayText(text) {
  showCaption(text);
  try {
    const r = await fetch('/api/speak', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }),
    });
    const d = await r.json();
    if (d.audio_url) await playAudio(d.audio_url);
  } catch { /* caption is enough */ }
}

/* Send a message (typed text or a recorded command) and speak the reply. */
async function submitChat(init) {
  setVisualState('thinking');
  try {
    const res = await fetch('/api/chat', init);
    const d = await res.json();
    if (res.status === 429) return;  // she's already answering someone
    if (d.error === 'no input') { await sayText("I didn't catch that."); return; }
    if (d.error) { showCaption('⚠ ' + d.error); return; }
    if (d.reply) showCaption(d.reply);
    if (d.audio_url) await playAudio(d.audio_url);
  } catch (err) {
    showCaption('Network error: ' + err.message);
  }
}

function sendText(text) {
  if (!text) return;
  return runExclusive(() => submitChat({
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text, voice: true }),
  }));
}

/* ── Microphone ──────────────────────────────────────────────────────────── */
async function ensureMic() {
  if (micStream) return true;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) return false;
  try {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const src = audioCtx.createMediaStreamSource(micStream);
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 1024;
    levelBuf = new Float32Array(analyser.fftSize);
    src.connect(analyser);
    if (audioCtx.state === 'suspended') await audioCtx.resume();
    return true;
  } catch (err) {
    showCaption(err.name === 'NotAllowedError' ? '⚠ Microphone access denied.' : '⚠ Mic error: ' + err.message);
    return false;
  }
}

function micLevel() {
  analyser.getFloatTimeDomainData(levelBuf);
  let peak = 0;
  for (let i = 0; i < levelBuf.length; i++) peak = Math.max(peak, Math.abs(levelBuf[i]));
  return peak;
}

function pickMime() {
  return ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg'].find(m => MediaRecorder.isTypeSupported(m)) || '';
}

/**
 * Record from the mic. Resolves { blob, peak, heard } when it stops:
 *   maxMs      hard limit
 *   vad        also stop after SILENCE_END_MS of quiet following speech, or
 *              NO_SPEECH_MS with none at all
 *   shouldStop polled every tick (tap-to-talk's second tap)
 */
function recordSegment({ maxMs, vad = false, shouldStop = null }) {
  return new Promise(resolve => {
    const mime = pickMime();
    const rec = new MediaRecorder(micStream, mime ? { mimeType: mime } : {});
    const chunks = [];
    let peak = 0, heard = false;
    const t0 = performance.now();
    let lastVoice = t0;

    rec.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };
    rec.onstop = () => resolve({ blob: new Blob(chunks, { type: mime || 'audio/webm' }), peak, heard });
    rec.start();

    const tick = setInterval(() => {
      const now = performance.now();
      const level = micLevel();
      peak = Math.max(peak, level);
      if (level > LEVEL_SPEECH) { heard = true; lastVoice = now; }
      const elapsed = now - t0;
      const done = elapsed >= maxMs
        || (shouldStop && shouldStop())
        || (vad && ((heard && now - lastVoice > SILENCE_END_MS) || (!heard && elapsed > NO_SPEECH_MS)));
      if (done) {
        clearInterval(tick);
        if (rec.state !== 'inactive') rec.stop();
      }
    }, 50);
  });
}

function audioForm(blob, name) {
  const fd = new FormData();
  fd.append('audio', blob, name);
  return fd;
}

/* ── Wake flow: greeting, then listen for one command ────────────────────── */
async function wakeFlow(greetingText, greetingAudio) {
  await say(greetingText, greetingAudio);
  if (!micStream) return;

  setVisualState('listening');
  $('mic-hint').textContent = 'LISTENING…';
  let seg;
  try { seg = await recordSegment({ maxMs: cfg.record_seconds * 1000, vad: true }); }
  finally { $('mic-hint').textContent = hintText(); }

  if (!seg.heard) { await sayText("I couldn't hear you clearly."); return; }
  await submitChat({ method: 'POST', body: audioForm(seg.blob, 'command.webm') });
}

/* ── Wake-word loop: short clips, transcribed by the server ──────────────── */
async function wakeLoop() {
  for (;;) {
    if (!started || !micStream || !wakeEnabled || busy || manualRecording) { await wait(300); continue; }

    const seg = await recordSegment({ maxMs: cfg.wake_seconds * 1000 });
    if (busy || manualRecording || !wakeEnabled) continue;   // something started meanwhile
    if (seg.peak < LEVEL_WAKE_CLIP) continue;                // silence: nothing to transcribe

    let d;
    try {
      d = await (await fetch('/api/wake', { method: 'POST', body: audioForm(seg.blob, 'wake.webm') })).json();
    } catch { await wait(1000); continue; }
    if (busy) continue;                                      // something else took over while it transcribed

    if (d.trigger === 'wake') runExclusive(() => wakeFlow(d.text, d.audio_url));
    else if (d.trigger === 'wifi') sendText('check wifi');
  }
}

/* ── Tap to talk ─────────────────────────────────────────────────────────── */
$('mic-btn').addEventListener('click', async () => {
  if (manualRecording) { manualStop = true; return; }
  if (busy) return;
  if (!(await ensureMic())) {
    showCaption('⚠ Mic needs HTTPS or localhost — use the ⌨ button to type instead.');
    return;
  }
  await runExclusive(async () => {
    manualRecording = true; manualStop = false;
    $('mic-btn').classList.add('active');
    $('mic-hint').textContent = 'LISTENING… TAP TO STOP';
    setVisualState('listening');
    let seg;
    try {
      seg = await recordSegment({ maxMs: MANUAL_MAX_MS, shouldStop: () => manualStop });
    } finally {
      manualRecording = false;
      $('mic-btn').classList.remove('active');
      $('mic-hint').textContent = hintText();
    }
    if (seg.peak < LEVEL_WAKE_CLIP) { await sayText("I couldn't hear you clearly."); return; }
    await submitChat({ method: 'POST', body: audioForm(seg.blob, 'recording.webm') });
  });
});

/* ── Typing ──────────────────────────────────────────────────────────────── */
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

/* ── "Always listening" switch ───────────────────────────────────────────── */
function hintText() { return wakeEnabled ? 'SAY “HEY DREAM” OR TAP' : 'TAP TO TALK'; }

$('wake-toggle').addEventListener('click', () => {
  wakeEnabled = !wakeEnabled;
  $('wake-toggle').classList.toggle('off', !wakeEnabled);
  $('mic-hint').textContent = hintText();
});

/* ── Events from app.py ──────────────────────────────────────────────────── */
function onSleep(isAsleep) {
  if (isAsleep) {
    if (busy) { pendingSleep = true; return; }   // let her finish talking first
    sleeping = true;
    setVisualState('sleeping');
  } else {
    pendingSleep = null;
    sleeping = false;
    if (!busy) setVisualState('idle');
  }
}

function connectSSE() {
  const sse = new EventSource('/events');

  sse.onopen = () => { $('dream-dot').className = 'on'; };
  sse.onerror = () => { $('dream-dot').className = 'warn'; setTimeout(connectSSE, 5000); };

  sse.onmessage = e => {
    let d;
    try { d = JSON.parse(e.data); } catch { return; }

    switch (d.type) {
      case 'state':
        // While this page drives the avatar itself (busy), it ignores the server's
        // state; talking is timed by the audio playing, so that pulse is ignored too.
        if (!busy && d.state !== 'talking') setVisualState(d.state);
        break;
      case 'transcript':
        if (d.role !== 'system') showCaption(d.text);
        break;
      case 'error':
        showCaption('⚠ ' + d.msg);
        break;
      case 'sleep':
        onSleep(d.sleeping);
        break;
      case 'wake':    // the sensor saw someone while she slept: greet, then listen
        if (started && !busy) runExclusive(() => wakeFlow(d.text, d.audio_url));
        break;
      case 'speak':   // e.g. the goodbye when someone leaves
        if (started && !busy) runExclusive(() => say(d.text, d.audio_url));
        break;
      case 'flirt':   // idle for a while: one flirty clip
        if (started && !busy && !sleeping) runExclusive(() => playOnce(d.clip, 'FLIRTING'));
        break;
    }
  };
}

/* ── Startup: same order as dream.py — intro clip, then the announcement ─── */
async function startup() {
  if (pools.intro.length) await playOnce(pools.intro[0], 'WAKING UP');
  if (!sleeping && cfg.startup_text) await sayText(cfg.startup_text);
}

$('splash').addEventListener('click', async () => {
  if (started) return;
  started = true;
  $('splash').classList.add('gone');
  // Ask for the mic alongside the startup sequence, not before it: the permission
  // prompt can sit there unanswered and she shouldn't stay silent meanwhile.
  ensureMic().then(micOk => {
    if (micOk) return;
    wakeEnabled = false;
    $('wake-toggle').classList.add('off');
    $('mic-hint').textContent = hintText();
  });
  runExclusive(startup);
});

/* ── Init ────────────────────────────────────────────────────────────────── */
async function loadConfig() {
  try { cfg = { ...cfg, ...(await (await fetch('/api/dream/config')).json()) }; } catch { /* defaults */ }
  if (cfg.sleeping) onSleep(true);
}

async function loadVideoPools() {
  try { pools = { ...pools, ...(await (await fetch('/api/videos')).json()) }; } catch { /* empty pools: playClip no-ops */ }
  curState = null;  // the state may already be set (asleep on load) but had no clips to show yet
  setVisualState(sleeping ? 'sleeping' : 'idle');
}

window.addEventListener('DOMContentLoaded', async () => {
  $('mic-hint').textContent = hintText();
  connectSSE();
  await loadConfig();
  await loadVideoPools();
  wakeLoop();
});
