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
// Speech vs silence follows the room: the quietest 50 ms in the last NOISE_WINDOW_MS is the
// background (fan, hiss), and only SPEECH_OVER_NOISE times that (at least SPEECH_RMS_MIN) is talking.
const SPEECH_RMS_MIN    = 0.01;
const SPEECH_OVER_NOISE = 3;
const NOISE_WINDOW_MS   = 2000;
const SILENCE_END_MS  = 1500;   // this long quiet after speech ends the command
const NO_SPEECH_MS    = 6000;   // this long with no speech at all gives up
const MANUAL_NO_SPEECH_MS = 10000;  // tap-to-talk waits a bit longer for you to start
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
let showThoughts = false;  // "Show thoughts" (💭): starts as config.json ShowThoughts, then this browser remembers
const thoughts = [];       // her unspoken thoughts, kept even while hidden
const MAX_THOUGHTS = 40;

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

/* Upright phone: each clip's portrait twin (videos/portrait/), swapped mid-clip when it's turned.
   curPath stays the landscape clip: that's what the mirrors (RITA) and pickClip go by. */
let portrait = {};
const tall = window.matchMedia('(orientation: portrait)');
const shown = path => (tall.matches && portrait[path]) || path;

tall.addEventListener('change', () => {
  if (!curPath || !front.src) return;
  const want = shown(curPath);
  if (new URL(want, location.href).href === front.src) return;
  const at = front.currentTime, old = front;
  back.loop = old.loop;
  back.onended = old.onended;
  back.onerror = old.onerror;
  back.src = want;
  back.oncanplay = () => {
    back.oncanplay = null;
    try { back.currentTime = at; } catch { /* starts from the top */ }
    back.play().catch(() => {});
    back.classList.add('showing');
    old.classList.remove('showing');
    old.onended = old.onerror = null;   // only the shown twin finishes a one-shot clip
    old.pause();
    [front, back] = [back, front];
  };
  back.load();
});

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
  back.src = shown(path);
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
  tellMirror({ clip: path, loop });
}

/* What she shows and says, for robots that mirror her (RITA: the same clip, her voice). */
function tellMirror(what) {
  if (!isBody) return;   // only where she is: a page on standby doesn't speak for her
  fetch('/api/dream/now', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(what),
  }).catch(() => {});
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
    tellMirror({ speech: url });
    const done = () => { if (currentAudio === a) { currentAudio = null; tellMirror({ speech_done: true }); } resolve(); };
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

/* Tell app.py how a line to another robot is going ('speaking', then 'done'). */
function peerLine(id, status) {
  return fetch('/api/dream/peer_line', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ id, status }),
  }).catch(() => {});
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
    if (d.clip) await playOnce(d.clip, 'REQUEST');   // an add-on's clip, played once
    if (d.listen) await listenForCommand();   // she asked yes or no: answer without "Hey DREAM"
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

/* Loudness (RMS, 0..1) of the latest mic samples: steadier than the peak for telling talk from noise. */
function micRms() {
  analyser.getFloatTimeDomainData(levelBuf);
  let sum = 0;
  for (let i = 0; i < levelBuf.length; i++) sum += levelBuf[i] * levelBuf[i];
  return Math.sqrt(sum / levelBuf.length);
}

/* The room's background loudness: the quietest reading of the last NOISE_WINDOW_MS. */
const noiseReadings = [];
function noiseFloor(now, rms) {
  noiseReadings.push([now, rms]);
  while (noiseReadings.length && now - noiseReadings[0][0] > NOISE_WINDOW_MS) noiseReadings.shift();
  return Math.min(...noiseReadings.map(r => r[1]));
}

function pickMime() {
  return ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg'].find(m => MediaRecorder.isTypeSupported(m)) || '';
}

/**
 * Record from the mic. Resolves { blob, peak, heard } when it stops:
 *   maxMs      hard limit
 *   vad        also stop after SILENCE_END_MS of quiet following speech, or
 *              noSpeechMs with none at all (quiet = near the room's own noise)
 *   shouldStop polled every tick (tap-to-talk's second tap)
 */
function recordSegment({ maxMs, vad = false, noSpeechMs = NO_SPEECH_MS, shouldStop = null }) {
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
      peak = Math.max(peak, micLevel());
      const rms = micRms();
      if (rms > Math.max(SPEECH_RMS_MIN, noiseFloor(now, rms) * SPEECH_OVER_NOISE)) { heard = true; lastVoice = now; }
      const elapsed = now - t0;
      const done = elapsed >= maxMs
        || (shouldStop && shouldStop())
        || (vad && ((heard && now - lastVoice > SILENCE_END_MS) || (!heard && elapsed > noSpeechMs)));
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
  await listenForCommand();
}

/* Record one command (stops on silence) and send it. Without a mic, type instead. */
async function listenForCommand() {
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
    if (!started || !isBody || !micStream || !wakeEnabled || busy || manualRecording) { await wait(300); continue; }

    const seg = await recordSegment({ maxMs: cfg.wake_seconds * 1000 });
    if (busy || manualRecording || !wakeEnabled) continue;   // something started meanwhile
    if (seg.peak < LEVEL_WAKE_CLIP) continue;                // silence: nothing to transcribe

    let d;
    try {
      d = await (await fetch('/api/wake', { method: 'POST', body: audioForm(seg.blob, 'wake.webm') })).json();
    } catch { await wait(1000); continue; }
    if (busy) continue;                                      // something else took over while it transcribed

    if (d.trigger === 'wake') runExclusive(() => wakeFlow(d.text, d.audio_url));
    else if (d.trigger === 'stir') runExclusive(() => say(d.text, d.audio_url));   // mumbles, still asleep
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
    $('mic-hint').textContent = 'LISTENING… STOPS WHEN YOU DO';
    setVisualState('listening');
    let seg;
    try {
      seg = await recordSegment({ maxMs: MANUAL_MAX_MS, vad: true, noSpeechMs: MANUAL_NO_SPEECH_MS,
                                 shouldStop: () => manualStop });  // stops when you stop talking, or on a second tap
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

/* ── "Show thoughts": TINA's reasoning and the robots' answers, never spoken ─ */
function renderThoughts() {
  const box = $('thoughts');
  box.hidden = !showThoughts;
  $('thoughts-toggle').classList.toggle('off', !showThoughts);
  if (!showThoughts) return;
  box.replaceChildren();
  if (!thoughts.length) {
    const p = document.createElement('p');
    p.className = 'empty';
    p.textContent = 'No thoughts yet. Ask her to tell a robot something, or hand TINA a job.';
    box.appendChild(p);
    return;
  }
  for (const t of thoughts) {
    const p = document.createElement('p');
    p.className = 'thought';
    p.textContent = t;
    box.appendChild(p);
  }
  box.scrollTop = box.scrollHeight;
}

function addThought(text) {
  thoughts.push(text);
  if (thoughts.length > MAX_THOUGHTS) thoughts.shift();
  renderThoughts();
}

function toggleThoughts() {
  showThoughts = !showThoughts;
  try { localStorage.setItem('dream.showThoughts', showThoughts ? '1' : '0'); } catch { /* not remembered */ }
  renderThoughts();
}
$('thoughts-toggle').addEventListener('click', toggleThoughts);

/* ── Her controls: hidden until Space (or a double-tap on her) shows them ─── */
/* The thoughts panel keeps its own switch (💭 or T), so it isn't hidden with them. */
function toggleUi() {
  document.body.classList.toggle('ui-hidden');
  if (document.activeElement && document.activeElement.tagName === 'BUTTON') document.activeElement.blur();
}

window.addEventListener('keydown', e => {
  const typing = /^(INPUT|TEXTAREA)$/.test(e.target.tagName) || e.target.isContentEditable;
  if (typing || e.repeat || e.ctrlKey || e.altKey || e.metaKey) return;
  if (e.code === 'Space') { e.preventDefault(); toggleUi(); }     // not a click on the focused button
  else if (e.code === 'KeyT') toggleThoughts();
});

// Phones have no Space bar: two taps on her face within 350 ms do the same.
let lastTap = 0;
stage.addEventListener('pointerup', () => {
  const now = Date.now();
  if (now - lastTap < 350) { toggleUi(); lastTap = 0; } else lastTap = now;
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
        if (started && isBody && !busy) runExclusive(() => wakeFlow(d.text, d.audio_url));
        break;
      case 'stir':    // the sensor nudged her but she sleeps on: a sleepy mumble
        if (started && isBody && !busy) runExclusive(() => say(d.text, d.audio_url));
        break;
      case 'speak':   // e.g. the goodbye when someone leaves
        if (started && isBody && !busy) runExclusive(() => say(d.text, d.audio_url));
        break;
      case 'peer_line':   // her line to KIDA (peer_talk.py): said like any reply; app.py waits for "done"
        if (started && !busy && !sleeping) runExclusive(async () => {
          await peerLine(d.id, 'speaking');
          try { await say(d.text, d.audio_url); } finally { await peerLine(d.id, 'done'); }
        });
        break;
      case 'thought':  // shown with "Show thoughts" on; never spoken
        addThought(d.text);
        break;
      case 'flirt':   // idle for a while: one flirty clip
        if (started && isBody && !busy && !sleeping) runExclusive(() => playOnce(d.clip, 'FLIRTING'));
        break;
    }
  };
}

/* ── Her body: the newest open page of hers is where she is ────────────────
   It shows her, speaks for her, listens for her and (with a camera) sees for her. Every other page
   waits on standby, silent, saying where she is; close the newest and the one before it takes over
   within a few seconds (close the DREAM proxy's window: she's back on this PC). scripts/dream_eyes.py. */
const PARAMS = new URLSearchParams(location.search);
const PAGE_ID = Math.random().toString(36).slice(2);
const PAGE_STARTED = Date.now();
const BODY_LABEL = PARAMS.get('body')
  || (/Mobi|Android|iPhone|iPad/.test(navigator.userAgent) ? 'a phone'
  : ['localhost', '127.0.0.1'].includes(location.hostname) ? 'this PC' : location.hostname);
let isBody = true;
let camVideo = null, camCanvas = null, camBusy = false;

function bodyNote() {
  let el = document.getElementById('body-note');
  if (!el) { el = document.createElement('div'); el.id = 'body-note'; document.body.appendChild(el); }
  return el;
}

function setBody(active, where) {
  bodyNote().textContent = active ? '' : `DREAM is with you on ${where}`;
  if (active === isBody) return;
  isBody = active;
  document.body.classList.toggle('standby', !active);
  if (!active && currentAudio) currentAudio.pause();   // her voice is somewhere else now
  if (active) setVisualState(sleeping ? 'sleeping' : 'idle');
}

async function bodyBeat() {
  if (!started) return;
  try {
    const d = await (await fetch('/api/body/beat', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ page: PAGE_ID, started: PAGE_STARTED, label: BODY_LABEL }),
    })).json();
    setBody(!!d.active, d.label);
  } catch { /* her server is gone: stay as we are */ }
}

/* The phone's own sensors (proximity, light, movement, battery): the Android app hands them over
   here, and they go to her only while this page is her body. Her sensor board keeps working too. */
window.dreamPhoneSensors = readings => {
  if (!started || !isBody) return;
  fetch('/api/body/sensors', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...readings, page: PAGE_ID, started: PAGE_STARTED, label: BODY_LABEL }),
  }).catch(() => {});
};

/* her eyes: this device's camera, a few small frames a second, while this page is her body */
async function ensureCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) return;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 } } });
    camVideo = document.createElement('video');
    camVideo.muted = true; camVideo.playsInline = true; camVideo.srcObject = stream;
    await camVideo.play();
    camCanvas = document.createElement('canvas');
  } catch { camVideo = null; }   // no camera, or not allowed: the proxy's or her own PC's then
}

function sendFrame() {
  if (!isBody || !camVideo || camBusy || !camVideo.videoWidth || document.hidden) return;
  const w = 480, h = Math.round(camVideo.videoHeight * w / camVideo.videoWidth);
  camCanvas.width = w; camCanvas.height = h;
  camCanvas.getContext('2d').drawImage(camVideo, 0, 0, w, h);
  camBusy = true;
  camCanvas.toBlob(blob => {
    const fd = new FormData();
    fd.append('frame', blob, 'eyes.jpg');
    fd.append('page', PAGE_ID);
    fd.append('started', String(PAGE_STARTED));
    fetch('/api/eyes/frame', { method: 'POST', body: fd })
      .then(r => r.json()).then(d => { bodyNote().dataset.eyes = d.eyes ? 'yes' : ''; })
      .catch(() => {}).finally(() => { camBusy = false; });
  }, 'image/jpeg', 0.7);
}

window.addEventListener('pagehide', () => {
  navigator.sendBeacon('/api/body/bye', JSON.stringify({ page: PAGE_ID }));
});

/* ── Startup: same order as dream.py — intro clip, then the announcement ─── */
async function startup() {
  if (pools.intro.length) await playOnce(pools.intro[Math.floor(Math.random() * pools.intro.length)], 'WAKING UP');
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
  bodyBeat();
  setInterval(bodyBeat, 1000);
  ensureCamera().then(() => { if (camVideo) setInterval(sendFrame, 350); });
});

/* ── Init ────────────────────────────────────────────────────────────────── */
async function loadConfig() {
  try { cfg = { ...cfg, ...(await (await fetch('/api/dream/config')).json()) }; } catch { /* defaults */ }
  if (cfg.sleeping) onSleep(true);
  let saved = null;
  try { saved = localStorage.getItem('dream.showThoughts'); } catch { /* no storage */ }
  showThoughts = saved === null ? !!cfg.show_thoughts : saved === '1';
  renderThoughts();
}

async function loadVideoPools() {
  try {
    const got = await (await fetch('/api/videos')).json();
    portrait = got.portrait || {};
    delete got.portrait;
    pools = { ...pools, ...got };
  } catch { /* empty pools: playClip no-ops */ }
  curState = null;  // the state may already be set (asleep on load) but had no clips to show yet
  setVisualState(sleeping ? 'sleeping' : 'idle');
}

window.addEventListener('DOMContentLoaded', async () => {
  $('mic-hint').textContent = hintText();
  connectSSE();
  await loadConfig();
  await loadVideoPools();
  wakeLoop();
  if (PARAMS.get('autostart') === '1') $('splash').click();
});
