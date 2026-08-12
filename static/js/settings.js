'use strict';

const $ = id => document.getElementById(id);

/* ── Alarm toggle ───────────────────────────────────────────────────────── */
function setAlarmUI(enabled, reachable) {
  const toggle = $('alarm-toggle');
  toggle.checked  = !!enabled;
  toggle.disabled = !reachable;
  $('alarm-status').textContent = reachable ? (enabled ? 'ON' : 'OFF') : 'unreachable';
}

async function loadAlarm() {
  try {
    const r = await fetch('/api/alarm');
    const d = await r.json();
    setAlarmUI(d.enabled, true);
  } catch {
    setAlarmUI(true, false);
  }
}

$('alarm-toggle').addEventListener('change', async e => {
  const wanted = e.target.checked;
  e.target.disabled = true;
  try {
    const r = await fetch('/api/alarm', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled: wanted }),
    });
    const d = await r.json();
    if (!r.ok) {
      $('alarm-status').textContent = 'unreachable';
      e.target.checked = !wanted;
    } else {
      setAlarmUI(d.enabled, true);
    }
  } catch {
    $('alarm-status').textContent = 'unreachable';
    e.target.checked = !wanted;
  } finally {
    e.target.disabled = false;
  }
});

/* ── NORA reachability ──────────────────────────────────────────────────── */
function setNoraUI(d) {
  $('nora-reachable').textContent = d.reachable ? 'yes' : 'no';
}

async function loadNora() {
  try {
    const r = await fetch('/api/nora');
    setNoraUI(await r.json());
  } catch { /* silent */ }
}

$('nora-recheck-btn').addEventListener('click', async () => {
  $('nora-reachable').textContent = 'checking…';
  try {
    await fetch('/api/nora/check', { method: 'POST' });
  } catch { /* silent */ }
  setTimeout(loadNora, 2000);
});

/* ── NORA serial command / text message ─────────────────────────────────── */
function flashNoraSendStatus(msg, isError) {
  const el = $('nora-send-status');
  el.textContent = msg;
  el.style.color = isError ? 'var(--danger)' : 'var(--ok)';
  setTimeout(() => { el.textContent = ''; }, 3000);
}

$('nora-serial-btn').addEventListener('click', async () => {
  const cmd = $('nora-serial-input').value.trim();
  if (!cmd) return;
  try {
    const r = await fetch('/api/nora/serial', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ cmd }),
    });
    const d = await r.json();
    flashNoraSendStatus(r.ok ? `Sent: ${cmd}` : (d.error || 'Failed'), !r.ok);
    if (r.ok) $('nora-serial-input').value = '';
  } catch (err) {
    flashNoraSendStatus(`Network error: ${err.message}`, true);
  }
});

$('nora-message-btn').addEventListener('click', async () => {
  const text = $('nora-message-input').value.trim();
  if (!text) return;
  try {
    const r = await fetch('/api/nora/message', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    const d = await r.json();
    flashNoraSendStatus(r.ok ? 'Message sent' : (d.error || 'Failed'), !r.ok);
    if (r.ok) $('nora-message-input').value = '';
  } catch (err) {
    flashNoraSendStatus(`Network error: ${err.message}`, true);
  }
});

/* ── Fleet heartbeat transport (WiFi/Bluetooth) ─────────────────────────── */
function syncFleetBtPortVisibility() {
  $('nora-fleet-bt-port').style.display =
    $('nora-fleet-mode-select').value === 'bluetooth' ? 'inline-block' : 'none';
}

async function loadFleetMode() {
  try {
    const r = await fetch('/api/nora/mode');
    const d = await r.json();
    $('nora-fleet-mode-select').value = d.mode || 'wifi';
    if (d.bt_port) $('nora-fleet-bt-port').value = d.bt_port;
    syncFleetBtPortVisibility();
  } catch { /* silent */ }
}

$('nora-fleet-mode-select').addEventListener('change', syncFleetBtPortVisibility);

$('nora-fleet-mode-btn').addEventListener('click', async () => {
  const mode    = $('nora-fleet-mode-select').value;
  const bt_port = $('nora-fleet-bt-port').value.trim();
  const status  = $('nora-fleet-mode-status');
  status.textContent = 'applying…';
  try {
    const r = await fetch('/api/nora/mode', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode, bt_port }),
    });
    const d = await r.json();
    status.textContent = r.ok ? `mode: ${d.mode}` : (d.error || 'failed');
    status.style.color = r.ok ? 'var(--ok)' : 'var(--danger)';
  } catch (err) {
    status.textContent = `Network error: ${err.message}`;
    status.style.color = 'var(--danger)';
  }
});

/* ── Live updates ───────────────────────────────────────────────────────── */
function connectSSE() {
  const sse = new EventSource('/events');
  sse.onmessage = e => {
    let d;
    try { d = JSON.parse(e.data); } catch { return; }
    if (d.type === 'alarm') setAlarmUI(d.enabled, true);
    if (d.type === 'nora') setNoraUI(d);
  };
}

window.addEventListener('DOMContentLoaded', () => {
  loadAlarm();
  loadNora();
  loadFleetMode();
  connectSSE();
  setInterval(loadNora, 15000);
});
