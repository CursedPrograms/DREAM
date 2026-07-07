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

/* ── NORA WiFi status ───────────────────────────────────────────────────── */
function setNoraUI(d) {
  $('nora-range').textContent     = d.in_range ? 'yes' : 'no';
  $('nora-connected').textContent = d.connected ? 'yes' : 'no';
}

async function loadNora() {
  try {
    const r = await fetch('/api/nora');
    setNoraUI(await r.json());
  } catch { /* silent */ }
}

$('nora-reconnect-btn').addEventListener('click', async () => {
  $('nora-connected').textContent = 'checking…';
  try {
    await fetch('/api/nora/connect', { method: 'POST' });
  } catch { /* silent */ }
  setTimeout(loadNora, 3000);
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
  connectSSE();
  setInterval(loadNora, 15000);
});
