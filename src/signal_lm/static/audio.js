const $ = id => document.getElementById(id);
const number = value => value === null ? '—' : new Intl.NumberFormat('fr-FR', {maximumFractionDigits: 4}).format(value);
let session = null, timer = null, latest = null, available = [], busy = false, stopping = false;
async function api(action, body = {}) {
  const response = await fetch('/api/audio/' + action, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const data = await response.json(); if(!response.ok) throw Error(data.error || 'Erreur de capture audio.'); return data;
}
function controls(active) {
  $('start-audio').disabled = active || !available.length || busy;
  $('stop-audio').disabled = !active || stopping;
  $('zone').disabled = $('device').disabled = $('devices-refresh').disabled = active || busy;
  $('live-badge').textContent = active ? 'CAPTURE ACTIVE · AUDIO PC' : 'CAPTURE ARRÊTÉE';
}
function deviceInfo() {
  const info = available.find(d => String(d.id) === $('device').value);
  $('device-info').textContent = info ? `${number(info.sample_rate)} Hz · ${info.channels} canaux · format natif` : 'Aucune sortie compatible.';
}
async function loadDevices() {
  $('audio-error').textContent = ''; busy = true; controls(false);
  try {
    const data = await api('devices'); available = data.devices;
    $('device').replaceChildren(...available.map(d => {const o=document.createElement('option');o.value=d.id;o.textContent=d.name;o.selected=d.default;return o;}));
    if(!available.length) throw Error('Aucune sortie WASAPI loopback disponible. Vérifiez les périphériques audio Windows.');
    deviceInfo();
  } catch(error) {available=[]; $('audio-error').textContent=error.message;}
  finally {busy=false; controls(false);}
}
function renderAudio() {
  if(!latest)return;
  const index=Number($('audio-channel').value), m=latest.measurements[index];
  $('audio-frequency').textContent=number(m.dominant_frequency_hz)+' Hz';
  $('audio-rms').textContent=number(m.rms)+' FS';$('audio-peak').textContent=number(m.peak_abs)+' FS';
  chart('wave',latest.waveform.time_s,latest.waveform.values[index],'#326ce5');
  chart('spectrum',latest.spectrum.frequency_hz,latest.spectrum.amplitude_by_channel[index],'#087e8b');
  $('audio-window').textContent=`${latest.capture.zone} · ${number(latest.sample_rate_hz)} Hz · ${latest.sample_count} échantillons`;
  $('audio-time').textContent='Fin de fenêtre : '+new Date(latest.capture.window_end_utc).toLocaleString('fr-FR');
  $('audio-quality').textContent=`${latest.capture.discontinuities} rupture(s) · ${latest.capture.clipped ? 'pleine échelle atteinte' : 'sous la pleine échelle'} · pas ${number(latest.frequency_bin_spacing_hz)} Hz`;
}
async function poll() {
  if(!session || stopping)return;
  const current=session;
  try {
    const frame=await api('frame',{session:current}); if(session!==current || stopping)return;
    if(frame.waiting) {
      $('audio-status').textContent=frame.message;
      $('live-badge').textContent='CAPTURE ACTIVE · EN ATTENTE';
      $('audio-time').textContent='En attente de données récentes ; les courbes précédentes restent figées.';
    } else {
      latest=frame;
      const count=frame.measurements.length;
      if($('audio-channel').options.length!==count) $('audio-channel').replaceChildren(...frame.measurements.map((m,i)=>{const o=document.createElement('option');o.value=i;o.textContent='Canal '+(i+1);return o;}));
      renderAudio(); $('audio-status').textContent='Audio en direct · '+frame.capture.zone;
      $('live-badge').textContent='CAPTURE ACTIVE · AUDIO PC';
    }
    timer=setTimeout(poll,250);
  } catch(error) {
    $('audio-error').textContent=error.message;
    await stopAudio();
  }
}
$('audio-form').onsubmit=async event=>{
  event.preventDefault(); if(session || busy)return;
  busy=true; controls(false); $('audio-error').textContent='';
  try {
    const data=await api('start',{device_id:Number($('device').value),zone:$('zone').value});
    session=data.session;latest=null;
    $('wave').replaceChildren();$('spectrum').replaceChildren();
    for(const id of ['audio-frequency','audio-rms','audio-peak']) $(id).textContent='—';
    $('audio-status').textContent='Capture active · remplissage de la fenêtre…';
  } catch(error) {$('audio-error').textContent=error.message;}
  finally {busy=false;controls(Boolean(session));}
  if(session)poll();
};
async function stopAudio() {
  clearTimeout(timer); if(!session)return;
  const current=session; stopping=true; controls(true);
  try {
    await api('stop',{session:current});session=null;
    $('audio-status').textContent='Capture arrêtée · dernière fenêtre figée';
  } catch(error) {
    $('audio-error').textContent='Arrêt non confirmé : '+error.message+'. Sans suivi, expiration automatique sous 15 secondes.';
    session=null;
  } finally {stopping=false;controls(false);}
}
$('stop-audio').onclick=stopAudio;$('devices-refresh').onclick=loadDevices;$('device').onchange=deviceInfo;$('audio-channel').onchange=renderAudio;
window.addEventListener('pagehide',()=>{clearTimeout(timer);if(session)fetch('/api/audio/stop',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session}),keepalive:true}).catch(()=>{});});
loadDevices();
