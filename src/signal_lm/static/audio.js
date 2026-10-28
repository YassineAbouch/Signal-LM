const $ = id => document.getElementById(id);
const number = value => value === null ? '—' : new Intl.NumberFormat('fr-FR', {maximumFractionDigits: 4}).format(value);
let session = null, timer = null, latest = null, available = [], busy = false, stopping = false;
let sourceMode = 'live', fileBusy = false;
async function api(action, body = {}) {
  const response = await fetch('/api/audio/' + action, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const data = await response.json(); if(!response.ok) throw Error(data.error || 'Erreur de capture audio.'); return data;
}
function controls(active) {
  $('start-audio').disabled = active || !available.length || busy;
  $('stop-audio').disabled = !active || stopping;
  $('zone').disabled = $('device').disabled = $('devices-refresh').disabled = active || busy;
  if(sourceMode==='live') $('live-badge').textContent = active ? 'CAPTURE ACTIVE · AUDIO PC' : 'CAPTURE ARRÊTÉE';
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
  const info=latest.audio_file || latest.capture;
  $('audio-window').textContent=`${info.zone} · ${number(latest.sample_rate_hz)} Hz · ${latest.sample_count} échantillons`;
  $('audio-time').textContent=latest.audio_file ? `${info.name} · ${number(info.window_start_s)}–${number(info.window_end_s)} s` : 'Fin de fenêtre : '+new Date(info.window_end_utc).toLocaleString('fr-FR');
  $('audio-quality').textContent=(latest.audio_file ? 'Première piste audio' : `${info.discontinuities} rupture(s)`) + ` · ${info.clipped ? 'pleine échelle atteinte' : 'sous la pleine échelle'} · pas ${number(latest.frequency_bin_spacing_hz)} Hz`;
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

async function switchSource(mode) {
  if(fileBusy || busy || stopping)return;
  if(session)await stopAudio();
  sourceMode=mode;
  $('audio-form').hidden=$('live-help').hidden=mode!=='live';
  $('audio-file-form').hidden=mode!=='file';
  $('audio-clock-help').textContent=mode==='file'?'La position est relative au début du fichier. La date affichée est celle de l’analyse ; la date d’acquisition du son est inconnue.':'L’heure affichée est locale ; les horodatages internes sont en UTC, estimés à la réception sur le PC.';
  for(const [key,value] of [['live','live'],['file','file']]) {$(key+'-tab').classList.toggle('selected',mode===value);$(key+'-tab').setAttribute('aria-pressed',String(mode===value));}
  $('live-badge').textContent=mode==='file'?'FICHIER AUDIO':'CAPTURE ARRÊTÉE';
  latest=null;$('wave').replaceChildren();$('spectrum').replaceChildren();
  for(const id of ['audio-frequency','audio-rms','audio-peak'])$(id).textContent='—';
  $('audio-window').textContent=mode==='file'?'Choisissez un fichier et une position à analyser.':'Démarrez la capture pour afficher le signal.';
  $('audio-time').textContent='Aucune fenêtre analysée';$('audio-quality').textContent='Aucune donnée';
  $('audio-status').textContent=mode==='file'?'Fichier audio · prêt':'Capture arrêtée';
}
$('live-tab').onclick=()=>switchSource('live');$('file-tab').onclick=()=>switchSource('file');
$('audio-file-form').onsubmit=async event=>{
  event.preventDefault();if(fileBusy)return;
  fileBusy=true;$('analyze-file').disabled=true;$('file-error').textContent='';
  $('live-badge').textContent='ANALYSE DU FICHIER…';
  $('audio-status').textContent='Décodage et analyse du fichier…';
  try {
    const file=$('audio-file').files[0];if(!file)throw Error('Choisissez un fichier audio.');
    if(file.size>50*1024*1024)throw Error('Fichier limité à 50 Mio.');
    const response=await fetch('/api/audio/file',{method:'POST',headers:{'Content-Type':'application/octet-stream',
      'X-Audio-Name':encodeURIComponent(file.name),'X-Audio-Zone':encodeURIComponent($('file-zone').value),
      'X-Audio-Start':$('file-start').value},body:file});
    const data=await response.json();if(!response.ok)throw Error(data.error || 'Fichier non lisible.');
    latest=data;$('audio-channel').replaceChildren(...data.measurements.map((m,i)=>{const o=document.createElement('option');o.value=i;o.textContent='Canal '+(i+1);return o;}));
    renderAudio();$('live-badge').textContent='FICHIER ANALYSÉ';
    $('audio-status').textContent='Analyse terminée à '+new Date(data.audio_file.analyzed_at_utc).toLocaleString('fr-FR')+' · date d’acquisition inconnue';
  } catch(error) {$('file-error').textContent=error.message;$('audio-status').textContent='Fichier non analysé';$('live-badge').textContent='ANALYSE ÉCHOUÉE';}
  finally {fileBusy=false;$('analyze-file').disabled=false;}
};
