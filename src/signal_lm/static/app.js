if (location.hash === '#model') location.replace('/model');
const $ = id => document.getElementById(id);
let source = 'demo', result = null;
let componentId = 0;
function updateComponents() {
  const rows = [...$('components').children];
  rows.forEach((row, index) => {
    row.querySelector('h4').textContent = `Composante ${index + 1}`;
    const remove = row.querySelector('button');
    remove.disabled = rows.length === 1;
    remove.setAttribute('aria-label', `Supprimer la composante ${index + 1}`);
  });
  $('component-count').textContent = `${rows.length} / 16 composantes · même canal, même axe temporel`;
  $('add-component').disabled = rows.length >= 16;
}
function addComponent(frequency = 120, amplitude = 1, phase = 0) {
  if ($('components').children.length >= 16) return;
  const row = document.createElement('section'); row.className = 'component-row';
  const id = ++componentId;
  row.innerHTML = `<div class="component-heading"><h4></h4><button type="button" class="remove-component">×</button></div><div class="component-inputs"><div><label for="frequency-${id}">Fréq. (Hz)</label><input id="frequency-${id}" data-key="frequency" type="number" min="0.001" step="any" required></div><div><label for="amplitude-${id}">Ampl. (a.u.)</label><input id="amplitude-${id}" data-key="amplitude" type="number" min="0.001" max="1000" step="any" required></div><div><label for="phase-${id}">Phase (°)</label><input id="phase-${id}" data-key="phase_deg" type="number" min="-360" max="360" step="any" required></div></div>`;
  [frequency, amplitude, phase].forEach((value, index) => { row.querySelectorAll('input')[index].value = value; });
  row.querySelector('button').onclick = () => { row.remove(); updateComponents(); };
  $('components').append(row); updateComponents();
}
$('add-component').onclick = () => addComponent(240, 0.5, 0);
addComponent();
function mode(value) {
  source = value;
  $('demo-fields').hidden = value !== 'demo'; $('csv-fields').hidden = value !== 'csv';
  for (const key of ['demo','csv']) { $(key+'-tab').classList.toggle('selected', key===value); $(key+'-tab').setAttribute('aria-pressed', String(key===value)); }
  $('components').querySelectorAll('input').forEach(input => { input.disabled = value !== 'demo'; });
  $('error').textContent = '';
}
$('demo-tab').onclick = () => mode('demo'); $('csv-tab').onclick = () => mode('csv');
function number(v) { return v === null ? '—' : new Intl.NumberFormat('fr-FR',{maximumFractionDigits:4}).format(v); }
function metric(id, value, unit) { $(id).replaceChildren(document.createTextNode(number(value)+' ')); const em=document.createElement('em'); em.textContent=unit; $(id).append(em); }

function render() {
  if(!result)return; const c=Number($('channel').value), m=result.measurements[c];
  metric('freq-value',m.dominant_frequency_hz,'Hz'); metric('rms-value',m.rms,m.unit); metric('amp-value',m.dominant_amplitude,m.unit);
  const components = result.waveform.components || [];
  $('component-toggle-label').hidden = components.length < 2;
  chart('wave',result.waveform.time_s,result.waveform.values[c],'#326ce5', $('show-components').checked && components.length > 1 ? components : []);chart('spectrum',result.spectrum.frequency_hz,result.spectrum.amplitude_by_channel[c],'#087e8b');
  $('wave-meta').textContent=`${number(result.sample_count)} échantillons · ${number(result.sample_rate_hz)} Hz · ${number(result.window_duration_s)} s`;
  $('wave-unit').textContent=`${m.channel} · ${m.unit} · aperçu ${result.waveform.display_samples} points` + (components.length > 1 ? ' · bleu : somme / pointillés : composantes' : '');
  $('spectrum-meta').textContent=`Pas fréquentiel : ${number(result.frequency_bin_spacing_hz)} Hz`;
}
$('channel').onchange=render;
$('show-components').onchange=render;
$('form').onsubmit=async event=>{
  event.preventDefault();$('error').textContent='';$('run').disabled=true;$('run').textContent='Analyse en cours…';$('status').textContent='Calcul des mesures…';
  try {
    const payload={source,sample_rate:Number($('rate').value)};
    if(source==='demo') payload.components=[...$('components').children].map(row=>Object.fromEntries([...row.querySelectorAll('input')].map(input=>[input.dataset.key,Number(input.value)])));
    if(source==='csv'){
      const file=$('file').files[0];if(!file)throw Error('Choisissez un fichier CSV.');if(file.size>1900000)throw Error('Fichier trop volumineux (limite proche de 2 Mio).');
      payload.csv=await file.text();payload.delimiter=$('delimiter').value==='\\t'?'\t':$('delimiter').value;
      payload.columns=$('columns').value.split(',').map(s=>s.trim()).filter(Boolean);payload.units=$('units').value.split(',').map(s=>s.trim()).filter(Boolean);
    }
    const response=await fetch('/api/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const data=await response.json();if(!response.ok)throw Error(data.error||'Analyse impossible.');
    result=data;$('channel').replaceChildren(...data.measurements.map((m,i)=>{const o=document.createElement('option');o.value=i;o.textContent=m.channel;return o;}));render();$('export').disabled=false;$('status').textContent='Analyse terminée · aucune prédiction métier';
  } catch(error){$('error').textContent=error.message;$('status').textContent='Analyse non effectuée';}
  finally{$('run').disabled=false;$('run').innerHTML='Analyser le signal <span>→</span>';}
};
$('export').onclick=()=>{if(!result)return;const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='signal-lm-analysis.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
fetch('/api/doctor').then(async response=>{if(!response.ok)throw Error();return response.json();}).then(d=>{$('gpu').textContent=d.cuda_available?'CUDA disponible · '+d.gpu:'Calcul CPU · CUDA indisponible';$('parameters').textContent=number(d.parameters)+' paramètres';$('runtime').textContent=`Python ${d.python} / PyTorch ${d.torch}`;}).catch(()=>{$('gpu').textContent='Diagnostic système indisponible';});
$('form').requestSubmit();
