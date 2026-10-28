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
function chart(id,x,y,color,overlays=[]) {
  const ns='http://www.w3.org/2000/svg'; const svg=document.createElementNS(ns,'svg'); svg.setAttribute('viewBox','0 0 800 190'); svg.setAttribute('role','img');
  const title=document.createElementNS(ns,'title'); title.textContent=id==='wave'?'Signal temporel, aperçu réduit':'Spectre d’amplitude';svg.append(title);
  const left=56, right=784, top=12, bottom=163;
  let ymin=Math.min(0,...y), ymax=Math.max(...y);
  for(const values of overlays){ymin=Math.min(ymin,...values);ymax=Math.max(ymax,...values);}
  if(ymax===ymin)ymax=ymin+1;
  const xmin=x[0], xmax=x[x.length-1]||1;
  const sx=v=>left+(v-xmin)/(xmax-xmin||1)*(right-left), sy=v=>bottom-(v-ymin)/(ymax-ymin)*(bottom-top);
  function element(tag,attrs,text){const e=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;svg.append(e);return e;}
  for(let i=0;i<=4;i++){let yy=top+(bottom-top)*i/4;element('line',{x1:left,y1:yy,x2:right,y2:yy,stroke:'#e8edf3','stroke-dasharray':'3 5'});element('text',{x:left-9,y:yy+4,'text-anchor':'end'},number(ymax-(ymax-ymin)*i/4));let xx=left+(right-left)*i/4;element('text',{x:xx,y:185,'text-anchor':i===4?'end':'middle'},number(xmin+(xmax-xmin)*i/4));}
  for(const values of overlays){element('polyline',{points:values.map((v,i)=>sx(x[i]).toFixed(2)+','+sy(v).toFixed(2)).join(' '),fill:'none',stroke:'#8b6eae','stroke-opacity':0.45,'stroke-width':1,'stroke-dasharray':'4 3'});}
  // Keep spectral maxima in each display bucket rather than dropping narrow peaks.
  let points=[]; const stride=Math.max(1,Math.ceil(x.length/1400));
  for(let i=0;i<x.length;i+=stride){let index=i;if(id==='spectrum')for(let j=i+1;j<Math.min(i+stride,x.length);j++)if(y[j]>y[index])index=j;points.push(sx(x[index]).toFixed(2)+','+sy(y[index]).toFixed(2));}
  element('polyline',{points:points.join(' '),fill:'none',stroke:color,'stroke-width':1.6,'stroke-linejoin':'round'});$(id).replaceChildren(svg);
}
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
