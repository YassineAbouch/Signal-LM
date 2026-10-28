const $ = id => document.getElementById(id);
let source = 'demo', result = null;
function mode(value) {
  source = value;
  $('demo-fields').hidden = value !== 'demo'; $('csv-fields').hidden = value !== 'csv';
  for (const key of ['demo','csv']) { $(key+'-tab').classList.toggle('selected', key===value); $(key+'-tab').setAttribute('aria-pressed', String(key===value)); }
  $('frequency').required = $('amplitude').required = value === 'demo';
  $('error').textContent = '';
}
$('demo-tab').onclick = () => mode('demo'); $('csv-tab').onclick = () => mode('csv');
function number(v) { return v === null ? '—' : new Intl.NumberFormat('fr-FR',{maximumFractionDigits:4}).format(v); }
function metric(id, value, unit) { $(id).replaceChildren(document.createTextNode(number(value)+' ')); const em=document.createElement('em'); em.textContent=unit; $(id).append(em); }
function chart(id,x,y,color) {
  const ns='http://www.w3.org/2000/svg'; const svg=document.createElementNS(ns,'svg'); svg.setAttribute('viewBox','0 0 800 190'); svg.setAttribute('role','img');
  const title=document.createElementNS(ns,'title'); title.textContent=id==='wave'?'Signal temporel, aperçu réduit':'Spectre d’amplitude';svg.append(title);
  const left=56, right=784, top=12, bottom=163;
  let ymin=Math.min(0,...y), ymax=Math.max(...y); if(ymax===ymin)ymax=ymin+1;
  const xmin=x[0], xmax=x[x.length-1]||1;
  const sx=v=>left+(v-xmin)/(xmax-xmin||1)*(right-left), sy=v=>bottom-(v-ymin)/(ymax-ymin)*(bottom-top);
  function element(tag,attrs,text){const e=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;svg.append(e);return e;}
  for(let i=0;i<=4;i++){let yy=top+(bottom-top)*i/4;element('line',{x1:left,y1:yy,x2:right,y2:yy,stroke:'#e8edf3','stroke-dasharray':'3 5'});element('text',{x:left-9,y:yy+4,'text-anchor':'end'},number(ymax-(ymax-ymin)*i/4));let xx=left+(right-left)*i/4;element('text',{x:xx,y:185,'text-anchor':i===4?'end':'middle'},number(xmin+(xmax-xmin)*i/4));}
  // Keep spectral maxima in each display bucket rather than dropping narrow peaks.
  let points=[]; const stride=Math.max(1,Math.ceil(x.length/1400));
  for(let i=0;i<x.length;i+=stride){let index=i;if(id==='spectrum')for(let j=i+1;j<Math.min(i+stride,x.length);j++)if(y[j]>y[index])index=j;points.push(sx(x[index]).toFixed(2)+','+sy(y[index]).toFixed(2));}
  element('polyline',{points:points.join(' '),fill:'none',stroke:color,'stroke-width':1.6,'stroke-linejoin':'round'});$(id).replaceChildren(svg);
}
function render() {
  if(!result)return; const c=Number($('channel').value), m=result.measurements[c];
  metric('freq-value',m.dominant_frequency_hz,'Hz'); metric('rms-value',m.rms,m.unit); metric('amp-value',m.dominant_amplitude,m.unit);
  chart('wave',result.waveform.time_s,result.waveform.values[c],'#326ce5');chart('spectrum',result.spectrum.frequency_hz,result.spectrum.amplitude_by_channel[c],'#087e8b');
  $('wave-meta').textContent=`${number(result.sample_count)} échantillons · ${number(result.sample_rate_hz)} Hz · ${number(result.window_duration_s)} s`;
  $('wave-unit').textContent=`${m.channel} · ${m.unit} · aperçu ${result.waveform.display_samples} points`;
  $('spectrum-meta').textContent=`Pas fréquentiel : ${number(result.frequency_bin_spacing_hz)} Hz`;
}
$('channel').onchange=render;
$('form').onsubmit=async event=>{
  event.preventDefault();$('error').textContent='';$('run').disabled=true;$('run').textContent='Analyse en cours…';$('status').textContent='Calcul des mesures…';
  try {
    const payload={source,sample_rate:Number($('rate').value),frequency:Number($('frequency').value),amplitude:Number($('amplitude').value)};
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
