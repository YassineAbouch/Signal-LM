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
