'use strict';
const $=id=>document.getElementById(id);
const tasks=['ECG','PPG','ABP'];
const colors={raw:'#245d80',output:'#138b74',trigger:'#c86a16',ai:'#f6c961',out:'#808a90',modified:'#8353b3',rescan:'#b52d45'};
const kinds={ai:'本次 AI 圈选',excluded:'新增停用',modified:'数值改动',triggers:'初筛',rescan:'回扫命中'};
const fmt=(v,d=3)=>Number(v).toLocaleString('zh-CN',{maximumFractionDigits:d});
const hh=v=>{v=Math.round(v);return [Math.floor(v/3600),Math.floor(v/60)%60,v%60].map(n=>String(n).padStart(2,'0')).join(':');};
let meta, group, selected=0, cursor=[0,1], wanted=null, loadedScript=null;
const opt=(label,value)=>new Option(label,value);
function element(tag,txt){const el=document.createElement(tag);el.textContent=txt;return el;}
function table(el,head,rows){el.replaceChildren();for(const [i,cells] of [head,...rows].entries()){const tr=document.createElement('tr');cells.forEach(s=>tr.append(element(i?'td':'th',s)));el.append(tr);}}
function b64(s){const bytes=Uint8Array.from(atob(s),c=>c.charCodeAt(0));return new Float64Array(bytes.buffer);}
function focus(caseId,span){const b=meta.blocks.find(b=>Object.values(b.ids).includes(caseId));if(!b)return;
  $('span').value='5';$('kind').value='ai';$('task').value=Object.keys(b.ids).find(k=>b.ids[k]===caseId);load(b.i,span);$('detail').scrollIntoView();}
function load(i,range=null){selected=Math.min(meta.blocks.length-1,Math.max(0,i));$('block').value=selected;wanted=range;if($('block').selectedIndex<0){$('blockFilter').value='all';$('block').replaceChildren();for(const b of meta.blocks)$('block').add(opt(String(b.i+1).padStart(3,'0')+' · '+hh(b.start)+'–'+hh(b.start+b.duration),b.i));$('block').value=selected;}group=null;window.patientView=null;updateEvents();$('status').textContent='正在读取连续段…';
  if(loadedScript)loadedScript.remove();const s=document.createElement('script');s.src='blocks/'+String(selected).padStart(3,'0')+'.js';s.onerror=()=>{$('status').textContent='连续段文件读取失败。'};document.body.append(s);loadedScript=s;}
window.receiveMeta=function(m){meta=m;
  $('lead').textContent=m.record+' · 有观测 '+hh(m.stored_s)+' · 首末时间跨度 '+hh(m.elapsed_s)+' · '+m.blocks.length+' 个连续段，各含三通道';
  const showBlocks=()=>{const sel=$('block'), filter=$('blockFilter').value;sel.replaceChildren();
    for(const b of m.blocks){const counts=Object.values(b.counts);
      if(filter!=='all'&&!counts.some(c=>filter==='action'?(c.modified||c.excluded):c[filter]))continue;
      sel.add(opt(String(b.i+1).padStart(3,'0')+' · '+hh(b.start)+'–'+hh(b.start+b.duration)+' · '+fmt(b.duration)+' 秒',b.i));}
    if(!sel.options.length)sel.add(opt('没有匹配的连续段',''));
  };
  showBlocks();
  $('blockFilter').onchange=()=>{showBlocks();if($('block').value!=='')load(Number($('block').value));else{$('status').textContent='没有匹配的连续段，波形仍显示当前段。';}};
  table($('stats'),['通道','初筛触发条数','实际改值（点）','新增停用（秒）','估计值（点）','形成测量（分钟）'],
    tasks.map(k=>{const s=m.stats[k];return [k,fmt(s.candidate_triggers),fmt(s.modified),fmt(s.new_invalid/m.blocks[0].fs),fmt(s.estimated),fmt(s.scored_samples/m.blocks[0].fs/60,2)];}));
  for(const k of tasks){const row=document.createElement('div');row.append(element('strong',k));const cv=document.createElement('canvas');cv.id='cv'+k;cv.setAttribute('aria-label',k+' 原始与交付波形');row.append(cv);$('waves').append(row);cv.onmousemove=e=>showSample(k,e);}
  for(const r of m.reviews){const box=document.createElement('div');box.className='panel';const b=m.blocks.find(b=>Object.values(b.ids).includes(r.case));const fs=b.fs;
    const range=r.span||r.seed;box.append(element('h3',r.task+' · 第 '+(b.i+1)+' 段 · '+fmt(range[0]/fs)+'–'+fmt(range[1]/fs)+' 秒'+(r.span?'（AI 疑似范围）':'（触发区间，判断未决）')));
    box.append(element('p',r.obs));if(r.cause)box.append(element('p','可能原因：'+r.cause));if(r.limitations)box.append(element('p',r.limitations));
    const btn=element('button','在三个通道上查看');btn.onclick=()=>focus(r.case,range);box.append(btn);
    for(const im of r.images){const a=element('a','　查看当时图片 ±'+im.flank+' 秒');a.href=im.image;box.append(a);}
    $('reviews').append(box);}
  for(const r of m.batch_reviews){const box=document.createElement('div');box.className='panel';box.append(element('h3',r.task),element('p',r.observation));
    if(r.concerns.length)box.append(element('p',r.concerns.join('；')));
    for(const [i,im] of r.pictures.entries()){const a=element('a','自检图 '+(i+1)+'　');a.href=im.image;box.append(a);}$('batchReviews').append(box);}
  table($('sweeps'),['通道','方法 ID','检查段数','原始命中','处理后数值命中','可用范围仍命中','新增命中'],
    m.rescans.map(r=>[r.task,r.rule_id.slice(0,10),r.cases,...['before','after_numeric','after_eligible','newly_triggered'].map(k=>r.counts[k]||0)]));
  $('cost').textContent='本次新增 '+m.api_executions+' 次模型请求，用量估算 $'+m.api_increment.toFixed(4)+'。真实恢复准确率待独立参照。';
  $('block').onchange=()=>load(Number($('block').value));const adjacent=delta=>{const sel=$('block'),option=sel.options[sel.selectedIndex+delta];if(option&&option.value!=='')load(Number(option.value));};
  $('prev').onclick=()=>adjacent(-1);$('next').onclick=()=>adjacent(1);
  $('map').onclick=e=>{const rect=$('map').getBoundingClientRect();const x=(e.clientX-rect.left)/rect.width;const t=(x-.09)/.895*m.elapsed_s;
    if(t<0||t>=m.elapsed_s)return;const b=m.blocks.find(b=>b.start<=t&&t<b.start+b.duration);
    if(!b){$('hint').textContent=hh(t)+' 位于采集缺口。';return;}
    $('hint').textContent='已定位至 '+hh(t)+'，第 '+(b.i+1)+' 段。';
    const s=Math.min(b.n-1,Math.floor((t-b.start)*b.fs));load(b.i,[s,s+1]);$('detail').scrollIntoView();};
  $('span').onchange=draw;$('position').oninput=()=>{if(!group)return;const b=meta.blocks[selected];const a=Math.min(b.n-1,Math.round(+$('position').value*b.fs));cursor=[a,a+1];draw();};
  for(const id of ['raw','out','trig'])$(id).onchange=draw;
  for(const id of ['task','kind'])$(id).onchange=()=>updateEvents(true);
  $('event').onchange=()=>{if($('event').value){cursor=JSON.parse($('event').value);draw();}};
  const first=m.reviews.find(r=>r.task==='PPG')||m.reviews[0];
  if(first){const b=m.blocks.find(b=>Object.values(b.ids).includes(first.case));load(b.i,first.span||first.seed);}else load(0);};
window.receiveBlock=function(payload){if(payload.i!==selected)return;group=payload.signals;const b=meta.blocks[selected];
  for(const k of tasks){const s=group[k];s.raw=b64(s.raw);s.output=s.output?b64(s.output):s.raw;}
  cursor=wanted||[Math.floor(b.n/2),Math.floor(b.n/2)+1];wanted=null;
  $('position').max=(b.n-1)/b.fs;$('position').step=1/b.fs;
  let ops=[];for(const k of tasks){for(const op of group[k].operations)ops.push(k+' · '+op.op+' · '+JSON.stringify(op.parameters)+' · '+JSON.stringify(op.ranges_samples)+' 点（段内） · '+(typeof op.explanation==='string'?op.explanation:JSON.stringify(op.explanation??'原因未提供')));}
  $('ops').textContent=ops.join('；')||'本段采用原样观测。';updateEvents();draw();};
function updateEvents(jump=false){const sel=$('event');sel.replaceChildren();if(!group)return;
  const name=$('kind').value,fs=meta.blocks[selected].fs;
  const list=name==='ai'?group[$('task').value].ai.map(v=>v.span):group[$('task').value][name];
  sel.add(opt('本段 '+list.length+' 项 · 选择位置',''));
  for(const [a,b,reason] of list)sel.add(opt(fmt(a/fs)+'–'+fmt(b/fs)+' s'+(reason?' · '+reason.slice(0,25):''),JSON.stringify([a,b])));
  if(jump&&list.length){const item=list[0].slice(0,2);sel.value=JSON.stringify(item);cursor=item;draw();}else{sel.value=JSON.stringify(cursor);if(sel.selectedIndex<0)sel.selectedIndex=0;}}
function range(){const b=meta.blocks[selected];if($('span').value==='block')return [0,b.n];const pad=Math.round(+$('span').value*b.fs);return [Math.max(0,cursor[0]-pad),Math.min(b.n,cursor[1]+pad)];}
function draw(){if(!group)return;const b=meta.blocks[selected];const [a,z]=range();$('position').value=(cursor[0]+cursor[1])/(2*b.fs);
  $('time').textContent=fmt(+$('position').value)+' s';$('status').textContent='第 '+(b.i+1)+' 段 · 段内 '+fmt(a/b.fs)+'–'+fmt(z/b.fs)+' 秒 · 距首个观测 '+hh(b.start+a/b.fs)+' · 采样率 '+b.fs+' Hz';
  for(const k of tasks)plot(k,a,z);window.patientView={ids:b.ids,target:cursor,range:[a,z]};}
function plot(k,a,z){const s=group[k],el=$('cv'+k),width=el.clientWidth,height=el.clientHeight,dpr=window.devicePixelRatio||1;
  el.width=width*dpr;el.height=height*dpr;const ctx=el.getContext('2d');ctx.scale(dpr,dpr);const l=48,r=width-7,top=10,bottom=height-23;
  let lo=Infinity,hi=-Infinity;for(const arr of [s.raw,s.output])for(let i=a;i<z;i++){let v=arr[i];if(Number.isFinite(v)){lo=Math.min(lo,v);hi=Math.max(hi,v);}}
  if(!Number.isFinite(lo)){lo=0;hi=1;}const pad=(hi-lo||1)*.08;lo-=pad;hi+=pad;
  const xp=i=>l+(i-a)/(z-a)*(r-l),yp=v=>bottom-(v-lo)/(hi-lo)*(bottom-top);
  const band=(ranges,color,y0,y1,alpha)=>{ctx.globalAlpha=alpha;ctx.fillStyle=color;for(const [u,v] of ranges){if(v<=a||u>=z)continue;const x=xp(Math.max(u,a));ctx.fillRect(x,y0,Math.max(.8,xp(Math.min(z,v))-x),y1-y0);}ctx.globalAlpha=1;};
  band(s.ai.map(q=>q.span),colors.ai,top,bottom,.35);if($('trig').checked)band(s.triggers,colors.trigger,top,top+6,.8);
  band(s.stops,colors.out,top,bottom,.12);for(const [u,v] of s.stops){if(v<=a||u>=z)continue;ctx.save();const x=xp(Math.max(a,u)),w=Math.max(.8,xp(Math.min(v,z))-x);
    ctx.beginPath();ctx.rect(x,top,w,bottom-top);ctx.clip();ctx.strokeStyle='#a1a7aa';ctx.lineWidth=.6;ctx.beginPath();for(let i=x-height;i<x+w;i+=8){ctx.moveTo(i,bottom);ctx.lineTo(i+height,top);}ctx.stroke();ctx.restore();}
  band(s.modified,colors.modified,bottom-6,bottom,.95);band(s.rescan,colors.rescan,bottom-12,bottom-7,.95);
  for(const [arr,on,color,dash] of [[s.raw,$('raw').checked,colors.raw,[]],[s.output,$('out').checked,colors.output,[4,3]]]){
    if(!on)continue;ctx.strokeStyle=color;ctx.setLineDash(dash);ctx.lineWidth=1.35;ctx.beginPath();
    if(z-a<=4*width){let move=true;for(let i=a;i<z;i++){let v=arr[i];if(!Number.isFinite(v)){move=true;continue;}if(move)ctx.moveTo(xp(i),yp(v));else ctx.lineTo(xp(i),yp(v));move=false;}}
    else{const step=Math.ceil((z-a)/(r-l));for(let i=a;i<z;i+=step){let mn=Infinity,mx=-Infinity;for(let j=i;j<Math.min(i+step,z);j++){if(Number.isFinite(arr[j])){mn=Math.min(mn,arr[j]);mx=Math.max(mx,arr[j]);}}if(Number.isFinite(mn)){ctx.moveTo(xp(i),yp(mn));ctx.lineTo(xp(i),yp(mx)+.4);}}}
    ctx.stroke();ctx.setLineDash([]);}
  ctx.font='11px system-ui';ctx.fillStyle='#4a5c6a';ctx.fillText(hi.toFixed(1),0,top+6);ctx.fillText(lo.toFixed(1),0,bottom);ctx.fillText(s.unit==='mmHg'?'mmHg':'源单位',1,(top+bottom)/2);
  const fs=meta.blocks[selected].fs;for(let i=0;i<=3;i++){const at=a+(z-a)*i/3;ctx.fillText(fmt(at/fs,2)+'s',Math.min(r-42,xp(at)),height-4);}}
function showSample(k,e){if(!group)return;const c=e.target,box=c.getBoundingClientRect(),[a,z]=range();const i=Math.min(z-1,Math.max(a,Math.floor(a+(e.clientX-box.left-48)/(box.width-55)*(z-a))));const s=group[k],fs=meta.blocks[selected].fs;
  $('hover').textContent=k+' · 第 '+i+' 点 · '+fmt(i/fs)+' 秒 · 原始值 '+fmt(s.raw[i],5)+' · 交付值 '+fmt(s.output[i],5)+' · '+(s.stops.some(([l,r])=>l<=i&&i<r)?'任务停用':'可用范围');}
window.addEventListener('resize',draw);
