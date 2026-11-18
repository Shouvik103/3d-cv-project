/* Read-only research views. Charts consume saved measurements, never demo numbers. */
(() => {
  const $=s=>document.querySelector(s), $$=s=>Array.from(document.querySelectorAll(s));
  const names={ours:'Our SfM',colmap:'COLMAP',vggt:'VGGT',car_s1:'Car',fountain_s2:'Fountain',temple_gt:'Temple',low_texture_dino:'Low-texture Dino',glossy_car:'Glossy car',sparse_car_5:'Sparse car · 5 views',blurred_temple:'Blurred Temple'};
  const colors={ours:'#247b69',colmap:'#bc842e',vggt:'#7582d5'};
  const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const finite=v=>v!==null&&v!==undefined&&typeof v==='number'&&Number.isFinite(v);
  const num=(v,d=3)=>finite(v)?v.toLocaleString(undefined,{maximumFractionDigits:d}):'—';
  const label=v=>names[v]||String(v??'').replaceAll('_',' ');
  let data=null,experiment='E1',analysisSignature='',analysisRequest=0;
  let comparisonSignature='',comparisonRequest=0;
  const stage=id=>data?.experiments.find(e=>e.id===id);
  const asset=(id,path)=>stage(id)?.assets.find(a=>a.path===`${id}/${path}`);
  const link=(text,url,download=false)=>{const a=document.createElement('a');a.className='secondary';a.textContent=text;a.href=url;if(download)a.href+=(url.includes('?')?'&':'?')+'download=1';else{a.target='_blank';a.rel='noopener';}return a;};
  const fillLinks=(host,items)=>{host.replaceChildren(...items.filter(Boolean));};
  const rowsFor=(id,scene)=>stage(id)?.rows.filter(r=>!scene||(r.scene||r.condition)===scene)||[];
  function wireTabs(selector,choose){
    const buttons=$$(selector);
    buttons.forEach((b,i)=>{b.addEventListener('click',()=>choose(b));b.addEventListener('keydown',e=>{
      let next;if(e.key==='ArrowRight')next=(i+1)%buttons.length;if(e.key==='ArrowLeft')next=(i+buttons.length-1)%buttons.length;if(e.key==='Home')next=0;if(e.key==='End')next=buttons.length-1;
      if(next!==undefined){e.preventDefault();buttons[next].focus();choose(buttons[next]);}
    });});
  }
  function showPage(page){
    if(!['workspace','methods','performance','experiments'].includes(page))page='workspace';
    $$('[data-page]').forEach(b=>{const selected=b.dataset.page===page;b.setAttribute('aria-selected',selected);b.tabIndex=selected?0:-1;$('#'+b.getAttribute('aria-controls')).hidden=!selected;});
    history.replaceState(null,'','#'+page+(page==='experiments'?'/'+experiment:''));
    if(page==='workspace')requestAnimationFrame(()=>window.reconstructionWorkbench?.viewer?.resize());
  }
  wireTabs('[data-page]',b=>showPage(b.dataset.page));
  $$('[data-open-page]').forEach(a=>a.addEventListener('click',e=>{e.preventDefault();showPage(a.dataset.openPage);$('#tab-'+a.dataset.openPage).focus();}));
  wireTabs('[data-performance]',b=>$$('[data-performance]').forEach(button=>{
    const selected=button===b;button.setAttribute('aria-selected',selected);button.tabIndex=selected?0:-1;$('#'+button.getAttribute('aria-controls')).hidden=!selected;
  }));
  const initial=location.hash.slice(1).split('/');if(/^E[1-7]$|^M[45]$/.test(initial[1]||''))experiment=initial[1];showPage(initial[0]);
  window.addEventListener('hashchange',()=>showPage(location.hash.slice(1).split('/')[0]));

  function chart(title,rows,key,{unit='',max=null,value=null,line=false,note=''}={}){
    const card=document.createElement('figure');card.className='panel chart-card';
    const head=document.createElement('figcaption');head.textContent=title;card.append(head);
    const reading=document.createElement('p');reading.className='chart-readout';reading.textContent=note||'Hover or focus a mark to inspect its value.';
    const get=r=>value?value(r):r[key];
    if(!rows.length){const p=document.createElement('p');p.className='muted chart-empty';p.textContent='No methods selected or no recorded values.';card.append(p);return card;}
    const actual=rows.map(get).filter(finite),ceiling=max??Math.max(1,...actual)*1.16;
    const W=600,left=133,right=74,plotW=W-left-right;
    let body='',height;
    const tip=r=>`${label(r.method)}${r.variant?' · '+label(r.variant):''}${r.views?' · '+r.views+' views':''}: ${num(get(r))}${finite(get(r))?unit:''}${!finite(get(r))?' (unavailable)':''}`;
    if(line){
      height=250;const top=20,bottom=199,categories=[...new Set(rows.map(r=>r.views))].sort((a,b)=>a-b);
      const x=n=>left+(categories.indexOf(n)/Math.max(1,categories.length-1))*plotW,y=n=>bottom-(n/ceiling)*(bottom-top);
      for(let i=0;i<=4;i++){const v=ceiling*i/4;body+=`<line class="chart-gridline" x1="${left}" y1="${y(v)}" x2="${W-right}" y2="${y(v)}"/><text class="chart-tick" x="${left-12}" y="${y(v)+4}" text-anchor="end">${esc(num(v,2))}</text>`;}
      categories.forEach(n=>{body+=`<text class="chart-tick" x="${x(n)}" y="222" text-anchor="middle">${n} views</text>`;});
      [...new Set(rows.map(r=>r.method))].forEach((method,i)=>{
        const series=rows.filter(r=>r.method===method).sort((a,b)=>a.views-b.views);let points=[];
        const flush=()=>{if(points.length>1)body+=`<polyline fill="none" stroke="${colors[method]}" stroke-width="2.5" points="${points.join(' ')}"/>`;points=[];};
        series.forEach(r=>{if(finite(get(r)))points.push(`${x(r.views)},${y(get(r))}`);else flush();});flush();
        series.filter(r=>finite(get(r))).forEach(r=>{body+=`<circle tabindex="0" role="img" aria-label="${esc(tip(r))}" data-tip="${esc(tip(r))}" cx="${x(r.views)}" cy="${y(get(r))}" r="5" fill="${colors[method]}"><title>${esc(tip(r))}</title></circle>`;});
        body+=`<circle cx="${left+i*135}" cy="244" r="4" fill="${colors[method]}"/><text class="chart-tick" x="${left+10+i*135}" y="248">${esc(label(method))}</text>`;
      });
    }else{
      height=Math.max(185,rows.length*44+55);
      for(let i=0;i<=4;i++){const x=left+plotW*i/4;body+=`<line class="chart-gridline" x1="${x}" y1="12" x2="${x}" y2="${height-35}"/><text class="chart-tick" x="${x}" y="${height-12}" text-anchor="middle">${esc(num(ceiling*i/4,2))}</text>`;}
      rows.forEach((r,i)=>{
        const v=get(r),y=25+i*44,caption=r.variant?label(r.variant):label(r.method),color=colors[r.method]||colors.ours;
        body+=`<text class="chart-label" x="${left-12}" y="${y+14}" text-anchor="end">${esc(caption)}</text>`;
        if(finite(v))body+=`<g tabindex="0" role="img" aria-label="${esc(tip(r))}" data-tip="${esc(tip(r))}"><rect x="${left}" y="${y}" width="${Math.max(0,v/ceiling*plotW)}" height="22" rx="4" fill="${color}"/>${v===0?`<circle cx="${left+3}" cy="${y+11}" r="3" fill="${color}"/>`:''}<text class="chart-value" x="${left+Math.max(0,v/ceiling*plotW)+9}" y="${y+15}">${esc(num(v))}${esc(unit)}</text><title>${esc(tip(r))}</title></g>`;
        else body+=`<text class="chart-tick" x="${left+8}" y="${y+14}">Unavailable</text>`;
      });
    }
    const host=document.createElement('div');host.innerHTML=`<svg viewBox="0 0 ${W} ${height}" role="img" aria-label="${esc(title)}"><title>${esc(title)}</title>${body}</svg>`;
    host.querySelectorAll('[data-tip]').forEach(mark=>{['pointerenter','focus'].forEach(event=>mark.addEventListener(event,()=>reading.textContent=mark.dataset.tip));});
    card.append(host,reading);return card;
  }
  const headings={scene:'Scene',condition:'Condition',method:'Method',variant:'Variant',status:'Run status',registered:'Cameras',total:'Input views',views:'Views',reference:'Reference',ate_percent_extent:'ATE % ↓',auc5:'AUC@5 ↑',auc10:'AUC@10 ↑',auc30:'AUC@30 ↑',median_rotation_deg:'Rotation error ° ↓',median_translation_deg:'Translation error ° ↓',median_reprojection_px:'Reproj. px ↓',reprojection_px:'Reproj. px ↓',points:'Exported points',mean_track_length:'Track length',cloud_chamfer_agreement:'Cloud agreement distance ↓',cloud_fscore_agreement:'Cloud agreement F ↑',seconds:'Seconds',triangles:'Mesh triangles',reprojection_statistic:'Residual statistic',depth_backend:'Depth backend',prior_attempts:'Prior attempts',peak_rss_mb:'Peak RSS MB',error:'Failure reason',reused:'Reused run',initial_fx:'Initial fx',final_fx:'Final fx',fused_views:'Fused views',faces:'Full faces',accuracy_percent_diagonal:'Accuracy % ↓',completeness_percent_diagonal:'Completeness % ↓',f005:'F@0.5% ↑',f01:'F@1% ↑',f02:'F@2% ↑',coverage:'Coverage',psnr_db:'PSNR dB ↑',ssim:'SSIM ↑'};
  function table(host,rows,columns=null){
    host.replaceChildren();if(!rows.length){host.textContent='No recorded rows for this selection.';return;}
    columns=columns||Object.keys(headings).filter(key=>rows.some(r=>key in r));
    const el=document.createElement('table'),caption=document.createElement('caption');caption.className='sr-only';caption.textContent='Recorded measurements. Click column headers to sort; missing values remain unavailable.';el.append(caption);
    const thead=document.createElement('thead'),tr=document.createElement('tr'),tbody=document.createElement('tbody');
    let sorted=rows.slice(),direction=1,last='';
    function draw(){tbody.replaceChildren();sorted.forEach(row=>{const line=document.createElement('tr');if(row.status==='failed')line.className='failed-row';columns.forEach(key=>{
      const cell=document.createElement('td');let value=row[key];
      if(key==='registered')value=finite(row.registered)?`${row.registered}/${row.total??'?'}`:'—';
      else if(key==='status'){value=row.status==='completed'&&finite(row.total)&&row.registered<row.total?'Completed · partial cameras':label(value);cell.className='status-cell';}
      else if(key==='reference')value=value==='gt'?'Ground truth':value==='colmap'?'COLMAP agreement':value;
      else if(typeof value==='number')value=num(value);
      else if(typeof value==='boolean')value=value?'Yes':'No';
      else if(['method','scene','condition','variant'].includes(key))value=label(value);
      cell.textContent=value??'—';line.append(cell);
    });tbody.append(line);});}
    columns.forEach(key=>{const th=document.createElement('th'),button=document.createElement('button');button.textContent=headings[key]||key;button.type='button';button.title='Sort by '+button.textContent;button.addEventListener('click',()=>{
      direction=last===key?-direction:1;last=key;thead.querySelectorAll('th').forEach(h=>h.removeAttribute('aria-sort'));th.setAttribute('aria-sort',direction===1?'ascending':'descending');
      sorted=rows.slice().sort((a,b)=>{if(a[key]==null)return b[key]==null?0:1;if(b[key]==null)return -1;return direction*(typeof a[key]==='number'?a[key]-b[key]:String(a[key]).localeCompare(String(b[key])));});draw();
    });th.append(button);tr.append(th);});thead.append(tr);el.append(thead,tbody);host.append(el);draw();
  }
  function csv(id,selector){const a=asset(id,'results.csv'),button=$(selector);if(a){button.href=a.url+'?download=1';button.hidden=false;}else button.hidden=true;}
  function metricCharts(host,rows,line=false){host.replaceChildren(
    chart('Cameras reconstructed / predicted (%) ↑',rows,'registration',{max:100,unit:'%',line,value:r=>finite(r.registered)&&r.total?r.registered/r.total*100:null,note:'Read registration with pose accuracy; VGGT cameras are predictions.'}),
    chart('Aligned camera ATE (% trajectory extent) ↓',rows,'ate_percent_extent',{unit:'%',line,note:'Only shared cameras; undefined Sim(3) alignment stays unavailable.'}),
    chart('Relative-pose AUC@5 ↑',rows,'auc5',{max:1,line,note:'Range 0–1. Missing camera pairs count as failures.'}),
    chart('End-to-end method time (seconds)',rows,'seconds',{unit:'s',line,note:'Includes loading and export. Available caches and hardware differ.'})
  );}
  function comparison(){
    const scene=$('#comparison-scene').value,selected=new Set($$('#method-filters input:checked').map(x=>x.value));
    const rows=rowsFor('E1',scene).filter(r=>selected.has(r.method));
    $('#comparison-reference').textContent=scene==='temple_gt'?'Temple · supplied GT cameras. All methods received the same 16 images. ATE alone can favor a method that reconstructs fewer views.':`${label(scene)} · agreement with an existing full-data COLMAP pseudo-reference. These camera errors are not independent ground-truth accuracy.`;
    metricCharts($('#comparison-charts'),rows);table($('#comparison-table'),rows,['method','status','registered','ate_percent_extent','auc5','median_reprojection_px','points','seconds','peak_rss_mb']);csv('E1','#comparison-csv');
  }
  function ablations(reset=false){
    const selected=stage($('#ablation-study').value);if(!selected)return;
    const choices=[...new Set(selected.rows.map(r=>r.scene||r.condition))],pick=$('#ablation-scene');
    if(reset){pick.replaceChildren(...choices.map(v=>new Option(label(v),v)));}
    const rows=rowsFor(selected.id,pick.value);$('#ablation-description').textContent=selected.description+' '+(rows[0]?.reference==='colmap'?'Reference: COLMAP agreement, not GT.':'Reference: supplied GT cameras.');
    metricCharts($('#ablation-charts'),rows,selected.id==='E3');table($('#ablation-table'),rows);csv(selected.id,'#ablation-csv');
  }
  function meshes(){
    const rows=stage('M5')?.rows||[];
    $('#mesh-charts').replaceChildren(chart('Surface accuracy (% GT diagonal) ↓',rows,'accuracy_percent_diagonal',{unit:'%'}),chart('Surface completeness (% GT diagonal) ↓',rows,'completeness_percent_diagonal',{unit:'%'}),chart('F-score at 1% of GT diagonal ↑',rows,'f01',{max:1}),chart('Fusion-held-out render coverage (%)',rows,'coverage',{max:100,unit:'%',value:r=>finite(r.coverage)?100*r.coverage:null,note:'Not an unseen-image test. Read beside masked PSNR and SSIM.'}));
    table($('#mesh-table'),rows,['method','status','fused_views','faces','accuracy_percent_diagonal','completeness_percent_diagonal','f005','f01','f02','coverage','psnr_db','ssim','seconds']);
    $('#mesh-comparison').src=asset('M5','side_by_side.png')?.url||'';
    fillLinks($('#mesh-links'),[['ours','M4'],['colmap','M5'],['vggt','M5']].map(([m,s])=>{const a=asset(s,`${m}/interactive_mesh.html`);return a&&link(label(m)+' · interactive mesh ↗',a.url);}).concat([asset('M5','results.csv')&&link('Mesh measurements CSV ↓',asset('M5','results.csv').url,true)]));
  }
  function archive(id){
    const selected=stage(id);if(!selected)return;experiment=id;
    $$('#experiment-nav button').forEach(b=>{const active=b.dataset.experiment===id;b.setAttribute('aria-current',active?'true':'false');});
    $('#experiment-status').textContent=id+' · '+selected.status;$('#experiment-title').textContent=selected.title;$('#experiment-description').textContent=selected.description;
    table($('#experiment-table'),selected.rows);
    fillLinks($('#experiment-downloads'),['csv','json'].map(ext=>{const a=asset(id,'results.'+ext);return a&&link(ext.toUpperCase()+' ↓',a.url,true);}));
    const gallery=$('#experiment-plots');gallery.replaceChildren();
    selected.assets.filter(a=>a.path.endsWith('.png')).forEach(a=>{
      const figure=document.createElement('figure');figure.className='panel figure-card';const anchor=document.createElement('a');anchor.href=a.url;anchor.target='_blank';anchor.rel='noopener';
      const img=document.createElement('img');img.src=a.url;img.loading='lazy';img.alt=label(a.path.replace(id+'/','').replace('.png',''));anchor.append(img);
      const caption=document.createElement('figcaption');caption.textContent=img.alt+' · Open original ↗';const pdf=selected.assets.find(p=>p.path===a.path.replace('.png','.pdf'));if(pdf)caption.append(' ',link('PDF ↓',pdf.url,true));figure.append(anchor,caption);gallery.append(figure);
    });
    fillLinks($('#experiment-records'),selected.assets.filter(a=>/\.(json|md|yaml|html)$/.test(a.path)).map(a=>link(a.path.slice(id.length+1),a.url,!a.path.endsWith('.html'))));
    if(!$('#experiments-panel').hidden)history.replaceState(null,'','#experiments/'+id);
  }
  function methodCards(){
    $('#method-cards').replaceChildren(...data.methods.map(method=>{
      const card=document.createElement('article');card.className='panel lab-card method-card';card.dataset.method=method.id;
      const pill=document.createElement('span');pill.className='method-kicker';pill.textContent=method.label;
      const title=document.createElement('h3');title.textContent=method.name;const device=document.createElement('span');device.className='device-label';device.textContent=method.device;
      const description=document.createElement('p');description.textContent=method.description;
      const steps=document.createElement('ol');steps.className='method-steps';method.steps.forEach(text=>{const li=document.createElement('li');li.textContent=text;steps.append(li);});
      const details=document.createElement('details');details.open=false;const summary=document.createElement('summary');summary.textContent='Outputs, calibration & limitations';details.append(summary);
      for(const [key,heading] of [['outputs','Outputs'],['calibration','Camera calibration'],['mesh','Dense reconstruction'],['limits','Limits'],['license','License']]){const h=document.createElement('h4'),p=document.createElement('p');h.textContent=heading;p.textContent=method[key];details.append(h,p);}
      details.append(link('Source & documentation ↗',method.source));card.append(pill,title,device,description,steps,details);return card;
    }));
    const runtime=data.vggt||{},facts=[['Classical runtime','CPU · pycolmap 4.2.1'],['Learned runtime',runtime.device?.toUpperCase()||'Not recorded'],['VGGT preprocessing',runtime.preprocessing||'Not recorded'],['VGGT precision',runtime.precision||'Not recorded'],['Model revision',runtime.model_revision?.slice(0,12)||'Not recorded'],['Recorded validation',data.verification?.tests?.passed?data.verification.tests.passed+' tests passed':'Not recorded']];
    $('#runtime-facts').replaceChildren(...facts.map(([heading,value])=>{const div=document.createElement('div'),dt=document.createElement('span'),dd=document.createElement('strong');dt.textContent=heading;dd.textContent=value;div.append(dt,dd);return div;}));
  }
  async function loadResearch(){
    try{
      const response=await fetch('/api/research');if(!response.ok)throw Error('Saved experiment data could not be loaded.');data=await response.json();methodCards();
      const guide=$('#metric-glossary');guide.replaceChildren();data.glossary.forEach(([term,meaning])=>{const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=term;dd.textContent=meaning;guide.append(dt,dd);});
      if(!data.available)throw Error('Saved experiment evidence is not installed on this copy. Uploads and the 3D workspace are available; add validation/experiments_* to view the recorded research.');
      $('#benchmark-date').textContent='Recorded · '+data.date;
      const count=data.experiments.reduce((sum,e)=>sum+e.rows.length,0),summary=[['03','Methods compared'],[String(data.experiments.length).padStart(2,'0'),'Experiment studies'],[String(count),'Recorded conditions¹'],['GT + agreement','Reference types']];
      $('#benchmark-summary').replaceChildren(...summary.map(([value,text])=>{const div=document.createElement('div'),n=document.createElement('strong'),p=document.createElement('span');n.textContent=value;p.textContent=text;div.append(n,p);return div;}));
      const note=document.createElement('p');note.className='summary-footnote';note.textContent='¹ Includes reused conditions and failed attempts; method failures are retained.';$('#benchmark-summary').after(note);
      $('#benchmark-notes').replaceChildren(...data.notes.map(text=>{const li=document.createElement('li');li.textContent=text;return li;}));
      $('#ablation-study').replaceChildren(...data.experiments.filter(e=>/^E[2-7]$/.test(e.id)).map(e=>new Option(`${e.id} · ${e.title}`,e.id)));
      $('#experiment-nav').replaceChildren(...data.experiments.map(e=>{const button=document.createElement('button');button.type='button';button.dataset.experiment=e.id;const number=document.createElement('b'),span=document.createElement('span');number.textContent=e.id;span.textContent=e.title;button.append(number,span);button.addEventListener('click',()=>archive(e.id));return button;}));
      comparison();ablations(true);meshes();archive(experiment);
    }catch(error){const status=$('#research-status');status.hidden=false;status.replaceChildren();const message=document.createElement('p'),retry=document.createElement('button');message.textContent=error.message;retry.className='secondary';retry.textContent='Retry loading';retry.addEventListener('click',()=>{status.hidden=true;loadResearch();});status.append(message,retry);}
  }
  $('#comparison-scene').addEventListener('change',comparison);$('#method-filters').addEventListener('change',comparison);$('#ablation-study').addEventListener('change',()=>ablations(true));$('#ablation-scene').addEventListener('change',()=>ablations());

  async function uploadComparison(force=false){
    const groups=window.reconstructionWorkbench?.comparisons||[],select=$('#upload-comparison'),previous=select.value;
    const ids=groups.map(g=>g.id).join(',');
    if(select.dataset.ids!==ids){select.replaceChildren(...groups.map(g=>new Option(`${g.name} · ${g.children.length} method runs`,g.id)));select.dataset.ids=ids;if(groups.some(g=>g.id===previous))select.value=previous;}
    if(!groups.length){$('#upload-comparison-note').textContent='Choose All three in Workspace to reconstruct the same images with all methods.';return;}
    const group=groups.find(g=>g.id===select.value);if(!group)return;
    const signature=JSON.stringify([group.id,group.children.map(j=>[j.id,j.status,j.progress,j.metrics])]);
    if(signature===comparisonSignature&&!force)return;comparisonSignature=signature;const request=++comparisonRequest;
    try{
      const response=await fetch(`/api/comparisons/${group.id}`);if(!response.ok)throw Error('Comparison data is not available yet.');const report=await response.json();if(request!==comparisonRequest)return;
      $('#upload-comparison-note').textContent=report.scope+' '+report.reference_scope+' '+report.timing_scope+(report.missing_children?' Some method runs have been deleted.':'');
      $('#upload-method-runs').replaceChildren(...report.rows.map(row=>{const card=document.createElement('article');card.className='panel lab-card method-card';card.dataset.method=row.method;const title=document.createElement('h3'),status=document.createElement('p'),button=document.createElement('button');title.textContent=label(row.method);status.textContent=`${row.status} · ${num(row.progress,1)}% · ${num(row.registered,0)}/${row.total} cameras`;button.type='button';button.className='secondary';button.textContent=row.cloud_url||row.mesh_url?'Open interactive result ↗':'Open run / log ↗';button.addEventListener('click',()=>{window.openReconstruction(row.id);showPage('workspace');});if(row.mesh_url){const img=document.createElement('img');img.className='result-thumb';img.src=`/api/jobs/${row.id}/preview`;img.alt=label(row.method)+' mesh, fitted independently';img.loading='lazy';card.append(img);}card.append(title,status,button);if(row.error){const p=document.createElement('p');p.textContent=row.error;card.append(p);}return card;}));
      $('#upload-comparison-charts').replaceChildren(
        chart('Registered / predicted cameras (%) ↑',report.rows,'registered',{unit:'%',max:100,value:r=>finite(r.registered)?100*r.registered/r.total:null}),
        chart('Total reconstruction job time (seconds)',report.rows,'seconds',{unit:'s',note:'Includes preparation, camera/depth inference, meshing and export.'}),
        chart('Camera ATE (%) · COLMAP agreement, not GT',report.rows,'ate_percent_extent',{unit:'%',note:'COLMAP is the reference. Undefined alignment stays unavailable.'}),
        chart('Generated mesh triangles',report.rows,'triangles',{note:'Different depth sources and incomplete surfaces affect this count.'})
      );
      table($('#upload-comparison-table'),report.rows,['method','status','registered','points','triangles','depth_backend','reprojection_px','reprojection_statistic','ate_percent_extent','auc5','seconds','peak_rss_mb','prior_attempts','error']);
      fillLinks($('#upload-comparison-downloads'),['json','csv','png'].map(ext=>link(ext.toUpperCase()+' comparison ↓',`/api/comparisons/${group.id}/files/comparison.${ext}`,true)));
    }catch(error){if(request===comparisonRequest){$('#upload-comparison-note').textContent=error.message;comparisonSignature='';}}
  }
  $('#upload-comparison').addEventListener('change',()=>uploadComparison(true));$('#refresh-comparison').addEventListener('click',()=>uploadComparison(true));
  window.addEventListener('reconstruction-library',()=>{if(!$('#uploads-panel').hidden)uploadComparison();});
  $('#subtab-uploads').addEventListener('click',()=>uploadComparison(true));

  async function inspect(force=false){
    const selected=window.reconstructionWorkbench?.selected,section=$('#run-analysis');
    if(!selected){section.hidden=true;analysisSignature='';analysisRequest++;return;}
    const signature=JSON.stringify([selected.type,selected.id,selected.job?.status,selected.job?.metrics,selected.job?.prior_attempts]);
    if(signature===analysisSignature&&!force)return;analysisSignature=signature;const request=++analysisRequest;section.hidden=false;
    const host=$('#analysis-content');host.textContent='Loading recorded run measurements…';
    try{
      const response=await fetch(`/api/${selected.type==='job'?'jobs':'examples'}/${selected.id}/analysis`);if(!response.ok)throw Error('Details are not available yet.');const result=await response.json();if(request!==analysisRequest)return;
      host.replaceChildren();const scope=document.createElement('p');scope.className='scope-note';scope.textContent=`${result.method}${result.mesh_backend?' + '+result.mesh_backend.toUpperCase()+' / TSDF':''} · ${result.reference}${result.prior_attempts?' '+result.prior_attempts+' previous attempt(s) retained locally.':''}`;host.append(scope);
      const sfm=result.reconstruction||{},mesh=result.mesh||{},stats=mesh.full||{},median=finite(sfm.median_reprojection_px),residual=median?sfm.median_reprojection_px:sfm.mean_reprojection_px;
      const facts=[['Reprojection · '+(median?'median':'mean'),finite(residual)?num(residual)+' px':null],['Mean track length',sfm.mean_track_length],['SfM time',finite(sfm.runtime_seconds)?num(sfm.runtime_seconds,1)+' s':null],['Mesh time',finite(mesh.timing_seconds?.total)?num(mesh.timing_seconds.total,1)+' s':null],['Mesh components',stats.components],['Boundary edges',stats.boundary_edges],['Watertight',stats.watertight===true?'Yes':stats.watertight===false?'No':null],['Total time',finite(result.elapsed_seconds)?num(result.elapsed_seconds,1)+' s':null]];
      const grid=document.createElement('div');grid.className='fact-grid';facts.forEach(([key,value])=>{const div=document.createElement('div'),span=document.createElement('span'),strong=document.createElement('strong');span.textContent=key;strong.textContent=value==null?'—':String(value);div.append(span,strong);grid.append(div);});host.append(grid);
      if(result.history?.length){const indices=[...new Set(Array.from({length:Math.min(12,result.history.length)},(_,i)=>Math.round(i*(result.history.length-1)/Math.max(1,Math.min(12,result.history.length)-1))))];const rows=indices.map(i=>({...result.history[i],method:'ours',variant:'Step '+(i+1)}));host.append(chart('Reconstruction growth · registered cameras',rows,'registered',{note:'Up to 12 recorded snapshots shown; all snapshots remain in raw measurements.'}));}
      if(mesh.timing_seconds){const rows=Object.entries(mesh.timing_seconds).filter(([key])=>key!=='total').map(([variant,seconds])=>({method:'ours',variant,seconds}));host.append(chart('Mesh stage timings (seconds)',rows,'seconds',{unit:'s'}));}
      const details=document.createElement('details'),summary=document.createElement('summary'),pre=document.createElement('pre');summary.textContent='Configuration, versions & raw measurements';pre.textContent=JSON.stringify(result,null,2);details.append(summary,pre);host.append(details);
    }catch(error){if(request===analysisRequest){host.textContent=error.message;analysisSignature='';}}
  }
  $('#refresh-analysis').addEventListener('click',()=>inspect(true));window.addEventListener('reconstruction-selection',()=>inspect());
  loadResearch();inspect();
})();
