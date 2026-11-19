/* Uploads, persistent local jobs and result selection. No external image upload. */
(() => {
  const $=selector=>document.querySelector(selector);
  const format=n=>Number(n||0).toLocaleString();
  const activeStatuses=new Set(['queued','running']);
  const state={files:[],urls:[],jobs:[],examples:[],comparisons:[],selected:null,kind:'cloud',preferredKind:null,loadedKey:null,loadingKey:null,request:0,viewer:null,busy:false};
  const methodNames={ours:'Our SfM',colmap:'COLMAP',all:'Compare both'};
  const statusNames={uploading:'Draft',queued:'Queued',running:'Reconstructing',completed:'Ready',partial:'Partial result',failed:'Failed',cancelled:'Cancelled',interrupted:'Interrupted'};
  const percent=job=>Math.max(0,Math.min(100,Math.round((Number(job.progress)||0)*10)/10));
  const statusLabel=job=>`${statusNames[job.status]||job.status}${activeStatuses.has(job.status)?' · '+percent(job)+'%':''}`;
  async function api(path,options={}){
    const response=await fetch(path,{...options,headers:{'X-Reconstruction-UI':'1',...(options.headers||{})}});
    if(!response.ok){let message;try{message=(await response.json()).error;}catch{}throw Error(message||`Request failed (${response.status})`);}
    return response.json();
  }
  const post=(path,data)=>api(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data||{})});
  function error(message){$('#form-error').textContent=message;$('#form-error').hidden=!message;}
  function setFiles(files){
    if(state.busy)return;
    const list=Array.from(files).sort((a,b)=>a.name.localeCompare(b.name,undefined,{numeric:true}));error('');
    const limit=200;
    if(list.length>limit){error(`Choose at most ${limit} photos for this method on the local runtime.`);$('#start-button').disabled=true;return;}
    const invalid=list.find(file=>! /\.(jpe?g|png|tiff?)$/i.test(file.name)||file.size>25*1024*1024);
    if(invalid){error(`${invalid.name}: use JPEG, PNG or TIFF files up to 25 MB each.`);return;}
    if(list.reduce((sum,file)=>sum+file.size,0)>1024**3){error('Keep the total upload under 1 GB.');return;}
    state.urls.forEach(URL.revokeObjectURL);state.files=list;state.urls=[];
    $('#photo-preview').replaceChildren();
    list.slice(0,14).forEach(file=>{const img=document.createElement('img');const url=URL.createObjectURL(file);state.urls.push(url);img.src=url;img.alt=file.name;img.title=file.name;$('#photo-preview').append(img);});
    if(list.length>14){const more=document.createElement('div');more.className='more';more.textContent='+'+(list.length-14);$('#photo-preview').append(more);}
    $('#selection-info').hidden=!list.length;$('#selection-count').textContent=`${list.length} photos · ${(list.reduce((s,f)=>s+f.size,0)/1024**2).toFixed(1)} MB`;
    $('#start-button').disabled=list.length<3;
    if(list.length&&list.length<3)error('Choose at least three overlapping views. We recommend 12–60 for an object.');
  }
  $('#images').addEventListener('change',e=>setFiles(e.target.files));
  $('#dropzone').addEventListener('click',()=>{if(!state.busy)$('#images').click();});
  $('#dropzone').addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();if(!state.busy)$('#images').click();}});
  ['dragenter','dragover'].forEach(name=>$('#dropzone').addEventListener(name,event=>{event.preventDefault();$('#dropzone').classList.add('dragover');}));
  ['dragleave','drop'].forEach(name=>$('#dropzone').addEventListener(name,event=>{event.preventDefault();$('#dropzone').classList.remove('dragover');if(name==='drop')setFiles(event.dataTransfer.files);}));
  $('#clear-images').addEventListener('click',()=>{setFiles([]);$('#images').value='';});
  $('#upload-form').addEventListener('submit',async event=>{
    event.preventDefault();if(state.busy||state.files.length<3)return;
    state.busy=true;error('');$('#start-button').disabled=true;$('#clear-images').disabled=true;$('#images').disabled=true;
    let job;
    try{
      const focal=$('#focal').value.trim();
      const calibration=['fx','fy','cx','cy'].map(k=>$('#camera-'+k).value.trim());
      if(calibration.some(Boolean)&&!calibration.every(Boolean))throw Error('Supply all four calibration values: fx, fy, cx and cy.');
      if(focal&&calibration.some(Boolean))throw Error('Use either focal length or full calibration, not both.');
      const [fx,fy,cx,cy]=calibration.map(Number);
      job=await post('/api/jobs',{quality_profile:$('#quality-profile').value,name:$('#run-name').value||'Untitled reconstruction',method:$('#reconstruction-method').value,depth_backend:$('#depth-backend').value,matching_ratio:Number($('#matching-ratio').value),camera_matrix:calibration.every(Boolean)?[[fx,0,cx],[0,fy,cy],[0,0,1]]:null,max_side:Number($('#resolution').value),make_mesh:$('#output-type').value==='mesh',mesh_quality:$('#mesh-quality').value,focal_px:focal?Number(focal):null});
      selectJob(job);
      for(let i=0;i<state.files.length;i++){
        const file=state.files[i];$('#upload-status').textContent=`Uploading ${i+1} of ${state.files.length} · ${file.name}`;
        await api(`/api/jobs/${job.id}/images/${i}?name=${encodeURIComponent(file.name)}`,{method:'PUT',body:file,headers:{'Content-Type':'application/octet-stream'}});
      }
      job=await post(`/api/jobs/${job.id}/start`);
      if(job.children){const child=await api(`/api/jobs/${job.children[0]}`);selectJob(child);}else selectJob(job);
      $('#upload-status').textContent=job.children?'Two independent method runs queued. Compare them in Performance → Your comparisons.':'Images saved locally. Reconstruction is running in the background.';await refresh();
    }catch(exc){error(exc.message);$('#upload-status').textContent='';if(job)await post(`/api/jobs/${job.id}/cancel`).catch(()=>{});}
    finally{state.busy=false;$('#start-button').disabled=state.files.length<3;$('#clear-images').disabled=false;$('#images').disabled=false;}
  });
  function meshQuality(){
    const notes={balanced:'Balanced: 640 px depth maps and a standard surface grid. Fastest mesh option.',high:'High: 960 px depth maps and a finer surface grid. More time and memory than Balanced.',ultra:'Ultra: 1280 px depth maps and the finest surface grid. Highest time and memory use.'};
    $('#mesh-quality-field').hidden=$('#output-type').value!=='mesh';
    $('#mesh-quality-help').textContent=notes[$('#mesh-quality').value]+' Uses original uploads; smaller photos are not enlarged.';
  }
  $('#mesh-quality').addEventListener('change',meshQuality);$('#output-type').addEventListener('change',meshQuality);meshQuality();
  $('#reconstruction-method').addEventListener('change',()=>{
    const method=$('#reconstruction-method').value;$('#method-help').textContent=method==='all'?'Creates two independent method runs from identical images, processed one at a time. Up to 200 photos; about 60 sharp, overlapping views are recommended.':method==='colmap'?'COLMAP cameras and sparse points, with optional stereo / TSDF meshing.':'Our incremental SfM on CPU, with optional stereo / TSDF meshing.';
    setFiles(state.files);
    $('#dropzone small').textContent=`JPG, PNG, TIFF · 3–200 photos · 25 MB each`;
  });
  $('#quality-profile').addEventListener('change',()=>{$('#quality-profile-help').textContent=$('#quality-profile').value==='quality'?'Full camera frames for classical methods, 12,000 SIFT features and exhaustive matching. Best suited to calibrated object photos; takes more CPU time.':'Choose about 60 sharp, overlapping views covering every side. Object quality uses alpha or a confident simple-background mask. Balanced meshing limits memory.';});
  function viewer(){
    if(!state.viewer){state.viewer=new WorkbenchViewer($('#canvas-host'));state.viewer.axes.visible=$('#axes').checked;state.viewer.pointSize($('#point-size').value);}
    return state.viewer;
  }
  function message(text,isError=false){$('#result-message').textContent=text;$('#result-message').hidden=!text;$('#result-message').dataset.error=String(isError);}
  function resetSelection(){state.request++;state.loadedKey=null;state.loadingKey=null;state.viewer?.clear();$('#viewer-empty').hidden=false;$('#viewer-loading').hidden=true;$('#geometry-count').textContent='';$('#appearance').disabled=true;$('#point-size-control').hidden=true;$('#fit-view').disabled=true;$('#downloads').replaceChildren();delete $('#downloads').dataset.signature;$('#downloads').hidden=true;$('#run-log').textContent='';$('#retry-controls').hidden=true;}
  function selectJob(job){
    const changed=state.selected?.id!==job.id||state.selected?.type!=='job'||state.selected?.job?.prior_attempts!==job.prior_attempts;
    if(changed){resetSelection();state.kind=state.preferredKind||'cloud';}
    state.selected={type:'job',id:job.id,job};renderJob(job);renderLibrary();
    if(changed)requestAnimationFrame(()=>$('#result-title').scrollIntoView({behavior:'smooth',block:'start'}));
  }
  function renderJob(job){
    $('#result-title').textContent=job.name;$('#result-status').textContent=methodNames[job.method||'ours']+' · '+(job.depth_backend==='mono'?'monocular':'SGBM')+' · '+statusLabel(job);$('#result-status').dataset.status=job.status;
    const running=activeStatuses.has(job.status);$('#progress-area').hidden=!(running||job.status==='uploading');$('#stage-message').textContent=job.message||'';$('#run-progress').value=job.progress||0;
    $('#retry-controls').hidden=running||job.status==='uploading';
    const retryKey=job.id+'-'+(job.prior_attempts||0);
    if($('#retry-controls').dataset.key!==retryKey){
      const method=job.method||'ours';$('#retry-controls').dataset.key=retryKey;$('#retry-ratio').value=job.matching_ratio??.85;$('#retry-ratio').disabled=method!=='ours';
      $('#retry-depth').value=job.depth_backend||'sgbm';$('#retry-quality').value=job.mesh_quality||'balanced';$('#retry-reuse').checked=false;$('#retry-reuse').disabled=!job.make_mesh||!job.artifacts?.poses;
    }
    $('#run-percent').textContent=percent(job)+'%';
    const stages=['preparing','reconstructing','meshing','finished'];let stage=stages.indexOf(job.stage);if(job.stage==='point_cloud')stage=1;if(job.stage==='exporting')stage=2;
    document.querySelectorAll('[data-stage]').forEach((el,i)=>el.classList.toggle('done',i<=stage));
    const m=job.metrics||{};renderMetrics(m.images||job.uploads.length,m.registered,m.dense_points||m.sparse_points,m.triangles);
    const artifacts=job.artifacts||{};const cloud=!!artifacts.cloud_view,mesh=!!artifacts.mesh_view;
    $('#cloud-tab').disabled=!cloud;$('#mesh-tab').disabled=!mesh;
    if(state.kind==='mesh'&&!mesh)state.kind='cloud';
    const failed=['failed','interrupted','cancelled'].includes(job.status);
    message([...(job.warnings||[]),...(failed?[job.message]:[]),...(job.status==='uploading'&&!state.busy?['This upload is incomplete. Cancel this draft and select your photos again.']:[])].join('\n'),failed);
    if(cloud||mesh){const kind=state.kind==='mesh'&&mesh?'mesh':'cloud';const revision=kind==='cloud'?(m.dense_points||m.sparse_points||0):(m.triangles||0);loadGeometry(`/api/jobs/${job.id}/files/${kind}_view`,`${job.id}-${kind}-${revision}`,kind);}
    $('#run-details').hidden=false;
    const photoKey=job.id+'-'+job.uploads.length;
    if($('#run-photos').dataset.key!==photoKey){$('#run-photos').replaceChildren();job.uploads.forEach(item=>{const img=document.createElement('img');img.src=`/api/jobs/${job.id}/images/${item.index}`;img.alt=item.name;img.title=item.name;img.loading='lazy';$('#run-photos').append(img);});$('#run-photos').dataset.key=photoKey;}
    const links=[];
    for(const [key,label] of [['mesh_glb','Mesh GLB'],['mesh_ply','Mesh PLY'],['mesh_obj','Mesh OBJ'],['full_mesh','Full mesh PLY'],['depth_sample_cloud','Depth sample cloud PLY'],['dense_cloud','Fused cloud PLY'],['sparse_cloud','Sparse cloud PLY'],['poses','Camera poses'],['mesh_summary','Mesh metrics'],['foreground_report','Foreground record'],['registration_diagnostics','Registration diagnostics'],['camera_prior','Lens calibration source']])if(artifacts[key])links.push([label,`/api/jobs/${job.id}/files/${key}?download=1`]);
    renderDownloads(links);
    window.dispatchEvent(new Event('reconstruction-selection'));
  }
  function renderMetrics(input,registered,points,triangles){
    [input,registered,points,triangles].forEach((value,index)=>{$('#metrics').children[index].querySelector('strong').textContent=value?format(value):'—';});
  }
  function renderDownloads(links){
    const signature=JSON.stringify(links);if($('#downloads').dataset.signature===signature)return;$('#downloads').dataset.signature=signature;
    $('#downloads').replaceChildren();$('#downloads').hidden=!links.length;if(!links.length)return;
    const label=document.createElement('span');label.textContent='TAKE YOUR RESULTS';$('#downloads').append(label);
    links.forEach(([text,url])=>{const a=document.createElement('a');a.textContent=text+' ↓';a.href=url;$('#downloads').append(a);});
  }
  async function loadGeometry(url,key,kind){
    $('#cloud-tab').classList.toggle('active',kind==='cloud');$('#mesh-tab').classList.toggle('active',kind==='mesh');
    if(state.loadedKey===key||state.loadingKey===key)return;
    const request=++state.request;state.loadingKey=key;$('#viewer-loading').hidden=false;
    try{
      const data=await api(url);if(request!==state.request)return;
      viewer().load(data);state.loadedKey=key;$('#viewer-empty').hidden=true;$('#fit-view').disabled=false;$('#appearance').disabled=kind!=='mesh';$('#appearance').hidden=kind!=='mesh';$('#appearance').value='colored';$('#point-size-control').hidden=kind!=='cloud';
      $('#geometry-label').textContent=kind==='mesh'?'TRIANGLE MESH':'POINT CLOUD';
      $('#geometry-count').textContent=kind==='mesh'?`${format(data.triangles)} triangles · ${format(data.total_points)} vertices`:`${format(data.displayed_points)} shown / ${format(data.total_points)} points`;
      if(state.selected?.type==='example')renderMetrics(60,60,kind==='cloud'?data.total_points:null,kind==='mesh'?data.triangles:null);
    }catch(exc){if(request===state.request)message(exc.message,true);}
    finally{if(request===state.request){state.loadingKey=null;$('#viewer-loading').hidden=true;}}
  }
  function selectExample(example){
    resetSelection();state.selected={type:'example',id:example.id,example};state.kind=state.preferredKind||'mesh';
    $('#result-title').textContent=example.name;$('#result-status').textContent='Example';$('#result-status').dataset.status='completed';$('#progress-area').hidden=true;$('#run-details').hidden=true;$('#cloud-tab').disabled=false;$('#mesh-tab').disabled=false;
    message('Existing reconstruction from 60 selected views. Measured stereo surfaces can have holes; coordinates use arbitrary display units.');renderMetrics(60,60,null,null);
    renderDownloads([['Cloud PLY',`/api/examples/${example.id}/cloud-ply`],['Mesh GLB',`/api/examples/${example.id}/glb`],['Mesh PLY',`/api/examples/${example.id}/ply`],['Mesh OBJ',`/api/examples/${example.id}/obj`]]);
    loadGeometry(`/api/examples/${example.id}/${state.kind}`,`${example.id}-${state.kind}`,state.kind);renderLibrary();
    requestAnimationFrame(()=>$('#result-title').scrollIntoView({behavior:'smooth',block:'start'}));
    window.dispatchEvent(new Event('reconstruction-selection'));
  }
  function changeKind(kind){
    state.kind=kind;state.preferredKind=kind;if(!state.selected)return;
    if(state.selected.type==='job')renderJob(state.selected.job);
    else loadGeometry(`/api/examples/${state.selected.id}/${kind}`,`${state.selected.id}-${kind}`,kind);
  }
  $('#mesh-tab').addEventListener('click',()=>changeKind('mesh'));$('#cloud-tab').addEventListener('click',()=>changeKind('cloud'));
  $('#appearance').addEventListener('change',event=>state.viewer?.appearance(event.target.value));
  function pointSize(value){
    const size=Math.max(.3,Math.min(3,Number(value)||1));$('#point-size').value=size;$('#point-size-value').textContent=size.toFixed(1)+'×';state.viewer?.pointSize(size);
    try{localStorage.setItem('reconstruction-point-size',String(size));}catch{}
  }
  let savedSize=1;try{savedSize=localStorage.getItem('reconstruction-point-size')||1;}catch{}
  pointSize(savedSize);$('#point-size').addEventListener('input',event=>pointSize(event.target.value));
  $('#axes').addEventListener('change',event=>{if(state.viewer)state.viewer.axes.visible=event.target.checked;});$('#fit-view').addEventListener('click',()=>state.viewer?.fit());
  $('#retry-reuse').addEventListener('change',()=>{$('#retry-ratio').disabled=$('#retry-reuse').checked||state.selected?.job?.method!=='ours';if($('#retry-reuse').checked)$('#retry-ratio').value=state.selected?.job?.matching_ratio??.85;});
  $('#cancel-run').addEventListener('click',async()=>{if(state.selected?.type!=='job')return;try{const job=await post(`/api/jobs/${state.selected.id}/cancel`);selectJob(job);await refresh();}catch(exc){message(exc.message,true);}});
  $('#retry-run').addEventListener('click',async()=>{if(state.selected?.type!=='job')return;$('#retry-run').disabled=true;try{const method=state.selected.job.method||'ours',settings={mesh_quality:$('#retry-quality').value,reuse_geometry:$('#retry-reuse').checked};settings.depth_backend=$('#retry-depth').value;if(method==='ours')settings.matching_ratio=Number($('#retry-ratio').value);const job=await post(`/api/jobs/${state.selected.id}/retry`,settings);selectJob(job);await refresh();}catch(exc){message(exc.message,true);}finally{$('#retry-run').disabled=false;}});
  async function log(){if(state.selected?.type!=='job')return;try{const value=await api(`/api/jobs/${state.selected.id}/log`);$('#run-log').textContent=value.text||'The worker log will appear when reconstruction starts.';}catch(exc){$('#run-log').textContent=exc.message;}}
  $('#refresh-log').addEventListener('click',log);$('#run-details').addEventListener('toggle',()=>{if($('#run-details').open)log();});
  function clearSelection(){
    resetSelection();state.selected=null;$('#result-title').textContent='Your next perspective';$('#result-status').textContent='Ready to begin';delete $('#result-status').dataset.status;
    $('#progress-area').hidden=true;$('#run-details').hidden=true;$('#cloud-tab').disabled=true;$('#mesh-tab').disabled=true;renderMetrics();message('');
    window.dispatchEvent(new Event('reconstruction-selection'));
  }
  let pendingDelete=null;
  function askDelete(item){
    pendingDelete=item;$('#delete-error').hidden=true;$('#confirm-delete').disabled=false;
    $('#delete-description').textContent=item.type==='example'?`Remove “${item.name}” from your library? The original image datasets and reconstruction files will remain on disk.`:`Delete “${item.name}”? A running reconstruction will be stopped. Its saved files will be moved to the local trash.`;
    $('#delete-dialog').showModal();
  }
  $('#confirm-delete').addEventListener('click',async()=>{
    if(!pendingDelete)return;const item=pendingDelete;$('#confirm-delete').disabled=true;
    try{
      await api(`/api/${item.type==='example'?'examples':'jobs'}/${item.id}`,{method:'DELETE'});
      if(state.selected?.id===item.id&&state.selected?.type===item.type)clearSelection();
      $('#delete-dialog').close();pendingDelete=null;await refresh();
    }catch(exc){$('#delete-error').textContent=exc.message;$('#delete-error').hidden=false;}
    finally{$('#confirm-delete').disabled=false;}
  });
  function renderLibrary(){
    const host=$('#run-library');host.replaceChildren();
    function card(name,subtitle,status,img,onClick,item,progress){
      const article=document.createElement('article');article.className='run-card';article.dataset.id=item.id;
      const button=document.createElement('button');button.type='button';button.className='card-open';
      const preview=document.createElement('div');preview.className='card-preview';if(img){const image=document.createElement('img');image.src=img;image.alt='';image.loading='lazy';preview.append(image);}else{preview.classList.add('empty');preview.textContent='◈';}
      const info=document.createElement('div');info.className='card-info';const title=document.createElement('strong');title.textContent=name;const line=document.createElement('p');const text=document.createElement('span');text.textContent=subtitle;const badge=document.createElement('b');badge.textContent=status;line.append(text,badge);info.append(title,line);button.append(preview,info);button.addEventListener('click',onClick);article.append(button);
      if(progress!==undefined){const bar=document.createElement('progress');bar.className='card-progress';bar.max=100;bar.value=progress;bar.setAttribute('aria-label',`${name}: ${progress}% complete`);article.append(bar);}
      const actions=document.createElement('div');actions.className='card-actions';const remove=document.createElement('button');remove.type='button';remove.className='delete-button';remove.textContent='Delete';remove.setAttribute('aria-label',`Delete ${name}`);remove.addEventListener('click',()=>askDelete(item));actions.append(remove);article.append(actions);host.append(article);
    }
    state.examples.forEach(example=>card(example.name,'Saved reconstruction','Explore ↗',`/api/examples/${example.id}/preview`,()=>selectExample(example),{type:'example',id:example.id,name:example.name}));
    state.jobs.forEach(job=>card(job.name,`${methodNames[job.method||'ours']} · ${job.uploads.length} photos`,statusLabel(job),job.artifacts?.mesh_view?`/api/jobs/${job.id}/preview`:job.uploads.length?`/api/jobs/${job.id}/images/${job.uploads[0].index}`:null,()=>selectJob(job),{type:'job',id:job.id,name:job.name},activeStatuses.has(job.status)?percent(job):undefined));
    if(!host.children.length){const p=document.createElement('p');p.className='muted';p.textContent='Your reconstructions will be saved here. Upload photos to begin.';host.append(p);}
  }
  async function refresh(){
    const data=await api('/api/jobs');state.jobs=data.jobs;state.examples=data.examples;state.comparisons=data.comparisons||[];renderLibrary();
    window.dispatchEvent(new Event('reconstruction-library'));
    if(state.selected?.type==='job'){const job=state.jobs.find(x=>x.id===state.selected.id);if(job){if(state.selected.job.prior_attempts!==job.prior_attempts)resetSelection();state.selected.job=job;renderJob(job);}else clearSelection();}
    else if(state.selected?.type==='example'&&!state.examples.some(x=>x.id===state.selected.id))clearSelection();
  }
  $('#refresh-runs').addEventListener('click',()=>refresh().catch(exc=>message(exc.message,true)));
  $('#browse-results').addEventListener('click',()=>$('#run-library').scrollIntoView({behavior:'smooth',block:'start'}));
  refresh().catch(exc=>message(exc.message,true));
  setInterval(()=>{if(!document.hidden)refresh().catch(()=>{});},2500);
  // A compact inspection surface for browser validation; contains no photo data.
  window.reconstructionWorkbench=state;
  window.openReconstruction=id=>{const job=state.jobs.find(j=>j.id===id);if(job)selectJob(job);};
})();
