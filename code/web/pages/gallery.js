/* The Pages edition reads a fixed snapshot and never calls the workbench API. */
(() => {
  const $ = selector => document.querySelector(selector);
  const number = n => n == null ? '—' : n.toLocaleString();
  const state = {viewer: null, selected: null};
  const tabs = {cloud: $('#cloud-tab'), mesh: $('#mesh-tab')};
  const cards = new Map();
  let request = 0;
  window.reconstructionWorkbench = state;

  async function read(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Could not load ${url} (${response.status}).`);
    return response.json();
  }

  function message(text) {
    const el = $('#result-message');
    el.textContent = text;
    el.hidden = !text;
  }

  async function showGeometry(kind) {
    const item = state.selected;
    const current = ++request;
    for (const [name, tab] of Object.entries(tabs)) {
      tab.classList.toggle('active', name === kind);
      tab.setAttribute('aria-pressed', String(name === kind));
    }
    $('#viewer-loading').hidden = false;
    $('#fit-view').disabled = true;
    $('#appearance').disabled = true;
    $('#point-size-control').hidden = true;
    message('');
    try {
      const data = await read(item[kind]);
      if (current !== request) return;
      state.viewer ??= new WorkbenchViewer($('#canvas-host'));
      state.viewer.load(data);
      state.viewer.axes.visible = $('#axes').checked;
      state.viewer.pointSize(Number($('#point-size').value));
      $('#viewer-empty').hidden = true;
      $('#fit-view').disabled = false;
      $('#appearance').hidden = $('#appearance').disabled = kind !== 'mesh';
      $('#appearance').value = 'colored';
      $('#point-size-control').hidden = kind !== 'cloud';
      $('#geometry-label').textContent = kind === 'mesh' ? 'TRIANGLE MESH' : 'POINT CLOUD';
      $('#geometry-count').textContent = kind === 'mesh'
        ? `${number(data.triangles)} triangles · ${number(data.total_points)} vertices`
        : `${number(data.displayed_points)} shown / ${number(data.total_points)} points`;
    } catch (error) {
      if (current === request) message(`3D preview unavailable. ${error.message}`);
    } finally {
      if (current === request) $('#viewer-loading').hidden = true;
    }
  }

  function select(item) {
    state.selected = item;
    for (const [id, button] of cards) button.setAttribute('aria-pressed', String(id === item.id));
    $('#result-title').textContent = item.name;
    $('#result-status').textContent = item.label;
    $('#result-status').dataset.status = item.status;
    $('#geometry-count').textContent = '';
    state.viewer?.clear();
    $('#viewer-empty').hidden = false;
    const metrics = $('#metrics').querySelectorAll('strong');
    [item.metrics.images, item.metrics.registered, item.metrics.points, item.metrics.triangles]
      .forEach((value, i) => { metrics[i].textContent = number(value); });
    const download = document.createElement('a');
    download.href = item.glb;
    download.download = `${item.id}.glb`;
    download.textContent = 'Download GLB';
    $('#downloads').replaceChildren(download);
    $('#downloads').hidden = false;
    for (const [kind, tab] of Object.entries(tabs)) tab.disabled = !item[kind];
    showGeometry(item.mesh ? 'mesh' : 'cloud');
    window.dispatchEvent(new Event('reconstruction-selection'));
  }

  function card(item) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'run-card';
    button.setAttribute('aria-pressed', 'false');
    const preview = document.createElement('div');
    preview.className = 'card-preview';
    const image = document.createElement('img');
    image.src = item.preview;
    image.alt = '';
    image.loading = 'lazy';
    preview.append(image);
    const details = document.createElement('div');
    details.className = 'card-info';
    const name = document.createElement('strong');
    name.textContent = item.name;
    const label = document.createElement('p');
    label.textContent = `${item.label === 'Saved example' ? 'Example · ' : ''}${number(item.metrics.images)} photos`;
    details.append(name, label);
    button.append(preview, details);
    button.addEventListener('click', () => select(item));
    cards.set(item.id, button);
    return button;
  }

  for (const [kind, tab] of Object.entries(tabs)) tab.addEventListener('click', () => showGeometry(kind));
  $('#fit-view').addEventListener('click', () => state.viewer?.fit());
  $('#appearance').addEventListener('change', event => state.viewer?.appearance(event.target.value));
  $('#axes').addEventListener('change', event => { if (state.viewer) state.viewer.axes.visible = event.target.checked; });
  $('#point-size').addEventListener('input', event => {
    const size = Number(event.target.value);
    $('#point-size-value').textContent = `${size.toFixed(1)}×`;
    state.viewer?.pointSize(size);
  });
  read('data/gallery.json').then(items => {
    $('#run-library').replaceChildren(...items.map(card));
    if (items.length) select(items[0]);
  }).catch(error => { $('#run-library').textContent = error.message; });
})();
