/* Presentation only: reported progress is the only source of percentage updates. */
(() => {
  const $ = selector => document.querySelector(selector);
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const loops = [], finite = new Map();
  const visible = el => !document.hidden && !reduced.matches && !!el?.getClientRects().length;
  const animate = (target, options) => window.anime?.animate(target, options);
  const stagger = (amount, options) => window.anime?.stagger(amount, options) || 0;
  function entrance(el, options = {}) {
    if (!visible(el)) return;
    finite.get(el)?.cancel();
    const animation = animate(el, {opacity: [0, 1], y: [10, 0], duration: 360, ease: 'out(3)', ...options,
      onComplete: () => finite.delete(el)});
    if (animation) finite.set(el, animation);
  }
  function loop(target, options, host = target) {
    const animation = animate(target, {...options, autoplay: false, loop: true});
    if (animation) loops.push({animation, host, inView: true});
  }
  function sync() {
    loops.forEach(({animation, host, inView}) => {
      if (visible(host) && inView && host.dataset.motionActive !== 'false') animation.resume();
      else animation.pause();
    });
    if (document.hidden || reduced.matches || !$('#workspace-panel').getClientRects().length) settleProgress();
    if (reduced.matches) {finite.forEach(animation => animation.complete()); finite.clear();}
  }
  const area = $('#progress-area'), bar = $('#run-progress'), output = $('#run-percent');
  output.setAttribute('aria-live', 'off');
  const processing = document.createElement('div');
  processing.className = 'processing-scene'; processing.setAttribute('aria-hidden', 'true');
  processing.innerHTML = `<div class="processing-signal">${'<i></i>'.repeat(18)}</div>`;
  $('.progress-caption').insertBefore(processing, $('.progress-actions'));
  const track = document.createElement('div'), fill = document.createElement('div'), sheen = document.createElement('i');
  track.className = 'motion-track'; track.setAttribute('aria-hidden', 'true');
  fill.className = 'motion-fill'; sheen.className = 'motion-sheen';
  fill.append(sheen); track.append(fill); bar.after(track); area.classList.add('has-motion-track');
  const runner = document.createElement('i'); runner.className = 'motion-runner'; fill.append(runner);
  loop(processing.querySelectorAll('.processing-signal i'), {scaleY: [.15, 1], opacity: [.35, .95], duration: 600,
    delay: stagger(70, {from: 'center'}), alternate: true, ease: 'inOutSine'}, area);
  loop(runner, {left: ['0%', '100%'], scale: [.6, 1.3, .6], duration: 2400, ease: 'inOut(3)'}, area);
  const stagePulse = {opacity: 1};
  loop(stagePulse, {opacity: [1, .45], alternate: true, duration: 1100, ease: 'inOutSine',
    onUpdate: () => {const active = $('.stage-list .active'); if (active) active.style.setProperty('--stage-pulse', stagePulse.opacity);}}, area);
  loop(sheen, {translateX: ['-100%', '100%'], duration: 1300, ease: 'linear'}, area);
  loop($('.empty-orbits'), {y: [-4, 4], duration: 4200, alternate: true, ease: 'inOutSine'}, $('#viewer-empty'));
  loop($('.empty-cube'), {rotateX: -22, rotateY: [0, 360], duration: 24000, ease: 'linear'}, $('#viewer-empty'));
  loop($('.empty-orbits').querySelectorAll(':scope>i>b'), {rotate: [0, 360], duration: 16000, delay: stagger(1500), ease: 'linear'}, $('#viewer-empty'));
  loop($('.loading-wireframe'), {rotate: [-15, 345], duration: 2800, ease: 'linear'}, $('#viewer-loading'));
  document.querySelectorAll('.logo-ring>b').forEach((ring, i) => loop(ring, {rotate: [i * 120, i * 120 + 360], duration: 18000 + i * 5000, ease: 'linear'}, $('.brand')));
  loop($('.logo-core'), {rotate: [0, -180], duration: 24000, ease: 'linear'}, $('.brand'));
  loop($('.brand-mark'), {'--brand-halo': [.06, .22], duration: 2200, alternate: true, ease: 'inOutSine'}, $('.brand'));
  loop($('.local-badge i'), {opacity: [.45, 1], scale: [.85, 1.15], alternate: true, duration: 1500, ease: 'inOutSine'});
  const heroMotion = document.createElement('div'); heroMotion.className = 'hero-motion'; heroMotion.setAttribute('aria-hidden', 'true');
  const svgNS = 'http://www.w3.org/2000/svg';
  const wave = document.createElementNS(svgNS, 'svg');
  wave.setAttribute('viewBox', '0 0 400 100'); wave.setAttribute('preserveAspectRatio', 'none');
  const waveY = (t, lane) => 50 + Math.sin(t * Math.PI * 2) * (lane ? -20 : 20);
  for (let lane = 0; lane < 2; lane++) {
    const path = document.createElementNS(svgNS, 'path');
    path.setAttribute('d', Array.from({length: 101}, (_, i) => `${i ? 'L' : 'M'}${i * 4},${waveY(i / 100, lane)}`).join(' '));
    path.setAttribute('class', 'hero-wave-path'); wave.append(path);
  }
  const particles = Array.from({length: 4}, (_, i) => {
    const dot = document.createElementNS(svgNS, 'circle'); dot.setAttribute('r', i % 2 ? '1.8' : '2.5');
    dot.setAttribute('class', 'hero-wave-dot'); wave.append(dot); return dot;
  });
  const waveState = {phase: 0};
  function drawWave() {
    particles.forEach((dot, i) => {
      const t = (waveState.phase + i / particles.length) % 1;
      dot.setAttribute('cx', String(t * 400)); dot.setAttribute('cy', String(waveY(t, i % 2)));
      dot.setAttribute('opacity', String(.12 + Math.sin(t * Math.PI) * .7));
    });
  }
  heroMotion.append(wave); $('.intro').append(heroMotion); drawWave();
  loop(waveState, {phase: [0, 1], duration: 11000, ease: 'linear', onUpdate: drawWave}, heroMotion);
  const workspacePanel = $('.result-panel');
  workspacePanel.dataset.motionActive = 'false';
  loop(workspacePanel, {'--edge-angle': ['0deg', '360deg'], duration: 9500, ease: 'linear'}, workspacePanel);
  let key = null, lastStatus = null, target = 0, value = {percent: 0}, progressAnimation = null, completedKey = null;
  function paint() {fill.style.width = value.percent + '%'; output.textContent = Math.round(value.percent * 10) / 10 + '%';}
  function settleProgress() {progressAnimation?.cancel(); progressAnimation = null; value.percent = target; paint();}
  function resetProgress() {
    settleProgress(); key = null; completedKey = null; lastStatus = null; target = 0;
    value.percent = 0; paint(); area.dataset.motionActive = 'false'; workspacePanel.dataset.motionActive = 'false'; sync();
  }
  function progress(job, stage) {
    const nextKey = job.id + ':' + (job.prior_attempts || 0), changed = key !== nextKey;
    const running = ['queued', 'running', 'uploading'].includes(job.status);
    const justCompleted = !changed && ['queued', 'running'].includes(lastStatus) && job.status === 'completed';
    if (changed) {settleProgress(); key = nextKey; completedKey = null;}
    if (justCompleted) completedKey = key;
    const previousTarget = target;
    target = job.status === 'completed' ? 100 : Math.max(0, Math.min(100, Number(job.progress) || 0));
    bar.value = target; bar.setAttribute('aria-valuetext', target + '% · ' + (job.message || job.status));
    area.dataset.motionStatus = job.status; area.dataset.motionActive = String(running);
    workspacePanel.dataset.motionActive = String(['queued', 'running'].includes(job.status));
    if (completedKey === key && job.status === 'completed') area.hidden = false;
    if ((changed && running) || (!running && justCompleted)) entrance(processing, {opacity: [0, 1], y: [12, 0], duration: 650});
    document.querySelectorAll('[data-stage]').forEach((el, i) => {
      el.classList.toggle('done', job.status === 'completed' || (stage >= 0 && i < stage));
      el.classList.toggle('active', running && i === Math.max(0, stage));
      el.style.removeProperty('--stage-pulse');
      if (running && i === Math.max(0, stage)) el.setAttribute('aria-current', 'step'); else el.removeAttribute('aria-current');
    });
    if (changed || (!running && !justCompleted) || !visible(area) || !window.anime) settleProgress();
    else if (previousTarget !== target) {
      progressAnimation?.cancel();
      progressAnimation = animate(value, {percent: target, duration: justCompleted ? 650 : 800, ease: 'out(3)', onUpdate: paint,
      onComplete: () => {paint(); if (justCompleted) entrance(track, {opacity: [1, .4, 1], scaleX: [.96, 1], y: 0, duration: 700});}});
    }
    lastStatus = job.status; sync();
  }
  window.workbenchMotion = {progress, resetProgress};
  const observer = new MutationObserver(records => {
    records.forEach(({target: el, oldValue}) => {
      if (oldValue !== null && !el.hidden && el.matches('[role=tabpanel]')) {
        entrance(el, {y: [22, 0], duration: 600});
        const parts = [...el.querySelectorAll(':scope>.panel, :scope>.summary-grid, :scope>.page-heading, :scope>.chart-grid>.chart-card')];
        parts.forEach((part, i) => entrance(part, {y: [24, 0], delay: i * 65, duration: 650}));
      }
    });
    sync();
  });
  observer.observe(document.body, {subtree: true, attributes: true, attributeOldValue: true, attributeFilter: ['hidden']});
  document.addEventListener('toggle', event => {
    const details = event.target;
    if (details.tagName === 'DETAILS' && details.open) [...details.children].filter(el => el.tagName !== 'SUMMARY').forEach((el, i) => entrance(el, {y: [-14, 0], delay: i * 55, duration: 450}));
  }, true);
  document.addEventListener('visibilitychange', sync); reduced.addEventListener('change', sync);
  // Decorative page atmosphere, isolated from the viewer and all interaction.
  const atmosphere = document.createElement('div');
  atmosphere.className = 'page-atmosphere'; atmosphere.setAttribute('aria-hidden', 'true'); atmosphere.inert = true;
  const contours = document.createElementNS(svgNS, 'svg');
  contours.setAttribute('viewBox', '0 0 1440 1000'); contours.setAttribute('preserveAspectRatio', 'none');
  const paths = [
    'M-180 120 C200 -70 470 360 820 140 S1260 -50 1620 240',
    'M-180 400 C220 100 610 640 940 400 S1350 210 1620 510',
    'M-180 680 C250 450 520 900 900 660 S1300 490 1620 760',
    'M-180 940 C200 740 530 1120 880 920 S1300 730 1620 1040'
  ];
  paths.forEach((d, i) => {
    const path = document.createElementNS(svgNS, 'path'); path.setAttribute('d', d); contours.append(path);
    loop(path, {x: [-25, 25], y: [15 + i * 8, -15 - i * 8], duration: 35000 + i * 6000,
      alternate: true, ease: 'inOutSine'}, contours);
  });
  atmosphere.append(contours); document.body.prepend(atmosphere);
  const shimmer = document.createElement('div'); shimmer.className = 'ambient-shimmer';
  for (let i = 0; i < 18; i++) {
    const point = document.createElement('i');
    point.style.left = ((i * 61.8034) % 100).toFixed(2) + '%';
    point.style.top = ((i * 37.731 + 9) % 100).toFixed(2) + '%';
    shimmer.append(point);
  }
  atmosphere.append(shimmer);
  loop([...shimmer.children], {opacity: [.1, .46], y: [-3, 3], scale: [.8, 1.1],
    duration: 6500, delay: stagger(340), alternate: true, ease: 'inOutSine'}, shimmer);
  const intersection = new IntersectionObserver(entries => {
    entries.forEach(entry => loops.filter(item => item.host === entry.target).forEach(item => {item.inView = entry.isIntersecting;})); sync();
  });
  [...new Set(loops.map(item => item.host))].forEach(el => intersection.observe(el));
  // An actual entrance sequence, plus a responsive spring on interactive controls.
  [...document.querySelectorAll('.brand, .lab-nav button, .intro h1, .intro p, .intro .secondary, .workspace>.panel')]
    .forEach((el, i) => entrance(el, {y: [24, 0], delay: i * 60, duration: 850, ease: 'out(4)'}));
  const hover = new WeakMap();
  document.addEventListener('pointerover', event => {
    const el = event.target.closest('button:not(:disabled), .run-card, .downloads a');
    if (!el || el.contains(event.relatedTarget) || !visible(el)) return;
    hover.get(el)?.cancel();
    const animation = animate(el, {scale: 1.035, duration: 450, ease: 'outElastic(1, .6)'});
    if (animation) hover.set(el, animation);
  });
  document.addEventListener('pointerout', event => {
    const el = event.target.closest('button, .run-card, .downloads a');
    if (!el || el.contains(event.relatedTarget)) return;
    hover.get(el)?.cancel();
    if (reduced.matches) {el.style.scale = '1'; el.style.transform = '';}
    else {const animation = animate(el, {scale: 1, duration: 350, ease: 'out(3)'}); if (animation) hover.set(el, animation);}
  });
  sync();
})();
