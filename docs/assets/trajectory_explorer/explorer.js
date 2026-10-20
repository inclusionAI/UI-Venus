/* Buildless, offline trajectory reader. All model content is rendered as text. */
"use strict";
(() => {
  const cases = window.LBS_TRAJECTORIES.cases;
  const $ = id => document.getElementById(id);
  const escape = value => String(value).replace(/[&<>"']/g, c => ({"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"}[c]));
  let current = cases[0];
  let step = Math.min(2, current.turns.length);
  const format = n => Number.isInteger(n) ? String(n) : Number(n).toFixed(2);
  const boxText = box => `[${box.map(format).join(", ")}]`;

  function setHash() {
    const value = `#${current.id}/turn-${step}`;
    if (location.hash !== value) history.replaceState(null, "", value);
  }

  function sourceImage(crop) {
    const parent = current.crops.find(c => `Crop ${c.number}` === crop.parent);
    return parent ? parent.image : current.original;
  }

  function overlays(width, height, crops, final, crop = null) {
    if (!$('show-overlays').checked) return "";
    const labelScale = width / 500;
    let marks = "";
    for (const c of crops) {
      const [x1, y1, x2, y2] = c.original_box;
      const cls = c.number === 2 ? " second" : "";
      const ly = Math.max(0, y1 - 21 * labelScale);
      marks += `<rect class="crop-region${cls}" x="${x1}" y="${y1}" width="${x2-x1}" height="${y2-y1}"/><rect class="region-label-bg${cls}" x="${x1}" y="${ly}" width="${22*labelScale}" height="${20*labelScale}"/><text class="region-label" x="${x1+7*labelScale}" y="${ly+14*labelScale}" style="font-size:${12*labelScale}px">${c.number}</text>`;
    }
    if (final && current.score.bbox) {
      const [x1, y1, x2, y2] = crop ? crop.original_box : [0, 0, width, height];
      const sx = width / (x2 - x1), sy = height / (y2 - y1);
      const [a, b, c, d] = current.score.bbox;
      const [px, py] = current.score.pred;
      const x = (px-x1)*sx, y = (py-y1)*sy, arm = width/70;
      marks += `<rect class="target-box" x="${(a-x1)*sx}" y="${(b-y1)*sy}" width="${(c-a)*sx}" height="${(d-b)*sy}"/><path class="prediction" d="M${x-arm},${y}H${x+arm} M${x},${y-arm}V${y+arm}"/>`;
    }
    return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${final && current.score.bbox ? 'Reference target box and recorded prediction. ' : ''}${crops.length ? 'Recorded crop regions; 1 is solid clay, 2 is dashed sage.' : 'Image coordinate overlay.'}" style="overflow:hidden">${marks}</svg>`;
  }

  function imageCard(image, title, dimensions, caption, overlay = "", crop = false) {
    return `<figure class="image-card${crop ? ' crop-card' : ''}"><figcaption class="image-label"><strong>${escape(title)}</strong><span>${dimensions.join(" × ")} px${crop ? '' : ' · original size'}</span></figcaption><button class="image-trigger" type="button" data-image="${escape(image)}" data-caption="${escape(title)}" aria-label="Enlarge ${escape(title)}"><span class="image-space"><img src="${escape(image)}" alt="${escape(title)} for ${escape(current.title)}" ${crop ? `width="${dimensions[0]}" height="${dimensions[1]}"` : ''}>${overlay}</span></button><p class="image-caption">${caption}</p></figure>`;
  }

  function renderImages() {
    const turn = current.turns[step-1];
    const final = step === current.turns.length;
    const seen = current.crops.filter(c => turn && c.turn <= turn.number);
    const received = seen.filter(c => c.turn === turn?.number);
    // At the final turn both crops have already been received; keep the last view.
    const visible = received.length ? received : (step ? seen.filter(c => c.turn === seen[seen.length-1]?.turn) : []);
    let html = imageCard(current.original, "Original image", current.dimensions,
      seen.length ? `${seen.map(c => `Region ${c.number}: ${escape(c.parent)} → Crop ${c.number}`).join(' · ')}. Select the image to inspect.` : "The complete scene before any crop is requested. Select the image to inspect.",
      overlays(...current.dimensions, seen, final));
    if (received.length > 1) html += `<p class="branch-note">Original ↗ Crop 1 &nbsp; / &nbsp; Original ↘ Crop 2<br>Both crops requested in turn ${turn.number}; neither crop uses the other.</p>`;
    for (const c of visible) {
      html += imageCard(c.image, `Crop ${c.number} · from ${c.parent}`, c.dimensions,
        `Parent rectangle: <code>${boxText(c.parent_box)}</code><br>Original-image rectangle: <code>${boxText(c.original_box)}</code>`,
        overlays(...c.dimensions, [], final, c), true);
      if (c.parent !== "Original") {
        html += `<p class="branch-note">Original → ${escape(c.parent)} → Crop ${c.number}<br>The action uses pixels in the resized parent crop.</p>`;
      }
    }
    if (final && current.score.bbox) html += '<p class="visual-note">Reference: sage outline &nbsp; · &nbsp; Prediction: clay +<br>Both are review overlays, added after the recorded run.</p>';
    if (!step) html += '<p class="visual-note">Choose a model turn above to follow its reasoning and returned visual evidence.</p>';
    $('image-evidence').innerHTML = html;
  }

  function splitContent(turn) {
    const text = turn.content;
    const closing = text.indexOf("</think>");
    const reasoning = closing >= 0 ? text.slice(0, closing).replace(/^\s*<think>/, "").trim() : turn.reasoning_content;
    const remainder = closing >= 0 ? text.slice(closing+8) : text;
    const answer = remainder.replace(/<action>[\s\S]*?<\/action>/g, "").replace(/<\/?answer>/g, "").trim();
    return {reasoning, answer};
  }

  function renderTrace() {
    if (!step) {
      $('turn-label').textContent = "Before observation";
      $('transcript').innerHTML = `<div class="editorial-block"><strong>How to read this case</strong><p>${escape(current.analysis)}</p></div><div class="trace-block action-block"><h4>Observation protocol</h4><p class="action-flow">Read the original image → reason → request crop(s) → inspect returned image(s) → answer.</p><p class="action-flow">A crop is an action selected by the model. The image returned by that action is the next observation.</p></div>`;
      return;
    }
    const turn = current.turns[step-1];
    const parts = splitContent(turn);
    const actions = current.crops.filter(c => c.turn === turn.number);
    $('turn-label').textContent = `Model turn ${turn.number}`;
    let html = parts.reasoning ? `<section class="trace-block reasoning-block"><h4>Model reasoning <span>Recorded text</span></h4><p class="reasoning-text">${escape(parts.reasoning)}</p></section>` : '';
    for (const c of actions) {
      const args = c.parent_box.map(format);
      html += `<section class="trace-block action-block"><h4>Crop action ${c.number} <span>Turn ${c.turn}</span></h4><p class="action-flow">${escape(c.parent)} → Crop ${c.number} · ${c.dimensions.join(' × ')} px returned</p><pre class="action-line">crop_image(${escape(c.parent)},
  x1=${args[0]}, y1=${args[1]},
  x2=${args[2]}, y2=${args[3]})</pre><details><summary>Exact recorded action & parent image</summary><pre class="action-line">${escape(c.action)}</pre><button type="button" class="parent-link" data-image="${escape(sourceImage(c))}" data-caption="${escape(c.parent)} · action ${c.number} parent">Inspect parent image ↗</button></details></section>`;
    }
    if (parts.answer) {
      html += `<section class="trace-block answer-block"><h4>Final answer <span>Recorded output</span></h4><p class="final-answer">${escape(parts.answer)}</p><p class="score-line">${escape(current.metric)} · ${escape(current.outcome)}<br>Reference: ${escape(current.reference)}${current.score.bbox ? '<br>Answer coordinates are normalized to [0, 1000] on the original image.' : ''}</p></section>`;
    }
    html += `<div class="editorial-block"><strong>Editorial observation · turn ${turn.number}</strong><p>${escape(turn.analysis)}</p></div><details class="provenance"><summary>Verbatim response with protocol tags</summary>${turn.reasoning_content ? `<p>Separate reasoning_content field</p><pre>${escape(turn.reasoning_content)}</pre>` : ''}<pre>${escape(turn.content)}</pre></details>`;
    $('transcript').innerHTML = html;
  }

  function renderStep({focus = false} = {}) {
    const tabs = [...$('step-list').querySelectorAll('[role=tab]')];
    tabs.forEach((tab, i) => {tab.setAttribute('aria-selected', String(i === step)); tab.tabIndex = i === step ? 0 : -1;});
    $('step-panel').setAttribute('aria-labelledby', `step-${step}`);
    $('previous-step').disabled = !step;
    $('next-step').disabled = step === current.turns.length;
    $('step-position').textContent = step ? `Model turn ${current.turns[step-1].number} of ${current.turns.length}` : 'Original input';
    renderImages();
    renderTrace();
    setHash();
    if (focus) tabs[step].focus();
  }

  function renderCase() {
    $('case-list').innerHTML = cases.map((c, i) => `<button type="button" class="case-button" data-case="${escape(c.id)}" aria-current="${c.id === current.id}"><span class="case-index">${String(i+1).padStart(2,'0')}</span><span><strong>${escape(c.title)}</strong><small>${escape(c.benchmark)}</small></span></button>`).join('');
    $('case-title').textContent = current.title;
    $('case-meta').textContent = `${current.benchmark} / ${current.behavior}`;
    $('case-analysis').textContent = current.analysis;
    $('question-text').textContent = current.question;
    $('figure-link').href = current.figure;
    $('sequence-count').textContent = `${current.turns.length} model turns · ${current.crops.length} crops`;
    const steps = [{top:'Input', label:'Original image'}, ...current.turns.map(t => {
      const crops = current.crops.filter(c => c.turn === t.number);
      return {top:`Turn ${t.number}`, label: crops.length ? crops.map(c=>`Crop ${c.number}`).join(' + ') : 'Final answer'};
    })];
    $('step-list').innerHTML = steps.map((s,i) => `<button type="button" id="step-${i}" class="step-tab" role="tab" aria-controls="step-panel" aria-selected="${i===step}" tabindex="${i===step ? 0 : -1}" data-step="${i}"><span>${s.top}</span><strong>${s.label}</strong></button>`).join('');
    $('interpretation-title').textContent = current.behavior;
    $('case-limitation').textContent = current.limitation;
    $('provenance-content').innerHTML = `<div class="record-links"><a href="${escape(current.source_record)}" download="${escape(current.id)}.json">Complete released record ↓</a><a href="${escape(current.figure)}">Verified static figure ↗</a><button type="button" id="share-case">Copy step link</button></div><p>Repository source: <code>${escape(current.source)}</code><br>Source SHA-256: <code>${escape(current.source_sha256)}</code></p><p>Original resolution: ${current.dimensions.join(' × ')} px. Original previews are reduced to at most 1600 px on either side. Returned crops retain their checked, reconstructed pixel dimensions before API JPEG encoding. All rectangles use the saved geometry.</p><p>Question and model responses are preserved from sanitized release records. The visible action summary uses the executor’s parent rectangle; expand it for the exact model action. Editorial analysis is written separately. The verbatim response retains its original protocol tags, including missing opening tags in some saved responses.</p><p>This is a curated illustration, not an estimate of benchmark accuracy. No model inference or new judgment was performed.</p><pre>${escape(JSON.stringify(current.score,null,2))}</pre>`;
    $('provenance-content').parentElement.open = false;
    renderStep();
  }

  function loadHash() {
    const match = location.hash.match(/^#([a-z0-9_-]+)\/turn-(\d+)$/);
    const selected = match && cases.find(c => c.id === match[1]);
    if (selected) {current = selected; step = Math.min(Number(match[2]), current.turns.length);}
    renderCase();
  }

  $('case-list').addEventListener('click', event => {
    const button = event.target.closest('[data-case]');
    if (!button) return;
    current = cases.find(c => c.id === button.dataset.case);
    step = Math.min(2, current.turns.length);
    renderCase();
    $('case-list').querySelector(`[data-case="${current.id}"]`).focus();
  });
  $('step-list').addEventListener('click', event => {
    const button = event.target.closest('[data-step]');
    if (button) {step = Number(button.dataset.step); renderStep();}
  });
  $('step-list').addEventListener('keydown', event => {
    if (!['ArrowLeft','ArrowRight','Home','End'].includes(event.key)) return;
    event.preventDefault();
    if (event.key === 'Home') step = 0;
    else if (event.key === 'End') step = current.turns.length;
    else step = (step + (event.key === 'ArrowRight' ? 1 : -1) + current.turns.length + 1) % (current.turns.length + 1);
    renderStep({focus:true});
  });
  $('previous-step').addEventListener('click', () => {if (step) {step--; renderStep();}});
  $('next-step').addEventListener('click', () => {if (step < current.turns.length) {step++; renderStep();}});
  $('show-overlays').addEventListener('change', renderImages);
  document.addEventListener('click', event => {
    const trigger = event.target.closest('[data-image]');
    if (trigger) {
      $('dialog-image').src = trigger.dataset.image;
      $('dialog-image').alt = trigger.dataset.caption;
      $('dialog-caption').textContent = trigger.dataset.caption;
      $('image-dialog').showModal();
    }
  });
  $('close-dialog').addEventListener('click', () => $('image-dialog').close());
  $('image-dialog').addEventListener('click', event => {if (event.target === $('image-dialog')) $('image-dialog').close();});
  $('provenance-content').addEventListener('click', async event => {
    if (event.target.id !== 'share-case') return;
    try {await navigator.clipboard.writeText(location.href); event.target.textContent = 'Step link copied';}
    catch {event.target.textContent = `Step: ${location.hash}`;}
  });
  window.addEventListener('hashchange', loadHash);
  loadHash();
})();
