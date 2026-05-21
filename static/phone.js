const state = {
  token: null,
  models: [],
  busy: false,
};

const byId = (id) => document.getElementById(id);

function setStatus(text) {
  byId('progressText').textContent = text;
}

function stripOutputsPrefix(value) {
  const normalized = String(value || '');
  return normalized.startsWith('outputs/') ? normalized.slice('outputs/'.length) : normalized;
}

function encodePath(value) {
  return String(value)
    .split('/')
    .filter(Boolean)
    .map((part) => encodeURIComponent(part))
    .join('/');
}

function readTokenFromUrl() {
  const params = new URLSearchParams(window.location.search);
  const fromUrl = params.get('token');
  if (!fromUrl) {
    return null;
  }
  params.delete('token');
  const query = params.toString();
  const next = `${window.location.pathname}${query ? `?${query}` : ''}`;
  window.history.replaceState({}, document.title, next);
  return fromUrl;
}

function loadToken() {
  const urlToken = readTokenFromUrl();
  if (urlToken) {
    sessionStorage.setItem('sana_phone_token', urlToken);
  }
  state.token = sessionStorage.getItem('sana_phone_token');
  if (state.token) {
    byId('tokenInput').value = state.token;
    byId('authStatus').textContent = 'Token ready.';
  }
}

function saveToken() {
  const token = byId('tokenInput').value.trim();
  if (!token) {
    byId('authStatus').textContent = 'Token is required.';
    return;
  }
  state.token = token;
  sessionStorage.setItem('sana_phone_token', token);
  byId('authStatus').textContent = 'Token saved.';
}

async function apiFetch(path, options = {}) {
  if (!state.token) {
    throw new Error('Set access token first.');
  }
  const headers = {
    'Content-Type': 'application/json',
    'X-Sana-Token': state.token,
    ...(options.headers || {}),
  };
  const response = await fetch(path, {
    ...options,
    headers,
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    const detail = payload.detail || `HTTP ${response.status}`;
    throw new Error(detail);
  }
  return response;
}

function readDimensions() {
  const [width, height] = byId('resolutionSelect').value.split('x').map((v) => Number(v));
  return { width, height };
}

function readDtype() {
  const mode = byId('precisionSelect').value;
  if (mode === 'float32') {
    return 'float32';
  }
  return 'float16';
}

function buildRequestBody() {
  const dimensions = readDimensions();
  return {
    prompt: byId('promptInput').value,
    negative_prompt: byId('negativeInput').value || null,
    model: byId('modelSelect').value,
    width: dimensions.width,
    height: dimensions.height,
    steps: Number(byId('stepsSelect').value),
    guidance: Number(byId('guidanceInput').value),
    seed: Number(byId('seedInput').value),
    dtype: readDtype(),
    attention_slicing: true,
  };
}

function setBusy(value) {
  state.busy = value;
  for (const id of ['generateButton', 'gridButton', 'safeRecoveryButton', 'refreshGalleryButton']) {
    byId(id).disabled = value;
  }
}

function showMetadata(obj) {
  byId('metadataContent').textContent = JSON.stringify(obj, null, 2);
  byId('metadataCard').classList.remove('hidden');
}

async function refreshModels() {
  const response = await fetch('/models');
  const payload = await response.json();
  state.models = payload.models || [];
  const select = byId('modelSelect');
  select.innerHTML = '';
  for (const model of state.models) {
    const option = document.createElement('option');
    option.value = model;
    option.textContent = model;
    select.appendChild(option);
  }
}

async function refreshStatus() {
  try {
    const response = await apiFetch('/api/phone/status');
    const status = await response.json();
    const panel = byId('statusPanel');
    panel.innerHTML = '';
    const rows = [
      ['Device', status.device],
      ['MPS available', String(status.mps_available)],
      ['Selected model', status.selected_model],
      ['Server URL', status.server_url],
      ['Phone mode', String(status.phone_mode)],
    ];
    for (const [label, value] of rows) {
      const row = document.createElement('div');
      const dt = document.createElement('dt');
      const dd = document.createElement('dd');
      dt.textContent = label;
      dd.textContent = value || 'n/a';
      row.appendChild(dt);
      row.appendChild(dd);
      panel.appendChild(row);
    }
  } catch (error) {
    byId('authStatus').textContent = `Status failed: ${error.message}`;
  }
}

async function refreshGallery() {
  try {
    const response = await apiFetch('/api/phone/gallery');
    const payload = await response.json();
    const gallery = byId('gallery');
    gallery.innerHTML = '';

    const items = payload.items || [];
    if (!items.length) {
      const empty = document.createElement('p');
      empty.className = 'muted';
      empty.textContent = 'No outputs yet.';
      gallery.appendChild(empty);
      return;
    }

    for (const item of items) {
      const card = document.createElement('article');
      card.className = 'item';

      const image = document.createElement('img');
      const imageRel = encodePath(stripOutputsPrefix(item.image_path));
      image.src = `/api/phone/file/${imageRel}?token=${encodeURIComponent(state.token)}`;
      image.alt = item.prompt || 'Generated output';
      image.loading = 'lazy';
      image.addEventListener('click', () => {
        window.open(`/api/phone/file/${imageRel}?token=${encodeURIComponent(state.token)}`, '_blank', 'noopener,noreferrer');
      });

      const meta = document.createElement('div');
      meta.className = 'item-meta';
      meta.textContent = `${item.prompt || 'Untitled'} | seed ${item.seed ?? 'n/a'} | ${item.steps ?? 'n/a'} steps`;

      const controls = document.createElement('div');
      controls.className = 'row';
      const metaButton = document.createElement('button');
      metaButton.type = 'button';
      metaButton.className = 'secondary';
      metaButton.textContent = 'Metadata';
      metaButton.addEventListener('click', async () => {
        const relMeta = encodePath(stripOutputsPrefix(item.metadata_path));
        const metaResponse = await apiFetch(`/api/phone/metadata/${relMeta}`);
        const metaPayload = await metaResponse.json();
        showMetadata(metaPayload);
      });

      const openButton = document.createElement('button');
      openButton.type = 'button';
      openButton.textContent = 'Open image';
      openButton.addEventListener('click', () => {
        window.open(`/api/phone/file/${imageRel}?token=${encodeURIComponent(state.token)}`, '_blank', 'noopener,noreferrer');
      });

      controls.appendChild(metaButton);
      controls.appendChild(openButton);

      card.appendChild(image);
      card.appendChild(meta);
      card.appendChild(controls);
      gallery.appendChild(card);
    }
  } catch (error) {
    setStatus(`Gallery failed: ${error.message}`);
  }
}

async function refreshPresets() {
  try {
    const response = await apiFetch('/api/phone/presets');
    const payload = await response.json();
    const select = byId('presetSelect');
    select.innerHTML = '';

    const defaultOption = document.createElement('option');
    defaultOption.value = '';
    defaultOption.textContent = 'Choose preset';
    select.appendChild(defaultOption);

    for (const preset of payload.items || []) {
      const option = document.createElement('option');
      option.value = preset.id;
      option.textContent = preset.name;
      option.dataset.payload = JSON.stringify(preset);
      select.appendChild(option);
    }
  } catch (error) {
    byId('authStatus').textContent = `Preset load failed: ${error.message}`;
  }
}

function applyPreset(preset) {
  byId('promptInput').value = preset.prompt || '';
  byId('negativeInput').value = preset.negative_prompt || '';
  byId('stepsSelect').value = String(preset.steps || 12);
  byId('guidanceInput').value = String(preset.guidance || 4.5);
  byId('guidanceValue').textContent = byId('guidanceInput').value;
  byId('seedInput').value = String(preset.seed || 42);
  byId('precisionSelect').value = preset.dtype === 'float32' ? 'float32' : 'auto';

  const res = `${preset.width || 512}x${preset.height || 512}`;
  byId('resolutionSelect').value = ['512x512', '768x768', '1024x1024'].includes(res) ? res : '512x512';

  if (preset.model && state.models.includes(preset.model)) {
    byId('modelSelect').value = preset.model;
  }
}

async function generateImage() {
  try {
    setBusy(true);
    setStatus('Generating image...');
    const response = await apiFetch('/api/phone/generate', {
      method: 'POST',
      body: JSON.stringify(buildRequestBody()),
    });
    const payload = await response.json();
    setStatus(`Generated: ${payload.image_path}`);
    if (payload.metadata) {
      showMetadata(payload.metadata);
    }
    await refreshGallery();
  } catch (error) {
    setStatus(`Generate failed: ${error.message}`);
  } finally {
    setBusy(false);
  }
}

function parseGridSeeds() {
  const values = byId('gridSeedsInput').value
    .split(',')
    .map((v) => Number(v.trim()))
    .filter((v) => Number.isInteger(v) && v >= 0);
  return values;
}

async function generateGrid() {
  try {
    setBusy(true);
    setStatus('Generating grid...');
    const body = buildRequestBody();
    const seeds = parseGridSeeds();
    if (!seeds.length) {
      throw new Error('Provide at least one valid grid seed.');
    }
    const response = await apiFetch('/api/phone/grid', {
      method: 'POST',
      body: JSON.stringify({
        ...body,
        seeds,
        columns: Number(byId('gridColumnsInput').value || 2),
      }),
    });
    const payload = await response.json();
    setStatus(`Grid ready: ${payload.grid_image_path}`);
    showMetadata(payload);
    await refreshGallery();
  } catch (error) {
    setStatus(`Grid failed: ${error.message}`);
  } finally {
    setBusy(false);
  }
}

function applySafeRecovery() {
  byId('resolutionSelect').value = '512x512';
  byId('stepsSelect').value = '8';
  byId('precisionSelect').value = 'float32';
  const fallbackModel = 'Efficient-Large-Model/Sana_600M_512px_diffusers';
  if (state.models.includes(fallbackModel)) {
    byId('modelSelect').value = fallbackModel;
  }
  setStatus('Safe Recovery applied: 512x512, 8 steps, Force FP32, 600M 512px model.');
}

function randomSeed() {
  const seed = Math.floor(Math.random() * 2147483647);
  byId('seedInput').value = String(seed);
}

async function saveCurrentPreset() {
  const name = window.prompt('Preset name');
  if (!name) {
    return;
  }
  const body = buildRequestBody();
  const id = `phone_${Date.now()}`;
  try {
    await apiFetch('/api/phone/presets', {
      method: 'POST',
      body: JSON.stringify({
        id,
        name,
        prompt: body.prompt,
        negative_prompt: body.negative_prompt,
        width: body.width,
        height: body.height,
        steps: body.steps,
        guidance: body.guidance,
        dtype: body.dtype,
        tags: ['phone'],
      }),
    });
    await refreshPresets();
    setStatus(`Preset saved: ${name}`);
  } catch (error) {
    setStatus(`Preset save failed: ${error.message}`);
  }
}

function wireEvents() {
  byId('saveTokenButton').addEventListener('click', async () => {
    saveToken();
    await refreshStatus();
    await refreshPresets();
    await refreshGallery();
  });
  byId('refreshStatusButton').addEventListener('click', refreshStatus);
  byId('refreshGalleryButton').addEventListener('click', refreshGallery);
  byId('generateButton').addEventListener('click', generateImage);
  byId('gridButton').addEventListener('click', generateGrid);
  byId('safeRecoveryButton').addEventListener('click', applySafeRecovery);
  byId('randomSeedButton').addEventListener('click', randomSeed);
  byId('savePresetButton').addEventListener('click', saveCurrentPreset);
  byId('closeMetadataButton').addEventListener('click', () => byId('metadataCard').classList.add('hidden'));
  byId('loadPresetButton').addEventListener('click', () => {
    const select = byId('presetSelect');
    const option = select.options[select.selectedIndex];
    if (!option || !option.dataset.payload) {
      return;
    }
    applyPreset(JSON.parse(option.dataset.payload));
  });
  byId('guidanceInput').addEventListener('input', () => {
    byId('guidanceValue').textContent = byId('guidanceInput').value;
  });
}

async function initialize() {
  wireEvents();
  loadToken();
  await refreshModels();
  if (state.token) {
    await refreshStatus();
    await refreshPresets();
    await refreshGallery();
  }
}

initialize();
