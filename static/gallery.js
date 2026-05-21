async function loadGallery() {
  const response = await fetch('/outputs');
  const data = await response.json();
  const root = document.getElementById('gallery');
  root.innerHTML = '';

  if (!data.items || data.items.length === 0) {
    root.innerHTML = '<div class="empty">No generated outputs found yet.</div>';
    return;
  }

  const encodePath = (value) => String(value)
    .split('/')
    .filter(Boolean)
    .map((part) => encodeURIComponent(part))
    .join('/');

  const stripOutputsPrefix = (value) => {
    const normalized = String(value || '');
    return normalized.startsWith('outputs/') ? normalized.slice('outputs/'.length) : normalized;
  };

  for (const item of data.items) {
    const card = document.createElement('article');
    card.className = 'card';
    const prompt = item.prompt || 'Untitled generation';
    const imagePath = encodePath(stripOutputsPrefix(item.image_path));
    const metadataPath = encodePath(stripOutputsPrefix(item.metadata_path));

    const image = document.createElement('img');
    image.src = `/outputs/${imagePath}`;
    image.alt = prompt;

    const content = document.createElement('div');

    const title = document.createElement('h2');
    title.textContent = prompt;
    content.appendChild(title);

    const meta = document.createElement('div');
    meta.className = 'meta';
    const metaRows = [
      ['Seed', item.seed ?? 'n/a'],
      ['Steps', item.steps ?? 'n/a'],
      ['Model', item.model ?? 'n/a'],
      ['Runtime', `${item.runtime_seconds ?? 'n/a'}s'],
    ];

    for (const [label, value] of metaRows) {
      const row = document.createElement('div');
      row.textContent = `${label}: ${value}`;
      meta.appendChild(row);
    }
    content.appendChild(meta);

    const actions = document.createElement('div');
    actions.className = 'actions';

    const imageLink = document.createElement('a');
    imageLink.href = `/outputs/${imagePath}`;
    imageLink.target = '_blank';
    imageLink.rel = 'noreferrer';
    imageLink.textContent = 'Open image';

    const metadataLink = document.createElement('a');
    metadataLink.href = `/metadata/${metadataPath}`;
    metadataLink.target = '_blank';
    metadataLink.rel = 'noreferrer';
    metadataLink.textContent = 'Open metadata';

    actions.appendChild(imageLink);
    actions.appendChild(metadataLink);

    card.appendChild(image);
    card.appendChild(content);
    card.appendChild(actions);
    root.appendChild(card);
  }
}

document.getElementById('refreshButton').addEventListener('click', loadGallery);
loadGallery();
