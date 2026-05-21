async function loadGallery() {
  const response = await fetch('/outputs');
  const data = await response.json();
  const root = document.getElementById('gallery');
  root.innerHTML = '';

  if (!data.items || data.items.length === 0) {
    root.innerHTML = '<div class="empty">No generated outputs found yet.</div>';
    return;
  }

  for (const item of data.items) {
    const card = document.createElement('article');
    card.className = 'card';
    const prompt = item.prompt || 'Untitled generation';
    card.innerHTML = `
      <img src="/file/${item.image_path}" alt="${prompt}">
      <div>
        <h2>${prompt}</h2>
        <div class="meta">
          <div>Seed: ${item.seed ?? 'n/a'}</div>
          <div>Steps: ${item.steps ?? 'n/a'}</div>
          <div>Model: ${item.model ?? 'n/a'}</div>
          <div>Runtime: ${item.runtime_seconds ?? 'n/a'}s</div>
        </div>
      </div>
      <div class="actions">
        <a href="/file/${item.image_path}" target="_blank" rel="noreferrer">Open image</a>
        <a href="/file/${item.metadata_path}" target="_blank" rel="noreferrer">Open metadata</a>
      </div>
    `;
    root.appendChild(card);
  }
}

document.getElementById('refreshButton').addEventListener('click', loadGallery);
loadGallery();
