// Noise texture inspector: each channel drawn tiled 2x2 so the wrap seams
// and repetition are visible. Values mapped as ±2σ → grayscale.
export function setupNoiseDebug(entries) {
  const checkbox = document.getElementById('noiseView')
  const overlay = document.getElementById('noiseDebug')
  let built = false
  checkbox.addEventListener('change', () => {
    if (checkbox.checked && !built) {
      built = true
      for (const entry of entries) {
        for (const [label, data] of Object.entries(entry.channels)) {
          overlay.appendChild(buildPanel(entry, label, data))
        }
      }
    }
    overlay.style.display = checkbox.checked ? 'flex' : 'none'
  })
}

function buildPanel(entry, label, data) {
  const size = entry.size
  const tile = document.createElement('canvas')
  tile.width = size
  tile.height = size
  const tileCtx = tile.getContext('2d')
  const img = tileCtx.createImageData(size, size)
  for (let i = 0; i < data.length; i++) {
    const v = Math.max(0, Math.min(255, 128 + data[i] * 64))
    img.data[i * 4] = img.data[i * 4 + 1] = img.data[i * 4 + 2] = v
    img.data[i * 4 + 3] = 255
  }
  tileCtx.putImageData(img, 0, 0)

  const view = document.createElement('canvas')
  view.width = size * 2
  view.height = size * 2
  const ctx = view.getContext('2d')
  for (const dx of [0, 1]) {
    for (const dy of [0, 1]) ctx.drawImage(tile, dx * size, dy * size)
  }

  const panel = document.createElement('div')
  const caption = document.createElement('div')
  caption.textContent = `${entry.name} / ${label} (${size}px, tiled 2x2)`
  panel.appendChild(caption)
  panel.appendChild(view)
  return panel
}
