export function setupUI() {
  const params = {}
  for (const id of ['wavelength', 'amplitude', 'choppiness', 'layers', 'spread', 'waveDir', 'dispersion', 'ripple', 'rippleScale', 'rippleAniso', 'rippleBias', 'sss', 'depth', 'caustics', 'sun', 'lean', 'foam', 'foamLife', 'foamScale', 'shore', 'slope', 'shoreCurve']) {
    const input = document.getElementById(id)
    const value = input.parentElement.querySelector('span')
    const update = () => {
      params[id] = parseFloat(input.value)
      value.textContent = input.value
    }
    input.addEventListener('input', update)
    update()
  }
  for (const id of ['pause', 'wireframe']) {
    const box = document.getElementById(id)
    params[id] = false
    box.addEventListener('change', () => { params[id] = box.checked })
  }
  return params
}
