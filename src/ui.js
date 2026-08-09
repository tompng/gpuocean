export function setupUI() {
  const params = { wireframe: false }
  for (const id of ['wavelength', 'amplitude', 'choppiness', 'layers', 'spread', 'dispersion']) {
    const input = document.getElementById(id)
    const value = input.parentElement.querySelector('span')
    const update = () => {
      params[id] = parseFloat(input.value)
      value.textContent = input.value
    }
    input.addEventListener('input', update)
    update()
  }
  const wireframe = document.getElementById('wireframe')
  wireframe.addEventListener('change', () => { params.wireframe = wireframe.checked })
  return params
}
