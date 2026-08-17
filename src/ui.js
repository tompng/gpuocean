// The control panel is generated from SPEC so presets, quality levels and the
// shader all name parameters the same way. Groups tagged (s) act only on the
// above-water view, (uw) only on the submerged view, and (b) on both.

// Quality rebuilds GPU resources, so it is not a plain uniform: main.js
// watches params.quality and rebuilds when it changes.
export const QUALITY = {
  low: { gridN: 256, ribbonCells: 70, maxLayers: 3, noiseSize: 256, foamSize: 256, capLayers: 0, lodScale: 0.45 },
  medium: { gridN: 384, ribbonCells: 105, maxLayers: 5, noiseSize: 512, foamSize: 512, capLayers: 6, lodScale: 0.7 },
  high: { gridN: 512, ribbonCells: 140, maxLayers: 8, noiseSize: 512, foamSize: 512, capLayers: 6, lodScale: 1 },
  ultra: { gridN: 768, ribbonCells: 190, maxLayers: 8, noiseSize: 1024, foamSize: 1024, capLayers: 6, lodScale: 1.5 },
}

const SPEC = [
  {
    group: 'scene', open: true, items: [
      { id: 'preset', type: 'select', options: ['sea', 'river', 'lake', 'lagoon'], value: 'sea' },
      { id: 'quality', type: 'select', options: ['low', 'medium', 'high', 'ultra'], value: 'high' },
    ],
  },
  {
    group: 'sky / time', open: true, items: [
      { id: 'timeOfDay', min: 0, max: 24, step: 0.05, value: 17.4 },
      { id: 'latitude', min: -66, max: 66, step: 1, value: 20 },
      { id: 'azimuth', min: 0, max: 360, step: 1, value: 180 },
      { id: 'skyTurbidity', min: 1, max: 20, step: 0.1, value: 2.6 },
      { id: 'rayleigh', min: 0, max: 4, step: 0.05, value: 1.1 },
      { id: 'intensity', min: 0, max: 3, step: 0.05, value: 1 },
    ],
  },
  {
    group: 'waves', items: [
      { id: 'wavelength', min: 1, max: 60, step: 0.5, value: 10 },
      { id: 'amplitude', min: 0, max: 2, step: 0.01, value: 0.2 },
      { id: 'choppiness', min: 0, max: 3, step: 0.05, value: 1.5 },
      { id: 'layers', min: 1, max: 8, step: 1, value: 5 },
      { id: 'spread', min: 0, max: 90, step: 1, value: 40 },
      { id: 'waveDir', min: -180, max: 180, step: 1, value: 0 },
      { id: 'dispersion', min: 0, max: 2, step: 0.05, value: 1 },
      { id: 'lean', min: 0, max: 3, step: 0.05, value: 0.5 },
    ],
  },
  {
    group: 'ripples', items: [
      { id: 'ripple', min: 0, max: 0.6, step: 0.01, value: 0.2 },
      { id: 'rippleScale', min: 0.05, max: 2.5, step: 0.01, value: 0.5 },
      { id: 'rippleAniso', min: 0, max: 1, step: 0.05, value: 0.8 },
      { id: 'rippleBias', min: 0, max: 1, step: 0.05, value: 0.8 },
    ],
  },
  {
    group: 'surface (s)', items: [
      { id: 'sCaustics', label: 'caustics', min: 0, max: 2, step: 0.05, value: 1 },
      { id: 'sChlorophyll', label: 'chlorophyll', min: 0, max: 2, step: 0.05, value: 1 },
      { id: 'sTurbidity', label: 'turbidity', min: 0, max: 1, step: 0.01, value: 0.12 },
      { id: 'sss', min: 0, max: 1.5, step: 0.05, value: 1.5 },
      { id: 'depth', min: 0.5, max: 40, step: 0.5, value: 8 },
    ],
  },
  {
    group: 'underwater (uw)', items: [
      { id: 'uwTurbidity', label: 'turbidity', min: 0, max: 1, step: 0.01, value: 0.35 },
      { id: 'uwFog', label: 'fog', min: 0, max: 3, step: 0.05, value: 1 },
      { id: 'uwCaustics', label: 'caustics', min: 0, max: 3, step: 0.05, value: 1.2 },
      { id: 'distortionStrength', min: 0, max: 2, step: 0.02, value: 0.4 },
      { id: 'distortionScale', min: 0.1, max: 4, step: 0.05, value: 1 },
      { id: 'particleDensity', min: 0, max: 2, step: 0.02, value: 0.35 },
      { id: 'rippleStrength', min: 0, max: 2, step: 0.02, value: 0.6 },
      { id: 'waterlineThickness', min: 0, max: 1, step: 0.01, value: 0.3 },
    ],
  },
  {
    group: 'both (b)', items: [
      { id: 'chlorophyll', min: 0, max: 2, step: 0.02, value: 0.4 },
    ],
  },
  {
    group: 'foam', items: [
      { id: 'foam', min: 0, max: 1, step: 0.02, value: 0.6 },
      { id: 'foamLife', min: 0.5, max: 12, step: 0.5, value: 4 },
      { id: 'foamScale', min: 0.5, max: 3, step: 0.05, value: 1 },
      { id: 'shoreWidth', min: 0.2, max: 4, step: 0.05, value: 1 },
      { id: 'lapOvershoot', min: 0, max: 2, step: 0.05, value: 0.6 },
      { id: 'noiseScale', min: 0.2, max: 4, step: 0.05, value: 1 },
      { id: 'noiseSpeed', min: 0, max: 3, step: 0.05, value: 1 },
      { id: 'laceLow', min: 0, max: 1, step: 0.01, value: 0.12 },
      { id: 'laceHigh', min: 0, max: 1, step: 0.01, value: 0.45 },
      { id: 'crestStart', min: 0, max: 1, step: 0.01, value: 0.5 },
      { id: 'crestFull', min: 0, max: 1, step: 0.01, value: 0.86 },
      { id: 'opacity', min: 0, max: 1, step: 0.02, value: 1 },
      { id: 'crestScale', min: 0.2, max: 4, step: 0.05, value: 1 },
      { id: 'surgeRate', min: 0, max: 3, step: 0.05, value: 1 },
      { id: 'contactWidth', min: 0, max: 2, step: 0.02, value: 0.5 },
      { id: 'streaks', min: 0, max: 1, step: 0.02, value: 0.5 },
      { id: 'persistence', min: 0, max: 2, step: 0.02, value: 1 },
    ],
  },
  {
    group: 'debug', items: [
      { id: 'pause', type: 'check' },
      { id: 'wireframe', type: 'check' },
      { id: 'noiseView', type: 'check', skipParam: true },
    ],
  },
]

// Presets change how the water LOOKS, not the coastline geometry: every entry
// is a parameter override applied over the sea defaults.
const PRESETS = {
  sea: {},
  river: {
    wavelength: 3.5, amplitude: 0.05, choppiness: 0.8, layers: 4, spread: 12, dispersion: 0.6,
    lean: 0.2, ripple: 0.34, rippleScale: 0.3, depth: 3,
    sTurbidity: 0.42, uwTurbidity: 0.72, chlorophyll: 0.95, sChlorophyll: 1.1,
    uwCaustics: 0.5, sCaustics: 0.45, uwFog: 1.7, particleDensity: 1.1,
    foam: 0.42, foamLife: 2, streaks: 0.85, surgeRate: 1.8, persistence: 0.6, shoreWidth: 0.6,
  },
  lake: {
    wavelength: 2.5, amplitude: 0.025, choppiness: 0.5, layers: 3, spread: 55, dispersion: 0.4,
    lean: 0.1, ripple: 0.4, rippleScale: 0.22, depth: 14,
    sTurbidity: 0.2, uwTurbidity: 0.34, chlorophyll: 0.62, sChlorophyll: 0.9,
    uwCaustics: 0.35, sCaustics: 0.3, uwFog: 1.25, particleDensity: 0.6,
    foam: 0.06, foamLife: 1.5, streaks: 0.15, surgeRate: 0.4, shoreWidth: 0.45, lapOvershoot: 0.2,
  },
  lagoon: {
    wavelength: 5, amplitude: 0.07, choppiness: 1, layers: 4, spread: 30, dispersion: 0.8,
    lean: 0.25, ripple: 0.26, depth: 4.5,
    sTurbidity: 0.04, uwTurbidity: 0.1, chlorophyll: 0.12, sChlorophyll: 0.75,
    uwCaustics: 2.1, sCaustics: 1.7, uwFog: 0.55, particleDensity: 0.12,
    foam: 0.34, foamLife: 3, streaks: 0.3, surgeRate: 0.8, shoreWidth: 1.3, lapOvershoot: 0.9,
  },
}

export function setupUI() {
  const params = {}
  const inputs = {}
  const root = document.getElementById('controls')

  for (const { group, items, open } of SPEC) {
    const section = document.createElement('section')
    const heading = document.createElement('h3')
    heading.textContent = group
    section.appendChild(heading)
    for (const item of items) section.appendChild(buildControl(item, params, inputs))
    section.hidden = open === undefined ? false : !open
    root.appendChild(section)
  }

  // Applying a preset writes through the inputs so the panel and the params
  // object can never disagree
  inputs.preset.addEventListener('change', () => {
    const overrides = PRESETS[inputs.preset.value] ?? {}
    for (const { items } of SPEC) {
      for (const item of items) {
        if (item.type || !(item.id in inputs)) continue
        const v = overrides[item.id] ?? item.value
        inputs[item.id].value = v
        inputs[item.id].dispatchEvent(new Event('input'))
      }
    }
  })

  return params
}

function buildControl(item, params, inputs) {
  const label = document.createElement('label')
  const name = document.createElement('span')
  name.textContent = item.label ?? item.id
  name.style.textAlign = 'left'
  label.appendChild(name)

  if (item.type === 'select') {
    const select = document.createElement('select')
    for (const o of item.options) {
      const option = document.createElement('option')
      option.value = o
      option.textContent = o
      select.appendChild(option)
    }
    select.value = item.value
    select.id = item.id
    params[item.id] = item.value
    select.addEventListener('change', () => { params[item.id] = select.value })
    label.appendChild(select)
    label.appendChild(document.createElement('span'))
    inputs[item.id] = select
    return label
  }

  if (item.type === 'check') {
    const box = document.createElement('input')
    box.type = 'checkbox'
    box.id = item.id
    if (!item.skipParam) {
      params[item.id] = false
      box.addEventListener('change', () => { params[item.id] = box.checked })
    }
    label.appendChild(box)
    label.appendChild(document.createElement('span'))
    inputs[item.id] = box
    return label
  }

  const input = document.createElement('input')
  input.type = 'range'
  input.min = item.min
  input.max = item.max
  input.step = item.step
  input.value = item.value
  input.id = item.id
  const readout = document.createElement('span')
  const update = () => {
    params[item.id] = parseFloat(input.value)
    readout.textContent = input.value
  }
  input.addEventListener('input', update)
  update()
  label.appendChild(input)
  label.appendChild(readout)
  inputs[item.id] = input
  return label
}

// Exponential moving average: a raw per-frame reciprocal is too jittery to read
export function setupFPS() {
  const el = document.getElementById('fps')
  let avg = 0
  let since = 0
  return dt => {
    if (dt <= 0) return
    avg = avg === 0 ? 1 / dt : avg + (1 / dt - avg) * 0.05
    // The average tracks every frame, but the text is only rewritten a few
    // times a second: repainting a digit 120 times a second just flickers
    since += dt
    if (since < 0.25) return
    since = 0
    el.textContent = `${avg.toFixed(0)} fps`
  }
}
