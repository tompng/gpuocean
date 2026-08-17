// The control panel is generated from SPEC so presets, quality levels and the
// shader all name parameters the same way. Groups tagged (s) act only on the
// above-water view, (uw) only on the submerged view, and (b) on both.

// Quality scales the near-field CELL SIZE, not the vertex count directly.
// gridN alone would be the wrong lever: the warp grows cells exponentially
// past linearCells, so the grid's extent is hyper-sensitive to it — halving
// gridN shrinks the ocean from 177 km to a 51 m puddle rather than coarsening
// it. Each row below pairs a cell size with the gridN and linearCells that
// hold the extent near 175 km, so only the detail changes.
// These are baked into vertex/index buffers, so a change rebuilds the Ocean;
// maxLayers and lodScale are per-frame uniforms. The CPU noise textures are
// deliberately not rescaled — regenerating them costs a visible hitch.
export const QUALITY = {
  low: { cell: 0.8, linearCells: 80, gridN: 340, ribbonCells: 70, maxLayers: 3, lodScale: 0.4 },
  medium: { cell: 0.55, linearCells: 116, gridN: 418, ribbonCells: 105, maxLayers: 5, lodScale: 0.7 },
  high: { cell: 0.4, linearCells: 160, gridN: 512, ribbonCells: 140, maxLayers: 8, lodScale: 1 },
  ultra: { cell: 0.28, linearCells: 229, gridN: 656, ribbonCells: 190, maxLayers: 8, lodScale: 1.6 },
}

const SPEC = [{
  group: 'scene',
  open: true,
  items: [{ id: 'preset', type: 'select', options: ['sea', 'river', 'lake', 'lagoon'], value: 'sea' }, {
    id: 'quality',
    type: 'select',
    options: ['low', 'medium', 'high', 'ultra'],
    value: 'high'
  },],
}, {
  group: 'sky / time',
  open: true,
  items: [{ id: 'timeOfDay', min: 0, max: 24, step: 0.05, value: 17.4 }, {
    id: 'latitude',
    min: -66,
    max: 66,
    step: 1,
    value: 20
  }, { id: 'azimuth', min: 0, max: 360, step: 1, value: 180 }, {
    id: 'skyTurbidity',
    min: 2,
    max: 10,
    step: 0.1,
    value: 2.6
  }, { id: 'rayleigh', min: 0, max: 2, step: 0.05, value: 1.1 }, {
    id: 'intensity',
    min: 0,
    max: 3,
    step: 0.05,
    value: 1
  },],
}, {
  group: 'clouds',
  open: true,
  items: [
    // Sky fraction is not linear in this: it is a threshold on a field with
    // sigma ~0.167 about 0.5, so 0.45 covers roughly a third and 0.75 most of it
    { id: 'cloudCover', min: 0, max: 1, step: 0.01, value: 0.45 },
    { id: 'cloudAltitude', min: 300, max: 6000, step: 50, value: 1800 },
    { id: 'cloudThickness', min: 100, max: 2500, step: 50, value: 700 },
    // extinction [1/m]; density * thickness is the vertical optical depth, and
    // 0.04 * 700 = 28 is a properly opaque cumulus (real ones run 10-50)
    { id: 'cloudDensity', min: 0, max: 0.12, step: 0.002, value: 0.04 },
    { id: 'cloudScale', min: 300, max: 4000, step: 50, value: 1400 },
    { id: 'cloudSoftness', min: 0.02, max: 0.6, step: 0.01, value: 0.2 },
    { id: 'cloudDetail', min: 0, max: 1, step: 0.05, value: 0.6 },
    { id: 'cloudAnvil', min: 0, max: 1, step: 0.05, value: 0.2 },
    { id: 'cloudAbsorption', min: 0, max: 1, step: 0.05, value: 0.15 },
    { id: 'cloudAmbient', min: 0, max: 2, step: 0.05, value: 1 },
    { id: 'cloudWindSpeed', min: 0, max: 30, step: 0.5, value: 7 },
    { id: 'cloudWindDir', min: -180, max: 180, step: 1, value: 0 },
    { id: 'cloudShear', min: 0, max: 1, step: 0.05, value: 0.25 },
    { id: 'cloudShadow', min: 0, max: 1, step: 0.05, value: 0.7 },
  ],
}, {
  group: 'waves',
  items: [{ id: 'wavelength', min: 1, max: 60, step: 0.5, value: 10 }, {
    id: 'amplitude',
    min: 0,
    max: 2,
    step: 0.01,
    value: 0.2
  }, { id: 'choppiness', min: 0, max: 3, step: 0.05, value: 1.5 }, {
    id: 'layers',
    min: 1,
    max: 8,
    step: 1,
    value: 5
  }, { id: 'spread', min: 0, max: 90, step: 1, value: 40 }, {
    id: 'waveDir',
    min: -180,
    max: 180,
    step: 1,
    value: 0
  }, { id: 'dispersion', min: 0, max: 2, step: 0.05, value: 1 }, {
    id: 'lean',
    min: 0,
    max: 3,
    step: 0.05,
    value: 0.5
  },],
}, {
  group: 'ripples',
  items: [{ id: 'ripple', min: 0, max: 0.6, step: 0.01, value: 0.2 }, {
    id: 'rippleScale',
    min: 0.05,
    max: 2.5,
    step: 0.01,
    value: 0.5
  }, { id: 'rippleAniso', min: 0, max: 1, step: 0.05, value: 0.8 }, {
    id: 'rippleBias',
    min: 0,
    max: 1,
    step: 0.05,
    value: 0.8
  },],
}, {
  group: 'surface (s)',
  items: [{ id: 'sCaustics', label: 'caustics', min: 0, max: 2, step: 0.05, value: 1 }, {
    id: 'sChlorophyll',
    label: 'chlorophyll',
    min: 0,
    max: 2,
    step: 0.05,
    value: 1
  }, { id: 'sTurbidity', label: 'turbidity', min: 0, max: 1, step: 0.01, value: 0.12 }, {
    id: 'sss',
    min: 0,
    max: 1.5,
    step: 0.05,
    value: 1.5
  }, { id: 'depth', min: 0.5, max: 40, step: 0.5, value: 8 },],
}, {
  group: 'underwater (uw)',
  items: [{ id: 'uwTurbidity', label: 'turbidity', min: 0, max: 1, step: 0.01, value: 0.35 }, {
    id: 'uwFog',
    label: 'fog',
    min: 0,
    max: 3,
    step: 0.05,
    value: 1
  }, {
    id: 'uwCaustics',
    label: 'caustics',
    min: 0,
    max: 3,
    step: 0.05,
    value: 1.2
  }, // 1.0 is the true refracted parallax; below that under-refracts
    { id: 'distortionStrength', min: 0, max: 2, step: 0.02, value: 1 }, {
      id: 'distortionScale',
      min: 0.1,
      max: 4,
      step: 0.05,
      value: 1
    }, { id: 'particleDensity', min: 0, max: 2, step: 0.02, value: 0.35 }, {
      id: 'rippleStrength',
      min: 0,
      max: 2,
      step: 0.02,
      value: 0.6
    }, { id: 'waterlineThickness', min: 0, max: 1, step: 0.01, value: 0.3 },],
}, {
  group: 'both (b)', items: [{ id: 'chlorophyll', min: 0, max: 2, step: 0.02, value: 0.4 },],
}, {
  group: 'foam',
  open: true,
  items: [{
    id: 'plate',
    type: 'select',
    options: ['blend', 'sparse', 'mid', 'dense', 'procedural'],
    value: 'blend'
  }, { id: 'foam', min: 0, max: 1, step: 0.02, value: 0.6 }, {
    id: 'foamLife',
    min: 0.5,
    max: 12,
    step: 0.5,
    value: 4
  }, { id: 'shoreWidth', min: 0.2, max: 4, step: 0.05, value: 1 }, {
    id: 'lapOvershoot',
    min: 0,
    max: 2,
    step: 0.05,
    value: 0.6
  }, { id: 'noiseScale', min: 0.2, max: 4, step: 0.05, value: 1 }, {
    id: 'noiseSpeed',
    min: 0,
    max: 3,
    step: 0.05,
    value: 1
  }, { id: 'laceLow', min: 0, max: 1, step: 0.01, value: 0.12 }, {
    id: 'laceHigh',
    min: 0,
    max: 1,
    step: 0.01,
    value: 0.35
  }, { id: 'crestStart', min: 0, max: 1, step: 0.01, value: 0.72 }, {
    id: 'crestFull',
    min: 0,
    max: 1,
    step: 0.01,
    value: 0.97
  }, { id: 'opacity', min: 0, max: 1, step: 0.02, value: 1 }, {
    id: 'crestScale',
    min: 0.2,
    max: 4,
    step: 0.05,
    value: 1
  }, { id: 'surgeRate', min: 0, max: 3, step: 0.05, value: 1 }, {
    id: 'contactWidth',
    min: 0,
    max: 2,
    step: 0.02,
    value: 0.5
  }, { id: 'streaks', min: 0, max: 1, step: 0.02, value: 0.5 }, {
    id: 'persistence',
    min: 0,
    max: 2,
    step: 0.02,
    value: 1
  },],
}, {
  group: 'debug',
  items: [{ id: 'pause', type: 'check' }, { id: 'wireframe', type: 'check' }, {
    id: 'noiseView',
    type: 'check',
    skipParam: true
  },],
},]

// Presets change how the water LOOKS, not the coastline geometry: every entry
// is a parameter override applied over the sea defaults.
const PRESETS = {
  sea: {
    cloudCover: 0.4,
    sCaustics: 0.1,
    foam: 0.34,
    foamLife: 3,
    streaks: 0.15,
    surgeRate: 0.4,
    shoreWidth: 0.45,
    lapOvershoot: 0.2,
    noiseScale: 0.01,
  }, river: {
    // overcast stratus: low, thick and dark
    cloudCover: 0.85, cloudAltitude: 900, cloudThickness: 1400, cloudDensity: 0.05,
    cloudAbsorption: 0.55, cloudSoftness: 0.4, cloudAnvil: 1, cloudShadow: 0.9,
    wavelength: 3.5,
    amplitude: 0.05,
    choppiness: 0.8,
    layers: 4,
    spread: 12,
    dispersion: 0.6,
    lean: 0.2,
    ripple: 0.34,
    rippleScale: 0.3,
    depth: 3,
    sTurbidity: 0.42,
    uwTurbidity: 0.72,
    chlorophyll: 0.95,
    sChlorophyll: 1.1,
    uwCaustics: 0.5,
    sCaustics: 0.45,
    uwFog: 1.7,
    particleDensity: 1.1,
    foam: 0.42,
    foamLife: 2,
    streaks: 0.85,
    surgeRate: 1.8,
    persistence: 0.6,
    shoreWidth: 0.6,
  }, lake: {
    cloudCover: 0.3, cloudScale: 2200, cloudWindSpeed: 3,
    wavelength: 2.5,
    amplitude: 0.025,
    choppiness: 0.5,
    layers: 3,
    spread: 55,
    dispersion: 0.4,
    lean: 0.1,
    ripple: 0.4,
    rippleScale: 0.22,
    depth: 14,
    sTurbidity: 0.2,
    uwTurbidity: 0.34,
    chlorophyll: 0.62,
    sChlorophyll: 0.9,
    uwCaustics: 0.35,
    sCaustics: 0.3,
    uwFog: 1.25,
    particleDensity: 0.6,
    foam: 0.06,
    foamLife: 1.5,
    streaks: 0.15,
    surgeRate: 0.4,
    shoreWidth: 0.45,
    lapOvershoot: 0.2,
  }, lagoon: {
    // high thin cirrus
    cloudCover: 0.22, cloudAltitude: 5200, cloudThickness: 400, cloudDensity: 0.012,
    cloudDetail: 0.9, cloudSoftness: 0.35, cloudAnvil: 0.8, cloudShadow: 0.3,
    wavelength: 5,
    amplitude: 0.07,
    choppiness: 1,
    layers: 4,
    spread: 30,
    dispersion: 0.8,
    lean: 0.25,
    ripple: 0.26,
    depth: 4.5,
    sTurbidity: 0.04,
    uwTurbidity: 0.1,
    chlorophyll: 0.12,
    sChlorophyll: 0.75,
    uwCaustics: 2.1,
    sCaustics: 1.7,
    uwFog: 0.55,
    particleDensity: 0.12,
    foam: 0.34,
    foamLife: 3,
    streaks: 0.3,
    surgeRate: 0.8,
    shoreWidth: 1.3,
    lapOvershoot: 0.9,
  },
}

const PANEL_KEY = 'gpuocean.panel'

export function setupUI () {
  const params = {}
  const inputs = {}
  const root = document.getElementById('controls')

  // Collapsed state persists: with this many controls, re-opening the sections
  // you were working in after every reload is the annoying part
  let openState = {}
  try {
    openState = JSON.parse(localStorage.getItem(PANEL_KEY) ?? '{}')
  } catch {
  }

  for (const { group, items, open } of SPEC) {
    const section = document.createElement('details')
    section.className = 'group'
    section.open = openState[group] ?? open ?? false
    const summary = document.createElement('summary')
    summary.textContent = group
    section.appendChild(summary)
    for (const item of items) section.appendChild(buildControl(item, params, inputs))
    section.addEventListener('toggle', () => {
      openState[group] = section.open
      try {
        localStorage.setItem(PANEL_KEY, JSON.stringify(openState))
      } catch {
      }
    })
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

function buildControl (item, params, inputs) {
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
    select.addEventListener('change', () => {
      params[item.id] = select.value
    })
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
      box.addEventListener('change', () => {
        params[item.id] = box.checked
      })
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
export function setupFPS () {
  const el = document.getElementById('fps')
  let avg = 0
  let since = 0
  return (dt, gpu) => {
    if (dt <= 0) return
    avg = avg === 0 ? 1 / dt : avg + (1 / dt - avg) * 0.05
    // The average tracks every frame, but the text is only rewritten a few
    // times a second: repainting a digit 120 times a second just flickers
    since += dt
    if (since < 0.25) return
    since = 0
    // fps is vsync-capped; the GPU figure is what actually reflects cost
    el.textContent = gpu ? `${avg.toFixed(0)} fps \u00b7 ${gpu}` : `${avg.toFixed(0)} fps`
  }
}
