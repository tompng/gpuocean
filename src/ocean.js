import { SLOPE } from './chain.js'

const GRAVITY = 9.81
const ISLAND_COLS = 96
const MAIN_COLS = 160
// σ/ρ of water [m^3/s^2] for the capillary dispersion c = sqrt(g/k + (σ/ρ)k)
const CAPILLARY_SIGMA_RHO = 7.4e-5
// Warped grid: uniform cells near the center, then exponential growth per
// cell out to beyond the horizon distance seen from the camera's max height
const GRID_N = 512
const CELL = 0.4
const LINEAR_CELLS = 160
const CELL_GROWTH = 1.12
const SCALE_RATIO = 0.68
const MAX_LAYERS = 8
const DIR_FRACS = [0, 0.9, -0.75, 0.45, -0.35, 0.7, -1, 0.2]
const UV_OFFSETS = [
  [0.11, 0.63], [0.42, 0.17], [0.78, 0.55], [0.05, 0.91],
  [0.33, 0.4], [0.66, 0.08], [0.9, 0.77], [0.24, 0.31],
]
const CAP_ANGLES = [0.4, -0.8, 1.7]
// Anisotropic ripple directions follow the gravity waves (fractions of spread),
// mimicking parasitic capillaries riding their parent waves
const CAP_ANISO_FRACS = [0, 0.45, -0.35]
const CAP_SCALES = [1, 0.72, 0.52]
const CAP_UV_OFFSETS = [
  [0.19, 0.47], [0.61, 0.83], [0.07, 0.29],
  [0.37, 0.71], [0.83, 0.13], [0.53, 0.59],
]
// Half-extent of the foam accumulation buffer around the origin [m]
const FOAM_REGION = 80
// Shore ribbon: vertex x is normalized over the ribbon band, whose material
// width must match SIM_SPAN + SIM_BAND in wave_common.wgsl
const RIBBON_SPAN = 28
const RIBBON_CELLS = 140
// Rise time of foam generation, roughly the crest's texel-crossing time [s]
const FOAM_RISE = 0.08
// Uniforms struct size in floats; must stay a multiple of 4 (16-byte struct
// alignment) and match the tail of Uniforms in wave_common.wgsl
const UNIFORM_FLOATS = 196

export class Ocean {
  constructor(device, code, waveTexture, capTexture, foamViews, filmFoamViews, foamPattern, foamPlates, simView, coastView, sdfView, mainTableView, format, opts = {}) {
    this.device = device
    this.gridN = GRID_N
    const sampleCount = opts.sampleCount ?? 4
    const module = device.createShaderModule({ code })
    // The grid and ribbon pipelines share one explicit layout so a single
    // bind group works across all of them ('auto' layouts are pipeline-bound)
    const bindLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: {} },
        { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 2, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 6, visibility: GPUShaderStage.FRAGMENT, texture: { viewDimension: '2d-array' } },
        { binding: 7, visibility: GPUShaderStage.VERTEX, texture: { sampleType: 'unfilterable-float' } },
        { binding: 8, visibility: GPUShaderStage.VERTEX, texture: { sampleType: 'unfilterable-float' } },
        { binding: 9, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 10, visibility: GPUShaderStage.VERTEX, texture: { sampleType: 'unfilterable-float' } },
      ],
    })
    const filmLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
      ],
    })
    const base = {
      layout: device.createPipelineLayout({ bindGroupLayouts: [bindLayout, filmLayout] }),
      multisample: { count: sampleCount },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    }
    const vertex = entryPoint => ({
      module,
      entryPoint,
      buffers: [{
        arrayStride: 12,
        attributes: [
          { shaderLocation: 0, offset: 0, format: 'float32x2' },
          { shaderLocation: 1, offset: 8, format: 'float32' },
        ],
      }],
    })
    const makePipeline = (vsEntry, fsEntry, topology) => device.createRenderPipeline({
      ...base,
      vertex: vertex(vsEntry),
      fragment: { module, entryPoint: fsEntry, targets: [{ format }] },
      primitive: { topology },
    })
    this.fillGridPipeline = makePipeline('vs_grid', 'fs', 'triangle-list')
    this.fillRibbonPipeline = makePipeline('vs', 'fs', 'triangle-list')
    this.wireGridPipeline = makePipeline('vs_grid', 'fs_wire', 'line-list')
    this.wireRibbonPipeline = makePipeline('vs', 'fs_wire', 'line-list')
    this.fillIslandPipeline = makePipeline('vs_island', 'fs', 'triangle-list')
    this.wireIslandPipeline = makePipeline('vs_island', 'fs_wire', 'line-list')
    this.fillLandPipeline = makePipeline('vs_land', 'fs_land', 'triangle-list')
    this.wireLandPipeline = makePipeline('vs_land', 'fs_wire', 'line-list')
    this.bindLayout = bindLayout
    this.filmGroups = filmFoamViews.map(view => device.createBindGroup({
      layout: filmLayout,
      entries: [{ binding: 0, resource: view }],
    }))

    this.uniform = device.createBuffer({
      size: UNIFORM_FLOATS * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    const sampler = device.createSampler({
      addressModeU: 'repeat',
      addressModeV: 'repeat',
      magFilter: 'linear',
      minFilter: 'linear',
      mipmapFilter: 'linear',
    })
    const entries = [
      { binding: 0, resource: { buffer: this.uniform } },
      { binding: 1, resource: sampler },
      { binding: 2, resource: waveTexture.createView() },
      { binding: 3, resource: capTexture.createView() },
    ]
    // One bind group per foam ping-pong texture
    const patternView = foamPattern.texture.createView()
    this.bindGroups = foamViews.map(view => device.createBindGroup({
      layout: this.bindLayout,
      entries: [
        ...entries,
        { binding: 4, resource: view },
        { binding: 5, resource: patternView },
        { binding: 6, resource: foamPlates.texture.createView({ dimension: '2d-array' }) },
        { binding: 7, resource: simView },
        { binding: 8, resource: coastView },
        { binding: 9, resource: sdfView },
        { binding: 10, resource: mainTableView },
      ],
    }))

    const [tri, line] = buildIndices(this.gridN, this.gridN)
    this.triIndices = createIndexBuffer(device, tri)
    this.lineIndices = createIndexBuffer(device, line)
    this.triCount = tri.length
    this.lineCount = line.length
    this.vertexBuffer = createVertexBuffer(device, buildVertices(this.gridN))

    const [ribTri, ribLine] = buildIndices(RIBBON_CELLS, this.gridN)
    this.ribbonTriIndices = createIndexBuffer(device, ribTri)
    this.ribbonLineIndices = createIndexBuffer(device, ribLine)
    this.ribbonTriCount = ribTri.length
    this.ribbonLineCount = ribLine.length
    this.ribbonVertexBuffer = createVertexBuffer(device, buildRibbonVertices(RIBBON_CELLS, this.gridN))

    const [islTri, islLine] = buildIndices(RIBBON_CELLS, ISLAND_COLS * 4)
    this.islandTriIndices = createIndexBuffer(device, islTri)
    this.islandLineIndices = createIndexBuffer(device, islLine)
    this.islandTriCount = islTri.length
    this.islandLineCount = islLine.length
    this.islandVertexBuffer = createVertexBuffer(device, buildIslandVertices(RIBBON_CELLS, ISLAND_COLS))

    this.time = 0
    this.phases = new Float64Array(MAX_LAYERS)
    this.capPhases = new Float64Array(CAP_ANGLES.length + CAP_ANISO_FRACS.length)
    this.uniformData = new Float32Array(UNIFORM_FLOATS)
    this.layerCache = []
  }

  render(pass, dt, params, noise, capNoise, viewProj, eye, sunDir, foamIndex, filmIndex, camDepth, lodScale) {
    const u = this.uniformData
    u.set(viewProj, 0)
    this.time += dt
    u[16] = eye[0]; u[17] = eye[1]; u[18] = eye[2]; u[19] = this.time
    u[20] = sunDir[0]; u[21] = sunDir[1]; u[22] = sunDir[2]
    const count = Math.round(params.layers)
    u[24] = count
    u[25] = params.choppiness
    u[26] = noise.size * noise.dispGradPerTexel
    u[27] = noise.size

    const spread = params.spread * Math.PI / 180
    let sq = 0
    for (let i = 0; i < count; i++) sq += SCALE_RATIO ** (2 * i)
    // amp_i ∝ λ_i keeps per-layer steepness constant; total variance = amplitude^2
    const ampNorm = params.amplitude / Math.sqrt(sq)
    let meanX = 0
    let meanZ = 0
    for (let i = 0; i < count; i++) {
      const lambda = params.wavelength * SCALE_RATIO ** i
      const tile = lambda * noise.wavesPerTile
      this.phases[i] += Math.sqrt(GRAVITY * lambda / (2 * Math.PI)) / tile * dt
      const angle = params.waveDir * Math.PI / 180 + DIR_FRACS[i] * spread
      meanX += SCALE_RATIO ** (2 * i) * Math.cos(angle)
      meanZ += SCALE_RATIO ** (2 * i) * Math.sin(angle)
      const o = 32 + i * 8
      u[o] = Math.cos(angle)
      u[o + 1] = Math.sin(angle)
      u[o + 2] = 1 / tile
      u[o + 3] = ampNorm * SCALE_RATIO ** i
      u[o + 4] = UV_OFFSETS[i][0] - this.phases[i]
      u[o + 5] = UV_OFFSETS[i][1]
      u[o + 6] = 0
      u[o + 7] = 0
      this.layerCache[i] = { dx: u[o], dz: u[o + 1], invL: u[o + 2], amp: u[o + 3], su: u[o + 4], sv: u[o + 5] }
    }
    this.layerCache.length = count

    // ripple slider is a slope amplitude; height amplitude = slope * λ / 2π
    const capNorm = params.ripple / Math.sqrt(CAP_SCALES.length) / (2 * Math.PI)
    const isoWeight = Math.sqrt(1 - params.rippleAniso)
    const anisoWeight = Math.sqrt(params.rippleAniso)
    for (let i = 0; i < this.capPhases.length; i++) {
      const aniso = i >= CAP_ANGLES.length
      const j = i % CAP_SCALES.length
      const lambda = params.rippleScale * CAP_SCALES[j]
      const tile = lambda * (aniso ? noise : capNoise).wavesPerTile
      const k = 2 * Math.PI / lambda
      this.capPhases[i] += Math.sqrt(GRAVITY / k + CAPILLARY_SIGMA_RHO * k) / tile * dt
      // anisotropic parasitic ripples follow the rotated gravity waves; the
      // isotropic wind ripples keep their own fixed directions
      const angle = aniso ? params.waveDir * Math.PI / 180 + CAP_ANISO_FRACS[j] * spread : CAP_ANGLES[j]
      const o = 96 + i * 8
      u[o] = Math.cos(angle)
      u[o + 1] = Math.sin(angle)
      u[o + 2] = 1 / tile
      u[o + 3] = capNorm * lambda * (aniso ? anisoWeight : isoWeight)
      u[o + 4] = CAP_UV_OFFSETS[i][0] - this.capPhases[i]
      u[o + 5] = CAP_UV_OFFSETS[i][1]
      u[o + 6] = 0
      u[o + 7] = 0
    }
    u[144] = capNoise.size
    u[145] = params.rippleBias
    u[146] = params.sss
    u[147] = 1 / Math.max(params.amplitude, 0.01)
    u[148] = params.depth
    u[149] = params.sCaustics
    // caustic web cells scale with the ripple wavelength; 0.6 is the tuned default
    u[150] = params.rippleScale / 0.6
    const meanLen = Math.hypot(meanX, meanZ) || 1
    u[151] = params.lean * meanX / meanLen
    u[152] = params.lean * meanZ / meanLen
    u[153] = params.foam
    u[154] = FOAM_REGION
    const life = params.foamLife * Math.max(params.persistence, 0.01)
    u[155] = Math.exp(-dt / life)
    u[156] = Math.exp(-dt / (life * 0.25))
    u[157] = Math.exp(-dt / FOAM_RISE)
    u[158] = params.foamLife
    u[159] = SLOPE
    // world foam window follows the camera, snapped to buffer texels so the
    // carried-over content resamples exactly; frozen while paused so the
    // accumulated shift never outruns the skipped foam passes
    const texel = 2 * FOAM_REGION / 512
    if (!this.foamC) this.foamC = [Math.round(eye[0] / texel) * texel, Math.round(eye[2] / texel) * texel]
    let fdx = 0
    let fdz = 0
    if (dt > 0) {
      const cx = Math.round(eye[0] / texel) * texel
      const cz = Math.round(eye[2] / texel) * texel
      fdx = (cx - this.foamC[0]) / (2 * FOAM_REGION)
      fdz = (cz - this.foamC[1]) / (2 * FOAM_REGION)
      this.foamC = [cx, cz]
    }
    u[28] = this.foamC[0]
    u[29] = this.foamC[1]
    u[30] = fdx
    u[31] = fdz
    u[23] = this.chain.islandArcStep
    u[165] = this.chain.zBase
    u[166] = this.chain.lastShift
    u[167] = this.chain.tCamSnap
    u[168] = camDepth
    // waterlineThickness is authored 0..1; as a port radius that is a few cm
    // to half a meter, which is the range over which the split reads as a lens
    u[169] = 0.02 + params.waterlineThickness * 0.5
    u[170] = params.uwTurbidity
    u[171] = params.uwFog
    u[172] = params.uwCaustics
    u[173] = params.distortionStrength
    u[174] = params.distortionScale
    u[175] = params.particleDensity
    u[176] = params.rippleStrength
    u[177] = params.sTurbidity
    u[178] = params.sChlorophyll
    u[179] = params.chlorophyll
    u[180] = lodScale
    u[181] = params.noiseScale
    u[182] = params.noiseSpeed
    // smoothstep(a, a, x) is a divide by zero in WGSL, and both ramps can be
    // collapsed from the sliders, so separate the endpoints here
    u[183] = params.laceLow
    u[184] = Math.max(params.laceHigh, params.laceLow + 1e-3)
    u[185] = params.crestStart
    u[186] = Math.max(params.crestFull, params.crestStart + 1e-3)
    u[187] = params.opacity
    u[188] = params.crestScale
    u[189] = Math.max(params.contactWidth, 1e-3)
    u[190] = params.streaks
    u[191] = params.shoreWidth
    u[192] = params.lapOvershoot
    u[193] = params.surgeRate
    u[160] = Math.exp(-dt / 0.5)
    u[161] = Math.min(dt, 0.033)
    u[162] = 2 * Math.PI / params.wavelength
    u[164] = 1
    this.device.queue.writeBuffer(this.uniform, 0, u)

    const wire = params.wireframe
    pass.setBindGroup(0, this.bindGroups[foamIndex])
    pass.setBindGroup(1, this.filmGroups[filmIndex])
    pass.setPipeline(wire ? this.wireGridPipeline : this.fillGridPipeline)
    pass.setVertexBuffer(0, this.vertexBuffer)
    pass.setIndexBuffer(wire ? this.lineIndices : this.triIndices, 'uint32')
    pass.drawIndexed(wire ? this.lineCount : this.triCount)
    pass.setPipeline(wire ? this.wireRibbonPipeline : this.fillRibbonPipeline)
    pass.setVertexBuffer(0, this.ribbonVertexBuffer)
    pass.setIndexBuffer(wire ? this.ribbonLineIndices : this.ribbonTriIndices, 'uint32')
    pass.drawIndexed(wire ? this.ribbonLineCount : this.ribbonTriCount)
    pass.setPipeline(wire ? this.wireIslandPipeline : this.fillIslandPipeline)
    pass.setVertexBuffer(0, this.islandVertexBuffer)
    pass.setIndexBuffer(wire ? this.islandLineIndices : this.islandTriIndices, 'uint32')
    pass.drawIndexed(wire ? this.islandLineCount : this.islandTriCount)
    pass.setPipeline(wire ? this.wireLandPipeline : this.fillLandPipeline)
    pass.setVertexBuffer(0, this.vertexBuffer)
    pass.setIndexBuffer(wire ? this.lineIndices : this.triIndices, 'uint32')
    pass.drawIndexed(wire ? this.lineCount : this.triCount)
  }
}

function warpAxis(i) {
  const a = Math.abs(i)
  const sign = Math.sign(i)
  if (a <= LINEAR_CELLS) return sign * a * CELL
  return sign * (LINEAR_CELLS * CELL + CELL * (CELL_GROWTH ** (a - LINEAR_CELLS) - 1) / (CELL_GROWTH - 1))
}

// Uniform lattice in pre-warp space; the vertex shader warps it around the
// camera (see warpVertex in ocean.wgsl), so the buffer never changes
function buildVertices(n) {
  const half = n / 2
  const data = new Float32Array((n + 1) * (n + 1) * 3)
  let p = 0
  for (let iz = 0; iz <= n; iz++) {
    for (let ix = 0; ix <= n; ix++) {
      data[p++] = (ix - half) * CELL
      data[p++] = (iz - half) * CELL
      data[p++] = 0
    }
  }
  return data
}

// The ribbon is uniform in normalized x across its band and reuses the
// grid's warped axis along z, so the shoreline stays fine near the camera
// and coarsens toward the horizon
function buildRibbonVertices(nx, nz) {
  const half = nz / 2
  const cellAt = i => {
    const a = Math.min(Math.abs(i - half), half - 1)
    return warpAxis(a + 1) - warpAxis(a)
  }
  const dxMaterial = RIBBON_SPAN / nx
  const data = new Float32Array((nx + 1) * (nz + 1) * 3)
  let p = 0
  for (let iz = 0; iz <= nz; iz++) {
    for (let ix = 0; ix <= nx; ix++) {
      data[p++] = ix / nx
      data[p++] = warpAxis(iz - half)
      data[p++] = Math.max(dxMaterial, cellAt(iz))
    }
  }
  return data
}

// Island ribbon: a closed loop of ISLAND_COLS columns (the last row wraps
// back to the first); pos.y is the chain column coordinate
// The ribbon subdivides alongshore beyond the sim columns (fractional
// column coordinates): the seam against the grid is an interpolation-error
// mismatch between tessellations, so the ribbon's cells must match the
// grid's density — finer than the grid buys nothing, and the sim itself
// can stay coarse.
function buildIslandVertices(nx, cols) {
  const SUB = 4
  const rows = cols * SUB
  const data = new Float32Array((nx + 1) * (rows + 1) * 3)
  let p = 0
  for (let r = 0; r <= rows; r++) {
    for (let ix = 0; ix <= nx; ix++) {
      data[p++] = ix / nx
      data[p++] = MAIN_COLS + r / SUB
      data[p++] = 1.4 / SUB
    }
  }
  return data
}

function createVertexBuffer(device, data) {
  const buffer = device.createBuffer({
    size: data.byteLength,
    usage: GPUBufferUsage.VERTEX,
    mappedAtCreation: true,
  })
  new Float32Array(buffer.getMappedRange()).set(data)
  buffer.unmap()
  return buffer
}

function buildIndices(nx, nz) {
  const tri = new Uint32Array(nx * nz * 6)
  let t = 0
  for (let z = 0; z < nz; z++) {
    for (let x = 0; x < nx; x++) {
      const a = z * (nx + 1) + x
      const b = a + 1
      const c = a + nx + 1
      const d = c + 1
      tri[t++] = a; tri[t++] = c; tri[t++] = b
      tri[t++] = b; tri[t++] = c; tri[t++] = d
    }
  }
  const line = new Uint32Array(2 * (nx * (nz + 1) + nz * (nx + 1)))
  let l = 0
  for (let z = 0; z <= nz; z++) {
    for (let x = 0; x < nx; x++) {
      const a = z * (nx + 1) + x
      line[l++] = a; line[l++] = a + 1
    }
  }
  for (let x = 0; x <= nx; x++) {
    for (let z = 0; z < nz; z++) {
      const a = z * (nx + 1) + x
      line[l++] = a; line[l++] = a + nx + 1
    }
  }
  return [tri, line]
}

function createIndexBuffer(device, data) {
  const buffer = device.createBuffer({
    size: data.byteLength,
    usage: GPUBufferUsage.INDEX,
    mappedAtCreation: true,
  })
  new Uint32Array(buffer.getMappedRange()).set(data)
  buffer.unmap()
  return buffer
}
