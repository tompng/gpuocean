import { SLOPE, REST_DEPTH } from './chain.js'
import { COPY_RATIO } from './noise.js'

const GRAVITY = 9.81
const ISLAND_COLS = 96
const MAIN_COLS = 160
// σ/ρ of water [m^3/s^2] for the capillary dispersion c = sqrt(g/k + (σ/ρ)k)
const CAPILLARY_SIGMA_RHO = 7.4e-5
// Warped grid: uniform cells near the center, then exponential growth per
// cell out to beyond the horizon distance seen from the camera's max height
const GRID_N = 512
// Water/land tiling: one TILE_K x TILE_K lattice instanced at power-of-two
// cell sizes. Ring assignment guarantees each tile's cell <= the shader's
// cellForDistance schedule at the tile's nearest point (must match WGSL)
const TILE_K = 16
const TILE_CELLS = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
const MAX_DIST = 8000
const MAX_TILES = 4096
const MAX_WALLS = 4096
const distForCell = c => 32 + (c - 0.25) / 0.08
const CELL = 0.4
const LINEAR_CELLS = 160
const CELL_GROWTH = 1.12
// Five layers x five copies span the old 8-layer component range as a
// fixed geometric comb; a lower layer count truncates the comb from the
// long-wavelength end (the finest bands go first) instead of stretching
// the spacing, so partial counts stay coherent
const LAYER_RATIO = (0.68 ** 7 / COPY_RATIO ** 4) ** 0.25
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

export class Ocean {
  constructor(device, code, waveTexture, capTexture, foamViews, filmFoamViews, foamPattern, simView, coastView, sdfView, mainTableView, format, opts = {}) {
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
    const ribbonVertex = entryPoint => ({
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
    const tileVertex = entryPoint => ({
      module,
      entryPoint,
      buffers: [
        {
          arrayStride: 8,
          attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x2' }],
        },
        {
          arrayStride: 12,
          stepMode: 'instance',
          attributes: [{ shaderLocation: 1, offset: 0, format: 'float32x3' }],
        },
      ],
    })
    // Back faces only exist where choppiness folds the surface over; culling
    // them lets the front-facing sheets win instead of showing the inverted
    // flap (the crease itself sits in the foam-generating region anyway)
    const makePipeline = (vsEntry, fsEntry, topology, vtx) => device.createRenderPipeline({
      ...base,
      vertex: vtx(vsEntry),
      fragment: { module, entryPoint: fsEntry, targets: [{ format }] },
      primitive: topology === 'triangle-list' ? { topology, cullMode: 'back' } : { topology },
    })
    this.fillGridPipeline = makePipeline('vs_grid', 'fs', 'triangle-list', tileVertex)
    this.fillRibbonPipeline = makePipeline('vs', 'fs', 'triangle-list', ribbonVertex)
    this.wireGridPipeline = makePipeline('vs_grid', 'fs_wire', 'line-list', tileVertex)
    this.wireRibbonPipeline = makePipeline('vs', 'fs_wire', 'line-list', ribbonVertex)
    this.fillIslandPipeline = makePipeline('vs_island', 'fs', 'triangle-list', ribbonVertex)
    this.wireIslandPipeline = makePipeline('vs_island', 'fs_wire', 'line-list', ribbonVertex)
    this.fillLandPipeline = makePipeline('vs_land', 'fs_land', 'triangle-list', tileVertex)
    this.wireLandPipeline = makePipeline('vs_land', 'fs_wire', 'line-list', tileVertex)
    this.bindLayout = bindLayout
    this.filmGroups = filmFoamViews.map(view => device.createBindGroup({
      layout: filmLayout,
      entries: [{ binding: 0, resource: view }],
    }))

    this.uniform = device.createBuffer({
      size: 672,
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
        { binding: 7, resource: simView },
        { binding: 8, resource: coastView },
        { binding: 9, resource: sdfView },
        { binding: 10, resource: mainTableView },
      ],
    }))

    const [tileTri, tileLine] = buildIndices(TILE_K, TILE_K)
    this.tileTriIndices = createIndexBuffer(device, tileTri)
    this.tileLineIndices = createIndexBuffer(device, tileLine)
    this.tileTriCount = tileTri.length
    this.tileLineCount = tileLine.length
    this.tileVertexBuffer = createVertexBuffer(device, buildTileVertices(TILE_K))
    this.waterInst = device.createBuffer({
      size: MAX_TILES * 12,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    })
    this.landInst = device.createBuffer({
      size: MAX_TILES * 12,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    })
    this.waterInstData = new Float32Array(MAX_TILES * 3)
    this.landInstData = new Float32Array(MAX_TILES * 3)
    // level-boundary walls: crack faces either way, so no culling
    const wallVertex = entryPoint => ({
      module,
      entryPoint,
      buffers: [
        {
          arrayStride: 4,
          attributes: [{ shaderLocation: 0, offset: 0, format: 'float32' }],
        },
        {
          arrayStride: 16,
          stepMode: 'instance',
          attributes: [{ shaderLocation: 1, offset: 0, format: 'float32x4' }],
        },
      ],
    })
    this.fillWallPipeline = device.createRenderPipeline({
      ...base,
      vertex: wallVertex('vs_wall'),
      fragment: { module, entryPoint: 'fs', targets: [{ format }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
    })
    this.wallVertexBuffer = createVertexBuffer(device, buildWallVertices(TILE_K))
    this.wallVertCount = (TILE_K / 2) * 3
    this.wallInst = device.createBuffer({
      size: MAX_WALLS * 16,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    })
    this.wallInstData = new Float32Array(MAX_WALLS * 4)

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
    this.uniformData = new Float32Array(168)
    this.layerCache = []
  }

  render(pass, dt, params, noise, capNoise, viewProj, eye, sunDir, foamIndex, filmIndex) {
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
    const ratio = LAYER_RATIO
    let sq = 0
    for (let i = 0; i < count; i++) sq += ratio ** (2 * i)
    // amp_i ∝ λ_i keeps per-layer steepness constant; total variance = amplitude^2
    const ampNorm = params.amplitude / Math.sqrt(sq)
    let meanX = 0
    let meanZ = 0
    for (let i = 0; i < count; i++) {
      const lambda = params.wavelength * ratio ** i
      const tile = lambda * noise.wavesPerTile
      this.phases[i] += Math.sqrt(GRAVITY * lambda / (2 * Math.PI)) / tile * dt
      const angle = params.waveDir * Math.PI / 180 + DIR_FRACS[i] * spread
      meanX += ratio ** (2 * i) * Math.cos(angle)
      meanZ += ratio ** (2 * i) * Math.sin(angle)
      const o = 32 + i * 8
      u[o] = Math.cos(angle)
      u[o + 1] = Math.sin(angle)
      u[o + 2] = 1 / tile
      u[o + 3] = ampNorm * ratio ** i
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
    u[149] = params.caustics
    // caustic web cells scale with the ripple wavelength; 0.6 is the tuned default
    u[150] = params.rippleScale / 0.6
    const meanLen = Math.hypot(meanX, meanZ) || 1
    u[151] = params.lean * meanX / meanLen
    u[152] = params.lean * meanZ / meanLen
    u[153] = params.foam
    u[154] = FOAM_REGION
    u[155] = Math.exp(-dt / params.foamLife)
    u[156] = Math.exp(-dt / (params.foamLife * 0.25))
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
    u[160] = Math.exp(-dt / 0.5)
    u[161] = Math.min(dt, 0.033)
    u[162] = 2 * Math.PI / params.wavelength
    u[164] = params.foamScale
    this.device.queue.writeBuffer(this.uniform, 0, u)

    const [waterCount, landCount, wallCount] = this.updateTileInstances(eye, viewProj, params)
    const wire = params.wireframe
    pass.setBindGroup(0, this.bindGroups[foamIndex])
    pass.setBindGroup(1, this.filmGroups[filmIndex])
    pass.setPipeline(wire ? this.wireGridPipeline : this.fillGridPipeline)
    pass.setVertexBuffer(0, this.tileVertexBuffer)
    pass.setVertexBuffer(1, this.waterInst)
    pass.setIndexBuffer(wire ? this.tileLineIndices : this.tileTriIndices, 'uint32')
    pass.drawIndexed(wire ? this.tileLineCount : this.tileTriCount, waterCount)
    pass.setPipeline(wire ? this.wireRibbonPipeline : this.fillRibbonPipeline)
    pass.setVertexBuffer(0, this.ribbonVertexBuffer)
    pass.setVertexBuffer(1, null)
    pass.setIndexBuffer(wire ? this.ribbonLineIndices : this.ribbonTriIndices, 'uint32')
    pass.drawIndexed(wire ? this.ribbonLineCount : this.ribbonTriCount)
    pass.setPipeline(wire ? this.wireIslandPipeline : this.fillIslandPipeline)
    pass.setVertexBuffer(0, this.islandVertexBuffer)
    pass.setIndexBuffer(wire ? this.islandLineIndices : this.islandTriIndices, 'uint32')
    pass.drawIndexed(wire ? this.islandLineCount : this.islandTriCount)
    pass.setPipeline(wire ? this.wireLandPipeline : this.fillLandPipeline)
    pass.setVertexBuffer(0, this.tileVertexBuffer)
    pass.setVertexBuffer(1, this.landInst)
    pass.setIndexBuffer(wire ? this.tileLineIndices : this.tileTriIndices, 'uint32')
    pass.drawIndexed(wire ? this.tileLineCount : this.tileTriCount, landCount)
    if (!wire && wallCount > 0) {
      pass.setPipeline(this.fillWallPipeline)
      pass.setVertexBuffer(0, this.wallVertexBuffer)
      pass.setVertexBuffer(1, this.wallInst)
      pass.draw(this.wallVertCount, wallCount)
    }
  }

  // World-anchored tile rings: level L covers the square annulus between
  // the finer levels' extent and the distance where the schedule allows
  // the next cell size; boundaries align to the coarser level's lattice so
  // rings partition the plane exactly (no overlap, no gap). Each tile is
  // frustum-tested and classified by the coast SDF: fully-land tiles skip
  // the water set (every fragment would be cut), deep-sea tiles skip the
  // land set (terrain can never rise above any visible water surface).
  updateTileInstances(eye, viewProj, params) {
    const planes = frustumPlanes(viewProj)
    const pad = 4 * params.amplitude + 2
    const yMin = -params.depth - 3
    const yMax = 3.5 + 4 * params.amplitude
    const cutSDF = -(REST_DEPTH / SLOPE + 4)
    const landSDF = -(4 * params.amplitude + 1) / SLOPE
    const sdf = this.coast.sampleSDF
    let water = 0
    let land = 0
    let walls = 0
    let inner = null
    for (let l = 0; l < TILE_CELLS.length; l++) {
      const cell = TILE_CELLS[l]
      const span = TILE_K * cell
      const isLast = l === TILE_CELLS.length - 1
      const outerD = isLast ? MAX_DIST : distForCell(TILE_CELLS[l + 1])
      const align = isLast ? span : span * 2
      const x0 = Math.floor((eye[0] - outerD) / align) * align
      const z0 = Math.floor((eye[2] - outerD) / align) * align
      const x1 = Math.ceil((eye[0] + outerD) / align) * align
      const z1 = Math.ceil((eye[2] + outerD) / align) * align
      for (let tz = z0; tz < z1; tz += span) {
        for (let tx = x0; tx < x1; tx += span) {
          if (inner && tx >= inner[0] && tx + span <= inner[2] && tz >= inner[1] && tz + span <= inner[3]) continue
          if (!boxVisible(planes, tx - pad, yMin, tz - pad, tx + span + pad, yMax, tz + span + pad)) continue
          let sMin = Infinity
          let sMax = -Infinity
          for (const [px, pz] of [[tx, tz], [tx + span, tz], [tx, tz + span], [tx + span, tz + span], [tx + span / 2, tz + span / 2]]) {
            const v = sdf(px, pz)
            if (v < sMin) sMin = v
            if (v > sMax) sMax = v
          }
          // the SDF is 1-Lipschitz: five samples bound the tile's range
          // within half a span
          if (sMin - span / 2 <= cutSDF && water < MAX_TILES) {
            this.waterInstData.set([tx / span, tz / span, cell], water * 3)
            water++
          }
          if (sMax + span / 2 >= landSDF && land < MAX_TILES) {
            this.landInstData.set([tx / span, tz / span, cell], land * 3)
            land++
          }
        }
      }
      // walls close the T-junction cracks along this level's outer
      // boundary, where the next (coarser) level takes over
      if (!isLast) {
        for (let x = x0; x < x1; x += span) {
          walls = this.pushWall(walls, x, z0, span, cell, 0, planes, pad, yMin, yMax, cutSDF, sdf)
          walls = this.pushWall(walls, x, z1, span, cell, 0, planes, pad, yMin, yMax, cutSDF, sdf)
        }
        for (let z = z0; z < z1; z += span) {
          walls = this.pushWall(walls, x0, z, span, cell, 1, planes, pad, yMin, yMax, cutSDF, sdf)
          walls = this.pushWall(walls, x1, z, span, cell, 1, planes, pad, yMin, yMax, cutSDF, sdf)
        }
      }
      inner = [x0, z0, x1, z1]
    }
    this.device.queue.writeBuffer(this.waterInst, 0, this.waterInstData, 0, water * 3)
    this.device.queue.writeBuffer(this.landInst, 0, this.landInstData, 0, land * 3)
    this.device.queue.writeBuffer(this.wallInst, 0, this.wallInstData, 0, walls * 4)
    return [water, land, walls]
  }

  pushWall(count, x, z, span, cell, axis, planes, pad, yMin, yMax, cutSDF, sdf) {
    if (count >= MAX_WALLS) return count
    const x1 = axis === 0 ? x + span : x
    const z1 = axis === 0 ? z : z + span
    if (!boxVisible(planes, Math.min(x, x1) - pad, yMin, Math.min(z, z1) - pad, Math.max(x, x1) + pad, yMax, Math.max(z, z1) + pad)) return count
    const sMin = Math.min(sdf(x, z), sdf(x1, z1), sdf((x + x1) / 2, (z + z1) / 2))
    if (sMin - span / 2 > cutSDF) return count
    this.wallInstData.set([x / cell, z / cell, cell, axis], count * 4)
    return count + 1
  }
}

function warpAxis(i) {
  const a = Math.abs(i)
  const sign = Math.sign(i)
  if (a <= LINEAR_CELLS) return sign * a * CELL
  return sign * (LINEAR_CELLS * CELL + CELL * (CELL_GROWTH ** (a - LINEAR_CELLS) - 1) / (CELL_GROWTH - 1))
}

// Local lattice indices of one tile; world position comes from the
// instance's integer tile index times the cell size, float-exact for
// power-of-two cells, so shared tile edges match bitwise
function buildTileVertices(k) {
  const data = new Float32Array((k + 1) * (k + 1) * 2)
  let p = 0
  for (let iz = 0; iz <= k; iz++) {
    for (let ix = 0; ix <= k; ix++) {
      data[p++] = ix
      data[p++] = iz
    }
  }
  return data
}

// One crack triangle per fine vertex pair along a tile edge: positions
// are the along-edge lattice offsets (even, odd, even)
function buildWallVertices(k) {
  const data = new Float32Array((k / 2) * 3)
  let p = 0
  for (let j = 0; j < k / 2; j++) {
    data[p++] = 2 * j
    data[p++] = 2 * j + 1
    data[p++] = 2 * j + 2
  }
  return data
}

// Gribb-Hartmann planes from the column-major viewProj (WebGPU z in [0,1])
function frustumPlanes(m) {
  const row = i => [m[i], m[4 + i], m[8 + i], m[12 + i]]
  const [r0, r1, r2, r3] = [row(0), row(1), row(2), row(3)]
  const add = (a, b, sg) => [a[0] + sg * b[0], a[1] + sg * b[1], a[2] + sg * b[2], a[3] + sg * b[3]]
  return [add(r3, r0, 1), add(r3, r0, -1), add(r3, r1, 1), add(r3, r1, -1), r2, add(r3, r2, -1)]
}

function boxVisible(planes, x0, y0, z0, x1, y1, z1) {
  for (const [a, b, c, d] of planes) {
    const px = a > 0 ? x1 : x0
    const py = b > 0 ? y1 : y0
    const pz = c > 0 ? z1 : z0
    if (a * px + b * py + c * pz + d < 0) return false
  }
  return true
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
