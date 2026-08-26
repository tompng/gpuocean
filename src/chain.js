import { bilinearWrap } from './noise.js'

// Heightless swash film: per alongshore column, a horizontal chain of
// material nodes whose rest state is exactly the still-water wedge between
// the REST_DEPTH isobath and the static shoreline. Forces use the Lagrangian
// shallow-water pressure form: each segment's water column is its conserved
// rest volume over its current length, and nodes feel -g * d(eta)/dx with
// eta = terrain + column + shock viscosity. A linear spring law is NOT
// usable here: with a linearly preloaded chain every uniform spacing
// balances the interior, so perturbations settle into piled-up states;
// the pressure form pins the equilibrium to "surface flat" uniquely.
// Rendering stays heightless — the sim contributes displacement only.
// The seaward node is driven by the local sea level riding the beach
// slope; the last node is the waterline tip.
const COLS = 256
const NODES = 64
const SUBSTEPS = 4
// Columns 0..MAIN_COLS-1 follow the mainland coast (z in ±80); the rest
// loop around the island. Must match wave_common.wgsl.
const MAIN_COLS = 160
const ISLAND_COLS = 96
// The scene's fixed beach slope and mainland curve
export const SLOPE = 0.15
const GRAVITY = 9.81
const FRICTION = 0.3
const VISC_Q = 0.25
// caps keep shock transients CFL-stable on centimeter-scale segments
const A_CAP = 25
const U_CAP = 6
const Q_CAP = 0.5
const MAX_DRIVE_SPEED = 5
// junction depth: waves hand over to the film at this isobath, and the film
// thickness runs from this value at the junction to zero at the tip
export const REST_DEPTH = 0.25

export class ChainSim {
  constructor(device, coast) {
    this.device = device
    this.coast = coast
    // the layout src/surface.js walks when it inverts the film mapping
    this.nodes = NODES
    this.cols = COLS
    this.mainCols = MAIN_COLS
    this.sample4 = new Float32Array(4)
    this.x = new Float32Array(COLS * NODES)
    this.u = new Float32Array(COLS * NODES)
    this.vol = new Float32Array(NODES - 1)
    this.eta = new Float32Array(NODES - 1)
    this.drive = new Float32Array(COLS)
    this.prevXi = new Float32Array(COLS)
    this.levelLP = new Float32Array(COLS)
    this.levelLP2 = new Float32Array(COLS)
    this.ve = new Float32Array(COLS)
    this.key = ''
    this.texData = new Float32Array(NODES * COLS * 4)
    this.texture = device.createTexture({
      size: [NODES, COLS],
      format: 'rgba32float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    })
    this.view = this.texture.createView()
    this.coastTexture = device.createTexture({
      size: [COLS, 1],
      format: 'rgba32float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    })
    this.coastView = this.coastTexture.createView()
  }

  // The chain works in s = signed normal distance from the static shoreline
  // (terr = slope * s), identical for every column; only the drive knows the
  // column's world position and landward normal.
  // Mainland columns form a window of fixed width whose center zBase
  // follows the camera alongshore in whole-column steps. The chain state
  // lives in s (coast-relative), so shifting the window is a plain row
  // copy — only each column's world geometry needs recomputing.
  mainlandGeometry(j) {
    const t = this.zBase + (j / (MAIN_COLS - 1) - 0.5) * 160
    const c = this.sample4
    this.coast.sampleMain(t, c)
    this.normal[j * 2] = c[2]
    this.normal[j * 2 + 1] = c[3]
    this.juncWorld[j * 2] = c[0] + c[2] * this.sJ
    this.juncWorld[j * 2 + 1] = c[1] + c[3] * this.sJ
  }

  shiftWindow(steps) {
    const s = Math.max(-MAIN_COLS, Math.min(steps, MAIN_COLS))
    this.zBase += s * (160 / (MAIN_COLS - 1))
    const copyRow = (dst, src) => {
      this.x.copyWithin(dst * NODES, src * NODES, (src + 1) * NODES)
      this.u.copyWithin(dst * NODES, src * NODES, (src + 1) * NODES)
      this.prevXi[dst] = this.prevXi[src]
      this.levelLP[dst] = this.levelLP[src]
      this.levelLP2[dst] = this.levelLP2[src]
    }
    // rows shifted in from beyond the window clone the edge column: a much
    // better initial state than rest, settling within a wave period
    if (s > 0) {
      for (let j = 0; j < MAIN_COLS; j++) copyRow(j, Math.min(j + s, MAIN_COLS - 1))
    } else {
      for (let j = MAIN_COLS - 1; j >= 0; j--) copyRow(j, Math.max(j + s, 0))
    }
    for (let j = 0; j < MAIN_COLS; j++) this.mainlandGeometry(j)
    return s
  }

  reset(params) {
    this.sJ = -REST_DEPTH / SLOPE
    this.Lr = REST_DEPTH / SLOPE / (NODES - 1)
    this.zBase = this.zBase || 0
    for (let k = 0; k < NODES - 1; k++) {
      this.vol[k] = this.Lr * (REST_DEPTH - (k + 0.5) * this.Lr * SLOPE)
    }
    this.juncWorld = new Float32Array(COLS * 2)
    this.normal = new Float32Array(COLS * 2)
    const island = this.coast.island
    this.islandArcStep = island.step
    const coast = new Float32Array(COLS * 4)
    for (let j = 0; j < COLS; j++) {
      if (j < MAIN_COLS) {
        this.mainlandGeometry(j)
      } else {
        const k = j - MAIN_COLS
        const px = island.P[k * 2]
        const pz = island.P[k * 2 + 1]
        const nx = island.N[k * 2]
        const nz = island.N[k * 2 + 1]
        // the coast table only serves the island ribbon; mainland entries stay 0
        coast[j * 4] = px; coast[j * 4 + 1] = pz
        coast[j * 4 + 2] = nx; coast[j * 4 + 3] = nz
        this.juncWorld[j * 2] = px + nx * this.sJ
        this.juncWorld[j * 2 + 1] = pz + nz * this.sJ
        this.normal[j * 2] = nx
        this.normal[j * 2 + 1] = nz
      }
      for (let i = 0; i < NODES; i++) this.x[j * NODES + i] = this.sJ + i * this.Lr
      this.drive[j] = this.sJ
    }
    this.device.queue.writeTexture(
      { texture: this.coastTexture }, coast, { bytesPerRow: COLS * 16 }, [COLS, 1])
    this.u.fill(0)
    this.prevXi.fill(0)
    this.levelLP.fill(0)
    this.levelLP2.fill(0)
    this.ve.fill(0)
  }

  update(dt, params, sampleLevel, camX, camZ) {
    const key = `${params.depth}`
    if (key !== this.key) {
      this.key = key
      this.reset(params)
    }
    const tCam = this.coast.nearestMainArc(camX, camZ)
    this.tCamSnap = Math.floor(tCam / 0.4 + 0.5) * 0.4
    this.lastShift = 0
    if (dt > 0) {
      const steps = Math.round((tCam - this.zBase) / (160 / (MAIN_COLS - 1)))
      if (steps !== 0) this.lastShift = this.shiftWindow(steps)
      // The junction is positionally driven, so the chain's own inertia
      // never filters the drive; the level's short-period components get
      // an explicit low-pass instead. Two cascaded poles, not one: ve below
      // differentiates the result, and after a single pole that derivative
      // is flat in frequency, feeding every short-wave chop into the film
      // as full-strength velocity
      const kLP = 1 - Math.exp(-dt / LEVEL_TAU)
      for (let j = 0; j < COLS; j++) {
        const level = sampleLevel(this.juncWorld[j * 2], this.juncWorld[j * 2 + 1])
        this.levelLP[j] += (level - this.levelLP[j]) * kLP
        this.levelLP2[j] += (this.levelLP[j] - this.levelLP2[j]) * kLP
        const xi = this.levelLP2[j] * params.swashDrive
        this.drive[j] = this.sJ + xi
        this.ve[j] = Math.max(-MAX_DRIVE_SPEED, Math.min((xi - this.prevXi[j]) / dt, MAX_DRIVE_SPEED))
        this.prevXi[j] = xi
      }
      const sub = Math.min(dt, 0.04) / SUBSTEPS
      for (let s = 0; s < SUBSTEPS; s++) {
        for (let j = 0; j < COLS; j++) this.stepColumn(j, sub, params)
      }
      for (let i = 0; i < NODES; i++) {
        smoothSeg(this.x, i, NODES, 0, MAIN_COLS, false, 0.04)
        smoothSeg(this.u, i, NODES, 0, MAIN_COLS, false, 0.04)
        smoothSeg(this.x, i, NODES, MAIN_COLS, COLS, true, 0.04)
        smoothSeg(this.u, i, NODES, MAIN_COLS, COLS, true, 0.04)
      }
    }

    for (let j = 0; j < COLS; j++) {
      const base = j * NODES
      const tip = this.x[base + NODES - 1]
      for (let i = 0; i < NODES; i++) {
        const o = (base + i) * 4
        this.texData[o] = this.x[base + i] - (this.sJ + i * this.Lr)
        this.texData[o + 1] = this.u[base + i]
        this.texData[o + 2] = tip
        this.texData[o + 3] = 0
      }
    }
    this.device.queue.writeTexture(
      { texture: this.texture }, this.texData, { bytesPerRow: NODES * 16, rowsPerImage: COLS }, [NODES, COLS])
  }

  stepColumn(j, sub, params) {
    const base = j * NODES
    const x = this.x
    const u = this.u
    const eta = this.eta
    const terr = s => Math.min(Math.max(SLOPE * s, -params.depth), 3)
    const lFloor = 0.4 * this.Lr
    for (let k = 0; k < NODES - 1; k++) {
      const L = Math.max(x[base + k + 1] - x[base + k], lFloor)
      const du = u[base + k + 1] - u[base + k]
      const q = du < 0 ? Math.min(VISC_Q * du * du, Q_CAP) : 0
      eta[k] = terr((x[base + k] + x[base + k + 1]) / 2) + this.vol[k] / L + q
    }
    for (let i = 1; i < NODES; i++) {
      const etaR = i < NODES - 1 ? eta[i] : terr(x[base + NODES - 1])
      // eta values sit at segment midpoints; the tip's own terrain is at the
      // tip itself, half a segment away — using the full segment length
      // would halve the tip's gravity and let it outrun the chain
      const dx = Math.max((i < NODES - 1 ? x[base + i + 1] - x[base + i - 1] : x[base + i] - x[base + i - 1]) / 2, this.Lr)
      let a = -GRAVITY * (etaR - eta[i - 1]) / dx
      a = Math.max(-A_CAP, Math.min(a, A_CAP))
      const fr = FRICTION * (1 + 3 * i / (NODES - 1))
      u[base + i] += (a - fr * u[base + i]) * sub
      u[base + i] = Math.max(-U_CAP, Math.min(u[base + i], U_CAP))
    }
    x[base] = this.drive[j]
    u[base] = this.ve[j]
    for (let i = 1; i < NODES; i++) x[base + i] += u[base + i] * sub
    const lMin = this.Lr * 0.2
    for (let i = 1; i < NODES; i++) {
      if (x[base + i] < x[base + i - 1] + lMin) {
        x[base + i] = x[base + i - 1] + lMin
        if (u[base + i] < u[base + i - 1]) u[base + i] = u[base + i - 1]
      }
    }
    const xMax = Math.min(13, 2.8 / SLOPE)
    if (x[base + NODES - 1] > xMax) {
      x[base + NODES - 1] = xMax
      if (u[base + NODES - 1] > 0) u[base + NODES - 1] = 0
    }
  }
}

// Alongshore smoothing per coast segment: the mainland is open (ends
// untouched), the island loop wraps
function smoothSeg(a, offset, stride, j0, j1, wrap, k) {
  const n = j1 - j0
  for (let j = wrap ? 0 : 1; j < (wrap ? n : n - 1); j++) {
    const o = offset + (j0 + j) * stride
    const prev = offset + (j0 + (j + n - 1) % n) * stride
    const next = offset + (j0 + (j + 1) % n) * stride
    a[o] += k * (a[prev] + a[next] - 2 * a[o])
  }
}

// The drive is the quasi-static shoreline response: the waterline rides
// the local sea LEVEL up the slope regardless of wave direction. The
// orbital displacement is deliberately NOT part of it — it is ~90 degrees
// ahead of the level and unfiltered it yanks the junction seaward at
// every large crest-to-trough flip, over-stretching the film.
// Per-stage time constant of the two-pole cascade; 0.64 puts the cascade's
// -3 dB point where the old single pole's tau = 1.0 had it
const LEVEL_TAU = 0.64

export function sampleWaveLevel(x, z, noise, waveField, layers, weights) {
  const size = noise.size
  const copies = waveField.data
  let hsum = 0
  for (const [i, l] of layers.entries()) {
    const u0 = (x * l.dx + z * l.dz) * l.invL + l.su
    const v0 = (-x * l.dz + z * l.dx) * l.invL + l.sv
    let sh = 0
    for (let k = 0; k < noise.heights.length; k++) {
      sh += copies[k * 4 + 2] * bilinearWrap(noise.heights[k], size, u0 + copies[k * 4], v0 + copies[k * 4 + 1])
    }
    hsum += l.amp * weights[i] * sh
  }
  return hsum / SLOPE
}
