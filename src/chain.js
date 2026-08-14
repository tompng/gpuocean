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
// The seaward node is driven by the wave's horizontal orbital displacement
// at the junction; the last node is the waterline tip.
const COLS = 256
const NODES = 64
const SUBSTEPS = 4
// Columns 0..MAIN_COLS-1 follow the mainland coast (z in ±80); the rest
// loop around the island. Must match wave_common.wgsl.
const MAIN_COLS = 160
const ISLAND_COLS = 96
const ISLAND_C = [-45, 15]
const ISLAND_R = 20
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
  constructor(device) {
    this.device = device
    this.x = new Float32Array(COLS * NODES)
    this.u = new Float32Array(COLS * NODES)
    this.vol = new Float32Array(NODES - 1)
    this.eta = new Float32Array(NODES - 1)
    this.drive = new Float32Array(COLS)
    this.prevXi = new Float32Array(COLS)
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
  reset(params) {
    this.sJ = -REST_DEPTH / SLOPE
    this.Lr = REST_DEPTH / SLOPE / (NODES - 1)
    for (let k = 0; k < NODES - 1; k++) {
      this.vol[k] = this.Lr * (REST_DEPTH - (k + 0.5) * this.Lr * SLOPE)
    }
    this.juncWorld = new Float32Array(COLS * 2)
    this.normal = new Float32Array(COLS * 2)
    const coast = new Float32Array(COLS * 4)
    for (let j = 0; j < COLS; j++) {
      let px, pz, nx, nz
      if (j < MAIN_COLS) {
        const z = (j / (MAIN_COLS - 1) - 0.5) * 160
        const d = dShoreXAt(z)
        const inv = 1 / Math.sqrt(1 + d * d)
        px = shoreXAt(z); pz = z
        nx = inv; nz = -d * inv
      } else {
        const th = (j - MAIN_COLS) / ISLAND_COLS * 2 * Math.PI
        const r = ISLAND_R + 3 * Math.sin(3 * th + 1)
        const dr = 9 * Math.cos(3 * th + 1)
        px = ISLAND_C[0] + Math.cos(th) * r
        pz = ISLAND_C[1] + Math.sin(th) * r
        // landward = inward normal of the polar curve r = R(theta)
        const ox = Math.cos(th) * r - -Math.sin(th) * dr
        const oz = Math.sin(th) * r - Math.cos(th) * dr
        const oi = 1 / Math.hypot(ox, oz)
        nx = -ox * oi; nz = -oz * oi
      }
      coast[j * 4] = px; coast[j * 4 + 1] = pz
      coast[j * 4 + 2] = nx; coast[j * 4 + 3] = nz
      this.juncWorld[j * 2] = px + nx * this.sJ
      this.juncWorld[j * 2 + 1] = pz + nz * this.sJ
      this.normal[j * 2] = nx
      this.normal[j * 2 + 1] = nz
      for (let i = 0; i < NODES; i++) this.x[j * NODES + i] = this.sJ + i * this.Lr
      this.drive[j] = this.sJ
    }
    this.device.queue.writeTexture(
      { texture: this.coastTexture }, coast, { bytesPerRow: COLS * 16 }, [COLS, 1])
    this.u.fill(0)
    this.prevXi.fill(0)
    this.ve.fill(0)
  }

  update(dt, params, sampleDispN) {
    const key = `${params.depth}`
    if (key !== this.key) {
      this.key = key
      this.reset(params)
    }
    if (dt > 0) {
      for (let j = 0; j < COLS; j++) {
        const xi = sampleDispN(this.juncWorld[j * 2], this.juncWorld[j * 2 + 1], this.normal[j * 2], this.normal[j * 2 + 1])
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

// must match shoreX / dShoreX in wave_common.wgsl
function shoreXAt(z) {
  return 10 + 0.6 * (6 * Math.sin(z * 0.041) + 3.5 * Math.sin(z * 0.093 + 1.7))
}

function dShoreXAt(z) {
  return 0.6 * (6 * 0.041 * Math.cos(z * 0.041) + 3.5 * 0.093 * Math.cos(z * 0.093 + 1.7))
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

// CPU replica of the layered horizontal wave displacement, projected onto
// the column's landward normal, with the same shallow amplification and
// waterline fade the vertex shader applies. The sample point is always the
// junction, whose depth is REST_DEPTH by construction, so those factors
// are constants of the sample depth.
export function sampleWaveDispN(x, z, nx, nz, noise, waveField, layers, chop, k0) {
  const tex = noise.channels.disp
  const size = noise.size
  const copies = waveField.data
  let dsum = 0
  for (const l of layers) {
    const u0 = (x * l.dx + z * l.dz) * l.invL + l.su
    const v0 = (-x * l.dz + z * l.dx) * l.invL + l.sv
    let s = 0
    for (let k = 0; k < 3; k++) {
      s += copies[k * 4 + 2] * bilinearWrap(tex, size, u0 + copies[k * 4], v0 + copies[k * 4 + 1])
    }
    dsum += chop * l.amp * (l.dx * nx + l.dz * nz) * s
  }
  const amp = Math.min(Math.max(1 / Math.tanh(k0 * REST_DEPTH), 1), 2.5)
  const t = Math.min(Math.max((-REST_DEPTH + 0.6) / 0.7, 0), 1)
  const wSea = 1 - t * t * (3 - 2 * t)
  return dsum * amp * wSea
}

function bilinearWrap(tex, size, u, v) {
  const x = (u - Math.floor(u)) * size
  const y = (v - Math.floor(v)) * size
  const x0 = Math.floor(x) % size
  const y0 = Math.floor(y) % size
  const x1 = (x0 + 1) % size
  const y1 = (y0 + 1) % size
  const fx = x - Math.floor(x)
  const fy = y - Math.floor(y)
  const a = tex[y0 * size + x0] * (1 - fx) + tex[y0 * size + x1] * fx
  const b = tex[y1 * size + x0] * (1 - fx) + tex[y1 * size + x1] * fx
  return a * (1 - fy) + b * fy
}
