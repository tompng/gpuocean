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
const GRAVITY = 9.81
const FRICTION = 0.15
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
    this.hJunc = new Float32Array(COLS)
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
  }

  reset(params) {
    this.xb = params.shore - REST_DEPTH / params.slope
    this.Lr = REST_DEPTH / params.slope / (NODES - 1)
    for (let k = 0; k < NODES - 1; k++) {
      this.vol[k] = this.Lr * (REST_DEPTH - (k + 0.5) * this.Lr * params.slope)
    }
    for (let j = 0; j < COLS; j++) {
      for (let i = 0; i < NODES; i++) this.x[j * NODES + i] = this.xb + i * this.Lr
    }
    this.u.fill(0)
    this.drive.fill(this.xb)
    this.hJunc.fill(REST_DEPTH)
    this.prevXi.fill(0)
    this.ve.fill(0)
  }

  update(dt, params, region, sampleDispX, sampleHeight) {
    const key = `${params.shore}|${params.slope}|${params.depth}`
    if (key !== this.key) {
      this.key = key
      this.reset(params)
    }
    if (dt > 0) {
      const terr = x => Math.min(Math.max(params.slope * (x - params.shore), -params.depth), 3)
      for (let j = 0; j < COLS; j++) {
        const z = (j / (COLS - 1) - 0.5) * 2 * region
        const xi = sampleDispX(this.xb, z)
        this.drive[j] = this.xb + xi
        this.ve[j] = Math.max(-MAX_DRIVE_SPEED, Math.min((xi - this.prevXi[j]) / dt, MAX_DRIVE_SPEED))
        this.prevXi[j] = xi
        // water column at the junction, replicating the shader's attenuation
        // and soft clamp so the film's seaward edge meets the wave surface
        const t = terr(this.drive[j])
        const dy = sampleHeight(this.drive[j], z) * shoreHeightScale(t) - t - 0.01
        this.hJunc[j] = 0.01 + 0.5 * (dy + Math.sqrt(dy * dy + 0.0225))
      }
      const sub = Math.min(dt, 0.04) / SUBSTEPS
      for (let s = 0; s < SUBSTEPS; s++) {
        for (let j = 0; j < COLS; j++) this.stepColumn(j, sub, params)
      }
      for (let i = 0; i < NODES; i++) {
        smoothStrided(this.x, i, NODES, COLS, 0.04)
        smoothStrided(this.u, i, NODES, COLS, 0.04)
      }
    }

    for (let j = 0; j < COLS; j++) {
      const base = j * NODES
      const tip = this.x[base + NODES - 1]
      for (let i = 0; i < NODES; i++) {
        const o = (base + i) * 4
        this.texData[o] = this.x[base + i] - (this.xb + i * this.Lr)
        this.texData[o + 1] = this.u[base + i]
        this.texData[o + 2] = tip
        this.texData[o + 3] = this.hJunc[j]
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
    const terr = xw => Math.min(Math.max(params.slope * (xw - params.shore), -params.depth), 3)
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
      u[base + i] += (a - FRICTION * u[base + i]) * sub
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
    const xMax = params.shore + Math.min(13, 2.8 / params.slope)
    if (x[base + NODES - 1] > xMax) {
      x[base + NODES - 1] = xMax
      if (u[base + NODES - 1] > 0) u[base + NODES - 1] = 0
    }
  }
}

// must match shoreHeightScale in wave_common.wgsl
function shoreHeightScale(ty) {
  const s = Math.min(Math.max((ty + 1.2) / 1.05, 0), 1)
  return 1 - 0.65 * s * s * (3 - 2 * s)
}

function smoothStrided(a, offset, stride, count, k) {
  for (let j = 1; j < count - 1; j++) {
    const o = offset + j * stride
    a[o] += k * (a[o - stride] + a[o + stride] - 2 * a[o])
  }
}

// CPU replica of the layered horizontal wave displacement along x, with the
// same shallow amplification and waterline fade the vertex shader applies,
// so the driven junction matches the rendered wave-side displacement
export function sampleWaveDispX(x, z, noise, waveField, layers, chop, k0, params) {
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
    dsum += chop * l.amp * l.dx * s
  }
  const ty = Math.min(Math.max(params.slope * (x - params.shore), -params.depth), 3)
  const amp = Math.min(Math.max(1 / Math.tanh(k0 * Math.max(-ty, 0.05)), 1), 2.5)
  const t = Math.min(Math.max((ty + 0.6) / 0.7, 0), 1)
  const wSea = 1 - t * t * (3 - 2 * t)
  return dsum * amp * wSea
}

// CPU replica of the layered wave height (without choppy displacement)
export function sampleWaveHeight(x, z, noise, waveField, layers) {
  const tex = noise.channels.height
  const size = noise.size
  const copies = waveField.data
  let h = 0
  for (const l of layers) {
    const u0 = (x * l.dx + z * l.dz) * l.invL + l.su
    const v0 = (-x * l.dz + z * l.dx) * l.invL + l.sv
    for (let k = 0; k < 3; k++) {
      h += l.amp * copies[k * 4 + 2] * bilinearWrap(tex, size, u0 + copies[k * 4], v0 + copies[k * 4 + 1])
    }
  }
  return h
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
