// Heightless swash film: per alongshore column, a horizontal chain of
// material nodes with compression-only springs. The seaward node is pinned
// to the wave's own waterline, so pushes propagate landward as compression
// waves (at the spring speed, low-passing the geometric contact-line speed)
// while retreat is free ballistics — water pushes, it never pulls. The last
// node is the waterline tip. No height dynamics at all: the film is a
// displacement field that carries foam, and rendering ends at the tip.
const COLS = 256
const NODES = 64
const SPAN = 24
const SUBSTEPS = 3
const EDGE_STEPS = 20
const GRAVITY = 9.81
const FRICTION = 0.15
// spring constant sets the compression-wave speed: c = L0 * sqrt(K) ~ 3 m/s
const SPRING_K = 60
const SPRING_D = 4
const MAX_EDGE_SPEED = 5

export class ChainSim {
  constructor(device) {
    this.device = device
    this.x = new Float32Array(COLS * NODES)
    this.u = new Float32Array(COLS * NODES)
    this.edge = new Float32Array(COLS)
    this.prevEdge = new Float32Array(COLS)
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
    this.x0 = params.shore - 10
    this.L0 = SPAN / (NODES - 1)
    for (let j = 0; j < COLS; j++) {
      for (let i = 0; i < NODES; i++) this.x[j * NODES + i] = this.x0 + i * this.L0
    }
    this.u.fill(0)
    this.edge.fill(this.x0)
    this.prevEdge.fill(this.x0)
    this.ve.fill(0)
  }

  update(dt, params, region, sampleHeight) {
    const key = `${params.shore}|${params.slope}|${params.depth}`
    if (key !== this.key) {
      this.key = key
      this.reset(params)
    }
    if (dt > 0) {
      const terr = x => Math.min(Math.max(params.slope * (x - params.shore), -params.depth), 3)
      const step = SPAN / EDGE_STEPS
      for (let j = 0; j < COLS; j++) {
        const z = (j / (COLS - 1) - 0.5) * 2 * region
        // the wave's own waterline: march landward from certainly-wet water
        // to the first point where the rendered surface dips below the sand,
        // using the same near-shore height attenuation as the shaders
        let edge = this.x0
        let prevOver = sampleHeight(this.x0, z) * shoreHeightScale(terr(this.x0)) - terr(this.x0)
        if (prevOver >= 0) {
          edge = this.x0 + SPAN
          for (let i = 1; i <= EDGE_STEPS; i++) {
            const x = this.x0 + i * step
            const t = terr(x)
            const over = sampleHeight(x, z) * shoreHeightScale(t) - t
            if (over < 0) {
              edge = x - step + step * prevOver / (prevOver - over)
              break
            }
            prevOver = over
          }
        }
        this.edge[j] = edge
        this.ve[j] = Math.max(-MAX_EDGE_SPEED, Math.min((edge - this.prevEdge[j]) / dt, MAX_EDGE_SPEED))
        this.prevEdge[j] = edge
      }
      smoothArray(this.edge, 0.1)
      const sub = dt / SUBSTEPS
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
        this.texData[o] = this.x[base + i] - (this.x0 + i * this.L0)
        this.texData[o + 1] = this.u[base + i]
        this.texData[o + 2] = tip
        this.texData[o + 3] = this.edge[j]
      }
    }
    this.device.queue.writeTexture(
      { texture: this.texture }, this.texData, { bytesPerRow: NODES * 16, rowsPerImage: COLS }, [NODES, COLS])
  }

  stepColumn(j, sub, params) {
    const base = j * NODES
    const x = this.x
    const u = this.u
    const terr = xw => Math.min(Math.max(params.slope * (xw - params.shore), -params.depth), 3)
    for (let i = 1; i < NODES; i++) {
      const gs = (terr(x[base + i] + 0.4) - terr(x[base + i] - 0.4)) / 0.8
      u[base + i] += (-GRAVITY * gs - FRICTION * u[base + i]) * sub
    }
    for (let k = 0; k < NODES - 1; k++) {
      const L = x[base + k + 1] - x[base + k]
      if (L < this.L0) {
        let f = SPRING_K * (this.L0 - L)
        const rel = u[base + k] - u[base + k + 1]
        if (rel > 0) f += SPRING_D * rel
        u[base + k] -= f * sub
        u[base + k + 1] += f * sub
      }
    }
    x[base] = this.edge[j]
    u[base] = this.ve[j]
    for (let i = 1; i < NODES; i++) x[base + i] += u[base + i] * sub
    const lMin = this.L0 * 0.1
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

function smoothArray(a, k) {
  for (let i = 1; i < a.length - 1; i++) {
    a[i] += k * (a[i - 1] + a[i + 1] - 2 * a[i])
  }
}

function smoothStrided(a, offset, stride, count, k) {
  for (let j = 1; j < count - 1; j++) {
    const o = offset + j * stride
    a[o] += k * (a[o - stride] + a[o + stride] - 2 * a[o])
  }
}

// CPU replica of the layered wave height (without choppy displacement),
// summing the wave-field copies over the gravity noise texture
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
