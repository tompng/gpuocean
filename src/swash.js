// 1D swash dynamics: one column per alongshore position. The driving front is
// the wave's own waterline (where the clamped wave surface meets the sand),
// found by marching the CPU wave replica. The film tip is pushed only while
// that edge is in contact; otherwise it coasts ballistically until the next
// wave catches it. Catching a retreating tip raises a collision burst.
const N = 256
const STEPS = 20
const GRAVITY = 9.81
const FRICTION = 0.15
const BURST_LIFETIME = 0.8
const MAX_EDGE_SPEED = 6

export class SwashSim {
  constructor(device) {
    this.device = device
    this.xt = new Float32Array(N)
    this.vt = new Float32Array(N)
    this.burst = new Float32Array(N)
    this.edge = new Float32Array(N)
    this.prevEdge = new Float32Array(N)
    this.started = false
    this.texData = new Float32Array(N * 4)
    this.texture = device.createTexture({
      size: [N, 1],
      format: 'rgba32float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    })
    this.view = this.texture.createView()
  }

  update(dt, params, region, sampleHeight) {
    const slope = params.slope
    const shore = params.shore
    const searchH = 2.2 * params.amplitude + 0.3
    const xMin = shore - searchH / slope
    const xMax = shore + Math.min(searchH / slope, 2.5 / slope)
    if (!this.started) {
      this.xt.fill(xMin)
      this.edge.fill(xMin)
      this.prevEdge.fill(xMin)
      this.started = true
    }
    if (dt > 0) {
      const step = (xMax - xMin) / STEPS
      for (let j = 0; j < N; j++) {
        const z = (j / (N - 1) - 0.5) * 2 * region
        // landward-most point still covered by the wave surface
        let edge = xMin
        let prevOver = 0
        for (let i = 0; i <= STEPS; i++) {
          const x = xMax - i * step
          const over = sampleHeight(x, z) - slope * (x - shore)
          if (over >= 0) {
            edge = i === 0 ? x : x + step * over / (over - prevOver)
            break
          }
          prevOver = over
        }
        this.edge[j] = edge
        const ve = Math.max(-MAX_EDGE_SPEED, Math.min((edge - this.prevEdge[j]) / dt, MAX_EDGE_SPEED))
        this.prevEdge[j] = edge

        if (edge >= this.xt[j]) {
          if (this.vt[j] < -0.3 && ve > 0.3) {
            this.burst[j] = Math.min(this.burst[j] + 0.25 * (ve - this.vt[j]), 2)
          }
          this.xt[j] = edge
          this.vt[j] = Math.max(this.vt[j], ve)
        } else {
          this.vt[j] += (-GRAVITY * slope - FRICTION * this.vt[j]) * dt
          this.xt[j] += this.vt[j] * dt
          if (this.xt[j] < edge) {
            this.xt[j] = edge
            this.vt[j] = 0
          }
        }
        this.burst[j] *= Math.exp(-dt / BURST_LIFETIME)
      }
      smoothLaplacian(this.xt, 0.15)
      smoothLaplacian(this.vt, 0.15)
      smoothLaplacian(this.burst, 0.1)
      smoothLaplacian(this.edge, 0.15)
    }

    for (let j = 0; j < N; j++) {
      this.texData[j * 4] = this.xt[j]
      this.texData[j * 4 + 1] = this.vt[j]
      this.texData[j * 4 + 2] = this.burst[j]
      this.texData[j * 4 + 3] = this.edge[j]
    }
    this.device.queue.writeTexture({ texture: this.texture }, this.texData, { bytesPerRow: N * 16 }, [N, 1])
  }
}

function smoothLaplacian(a, k) {
  for (let i = 1; i < a.length - 1; i++) {
    a[i] += k * (a[i - 1] + a[i + 1] - 2 * a[i])
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
