import { normalize } from './mat4.js'

const GRAVITY = 9.81
const PATCH_SIZE = 100
const SCALE_RATIO = 0.68
const MAX_LAYERS = 8
const DIR_FRACS = [0, 0.9, -0.75, 0.45, -0.35, 0.7, -1, 0.2]
const UV_OFFSETS = [
  [0.11, 0.63], [0.42, 0.17], [0.78, 0.55], [0.05, 0.91],
  [0.33, 0.4], [0.66, 0.08], [0.9, 0.77], [0.24, 0.31],
]
const SUN_DIR = normalize([0.6, 0.35, -0.7])

export class Ocean {
  constructor(device, code, waveTexture, format, opts = {}) {
    this.device = device
    this.gridN = opts.gridN ?? 256
    const sampleCount = opts.sampleCount ?? 4
    const module = device.createShaderModule({ code })
    const base = {
      layout: 'auto',
      vertex: { module, entryPoint: 'vs' },
      multisample: { count: sampleCount },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    }
    this.fillPipeline = device.createRenderPipeline({
      ...base,
      fragment: { module, entryPoint: 'fs', targets: [{ format }] },
      primitive: { topology: 'triangle-list' },
    })
    this.wirePipeline = device.createRenderPipeline({
      ...base,
      fragment: { module, entryPoint: 'fs_wire', targets: [{ format }] },
      primitive: { topology: 'line-list' },
    })

    this.uniform = device.createBuffer({
      size: 384,
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
    ]
    this.fillBindGroup = device.createBindGroup({ layout: this.fillPipeline.getBindGroupLayout(0), entries })
    this.wireBindGroup = device.createBindGroup({ layout: this.wirePipeline.getBindGroupLayout(0), entries })

    const [tri, line] = buildIndices(this.gridN)
    this.triIndices = createIndexBuffer(device, tri)
    this.lineIndices = createIndexBuffer(device, line)
    this.triCount = tri.length
    this.lineCount = line.length

    this.phases = new Float64Array(MAX_LAYERS)
    this.uniformData = new Float32Array(96)
  }

  render(pass, dt, params, noise, viewProj, eye) {
    const u = this.uniformData
    u.set(viewProj, 0)
    u[16] = eye[0]; u[17] = eye[1]; u[18] = eye[2]; u[19] = 0
    u[20] = SUN_DIR[0]; u[21] = SUN_DIR[1]; u[22] = SUN_DIR[2]; u[23] = PATCH_SIZE
    const count = Math.round(params.layers)
    u[24] = count
    u[25] = params.choppiness
    u[26] = noise.size * noise.dispGradPerTexel
    u[27] = noise.size
    u[28] = this.gridN

    const spread = params.spread * Math.PI / 180
    let sq = 0
    for (let i = 0; i < count; i++) sq += SCALE_RATIO ** (2 * i)
    // amp_i ∝ λ_i keeps per-layer steepness constant; total variance = amplitude^2
    const ampNorm = params.amplitude / Math.sqrt(sq)
    for (let i = 0; i < count; i++) {
      const lambda = params.wavelength * SCALE_RATIO ** i
      const tile = lambda * noise.wavesPerTile
      this.phases[i] += Math.sqrt(GRAVITY * lambda / (2 * Math.PI)) / tile * dt
      const angle = DIR_FRACS[i] * spread
      const o = 32 + i * 8
      u[o] = Math.cos(angle)
      u[o + 1] = Math.sin(angle)
      u[o + 2] = 1 / tile
      u[o + 3] = ampNorm * SCALE_RATIO ** i
      u[o + 4] = UV_OFFSETS[i][0] - this.phases[i]
      u[o + 5] = UV_OFFSETS[i][1]
      u[o + 6] = 0
      u[o + 7] = 0
    }
    this.device.queue.writeBuffer(this.uniform, 0, u)

    pass.setPipeline(params.wireframe ? this.wirePipeline : this.fillPipeline)
    pass.setBindGroup(0, params.wireframe ? this.wireBindGroup : this.fillBindGroup)
    pass.setIndexBuffer(params.wireframe ? this.lineIndices : this.triIndices, 'uint32')
    pass.drawIndexed(params.wireframe ? this.lineCount : this.triCount)
  }
}

function buildIndices(n) {
  const tri = new Uint32Array(n * n * 6)
  let t = 0
  for (let z = 0; z < n; z++) {
    for (let x = 0; x < n; x++) {
      const a = z * (n + 1) + x
      const b = a + 1
      const c = a + n + 1
      const d = c + 1
      tri[t++] = a; tri[t++] = c; tri[t++] = b
      tri[t++] = b; tri[t++] = c; tri[t++] = d
    }
  }
  const line = new Uint32Array(4 * n * (n + 1))
  let l = 0
  for (let z = 0; z <= n; z++) {
    for (let x = 0; x < n; x++) {
      const a = z * (n + 1) + x
      line[l++] = a; line[l++] = a + 1
    }
  }
  for (let x = 0; x <= n; x++) {
    for (let z = 0; z < n; z++) {
      const a = z * (n + 1) + x
      line[l++] = a; line[l++] = a + n + 1
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
