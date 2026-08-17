// Step 1: blend a few copies of the noise texture scrolling at different speeds
// into one texture per frame, approximating in-band dispersion.
const COPY_FACTORS = [-0.65, -0.3, 0.1, 0.4, 0.7]
// Crest-parallel drift: advances oblique spectral components differently from
// straight-ahead ones, imitating directional dispersion within one layer
const COPY_FACTORS_Y = [0.5, -0.35, -0.65, 0.2, 0.35]
const COPY_OFFSETS = [[0.13, 0.71], [0.53, 0.29], [0.87, 0.61], [0.31, 0.07], [0.67, 0.43]]

export class WaveField {
  // The noise is band-limited, so instead of mipmaps the renderers apply an
  // analytic per-layer attenuation matched to their sampling footprint.
  constructor(device, code, noise) {
    this.device = device
    this.size = noise.size
    this.texture = device.createTexture({
      size: [this.size, this.size],
      format: 'rgba16float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
    })
    this.view = this.texture.createView()

    const module = device.createShaderModule({ code })
    this.pipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format: 'rgba16float' }] },
    })
    this.uniform = device.createBuffer({
      size: COPY_FACTORS.length * 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    const sampler = device.createSampler({
      addressModeU: 'repeat',
      addressModeV: 'repeat',
      magFilter: 'linear',
      minFilter: 'linear',
    })
    // single-texture noises (capillary) bind the same texture to all copies
    const texs = noise.textures ?? COPY_FACTORS.map(() => noise.texture)
    this.bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniform } },
        { binding: 1, resource: sampler },
        ...texs.map((t, i) => ({ binding: 2 + i, resource: t.createView() })),
      ],
    })

    this.phases = COPY_FACTORS.map(() => 0)
    this.phasesY = COPY_FACTORS.map(() => 0)
    this.data = new Float32Array(COPY_FACTORS.length * 4)
  }

  // texFreq: scroll speed in tiles per second matching the texture's dominant wave
  update(dt, texFreq, dispersion) {
    const weight = 1 / Math.sqrt(COPY_FACTORS.length)
    for (let i = 0; i < COPY_FACTORS.length; i++) {
      this.phases[i] += COPY_FACTORS[i] * dispersion * texFreq * dt
      this.phasesY[i] += COPY_FACTORS_Y[i] * dispersion * texFreq * dt
      this.data.set([COPY_OFFSETS[i][0] - this.phases[i], COPY_OFFSETS[i][1] - this.phasesY[i], weight, 0], i * 4)
    }
    this.device.queue.writeBuffer(this.uniform, 0, this.data)
  }

  render(encoder) {
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: this.view, loadOp: 'clear', storeOp: 'store' }],
    })
    pass.setPipeline(this.pipeline)
    pass.setBindGroup(0, this.bindGroup)
    pass.draw(3)
    pass.end()
  }
}
