// Step 1: blend a few copies of the noise texture scrolling at different speeds
// into one texture per frame, approximating in-band dispersion.
const COPY_FACTORS = [-0.6, 0.15, 0.7]
const COPY_OFFSETS = [[0.13, 0.71], [0.53, 0.29], [0.87, 0.61]]

export class WaveField {
  constructor(device, code, mipCode, noise) {
    this.device = device
    this.size = noise.size
    this.mipLevelCount = Math.log2(this.size) + 1
    this.texture = device.createTexture({
      size: [this.size, this.size],
      format: 'rgba16float',
      mipLevelCount: this.mipLevelCount,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
    })
    this.levelViews = []
    for (let i = 0; i < this.mipLevelCount; i++) {
      this.levelViews.push(this.texture.createView({ baseMipLevel: i, mipLevelCount: 1 }))
    }

    const module = device.createShaderModule({ code })
    this.pipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format: 'rgba16float' }] },
    })
    this.uniform = device.createBuffer({
      size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    const sampler = device.createSampler({
      addressModeU: 'repeat',
      addressModeV: 'repeat',
      magFilter: 'linear',
      minFilter: 'linear',
    })
    this.bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniform } },
        { binding: 1, resource: sampler },
        { binding: 2, resource: noise.texture.createView() },
      ],
    })

    const mipModule = device.createShaderModule({ code: mipCode })
    this.mipPipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: mipModule, entryPoint: 'vs' },
      fragment: { module: mipModule, entryPoint: 'fs', targets: [{ format: 'rgba16float' }] },
    })
    const mipSampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' })
    this.mipBindGroups = []
    for (let i = 1; i < this.mipLevelCount; i++) {
      this.mipBindGroups.push(device.createBindGroup({
        layout: this.mipPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: mipSampler },
          { binding: 1, resource: this.levelViews[i - 1] },
        ],
      }))
    }

    this.phases = [0, 0, 0]
    this.data = new Float32Array(12)
  }

  // texFreq: scroll speed in tiles per second matching the texture's dominant wave
  update(dt, texFreq, dispersion) {
    const weight = 1 / Math.sqrt(COPY_FACTORS.length)
    for (let i = 0; i < COPY_FACTORS.length; i++) {
      this.phases[i] += COPY_FACTORS[i] * dispersion * texFreq * dt
      this.data.set([COPY_OFFSETS[i][0] - this.phases[i], COPY_OFFSETS[i][1], weight, 0], i * 4)
    }
    this.device.queue.writeBuffer(this.uniform, 0, this.data)
  }

  render(encoder) {
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: this.levelViews[0], loadOp: 'clear', storeOp: 'store' }],
    })
    pass.setPipeline(this.pipeline)
    pass.setBindGroup(0, this.bindGroup)
    pass.draw(3)
    pass.end()
    for (let i = 1; i < this.mipLevelCount; i++) {
      const mipPass = encoder.beginRenderPass({
        colorAttachments: [{ view: this.levelViews[i], loadOp: 'clear', storeOp: 'store' }],
      })
      mipPass.setPipeline(this.mipPipeline)
      mipPass.setBindGroup(0, this.mipBindGroups[i - 1])
      mipPass.draw(3)
      mipPass.end()
    }
  }
}
