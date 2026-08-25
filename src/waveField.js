// Step 1: blend the noise copies scrolling at different speeds into one
// texture per frame. A comb noise set supplies its own physical per-copy
// speeds and weights; single-texture noises (capillary) fall back to these
// in-band dispersion-approximation factors and uniform weights.
const COPY_FACTORS = [-0.65, -0.3, 0.1, 0.4, 0.7]
// Artificial speed spread scaled by the dispersion parameter: physical
// speed differences between neighbouring comb bands are only a few percent,
// which reads as frozen patterns. Zero mean under the copies' variance
// weights, so the spread never shifts the apparent propagation speed.
const DISPERSION_JITTER = [0.55, -0.45, 0.3, -0.6, -0.37]
// Crest-parallel drift: advances oblique spectral components differently from
// straight-ahead ones, imitating directional dispersion within one layer
const COPY_FACTORS_Y = [0.5, -0.35, -0.65, 0.2, 0.35]
const COPY_OFFSETS = [[0.13, 0.71], [0.53, 0.29], [0.87, 0.61], [0.31, 0.07], [0.67, 0.43]]

// 2x2 box downsample, one level per pass. A linear tap at the destination
// texel's centre lands exactly between the four source texels, so the filter
// is the box average with no explicit weights.
const MIP_CODE = /* wgsl */`
@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var src: texture_2d<f32>;
@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let xy = vec2f(vec2u((vi << 1u) & 2u, vi & 2u));
  return vec4f(xy * 2.0 - 1.0, 0.0, 1.0);
}
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let n = vec2f(textureDimensions(src, 0)) * 0.5;
  return textureSampleLevel(src, samp, pos.xy / n, 0.0);
}
`

export class WaveField {
  // The noise is band-limited, so instead of mipmaps the renderers apply an
  // analytic per-layer attenuation matched to their sampling footprint.
  // opts.mips: keep a mip chain. The renderers replace mip filtering with an
  // analytic per-band attenuation, so this is not about aliasing — it is for
  // the taps whose footprint is several texels wide, which read level 0
  // incoherently and thrash the cache.
  constructor(device, code, noise, opts = {}) {
    this.device = device
    this.size = noise.size
    this.mipCount = opts.mips ? Math.log2(this.size) + 1 : 1
    this.texture = device.createTexture({
      size: [this.size, this.size],
      format: 'rgba16float',
      mipLevelCount: this.mipCount,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
    })
    this.view = this.texture.createView()
    this.target = this.texture.createView({ baseMipLevel: 0, mipLevelCount: 1 })

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

    this.base = noise.copySpeeds ?? COPY_FACTORS.map(() => 0)
    this.jitter = noise.copySpeeds ? DISPERSION_JITTER : COPY_FACTORS
    this.weights = noise.copyWeights ?? COPY_FACTORS.map(() => 1 / Math.sqrt(COPY_FACTORS.length))
    this.phases = COPY_FACTORS.map(() => 0)
    this.phasesY = COPY_FACTORS.map(() => 0)
    this.data = new Float32Array(COPY_FACTORS.length * 4)
    this.mipTargets = []
    if (this.mipCount > 1) this.initMips(device, sampler)
  }

  initMips(device, sampler) {
    const module = device.createShaderModule({ code: MIP_CODE })
    this.mipPipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format: 'rgba16float' }] },
    })
    this.mipTargets = []
    this.mipBinds = []
    for (let level = 1; level < this.mipCount; level++) {
      this.mipTargets.push(this.texture.createView({ baseMipLevel: level, mipLevelCount: 1 }))
      this.mipBinds.push(device.createBindGroup({
        layout: this.mipPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: sampler },
          { binding: 1, resource: this.texture.createView({ baseMipLevel: level - 1, mipLevelCount: 1 }) },
        ],
      }))
    }
  }

  // texFreq: scroll speed in tiles per second matching the texture's dominant wave
  update(dt, texFreq, dispersion) {
    for (let i = 0; i < COPY_FACTORS.length; i++) {
      this.phases[i] += (this.base[i] + dispersion * this.jitter[i]) * texFreq * dt
      this.phasesY[i] += COPY_FACTORS_Y[i] * dispersion * texFreq * dt
      this.data.set([COPY_OFFSETS[i][0] - this.phases[i], COPY_OFFSETS[i][1] - this.phasesY[i], this.weights[i], 0], i * 4)
    }
    this.device.queue.writeBuffer(this.uniform, 0, this.data)
  }

  render(encoder) {
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: this.target, loadOp: 'clear', storeOp: 'store' }],
    })
    pass.setPipeline(this.pipeline)
    pass.setBindGroup(0, this.bindGroup)
    pass.draw(3)
    pass.end()
    for (let i = 0; i < this.mipTargets.length; i++) {
      const p = encoder.beginRenderPass({
        colorAttachments: [{ view: this.mipTargets[i], loadOp: 'clear', storeOp: 'store' }],
      })
      p.setPipeline(this.mipPipeline)
      p.setBindGroup(0, this.mipBinds[i])
      p.draw(3)
      p.end()
    }
  }
}
