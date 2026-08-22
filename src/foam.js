// Foam accumulation buffer: a persistent world-space texture around the
// origin, updated by ping-pong. Each frame the previous foam decays and new
// foam is generated where the wave-field Jacobian marks breaking.
const SIZE = 512

export class FoamSim {
  constructor(device, code, size = [SIZE, SIZE]) {
    this.device = device
    this.textures = [0, 1].map(() => device.createTexture({
      size,
      format: 'rgba16float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
    }))
    this.views = this.textures.map(t => t.createView())
    const module = device.createShaderModule({ code })
    this.pipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format: 'rgba16float' }] },
    })
    this.index = 0
  }

  bind(uniformBuffer, waveTexture, simView, sdfView, diskBuffer) {
    const sampler = this.device.createSampler({
      addressModeU: 'repeat',
      addressModeV: 'repeat',
      magFilter: 'linear',
      minFilter: 'linear',
      mipmapFilter: 'linear',
    })
    this.bindGroups = [0, 1].map(src => this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: sampler },
        // the film pass needs no wave field; the wave pass needs no sim
        ...(waveTexture ? [{ binding: 2, resource: waveTexture.createView() }] : []),
        { binding: 3, resource: this.views[src] },
        ...(simView ? [{ binding: 7, resource: simView }] : []),
        ...(sdfView ? [{ binding: 9, resource: sdfView }] : []),
        ...(diskBuffer ? [{ binding: 11, resource: { buffer: diskBuffer } }] : []),
      ],
    }))
  }

  render(encoder) {
    const dst = this.index ^ 1
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: this.views[dst], loadOp: 'clear', storeOp: 'store' }],
    })
    pass.setPipeline(this.pipeline)
    pass.setBindGroup(0, this.bindGroups[this.index])
    pass.draw(3)
    pass.end()
    this.index = dst
  }
}
