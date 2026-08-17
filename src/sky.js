export class Sky {
  constructor(device, code, format, opts = {}) {
    this.device = device
    const module = device.createShaderModule({ code })
    this.pipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format }] },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'always' },
      multisample: { count: opts.sampleCount ?? 4 },
    })
    this.uniform = device.createBuffer({
      size: 144,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.uniform } }],
    })
    this.data = new Float32Array(36)
  }

  render(pass, invViewProj, eye, sunDir, moonDir, camDepth, lensR, turbidity, chlorophyll, sky) {
    this.data.set(invViewProj, 0)
    this.data[16] = eye[0]; this.data[17] = eye[1]; this.data[18] = eye[2]
    this.data[19] = lensR
    this.data[20] = sunDir[0]; this.data[21] = sunDir[1]; this.data[22] = sunDir[2]
    this.data[23] = camDepth
    this.data[24] = turbidity
    this.data[25] = chlorophyll
    this.data[26] = sky.skyTurbidity
    this.data[27] = sky.rayleigh
    this.data[28] = sky.intensity
    this.data[32] = moonDir[0]; this.data[33] = moonDir[1]; this.data[34] = moonDir[2]
    this.device.queue.writeBuffer(this.uniform, 0, this.data)
    pass.setPipeline(this.pipeline)
    pass.setBindGroup(0, this.bindGroup)
    pass.draw(3)
  }
}
