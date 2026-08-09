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
      size: 96,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.uniform } }],
    })
    this.data = new Float32Array(24)
  }

  render(pass, invViewProj, eye, sunDir) {
    this.data.set(invViewProj, 0)
    this.data[16] = eye[0]; this.data[17] = eye[1]; this.data[18] = eye[2]
    this.data[20] = sunDir[0]; this.data[21] = sunDir[1]; this.data[22] = sunDir[2]
    this.device.queue.writeBuffer(this.uniform, 0, this.data)
    pass.setPipeline(this.pipeline)
    pass.setBindGroup(0, this.bindGroup)
    pass.draw(3)
  }
}
