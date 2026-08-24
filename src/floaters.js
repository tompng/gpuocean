// Verification rig for src/surface.js: spheres dropped on the water, held at
// the queried height and carried by the queried velocity. Each is white above
// its own equator and red below, so the query is right exactly when the
// colour split sits on the waterline — a wrong height shows as a sphere
// wearing a red or white band, and a wrong horizontal inversion shows as the
// split swimming up and down as the wave passes under it.

import { createSurfaceSample } from './surface.js'

const CODE = /* wgsl */`
struct U {
  viewProj: mat4x4f,
  sunDir: vec3f,
  pad: f32,
}
@group(0) @binding(0) var<uniform> u: U;

struct VSOut {
  @builtin(position) clip: vec4f,
  @location(0) nrm: vec3f,
  @location(1) side: f32,
}

@vertex
fn vs(@location(0) p: vec3f, @location(1) inst: vec4f) -> VSOut {
  var out: VSOut;
  out.clip = u.viewProj * vec4f(inst.xyz + p * inst.w, 1.0);
  out.nrm = p;
  out.side = p.y;
  return out;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4f {
  let n = normalize(in.nrm);
  let lit = 0.35 + 0.65 * max(dot(n, u.sunDir), 0.0);
  let base = select(vec3f(0.8, 0.12, 0.08), vec3f(0.97, 0.96, 0.92), in.side > 0.0);
  return vec4f(pow(base * lit, vec3f(1.0 / 2.2)), 1.0);
}
`

const COUNT = 24
const RADIUS = 0.35
// respawn ring around the camera target
const SPAWN_MIN = 8
const SPAWN_MAX = 40
const KEEP_MAX = 70

export class Floaters {
  constructor(device, format, opts = {}) {
    const sampleCount = opts.sampleCount ?? 4
    const mesh = uvSphere(16, 10)
    this.vertexCount = mesh.length / 3
    this.vb = device.createBuffer({ size: mesh.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST })
    device.queue.writeBuffer(this.vb, 0, mesh)
    this.instData = new Float32Array(COUNT * 4)
    this.ib = device.createBuffer({ size: this.instData.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST })
    this.uniform = device.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
    this.uniformData = new Float32Array(20)
    const module = device.createShaderModule({ code: CODE })
    this.pipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: {
        module,
        entryPoint: 'vs',
        buffers: [
          { arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] },
          { arrayStride: 16, stepMode: 'instance', attributes: [{ shaderLocation: 1, offset: 0, format: 'float32x4' }] },
        ],
      },
      fragment: { module, entryPoint: 'fs', targets: [{ format }] },
      primitive: { topology: 'triangle-list', cullMode: 'back' },
      multisample: { count: sampleCount },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    })
    this.bind = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.uniform } }],
    })
    this.device = device
    this.pos = new Float32Array(COUNT * 2)
    this.seq = new Int32Array(COUNT)
    this.spawned = false
    this.hit = createSurfaceSample()
  }

  update(dt, query, cx, cz) {
    if (!this.spawned) {
      for (let i = 0; i < COUNT; i++) this.respawn(i, cx, cz, query)
      this.spawned = true
    }
    for (let i = 0; i < COUNT; i++) {
      let x = this.pos[i * 2]
      let z = this.pos[i * 2 + 1]
      const h = query.sample(x, z, this.hit)
      x += h.vx * dt
      z += h.vz * dt
      // a floater that has run up onto dry sand, or drifted out of view, goes
      // back into the ring
      if (h.depth < 0.12 || Math.hypot(x - cx, z - cz) > KEEP_MAX) {
        this.respawn(i, cx, cz, query)
        x = this.pos[i * 2]
        z = this.pos[i * 2 + 1]
        query.sample(x, z, h)
      }
      this.pos[i * 2] = x
      this.pos[i * 2 + 1] = z
      const o = i * 4
      this.instData[o] = x
      this.instData[o + 1] = h.y
      this.instData[o + 2] = z
      this.instData[o + 3] = RADIUS
    }
    this.device.queue.writeBuffer(this.ib, 0, this.instData)
  }

  // A deterministic spiral rather than random, so a reload puts them in the
  // same places and two runs can be compared frame by frame. The per-floater
  // counter advances on every attempt: a fixed spot that happens to be dry
  // land would otherwise be picked again the instant it is rejected.
  respawn(i, cx, cz, query) {
    for (let k = 0; k < 12; k++) {
      const n = this.seq[i]++
      const a = i * 2.39996 + n * 0.7
      const r = SPAWN_MIN + (SPAWN_MAX - SPAWN_MIN) * ((n * 0.6180339887) % 1)
      this.pos[i * 2] = cx + Math.cos(a) * r
      this.pos[i * 2 + 1] = cz + Math.sin(a) * r
      if (query.sample(this.pos[i * 2], this.pos[i * 2 + 1], this.hit).depth > 0.5) return
    }
  }

  render(pass, viewProj, sunDir) {
    this.uniformData.set(viewProj, 0)
    this.uniformData.set(sunDir, 16)
    this.device.queue.writeBuffer(this.uniform, 0, this.uniformData)
    pass.setPipeline(this.pipeline)
    pass.setBindGroup(0, this.bind)
    pass.setVertexBuffer(0, this.vb)
    pass.setVertexBuffer(1, this.ib)
    pass.draw(this.vertexCount, COUNT)
  }
}

function uvSphere(segments, rings) {
  const at = (i, j) => {
    const phi = (j / rings) * Math.PI
    const theta = (i / segments) * 2 * Math.PI
    return [Math.sin(phi) * Math.cos(theta), Math.cos(phi), Math.sin(phi) * Math.sin(theta)]
  }
  const out = []
  for (let j = 0; j < rings; j++) {
    for (let i = 0; i < segments; i++) {
      const a = at(i, j)
      const b = at(i + 1, j)
      const c = at(i + 1, j + 1)
      const d = at(i, j + 1)
      out.push(...a, ...d, ...c, ...a, ...c, ...b)
    }
  }
  return new Float32Array(out)
}
