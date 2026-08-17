// One uniform block bound into BOTH the sky pass and the ocean pass, so the
// deck the water reflects is provably the deck the sky drew. The layout must
// match CloudUniforms in shaders/atmosphere.wgsl, which declares the block in
// text shared by both modules — that is what makes the binding number and the
// layout identical on both sides by construction rather than by agreement.
export const CLOUD_BYTES = 80

// The quality tier picks the march length, and it is GLOBAL rather than per call
// site on purpose. Varying the filter width leaves the shapes agreeing; varying
// the step count changes the integral itself, so a reflection marched at a
// different step count than the sky would visibly disagree with it.
// The march length. Full 8 steps is reserved for ultra: at the default it is
// not distinguishable from 4 on a moving deck, and it is the dominant cost.
export const CLOUD_LOD = { low: 2, medium: 2, high: 1, ultra: 0 }

export class Clouds {
  constructor(device) {
    this.device = device
    this.buffer = device.createBuffer({
      size: CLOUD_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.data = new Float32Array(CLOUD_BYTES / 4)
    // Wind is integrated here rather than evaluated as wind * time in the
    // shader: sky.wgsl's uniforms have no clock, and integrating means the
    // windSpeed slider sets a velocity instead of teleporting the deck.
    this.drift = [0, 0]
  }

  update(dt, p, eye, heightPx, fovY) {
    const dir = p.cloudWindDir * Math.PI / 180
    this.drift[0] += Math.sin(dir) * p.cloudWindSpeed * dt
    this.drift[1] += Math.cos(dir) * p.cloudWindSpeed * dt
    const d = this.data
    d[0] = eye[0]; d[1] = eye[1]; d[2] = eye[2]
    d[4] = this.drift[0]; d[5] = this.drift[1]
    d[6] = p.cloudCover
    d[7] = p.cloudAltitude
    d[8] = p.cloudThickness
    d[9] = p.cloudDensity
    d[10] = p.cloudScale
    d[11] = p.cloudSoftness
    d[12] = p.cloudDetail
    d[13] = p.cloudAnvil
    d[14] = p.cloudAbsorption
    d[15] = p.cloudAmbient
    // Stands in for the mip chain analytic noise does not have
    d[16] = 2 * Math.tan(fovY / 2) / heightPx
    d[17] = CLOUD_LOD[p.quality] ?? 0
    d[18] = p.cloudShear
    d[19] = p.cloudShadow
    this.device.queue.writeBuffer(this.buffer, 0, d)
  }
}
