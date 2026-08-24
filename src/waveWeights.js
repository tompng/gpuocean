// Per-slot spatial weights: channel i scales wave slot i's amplitude at a
// world position, so a region can lose one swell direction while keeping the
// others, or lose them all and go calm. Baked once over a fixed square around
// the origin and sampled clamp-to-edge, so the rim value extends outward.
//
// The weight is a continuous function of world position only, which is what
// keeps neighbouring mesh tiles bitwise-consistent at shared vertices — the
// slot directions themselves stay global for the same reason.
const SIZE = 128
// must match WEIGHT_EXTENT in wave_common.wgsl
const EXTENT = 384

export class WaveWeights {
  // fn(x, z) -> [w0, w1, w2, w3], each in [0, 1]
  constructor(device, fn) {
    this.data = new Uint8Array(SIZE * SIZE * 4)
    for (let iy = 0; iy < SIZE; iy++) {
      for (let ix = 0; ix < SIZE; ix++) {
        const w = fn(texelCenter(ix), texelCenter(iy))
        for (let c = 0; c < 4; c++) {
          this.data[(iy * SIZE + ix) * 4 + c] = Math.round(255 * Math.min(Math.max(w[c], 0), 1))
        }
      }
    }
    this.texture = device.createTexture({
      size: [SIZE, SIZE],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    })
    device.queue.writeTexture({ texture: this.texture }, this.data, { bytesPerRow: SIZE * 4 }, [SIZE, SIZE])
    this.view = this.texture.createView()
    this.sampler = device.createSampler({
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
      magFilter: 'linear',
      minFilter: 'linear',
    })
  }

  // CPU replica of the shader's fetch, for the chain drive
  sample(x, z, out) {
    const fx = (x / (2 * EXTENT) + 0.5) * SIZE - 0.5
    const fy = (z / (2 * EXTENT) + 0.5) * SIZE - 0.5
    const x0 = clampIndex(Math.floor(fx))
    const y0 = clampIndex(Math.floor(fy))
    const x1 = clampIndex(x0 + 1)
    const y1 = clampIndex(y0 + 1)
    const ax = Math.min(Math.max(fx - Math.floor(fx), 0), 1)
    const ay = Math.min(Math.max(fy - Math.floor(fy), 0), 1)
    for (let c = 0; c < 4; c++) {
      const a = this.data[(y0 * SIZE + x0) * 4 + c] * (1 - ax) + this.data[(y0 * SIZE + x1) * 4 + c] * ax
      const b = this.data[(y1 * SIZE + x0) * 4 + c] * (1 - ax) + this.data[(y1 * SIZE + x1) * 4 + c] * ax
      out[c] = (a * (1 - ay) + b * ay) / 255
    }
    return out
  }
}

function texelCenter(i) {
  return ((i + 0.5) / SIZE - 0.5) * 2 * EXTENT
}

function clampIndex(i) {
  return Math.min(Math.max(i, 0), SIZE - 1)
}
