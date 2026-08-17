// GPU-side pass timing via timestamp-query.
//
// Wall-clock frame timing measures how long the browser waited for vsync, not
// how long the GPU worked: below the refresh interval every frame reads the
// same number, and above it the value snaps to multiples of the interval. These
// timestamps are written by the GPU around each pass, so they show the real
// cost whether it is well under the refresh interval or well over.
//
// Results are averaged, because the readback is one frame behind and browsers
// quantise timestamps (Chrome to ~100 ns-scale buckets) to close timing side
// channels — a single sample is much noisier than the mean.
const NS_PER_MS = 1e6

export class GPUTimer {
  constructor(device, enabled, passes) {
    this.enabled = enabled
    this.passes = passes
    this.ms = passes.map(() => 0)
    if (!enabled) return
    this.device = device
    const count = passes.length * 2
    this.querySet = device.createQuerySet({ type: 'timestamp', count })
    this.resolve = device.createBuffer({
      size: count * 8,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    })
    // One spare buffer so a frame never waits on the previous mapping
    this.readback = [0, 1].map(() => device.createBuffer({
      size: count * 8,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    }))
    this.free = [...this.readback]
    this.samples = passes.map(() => [])
  }

  // Pass this into beginRenderPass as timestampWrites
  writes(index) {
    if (!this.enabled) return undefined
    return {
      querySet: this.querySet,
      beginningOfPassWriteIndex: index * 2,
      endOfPassWriteIndex: index * 2 + 1,
    }
  }

  // Call after the passes are encoded, before submit
  resolveInto(encoder) {
    if (!this.enabled || this.free.length === 0) return
    this.pending = this.free.pop()
    encoder.resolveQuerySet(this.querySet, 0, this.passes.length * 2, this.resolve, 0)
    encoder.copyBufferToBuffer(this.resolve, 0, this.pending, 0, this.pending.size)
  }

  // Call after submit; the map resolves a frame or two later
  read() {
    if (!this.enabled || !this.pending) return
    const buf = this.pending
    this.pending = null
    buf.mapAsync(GPUMapMode.READ).then(() => {
      const t = new BigUint64Array(buf.getMappedRange().slice(0))
      buf.unmap()
      this.free.push(buf)
      for (let i = 0; i < this.passes.length; i++) {
        const dt = Number(t[i * 2 + 1] - t[i * 2]) / NS_PER_MS
        // A zero or absurd delta means the query never landed; drop it
        if (!(dt > 0 && dt < 1000)) continue
        const s = this.samples[i]
        s.push(dt)
        if (s.length > 120) s.shift()
        this.ms[i] = s.reduce((a, b) => a + b, 0) / s.length
      }
    }).catch(() => { this.free.push(buf) })
  }

  reset() {
    if (this.enabled) this.samples = this.passes.map(() => [])
  }

  get totalMs() {
    return this.ms.reduce((a, b) => a + b, 0)
  }

  label() {
    if (!this.enabled) return ''
    return this.passes.map((p, i) => `${p} ${this.ms[i].toFixed(2)}`).join(' · ') + ' ms'
  }
}
