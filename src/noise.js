import { floatToHalf } from './gpu.js'

// Gravity-wave noise texture.
// R: height, G: horizontal displacement, B: dh/dx, A: dh/dy.
// For η = A cos(kx − ωt) traveling +x, the particle displacement is −A sin(kx − ωt),
// so G is the NEGATED x-cumsum of height (positive cumsum would sharpen troughs, not crests).
// Height is band-passed along x: the cumsum has 1/k gain, so without a band-pass
// tile-sized components would dominate the displacement channel.
export function generateGravityNoiseTexture(device, opts = {}) {
  const size = opts.size ?? 512
  const sigmaAlong = opts.sigmaAlong ?? 3
  const sigmaAlongWide = opts.sigmaAlongWide ?? 9
  const sigmaCross = opts.sigmaCross ?? 24
  const random = mulberry32(opts.seed ?? 12345)

  const n = size * size
  const noise = new Float32Array(n)
  for (let i = 0; i < n; i++) noise[i] = random() * 2 - 1

  const h = smoothAxisX(noise, size, sigmaAlong)
  const wide = smoothAxisX(noise, size, sigmaAlongWide)
  for (let i = 0; i < n; i++) h[i] -= wide[i]
  smoothAxisYInPlace(h, size, sigmaCross)
  normalizeVariance(h)

  const d = new Float32Array(n)
  for (let y = 0; y < size; y++) {
    const row = y * size
    let mean = 0
    for (let x = 0; x < size; x++) mean += h[row + x]
    mean /= size
    let c = 0
    for (let x = 0; x < size; x++) {
      h[row + x] -= mean
      c += h[row + x]
      d[row + x] = c
    }
    let dMean = 0
    for (let x = 0; x < size; x++) dMean += d[row + x]
    dMean /= size
    for (let x = 0; x < size; x++) d[row + x] -= dMean
  }
  let dSq = 0
  for (let i = 0; i < n; i++) dSq += d[i] * d[i]
  const sigmaD = Math.sqrt(dSq / n)
  for (let i = 0; i < n; i++) d[i] /= -sigmaD

  const hx = new Float32Array(n)
  const hy = new Float32Array(n)
  for (let y = 0; y < size; y++) {
    const up = ((y + size - 1) % size) * size
    const down = ((y + 1) % size) * size
    const row = y * size
    for (let x = 0; x < size; x++) {
      const left = row + (x + size - 1) % size
      const right = row + (x + 1) % size
      hx[row + x] = (h[right] - h[left]) * 0.5
      hy[row + x] = (h[down + x] - h[up + x]) * 0.5
    }
  }

  // Dominant wavenumber from the spectral moment: k_rms = sqrt(E[h_x^2] / E[h^2])
  let hSq = 0, hxSq = 0
  for (let i = 0; i < n; i++) {
    hSq += h[i] * h[i]
    hxSq += hx[i] * hx[i]
  }
  const kRms = Math.sqrt(hxSq / hSq)

  const data = new Uint16Array(n * 4)
  for (let i = 0; i < n; i++) {
    data[i * 4] = floatToHalf(h[i])
    data[i * 4 + 1] = floatToHalf(d[i])
    data[i * 4 + 2] = floatToHalf(hx[i])
    data[i * 4 + 3] = floatToHalf(hy[i])
  }
  const texture = device.createTexture({
    size: [size, size],
    format: 'rgba16float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  })
  device.queue.writeTexture({ texture }, data, { bytesPerRow: size * 8 }, [size, size])

  return {
    texture,
    size,
    wavesPerTile: size * kRms / (2 * Math.PI),
    dispGradPerTexel: -1 / sigmaD,
  }
}

// Weights 5a^|k| - 4a^(2|k|) + a^(3|k|): with f(x) = a^(3x) - 4a^(2x) + 5a^x,
// f'(0) = f'''(0) = 0, so f(-|x|) approximates a Gaussian using three one-pole filters.
function comboSmoothLine(line, out, sigma, tmp) {
  const a = Math.exp(-1 / sigma)
  const decays = [a, a * a, a * a * a]
  const coeffs = [5, -4, 1]
  const n = line.length
  out.fill(0)
  let gain = 0
  for (let k = 0; k < 3; k++) {
    const decay = decays[k]
    const coeff = coeffs[k]
    gain += coeff * (1 + decay) / (1 - decay)
    let s = 0
    for (let i = 0; i < n; i++) s = s * decay + line[i]
    for (let i = 0; i < n; i++) {
      s = s * decay + line[i]
      tmp[i] = s
    }
    s = 0
    for (let i = n - 1; i >= 0; i--) s = s * decay + line[i]
    for (let i = n - 1; i >= 0; i--) {
      s = s * decay + line[i]
      tmp[i] += s
    }
    for (let i = 0; i < n; i++) out[i] += coeff * (tmp[i] - line[i])
  }
  for (let i = 0; i < n; i++) out[i] /= gain
}

function smoothAxisX(src, size, sigma) {
  const dst = new Float32Array(src.length)
  const line = new Float32Array(size)
  const out = new Float32Array(size)
  const tmp = new Float32Array(size)
  for (let y = 0; y < size; y++) {
    const off = y * size
    line.set(src.subarray(off, off + size))
    comboSmoothLine(line, out, sigma, tmp)
    dst.set(out, off)
  }
  return dst
}

function smoothAxisYInPlace(data, size, sigma) {
  const line = new Float32Array(size)
  const out = new Float32Array(size)
  const tmp = new Float32Array(size)
  for (let x = 0; x < size; x++) {
    for (let y = 0; y < size; y++) line[y] = data[y * size + x]
    comboSmoothLine(line, out, sigma, tmp)
    for (let y = 0; y < size; y++) data[y * size + x] = out[y]
  }
}

function normalizeVariance(data) {
  let sq = 0
  for (let i = 0; i < data.length; i++) sq += data[i] * data[i]
  const sigma = Math.sqrt(sq / data.length)
  for (let i = 0; i < data.length; i++) data[i] /= sigma
}

function mulberry32(seed) {
  let s = seed >>> 0
  return () => {
    let t = (s += 0x6d2b79f5)
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}
