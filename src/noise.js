import { floatToHalf } from './gpu.js'

// Gravity-wave noise texture set.
// R: height, G: horizontal displacement, B: dh/dx, A: dh/dy.
// For η = A cos(kx − ωt) traveling +x, the particle displacement is −A sin(kx − ωt),
// so G is the NEGATED x-cumsum of height (positive cumsum would sharpen troughs, not crests).
// Height is band-passed along the travel axis: the cumsum has 1/k gain, so
// without a band-pass tile-sized components would dominate the displacement.
//
// Built in the frequency domain, which makes the band orientation a free
// parameter with exact tiling: the transfer is evaluated at frequencies
// rotated by the variant's angle, and the displacement cumsum divides by
// 1 - e^{-iω} along the rotated travel axis. Three independent variants
// (straight and ±ANGLE) feed the wave field's three scrolled copies, so a
// still frame shows 3 pattern orientations per layer instead of one.
const VARIANT_ANGLE = 10 * Math.PI / 180
const VARIANT_SEEDS = [23456, 12345, 34567]

export function generateGravityNoiseSet(device, opts = {}) {
  const size = opts.size ?? 512
  const angles = opts.angles ?? [-VARIANT_ANGLE, 0, VARIANT_ANGLE]
  const variants = angles.map((angle, i) =>
    gravityChannels(size, opts, VARIANT_SEEDS[i], angle))
  // one dispGradPerTexel serves all variants, so every d channel divides by
  // the SAME sigma (the straight variant's); the ~1% per-variant variance
  // difference is harmless, a mismatched gradient scale is not
  const sigmaD = variants[1].sigmaD
  const textures = []
  const heights = []
  for (const v of variants) {
    for (let i = 0; i < v.d.length; i++) v.d[i] /= -sigmaD
    const [hx, hy] = gradients(v.h, size)
    textures.push(createNoiseTexture(device, size, v.h, v.d, hx, hy))
    heights.push(v.h)
  }
  const [hx0] = gradients(variants[1].h, size)
  return {
    textures,
    size,
    wavesPerTile: wavesPerTile(variants[1].h, hx0, size),
    dispGradPerTexel: -1 / sigmaD,
    heights,
    variants: angles.map((angle, i) => ({
      name: `gravity ${Math.round(angle * 180 / Math.PI)}deg`,
      channels: { height: variants[i].h, disp: variants[i].d },
    })),
  }
}

function gravityChannels(size, opts, seed, angle) {
  const sigmaAlong = opts.sigmaAlong ?? 3
  const sigmaAlongWide = opts.sigmaAlongWide ?? 9
  const sigmaCross = opts.sigmaCross ?? 14
  const random = mulberry32(seed)

  const n = size * size
  const re = new Float64Array(n)
  const im = new Float64Array(n)
  re.set(randomArray(random, n))
  fft2d(re, im, size, false)

  const ca = Math.cos(angle)
  const sa = Math.sin(angle)
  const dRe = new Float64Array(n)
  const dIm = new Float64Array(n)
  for (let ky = 0; ky < size; ky++) {
    const wy = 2 * Math.PI * (ky > size / 2 ? ky - size : ky) / size
    for (let kx = 0; kx < size; kx++) {
      const i = ky * size + kx
      const wx = 2 * Math.PI * (kx > size / 2 ? kx - size : kx) / size
      const wAlong = wx * ca + wy * sa
      const wCross = -wx * sa + wy * ca
      const g = (comboTransferAt(sigmaAlong, wAlong) - comboTransferAt(sigmaAlongWide, wAlong))
        * comboTransferAt(sigmaCross, wCross)
      const hr = re[i] * g
      const hi = im[i] * g
      re[i] = hr
      im[i] = hi
      // cumsum along the travel axis; the band-pass vanishes quadratically
      // at wAlong = 0, so the 1/w pole is harmless away from exact zero
      const ar = 1 - Math.cos(wAlong)
      const ai = Math.sin(wAlong)
      const m = ar * ar + ai * ai
      if (m > 1e-12) {
        dRe[i] = (hr * ar + hi * ai) / m
        dIm[i] = (hi * ar - hr * ai) / m
      } else {
        re[i] = 0
        im[i] = 0
      }
    }
  }
  fft2d(re, im, size, true)
  fft2d(dRe, dIm, size, true)

  const h = new Float32Array(n)
  for (let i = 0; i < n; i++) h[i] = re[i]
  let hSq = 0
  for (let i = 0; i < n; i++) hSq += h[i] * h[i]
  const sigmaH = Math.sqrt(hSq / n)
  const d = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    h[i] /= sigmaH
    d[i] = dRe[i] / sigmaH
  }
  let dSq = 0
  for (let i = 0; i < n; i++) dSq += d[i] * d[i]
  return { h, d, sigmaD: Math.sqrt(dSq / n) }
}

// Isotropic band-passed noise for capillary ripples (normal perturbation only).
// R: height, G: unused, B: dh/dx, A: dh/dy.
export function generateCapillaryNoiseTexture(device, opts = {}) {
  const size = opts.size ?? 256
  const sigmaSmall = opts.sigmaSmall ?? 2
  const sigmaLarge = opts.sigmaLarge ?? 6
  const random = mulberry32(opts.seed ?? 54321)

  const n = size * size
  const h = bandpass2D(randomArray(random, n), size, sigmaSmall, sigmaLarge)

  const [hx, hy] = gradients(h, size)
  return {
    texture: createNoiseTexture(device, size, h, null, hx, hy),
    size,
    wavesPerTile: wavesPerTile(h, hx, size),
    dispGradPerTexel: 0,
    channels: { height: h },
  }
}

// Sea-foam density pattern: a bubble-raft web (convergence lines where foam
// collects) times granular clumping noise. Rendered with a threshold driven
// by the accumulated foam age, so it must carry a full gradient of densities.
export function generateFoamPatternTexture(device, opts = {}) {
  const size = opts.size ?? 256
  const random = mulberry32(opts.seed ?? 777)

  const n = size * size
  const web = bandpass2D(randomArray(random, n), size, 2, 6)
  const mid = bandpass2D(randomArray(random, n), size, 3, 9)
  const fine = bandpass2D(randomArray(random, n), size, 1, 2.5)
  const density = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    const w = Math.max(0, 1 - 1.2 * Math.abs(web[i]))
    density[i] = Math.min(Math.max(0.55 * w * w + 0.25 + 0.18 * mid[i] + 0.12 * fine[i], 0), 1)
  }

  const mipCount = Math.log2(size) + 1
  const texture = device.createTexture({
    size: [size, size],
    format: 'rgba16float',
    mipLevelCount: mipCount,
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  })
  let cur = density
  let s = size
  for (let level = 0; level < mipCount; level++) {
    const data = new Uint16Array(s * s * 4)
    for (let i = 0; i < s * s; i++) data[i * 4] = floatToHalf(cur[i])
    device.queue.writeTexture({ texture, mipLevel: level }, data, { bytesPerRow: s * 8 }, [s, s])
    if (s > 1) {
      const half = s / 2
      const next = new Float32Array(half * half)
      for (let y = 0; y < half; y++) {
        for (let x = 0; x < half; x++) {
          const o = 2 * y * s + 2 * x
          next[y * half + x] = (cur[o] + cur[o + 1] + cur[o + s] + cur[o + s + 1]) / 4
        }
      }
      cur = next
      s = half
    }
  }

  return { texture, size, channels: { pattern: density } }
}

function bandpass2D(src, size, sigmaSmall, sigmaLarge) {
  const a = smoothAxisX(src, size, sigmaSmall)
  smoothAxisYInPlace(a, size, sigmaSmall)
  const b = smoothAxisX(src, size, sigmaLarge)
  smoothAxisYInPlace(b, size, sigmaLarge)
  for (let i = 0; i < a.length; i++) a[i] -= b[i]
  normalizeVariance(a)
  return a
}

function randomArray(random, n) {
  const data = new Float32Array(n)
  for (let i = 0; i < n; i++) data[i] = random() * 2 - 1
  return data
}

function gradients(h, size) {
  const n = size * size
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
  return [hx, hy]
}

// Dominant wavenumber from the spectral moment: k_rms = sqrt(E[h_x^2] / E[h^2])
function wavesPerTile(h, hx, size) {
  let hSq = 0, hxSq = 0
  for (let i = 0; i < h.length; i++) {
    hSq += h[i] * h[i]
    hxSq += hx[i] * hx[i]
  }
  return size * Math.sqrt(hxSq / hSq) / (2 * Math.PI)
}

function createNoiseTexture(device, size, r, g, b, a) {
  const n = size * size
  const data = new Uint16Array(n * 4)
  for (let i = 0; i < n; i++) {
    data[i * 4] = floatToHalf(r[i])
    data[i * 4 + 1] = g ? floatToHalf(g[i]) : 0
    data[i * 4 + 2] = floatToHalf(b[i])
    data[i * 4 + 3] = floatToHalf(a[i])
  }
  const texture = device.createTexture({
    size: [size, size],
    format: 'rgba16float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  })
  device.queue.writeTexture({ texture }, data, { bytesPerRow: size * 8 }, [size, size])
  return texture
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

// Analytic DTFT of the (infinite two-sided) combo kernel at frequency w;
// the circular kernel differs by the wrapped tail, negligible for sigma << size
function comboTransferAt(sigma, w) {
  const a = Math.exp(-1 / sigma)
  const cw = Math.cos(w)
  let num = 0
  let gain = 0
  const coeffs = [5, -4, 1]
  for (let k = 0; k < 3; k++) {
    const d = a ** (k + 1)
    num += coeffs[k] * (1 - d * d) / (1 - 2 * d * cw + d * d)
    gain += coeffs[k] * (1 + d) / (1 - d)
  }
  return num / gain
}

function fft1d(re, im, inverse) {
  const n = re.length
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1
    for (; j & bit; bit >>= 1) j ^= bit
    j ^= bit
    if (i < j) {
      let t = re[i]; re[i] = re[j]; re[j] = t
      t = im[i]; im[i] = im[j]; im[j] = t
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (inverse ? 2 : -2) * Math.PI / len
    const wr = Math.cos(ang)
    const wi = Math.sin(ang)
    for (let i = 0; i < n; i += len) {
      let cr = 1
      let ci = 0
      for (let j = 0; j < len >> 1; j++) {
        const a = i + j
        const b = a + (len >> 1)
        const tr = re[b] * cr - im[b] * ci
        const ti = re[b] * ci + im[b] * cr
        re[b] = re[a] - tr
        im[b] = im[a] - ti
        re[a] += tr
        im[a] += ti
        const t = cr * wr - ci * wi
        ci = cr * wi + ci * wr
        cr = t
      }
    }
  }
  if (inverse) {
    for (let i = 0; i < n; i++) {
      re[i] /= n
      im[i] /= n
    }
  }
}

function fft2d(re, im, size, inverse) {
  const lr = new Float64Array(size)
  const li = new Float64Array(size)
  for (let y = 0; y < size; y++) {
    const off = y * size
    lr.set(re.subarray(off, off + size))
    li.set(im.subarray(off, off + size))
    fft1d(lr, li, inverse)
    re.set(lr, off)
    im.set(li, off)
  }
  for (let x = 0; x < size; x++) {
    for (let y = 0; y < size; y++) {
      lr[y] = re[y * size + x]
      li[y] = im[y * size + x]
    }
    fft1d(lr, li, inverse)
    for (let y = 0; y < size; y++) {
      re[y * size + x] = lr[y]
      im[y * size + x] = li[y]
    }
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
