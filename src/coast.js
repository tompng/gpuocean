// The authored coastline and its baked forms. The shapes below are the
// authoring surface: a mainland polyline (any non-self-intersecting curve,
// straight beyond |z| ~ 350 so it matches the far-field fallback) and
// closed island loops. Baking produces a signed-distance texture for the
// terrain and arclength tables for the ribbons and chain columns.
//
// Conventions: dense samples run in order with land on the (dz, -dx) side
// of the tangent — the mainland in +z order with land at +x, islands
// wound clockwise (land inside).

const SDF_SIZE = 512
// must match SDF_EXTENT / BASE_SHORE_X in wave_common.wgsl
const SDF_EXTENT = 384
const BASE_SHORE_X = 10
export const MAIN_TABLE_N = 2048
export const MAIN_TABLE_STEP = 0.8
const MAIN_COLS = 160
const ISLAND_COLS = 96

function mainlandX(z) {
  const w = Math.min(Math.abs(z) / 330, 1)
  const fade = 1 - w * w * (3 - 2 * w)
  const curve = 0.6 * (6 * Math.sin(z * 0.041) + 3.5 * Math.sin(z * 0.093 + 1.7))
  const cape = -26 * Math.exp(-(((z + 70) / 38) ** 2))
  return BASE_SHORE_X + fade * (curve + cape)
}

function islandPoint(s) {
  // clockwise wobbly circle; s in [0, 1)
  const th = -s * 2 * Math.PI
  const r = 20 + 3 * Math.sin(3 * th + 1)
  return [-45 + Math.cos(th) * r, 15 + Math.sin(th) * r]
}

// Resample an ordered dense point list to uniform arclength; returns
// positions, landward normals, and the step length
function resample(pts, count, closed) {
  const cum = [0]
  for (let i = 1; i < pts.length; i++) {
    cum.push(cum[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]))
  }
  const total = closed
    ? cum[pts.length - 1] + Math.hypot(pts[0][0] - pts[pts.length - 1][0], pts[0][1] - pts[pts.length - 1][1])
    : cum[pts.length - 1]
  const step = total / (closed ? count : count - 1)
  const P = new Float32Array(count * 2)
  const N = new Float32Array(count * 2)
  let seg = 0
  for (let i = 0; i < count; i++) {
    const t = i * step
    while (seg < pts.length - 2 && cum[seg + 1] < t) seg++
    const a = pts[seg]
    const b = pts[Math.min(seg + 1, pts.length - 1)]
    const len = Math.max(cum[Math.min(seg + 1, pts.length - 1)] - cum[seg], 1e-6)
    const f = Math.min((t - cum[seg]) / len, 1)
    P[i * 2] = a[0] + (b[0] - a[0]) * f
    P[i * 2 + 1] = a[1] + (b[1] - a[1]) * f
  }
  for (let i = 0; i < count; i++) {
    const i0 = closed ? (i + count - 1) % count : Math.max(i - 1, 0)
    const i1 = closed ? (i + 1) % count : Math.min(i + 1, count - 1)
    const dx = P[i1 * 2] - P[i0 * 2]
    const dz = P[i1 * 2 + 1] - P[i0 * 2 + 1]
    const inv = 1 / Math.max(Math.hypot(dx, dz), 1e-6)
    N[i * 2] = dz * inv
    N[i * 2 + 1] = -dx * inv
  }
  return { P, N, step, total }
}

export function buildCoast(device) {
  // mainland: dense sampling of the authored graph, arclength-resampled
  const zSpan = (MAIN_TABLE_N - 1) * MAIN_TABLE_STEP / 2 + 40
  const dense = []
  for (let z = -zSpan; z <= zSpan; z += 0.5) dense.push([mainlandX(z), z])
  const main = resample(dense, MAIN_TABLE_N, false)
  // center the arclength parameter on the point nearest z = 0
  let center = 0
  for (let i = 0; i < MAIN_TABLE_N; i++) {
    if (Math.abs(main.P[i * 2 + 1]) < Math.abs(main.P[center * 2 + 1])) center = i
  }
  main.t0 = -center * main.step

  const islDense = []
  for (let s = 0; s < 1; s += 1 / 512) islDense.push(islandPoint(s))
  const island = resample(islDense, ISLAND_COLS, true)

  const coast = {
    main,
    island,
    islandFine: resample(islDense, ISLAND_COLS * 4, true),
    // arclength t -> P, N; straight along the end tangents beyond the table
    sampleMain(t, out) {
      const f = (t - main.t0) / main.step
      const fc = Math.min(Math.max(f, 0), MAIN_TABLE_N - 1)
      const j0 = Math.min(Math.floor(fc), MAIN_TABLE_N - 2)
      const a = fc - j0
      let px = main.P[j0 * 2] * (1 - a) + main.P[j0 * 2 + 2] * a
      let pz = main.P[j0 * 2 + 1] * (1 - a) + main.P[j0 * 2 + 3] * a
      const nx = main.N[j0 * 2] * (1 - a) + main.N[j0 * 2 + 2] * a
      const nz = main.N[j0 * 2 + 1] * (1 - a) + main.N[j0 * 2 + 3] * a
      const inv = 1 / Math.max(Math.hypot(nx, nz), 1e-6)
      const over = (f - fc) * main.step
      // tangent = landward normal rotated -90: (-nz, nx)
      px += -nz * inv * over
      pz += nx * inv * over
      out[0] = px; out[1] = pz; out[2] = nx * inv; out[3] = nz * inv
    },
    nearestMainArc(x, z) {
      let best = 0
      let bd = Infinity
      for (let i = 0; i < MAIN_TABLE_N; i += 4) {
        const d = (main.P[i * 2] - x) ** 2 + (main.P[i * 2 + 1] - z) ** 2
        if (d < bd) { bd = d; best = i }
      }
      return main.t0 + best * main.step
    },
  }

  bakeSDF(device, coast)
  uploadMainTable(device, coast)
  return coast
}

// Brute-force signed distance to the dense coast samples, bucketed for
// speed; the sign comes from the nearest sample's landward normal
function bakeSDF(device, coast) {
  const pts = []
  const { main, island } = coast
  for (let i = 0; i < MAIN_TABLE_N; i++) {
    if (Math.abs(main.P[i * 2]) < SDF_EXTENT + 60 && Math.abs(main.P[i * 2 + 1]) < SDF_EXTENT + 60) {
      pts.push([main.P[i * 2], main.P[i * 2 + 1], main.N[i * 2], main.N[i * 2 + 1]])
    }
  }
  // the column table's ~1.3m island spacing would leave point-sampling
  // bumps at the sdf texel size, so the sdf uses its own finer resample
  const islFine = coast.islandFine
  for (let i = 0; i < islFine.P.length / 2; i++) {
    pts.push([islFine.P[i * 2], islFine.P[i * 2 + 1], islFine.N[i * 2], islFine.N[i * 2 + 1]])
  }
  const BUCKET = 24
  const NB = Math.ceil(2 * (SDF_EXTENT + 80) / BUCKET)
  const buckets = Array.from({ length: NB * NB }, () => [])
  const bIndex = (x, z) => {
    const bx = Math.min(Math.max(Math.floor((x + SDF_EXTENT + 80) / BUCKET), 0), NB - 1)
    const bz = Math.min(Math.max(Math.floor((z + SDF_EXTENT + 80) / BUCKET), 0), NB - 1)
    return bz * NB + bx
  }
  for (const p of pts) buckets[bIndex(p[0], p[1])].push(p)

  const data = new Uint16Array(SDF_SIZE * SDF_SIZE)
  const raw = new Float32Array(SDF_SIZE * SDF_SIZE)
  const texel = 2 * SDF_EXTENT / SDF_SIZE
  for (let iz = 0; iz < SDF_SIZE; iz++) {
    const z = -SDF_EXTENT + (iz + 0.5) * texel
    for (let ix = 0; ix < SDF_SIZE; ix++) {
      const x = -SDF_EXTENT + (ix + 0.5) * texel
      let bd = Infinity
      let bp = null
      const bx = Math.floor((x + SDF_EXTENT + 80) / BUCKET)
      const bz = Math.floor((z + SDF_EXTENT + 80) / BUCKET)
      for (let ring = 0; ring < NB; ring++) {
        if (bd < ((ring - 1) * BUCKET) ** 2 && ring > 1) break
        for (let dz = -ring; dz <= ring; dz++) {
          for (let dx = -ring; dx <= ring; dx++) {
            if (Math.max(Math.abs(dx), Math.abs(dz)) !== ring) continue
            const cx = bx + dx
            const cz = bz + dz
            if (cx < 0 || cz < 0 || cx >= NB || cz >= NB) continue
            for (const p of buckets[cz * NB + cx]) {
              const d = (p[0] - x) ** 2 + (p[1] - z) ** 2
              if (d < bd) { bd = d; bp = p }
            }
          }
        }
      }
      let sd = Math.sqrt(bd)
      if (bp && (x - bp[0]) * bp[2] + (z - bp[1]) * bp[3] < 0) sd = -sd
      raw[iz * SDF_SIZE + ix] = sd
      data[iz * SDF_SIZE + ix] = f16(sd)
    }
  }
  const texture = device.createTexture({
    size: [SDF_SIZE, SDF_SIZE],
    format: 'r16float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  })
  device.queue.writeTexture({ texture }, data, { bytesPerRow: SDF_SIZE * 2 }, [SDF_SIZE, SDF_SIZE])
  coast.sdfView = texture.createView()
  // CPU-side replica of the shader's coastSDF (bilinear + far-field blend),
  // for classifying tiles as land or sea
  coast.sampleSDF = (x, z) => {
    const u = (x / (2 * SDF_EXTENT) + 0.5) * SDF_SIZE - 0.5
    const v = (z / (2 * SDF_EXTENT) + 0.5) * SDF_SIZE - 0.5
    const cl = (a) => Math.min(Math.max(a, 0), SDF_SIZE - 1)
    const u0 = cl(Math.floor(u)); const v0 = cl(Math.floor(v))
    const u1 = cl(u0 + 1); const v1 = cl(v0 + 1)
    const fu = Math.min(Math.max(u - u0, 0), 1); const fv = Math.min(Math.max(v - v0, 0), 1)
    const baked = (raw[v0 * SDF_SIZE + u0] * (1 - fu) + raw[v0 * SDF_SIZE + u1] * fu) * (1 - fv)
                + (raw[v1 * SDF_SIZE + u0] * (1 - fu) + raw[v1 * SDF_SIZE + u1] * fu) * fv
    const m = Math.max(Math.abs(x), Math.abs(z))
    const t = Math.min(Math.max((m - (SDF_EXTENT - 48)) / 40, 0), 1)
    const far = t * t * (3 - 2 * t)
    return baked * (1 - far) + (x - BASE_SHORE_X) * far
  }
}

function uploadMainTable(device, coast) {
  const { main } = coast
  const data = new Float32Array(MAIN_TABLE_N * 4)
  for (let i = 0; i < MAIN_TABLE_N; i++) {
    data[i * 4] = main.P[i * 2]
    data[i * 4 + 1] = main.P[i * 2 + 1]
    data[i * 4 + 2] = main.N[i * 2]
    data[i * 4 + 3] = main.N[i * 2 + 1]
  }
  const texture = device.createTexture({
    size: [MAIN_TABLE_N, 1],
    format: 'rgba32float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  })
  device.queue.writeTexture({ texture }, data, { bytesPerRow: MAIN_TABLE_N * 16 }, [MAIN_TABLE_N, 1])
  coast.mainTableView = texture.createView()
}

const f16F32 = new Float32Array(1)
const f16U32 = new Uint32Array(f16F32.buffer)

function f16(v) {
  f16F32[0] = v
  const b = f16U32[0]
  const sign = (b >>> 16) & 0x8000
  let exp = ((b >>> 23) & 0xff) - 127 + 15
  const mant = (b >>> 13) & 0x3ff
  if (exp <= 0) return sign
  if (exp >= 31) return sign | 0x7bff
  return sign | (exp << 10) | mant
}
