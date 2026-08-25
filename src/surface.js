// CPU replica of the rendered water surface: given a WORLD position, the
// height there and the velocity of the water particle sitting on it, so a
// scene can float things. Covers both the open sea and the swash film.
//
// The rendered mesh is Lagrangian — a vertex at material xz is drawn at
// xz + disp(xz) — so a world query has to invert that map. Offshore the
// fixed point p <- X - disp(p) does it: the surface is never folded (the
// renderer would break first), so |grad disp| < 1 and the iteration
// contracts. On the film the inversion is exact instead: the chain's node
// positions live on the CPU, so the material coordinate follows from a
// scan along the column.
//
// The velocity is a difference between two SNAPSHOTS of the scroll phases
// taken at the same material point, which is what a particle riding the
// surface does. Rebuilding it from the phase rates would mean a third copy
// of the dispersion bookkeeping that ocean.js and waveField.js already own.

import { bilinearWrap } from './noise.js'
import { REST_DEPTH, SLOPE } from './chain.js'

// must match wave_common.wgsl
const SIM_SPAN = 24
const SIM_BAND = 4
// must match ocean.wgsl
const COPY_FINE = 1.75
// The floor of cellForDistance: the mesh never resolves finer than this, so a
// slot the attenuation kills here is drawn nowhere. Attenuating on the floor
// rather than on the query's own distance keeps the surface a function of
// world position alone — the same point must not change height because the
// camera moved — and inside the floor's range (32 m, which is where anything
// small enough to float is visible) the two agree exactly.
const FINEST_CELL = 0.25
// the junction translation band of ribbonVertex
const JUNCTION_BLEND = 12

// Three passes leave a residue far below the wave amplitude. A fixed count
// keeps the query stateless: warm-starting from the previous answer would
// converge sooner but then every caller has to carry material coordinates
// and decide when its own motion has invalidated them.
const INVERT_ITERS = 3

const COPIES = 5

export class SurfaceQuery {
  constructor(noise, waveField, ocean, coast, chain, weights) {
    this.noise = noise
    this.waveField = waveField
    this.ocean = ocean
    this.coast = coast
    this.chain = chain
    this.weights = weights
    this.cur = emptySnapshot()
    this.prev = emptySnapshot()
    this.dt = 0
    this.wgt = new Float32Array(4)
    this.now = new Float32Array(3)
    this.was = new Float32Array(3)
  }

  update(dt) {
    const cache = this.ocean.layerCache
    const field = this.ocean.fieldCache
    if (!cache.length || !field) return
    const spent = this.prev
    this.prev = this.cur
    this.cur = spent
    const s = this.cur
    for (let i = 0; i < s.layers.length; i++) Object.assign(s.layers[i], cache[i])
    const d = this.waveField.data
    for (let k = 0; k < COPIES; k++) {
      s.off[k * 2] = d[k * 4]
      s.off[k * 2 + 1] = d[k * 4 + 1]
      s.w[k] = d[k * 4 + 2]
    }
    Object.assign(s, field)
    s.valid = true
    this.dt = this.prev.valid ? dt : 0
  }

  terrainHeight(x, z) {
    return Math.min(Math.max(SLOPE * this.coast.sampleSDF(x, z), -this.cur.seaDepth), 3)
  }

  // Material xz -> wave height and horizontal displacement, mirroring
  // sampleWaves() in ocean.wgsl. The band attenuation is part of that: the
  // finest slots are damped to near nothing as GEOMETRY even right under the
  // camera (the fragment normals carry them instead), so a floater that used
  // the unattenuated field would ride several centimetres of short-wave
  // wiggle that is nowhere on the drawn water.
  fieldAt(px, pz, s, wgt, out) {
    const { heights, disps, size } = this.noise
    let h = 0
    let dx = 0
    let dz = 0
    for (let i = 0; i < s.layers.length; i++) {
      const l = s.layers[i]
      const att = 1 - smoothstep(2, 6, FINEST_CELL * l.invL * size * COPY_FINE)
      const w = wgt[i] * att
      if (w <= 0 || l.amp <= 0) continue
      const u0 = (px * l.dx + pz * l.dz) * l.invL + l.su
      const v0 = (-px * l.dz + pz * l.dx) * l.invL + l.sv
      let sh = 0
      let sd = 0
      for (let k = 0; k < COPIES; k++) {
        const uu = u0 + s.off[k * 2]
        const vv = v0 + s.off[k * 2 + 1]
        sh += s.w[k] * bilinearWrap(heights[k], size, uu, vv)
        sd += s.w[k] * bilinearWrap(disps[k], size, uu, vv)
      }
      const amp = l.amp * w
      h += amp * sh
      const c = s.choppiness * amp * sd
      dx += c * l.dx
      dz += c * l.dz
    }
    const eta = Math.max(h * s.ampInv, 0)
    const lean = eta * eta / (1 + eta) / s.ampInv
    dx += s.leanX * lean
    dz += s.leanY * lean
    const ty0 = this.terrainHeight(px, pz)
    const wSea = 1 - smoothstep(-0.6, 0.1, ty0)
    const shallow = clamp(1 / Math.tanh(s.waveK * Math.max(-ty0, 0.05)), 1, 2.5)
    const m = shallow * wSea
    out[0] = h
    out[1] = dx * m
    out[2] = dz * m
  }

  // out: { y, vx, vy, vz, depth, film }. depth inherits the renderer's
  // softClamp floor, so dry sand reads 0.1 rather than 0.
  sample(x, z, out) {
    out.y = 0
    out.vx = 0
    out.vy = 0
    out.vz = 0
    out.depth = 0
    out.film = 0
    const s = this.cur
    if (!s.valid) return out

    let px = x
    let pz = z
    for (let it = 0; it < INVERT_ITERS; it++) {
      this.weights.sample(px, pz, this.wgt)
      this.fieldAt(px, pz, s, this.wgt, this.now)
      px = x - this.now[1]
      pz = z - this.now[2]
    }
    this.weights.sample(px, pz, this.wgt)
    this.fieldAt(px, pz, s, this.wgt, this.now)

    const ty = this.terrainHeight(x, z)
    let y = softClamp(this.now[0], ty)
    let vx = 0
    let vy = 0
    let vz = 0
    if (this.dt > 0) {
      this.fieldAt(px, pz, this.prev, this.wgt, this.was)
      vx = (this.now[1] - this.was[1]) / this.dt
      vz = (this.now[2] - this.was[2]) / this.dt
      vy = (y - softClamp(this.was[0], ty)) / this.dt
    }

    const f = this.filmAt(x, z, ty)
    if (f) {
      y += (f.y - y) * f.blend
      vx += (f.vx - vx) * f.blend
      vy += (f.vy - vy) * f.blend
      vz += (f.vz - vz) * f.blend
      out.film = f.blend
    }
    out.y = y
    out.vx = vx
    out.vy = vy
    out.vz = vz
    out.depth = Math.max(y - ty, 0)
    return out
  }

  // The film's world mapping, inverted. Returns null offshore of the
  // handover band and landward of the waterline tip, where the ribbon does
  // not render and the wave branch already has the answer.
  filmAt(x, z, ty) {
    const chain = this.chain
    if (!chain?.juncWorld) return null
    const sJ = chain.sJ
    // sampleSDF is the same signed normal distance the chain works in, so it
    // rejects the open sea before the per-column search runs
    if (this.coast.sampleSDF(x, z) < sJ - SIM_BAND - 2) return null

    const col = nearestColumn(chain, x, z)
    if (!col) return null
    const { nx, nz, px, pz, j0, j1, a } = col
    const s = (x - px) * nx + (z - pz) * nz

    const nodes = chain.nodes
    const step = SIM_SPAN / (nodes - 1)
    const dispAt = i => {
      const rest = sJ + i * chain.Lr
      const d0 = chain.x[j0 * nodes + i] - rest
      const d1 = chain.x[j1 * nodes + i] - rest
      return d0 + (d1 - d0) * a
    }
    const velAt = i => {
      const v0 = chain.u[j0 * nodes + i]
      const v1 = chain.u[j1 * nodes + i]
      return v0 + (v1 - v0) * a
    }
    // The ribbon renders the film near the junction as the rest profile
    // TRANSLATED by the junction displacement, so the world mapping to
    // invert is that one, not the raw node positions.
    const dJ = dispAt(0)
    const worldS = (i, d) => sJ + i * chain.Lr + mixJunction(dJ, d, i * step)

    let b = 0
    let disp = dJ
    let vel = velAt(0)
    let prevS = worldS(0, dJ)
    if (s <= prevS) {
      // seaward of the junction the mapping is identity plus the junction
      // displacement, so b follows directly
      b = s - sJ - dJ
      if (b < -SIM_BAND) return null
    } else {
      let found = false
      for (let i = 1; i < nodes; i++) {
        const d = dispAt(i)
        const si = worldS(i, d)
        if (s <= si) {
          const t = si > prevS ? (s - prevS) / (si - prevS) : 0
          b = (i - 1 + t) * step
          const dPrev = dispAt(i - 1)
          disp = dPrev + (d - dPrev) * t
          const vPrev = velAt(i - 1)
          vel = vPrev + (velAt(i) - vPrev) * t
          found = true
          break
        }
        prevS = si
      }
      // landward of the tip: dry sand, nothing rendered
      if (!found) return null
    }

    const tTip = clamp(b / SIM_SPAN, 0, 1)
    const tyJ = SLOPE * (sJ + dJ)
    const y = Math.max(ty, tyJ) + REST_DEPTH * (1 - tTip)
    // A film particle rides the wedge, so its rise follows the slope times
    // its shore-normal speed; its own thickness is a material constant
    const vs = mixJunction(velAt(0), vel, b)
    return {
      y,
      vx: vs * nx,
      vy: SLOPE * vs,
      vz: vs * nz,
      blend: smoothstep(-SIM_BAND, 0, b),
    }
  }
}

// The two chain columns bracketing (x, z) alongshore, and the fraction
// between them. Columns are searched by their shoreline point; mainland and
// island columns are separate runs and never pair up.
function nearestColumn(chain, x, z) {
  const sJ = chain.sJ
  const shore = (j, out) => {
    out[0] = chain.juncWorld[j * 2] - chain.normal[j * 2] * sJ
    out[1] = chain.juncWorld[j * 2 + 1] - chain.normal[j * 2 + 1] * sJ
  }
  const p = chain.scratchA ??= new Float32Array(2)
  const q = chain.scratchB ??= new Float32Array(2)
  let best = -1
  let bd = Infinity
  for (let j = 0; j < chain.cols; j++) {
    shore(j, p)
    const d = (p[0] - x) ** 2 + (p[1] - z) ** 2
    if (d < bd) { bd = d; best = j }
  }
  if (best < 0) return null
  const island = best >= chain.mainCols
  const lo = island ? chain.mainCols : 0
  const hi = island ? chain.cols : chain.mainCols
  const wrap = j => island ? lo + ((j - lo) % (hi - lo) + (hi - lo)) % (hi - lo) : Math.min(Math.max(j, lo), hi - 1)
  shore(best, p)
  const prev = wrap(best - 1)
  const next = wrap(best + 1)
  shore(prev, q)
  const dPrev = (q[0] - x) ** 2 + (q[1] - z) ** 2
  shore(next, q)
  const dNext = (q[0] - x) ** 2 + (q[1] - z) ** 2
  const other = dPrev < dNext ? prev : next
  shore(other, q)
  const ex = q[0] - p[0]
  const ez = q[1] - p[1]
  const len2 = ex * ex + ez * ez
  const a = len2 > 1e-9 ? clamp(((x - p[0]) * ex + (z - p[1]) * ez) / len2, 0, 1) : 0
  const nx0 = chain.normal[best * 2]
  const nz0 = chain.normal[best * 2 + 1]
  const nx1 = chain.normal[other * 2]
  const nz1 = chain.normal[other * 2 + 1]
  let nx = nx0 + (nx1 - nx0) * a
  let nz = nz0 + (nz1 - nz0) * a
  const inv = 1 / Math.max(Math.hypot(nx, nz), 1e-6)
  nx *= inv
  nz *= inv
  return { nx, nz, px: p[0] + ex * a, pz: p[1] + ez * a, j0: best, j1: other, a }
}

function mixJunction(atJunction, value, b) {
  const w = smoothstep(0, JUNCTION_BLEND, b)
  return atJunction + (value - atJunction) * w
}

function softClamp(height, ty) {
  const dy = height - (ty + 0.1)
  return ty + 0.1 + 0.5 * (dy + Math.sqrt(dy * dy + 0.0225))
}

function smoothstep(a, b, x) {
  const t = clamp((x - a) / (b - a), 0, 1)
  return t * t * (3 - 2 * t)
}

function clamp(x, a, b) {
  return Math.min(Math.max(x, a), b)
}

function emptySnapshot() {
  return {
    layers: [0, 1, 2, 3].map(() => ({ dx: 1, dz: 0, invL: 0, amp: 0, su: 0, sv: 0 })),
    off: new Float32Array(COPIES * 2),
    w: new Float32Array(COPIES),
    choppiness: 0,
    ampInv: 1,
    waveK: 1,
    leanX: 0,
    leanY: 0,
    seaDepth: 8,
    valid: false,
  }
}

export function createSurfaceSample() {
  return { y: 0, vx: 0, vy: 0, vz: 0, depth: 0, film: 0 }
}
