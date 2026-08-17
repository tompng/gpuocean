@group(0) @binding(3) var capTex: texture_2d<f32>;
@group(0) @binding(4) var foamTex: texture_2d<f32>;
@group(0) @binding(5) var foamPatTex: texture_2d<f32>;
// Photographic foam plates, ordered along the coverage ramp:
// 0 sparse lace, 1 mid sheets with flow streaks, 2 dense bubble raft
@group(0) @binding(6) var foamPlates: texture_2d_array<f32>;
// Coast table: per column (P.x, P.z, N.x, N.z), written by chain.js
@group(0) @binding(8) var coastTex: texture_2d<f32>;
// Linear radiance of the submerged scene, rendered in its own pass. rgb is the
// lit floor, a is a submerged mask used to reject taps that land above water.
@group(0) @binding(9) var refrTex: texture_2d<f32>;
@group(0) @binding(10) var refrSamp: sampler;
@group(1) @binding(0) var filmFoamTex: texture_2d<f32>;

struct VSOut {
  @builtin(position) clip: vec4f,
  // undisplaced rest world position: anchors wave sampling and world foam
  @location(0) gridXZ: vec2f,
  @location(1) world: vec3f,
  // fragments with cut > 0 are discarded: the grid mesh ends just past the
  // junction, hidden under the shore ribbon's overlap
  @location(2) cut: f32,
  // film material coordinate (band b, column); the grid parks it far seaward
  @location(3) st: vec2f,
  // wave-sampling coordinate for fragment normals: on the ribbons it blends
  // to the REST-compressed position so that at rest it coincides with the
  // world position — the scroll waves' normal Jacobians then stay continuous
  // across the junction instead of kinking into ripple-strength stripes
  @location(4) waveXZ: vec2f,
  // film stretch: rendered world meters per band meter (rest ~0.07)
  @location(5) stretch: f32,
}

fn coastAt(col: f32) -> vec4f {
  let c = wrapCol(col);
  var j0 = i32(floor(c));
  var j1 = j0 + 1;
  if (j0 >= MAIN_COLS) {
    if (j1 >= SIM_COLS) { j1 = MAIN_COLS; }
  } else {
    j1 = min(j1, MAIN_COLS - 1);
  }
  let a = c - floor(c);
  return mix(textureLoad(coastTex, vec2i(j0, 0), 0), textureLoad(coastTex, vec2i(j1, 0), 0), a);
}

fn filmFoamAt(b: f32, col: f32) -> vec4f {
  let fx = clamp((b + SIM_BAND) / (SIM_BAND + SIM_SPAN) * 127.0, 0.0, 127.0);
  let c = wrapCol(col);
  var j0 = i32(floor(c));
  var j1 = j0 + 1;
  if (j0 >= MAIN_COLS) {
    if (j1 >= SIM_COLS) { j1 = MAIN_COLS; }
  } else {
    j1 = min(j1, MAIN_COLS - 1);
  }
  let i0 = i32(floor(fx));
  let i1 = min(i0 + 1, 127);
  let a = fx - floor(fx);
  let fb = c - floor(c);
  return mix(
    mix(textureLoad(filmFoamTex, vec2i(i0, j0), 0), textureLoad(filmFoamTex, vec2i(i1, j0), 0), a),
    mix(textureLoad(filmFoamTex, vec2i(i0, j1), 0), textureLoad(filmFoamTex, vec2i(i1, j1), 0), a), fb);
}

// Caustic web on the sand: bright filaments along the zero-crossing lines of
// two drifting noise fields. Shared by the surface (sampled at the refracted
// bottom point, so the pattern swims with the surface) and by the sea floor
// seen directly from underwater, so both show the same web.
fn causticWeb(xz: vec2f) -> f32 {
  let cs = textureSample(capTex, samp, xz / (13.0 * u.causticScale) + vec2f(0.023, 0.011) * u.time).x
         + textureSample(capTex, samp, xz / (8.7 * u.causticScale) + vec2f(-0.017, 0.019) * u.time).x;
  return pow(max(0.0, 1.0 - 0.6 * abs(cs)), 4.0);
}

// Foam density at a point, built from the three plates.
//
// The plates span the coverage ramp — lace between dark water, then broken
// sheets with holes and flow edges, then near-solid raft — and the SAME
// coverage that picks between them also sets the erosion threshold below, so
// plate and cut always agree.
//
// Their densities are blended and the result eroded once. Eroding each plate
// and cross-fading the three masks instead would give half-grey ghost foam
// everywhere two overlap, because the masks are near-binary.
fn foamPlate(xz: vec2f, flow: vec2f, cover: f32, lod: f32) -> f32 {
  let s = 1.0 / (5.0 * u.noiseScale);
  let drift = 0.25 * u.noiseSpeed * u.time;
  // Sparse never drifts: it is the stranded residue at the high-water mark and
  // has to stay locked to the water it was deposited on
  let sparse = textureSampleLevel(foamPlates, samp, xz * s, 0, lod).r;
  // Mid stretches ALONG the flow only. Stretching along by k while also
  // squeezing across by k is area-preserving but distorts every feature by k^2
  // — at k = 2.5 a round bubble becomes a 6:1 smear, and since this plate
  // already carries flow streaks photographically it double-counts them.
  // Elongating one axis keeps the cross-flow scale honest.
  let k = 1.0 + 1.2 * u.streaks;
  let along = dot(xz, flow) / k + drift;
  let across = dot(xz, vec2f(-flow.y, flow.x));
  let mid = textureSampleLevel(foamPlates, samp, vec2f(along, across) * s, 1, lod).r;
  // The raft on a fresh crest is the part that visibly churns, so it drifts
  // faster; 2.2 keeps it inherently finer than the lace
  let dense = textureSampleLevel(foamPlates, samp, (xz + flow * (drift * 1.7)) * (s * 2.2 / u.crestScale), 2, lod).r;
  // plateSel < 0 is the coverage blend. Otherwise a single plate is forced, so
  // each can be inspected on its own; the erosion threshold still follows
  // coverage, so a forced plate is shown eroded exactly as it would be in the
  // blend. The branch is on a uniform, and the samples are already taken above,
  // so no texture call sits in non-uniform control flow.
  if (u.plateSel >= 0.0) {
    let sel = i32(u.plateSel + 0.5);
    if (sel == 0) { return sparse; }
    if (sel == 1) { return mid; }
    if (sel == 2) { return dense; }
    // the procedural pattern this replaced, kept for an A/B against the plates
    return textureSampleLevel(foamPatTex, samp, xz * s, lod).r;
  }
  return mix(mix(sparse, mid, smoothstep(u.laceLow, u.laceHigh, cover)),
             dense, smoothstep(u.crestStart, u.crestFull, cover));
}

// Particulate suspended in the column. Not a volumetric march: the motes ride
// the fog weight, so they thicken with path length the way real particulate
// does, without a second pass.
fn suspended(xz: vec2f, fogAmount: f32) -> f32 {
  let m = textureSample(capTex, samp, xz * 2.6 + vec2f(0.021, -0.014) * u.time).x;
  return u.particleDensity * 0.11 * smoothstep(0.1, 0.75, m) * fogAmount;
}

struct VSIn {
  @location(0) pos: vec2f,
  @location(1) cell: f32,
}

struct WaveSample {
  height: f32,
  disp: vec2f,
}

fn sampleWaves(xz: vec2f, cell: f32) -> WaveSample {
  var height = 0.0;
  var disp = vec2f(0.0);
  for (var i = 0; i < i32(u.numLayers); i++) {
    let l = u.layers[i];
    // The noise is band-limited, so a smooth attenuation on the layer's
    // texel footprint stands in for mip filtering: coarse cells fade the
    // layer out instead of aliasing vertex heights, with no level seams.
    // The cutoff sits well below the band's Nyquist (~14 texels): with
    // fewer than ~5 vertices per wave the geometry reads as polygons, so
    // the height dies early and the fragment normals carry the detail.
    let att = 1.0 - smoothstep(2.0, 6.0, cell * l.dirScaleAmp.z * u.hGrad);
    let s = textureSampleLevel(waveTex, samp, layerUV(xz, i), 0.0);
    height += l.dirScaleAmp.w * s.x * att;
    disp += (u.choppiness * l.dirScaleAmp.w * s.y * att) * l.dirScaleAmp.xy;
  }
  // Forward displacement through a convex ramp of crest-relative height:
  // only tall crests lean (a linear ramp would shear every scale by the same
  // angle, reading as wind-carved dunes), and f' saturates to bound the
  // front-face compression
  let eta = max(height * u.ampInv, 0.0);
  disp += vec2f(u.leanX, u.leanY) * (eta * eta / (1.0 + eta) / u.ampInv);
  // Horizontal orbital displacement with shallow amplification (≈ 1/tanh(kd)),
  // fading through the waterline band
  let ty0 = terrainHeight(xz);
  let wSea = 1.0 - smoothstep(-0.6, 0.1, ty0);
  let shallowAmp = clamp(1.0 / tanh(u.waveK * max(-ty0, 0.05)), 1.0, 2.5);
  return WaveSample(height, disp * shallowAmp * wSea);
}

fn softClamp(height: f32, ty: f32) -> f32 {
  // full wave motion, but the surface never sinks below the sand (kept a
  // hair above so the wetted film stays visible)
  let dy = height - (ty + 0.1);
  return ty + 0.1 + 0.5 * (dy + sqrt(dy * dy + 0.0225));
}

// The grid/land lattice is static and uniform; the vertex shader maps it
// around the camera with a radial warp — identity out to warpLinear, then
// exponentially growing cells. The center snaps to the lattice pitch so
// near vertices sit on a fixed world lattice (no swimming); far vertices
// slide with the camera, which is invisible because every field they
// sample is a function of world position.
// Hard cap on the screen-space refraction offset, in uv units
const REFR_CLAMP: f32 = 0.08;

const WARP_GROWTH: f32 = 1.12;

fn warpVertex(p: vec2f) -> vec3f {
  let snap = floor(u.cameraPos.xz / u.warpCell + 0.5) * u.warpCell;
  let r = length(p);
  if (r <= u.warpLinear) {
    return vec3f(snap + p, u.warpCell);
  }
  let k = min((r - u.warpLinear) / u.warpCell, 110.0);
  let g = pow(WARP_GROWTH, k);
  let rw = u.warpLinear + u.warpCell * (g - 1.0) / (WARP_GROWTH - 1.0);
  return vec3f(snap + p * (rw / r), u.warpCell * g);
}

// the grid's cell size at a given world distance from the camera: the
// geometric cell growth sums to an exactly linear distance-cell relation
fn warpCellAt(dist: f32) -> f32 {
  return WARP_CELL + max((WARP_GROWTH - 1.0) * (dist - WARP_LINEAR), 0.0);
}

// Open-ocean grid: pure scroll waves. Across the shore ribbon's seaward
// band it dives below the sand and is cut just past the junction, so the
// ribbon always covers it; at the band's seaward edge both meshes evaluate
// the same surface, so the overlap seam has matching shape and color.
@vertex
fn vs_grid(in: VSIn) -> VSOut {
  let wv = warpVertex(in.pos);
  let xz = wv.xy;
  let w = sampleWaves(xz, wv.z);
  let dispXZ = xz + w.disp;
  let ty = terrainHeight(dispXZ);
  // The cut keys on the DISPLACED position: a vertex materially seaward
  // of the band can displace landward past the film's junction and poke
  // out of the ribbon's cover, so it is judged by where it lands — the
  // extra discards open no gap, the ribbon's wave side and film span
  // everything landward of the junction continuously. The dive-under
  // stays on the MATERIAL position: the landed field folds along steep
  // displacement gradients, and a sink keyed on it opens pit walls along
  // wave fronts outside the ribbon's cover.
  let sOff = coastSDF(xz);
  let sOffD = coastSDF(dispXZ);
  let sJ0 = -REST_DEPTH / u.slope;
  // Full height like the ribbon's wave side, so the two surfaces agree at
  // the overlap band's seaward edge; inside the band the dive-under ramp
  // keeps the grid below the ribbon's blend toward the film
  var y = softClamp(w.height, ty);
  // The dive-under ramp is LINEAR: a smoothstep starts flat, leaving the
  // grid coincident with the ribbon deep into the band, where differing
  // tessellations let coarse grid cells poke through as shading stripes.
  // It fades out as cells outgrow the band: a coarse cell straddling the
  // ramp would open a visible pit, and far away the attenuated waves leave
  // the grid nearly coplanar with the ribbon anyway.
  y -= 1.5 * clamp((sOff - (sJ0 - SIM_BAND)) / SIM_BAND, 0.0, 1.0) * (1.0 - smoothstep(1.0, 3.0, wv.z));
  var out: VSOut;
  out.world = vec3f(dispXZ.x, y, dispXZ.y);
  out.gridXZ = xz;
  out.cut = sOffD - sJ0;
  out.st = vec2f(-1000.0, 0.0);
  out.waveXZ = xz;
  out.stretch = 1.0;
  out.clip = u.viewProj * vec4f(out.world, 1.0);
  return out;
}

// ribbon row pitch (28m band / 140 cells); the skirt spans two rows
const SKIRT_W: f32 = 0.2;
const SKIRT_DROP: f32 = 0.1;

// Shore ribbons cover the junction band and the film, ending exactly at
// the chain's material domain end (the waterline tip), so nothing renders
// landward of the tip. Vertex x is normalized over the ribbon's band; the
// film's world position lies along the column's landward normal at its
// displaced normal-distance s, so the swash runs shore-perpendicular.
fn ribbonVertex(b: f32, col: f32, coastP: vec2f, coastN: vec2f, cell: f32) -> VSOut {
  // The wave side anchors to the UNCOMPRESSED material position (band
  // meters along the normal) — the rest-compressed mapping is only for
  // placing the film. Wherever the wave side still renders (the handover
  // band, the faded segment ends), compressed anchors would smear the
  // world foam into streaks and kink the normals.
  let matWorld = coastP + coastN * (-REST_DEPTH / u.slope + b);
  // The ribbon's own lattice stays fine near the coast no matter where the
  // camera is, so the vs height attenuation also takes the open grid's cell
  // at this distance — otherwise a far coast keeps full wave height against
  // the grid's flattened surface and the seam shows a step
  let cellW = max(cell, warpCellAt(distance(u.cameraPos.xz, matWorld)));
  let w = sampleWaves(matWorld, cellW);
  let sb = simBlend(b);
  let chain = simState(b, col);
  let chainWorld = coastP + coastN * (simRestS(b) + chain.x);
  let dispXZ = mix(matWorld + w.disp, chainWorld, sb);
  let ty = terrainHeight(dispXZ);
  // The wave height carries at full strength up to the handover — the sb
  // mix below already ramps it out toward the film, whose junction rides
  // the (low-passed) local sea level itself; an extra (1 - sb) here would
  // attenuate the band twice and sag it below both neighbours
  let yWave = softClamp(w.height, ty);
  // Film thickness tapers from the junction's still-water column to zero
  // at the tip, so the junction sits exactly at sea level and at rest
  // terrain + thickness cancels to the flat sea. Seaward of the junction
  // (the blend ramp) the terrain keeps dropping while the column stays at
  // the junction value, so clamp the film's terrain at the junction's —
  // the extrapolation is then flat at sea level instead of sagging below
  let sJ = -REST_DEPTH / u.slope + simState(0.0, col).x;
  let tyJ = u.slope * sJ;
  let tyF = max(ty, tyJ);
  // The junction column is NOT its depth below the static z=0 level: the
  // junction moves because the local sea level moved with it, so the
  // column stays the still-water depth. Measured against z=0 the whole
  // film abruptly dries on run-up and deepens on run-down.
  let tTip = clamp(b / SIM_SPAN, 0.0, 1.0);
  var y = mix(yWave, tyF + REST_DEPTH * (1.0 - tTip), sb);
  // Skirt: two rows beyond the band's seaward edge; only the outer one
  // folds down below the grid, the inner one stays a FLAT margin. At the
  // edge itself the two tessellations disagree by interpolation error, and
  // a bare ribbon edge shows see-through slivers wherever the grid
  // interpolates below it. The flat margin matters because the grid's
  // dive-under drops per VERTEX: its surface starts bending one cell
  // before the ramp line, and if the skirt also folded immediately the two
  // surfaces would X-cross and cut a groove along the seam (worst in calm
  // water) — the overlap must run parallel before either side sinks.
  y -= SKIRT_DROP * clamp((-SIM_BAND - SKIRT_W - b) / SKIRT_W, 0.0, 1.0);
  var out: VSOut;
  out.world = vec3f(dispXZ.x, y, dispXZ.y);
  out.gridXZ = matWorld;
  out.cut = -1.0;
  out.st = vec2f(b, col);
  out.waveXZ = mix(matWorld, coastP + coastN * simRestS(b), sb);
  let eS = 1.0;
  out.stretch = abs(simRestS(b + eS) + simState(b + eS, col).x
                  - (simRestS(b - eS) + simState(b - eS, col).x)) / (2.0 * eS);
  out.clip = u.viewProj * vec4f(out.world, 1.0);
  return out;
}

// Mainland coast table: per arclength entry (P.x, P.z, N.x, N.z), with
// straight extrapolation along the end tangents beyond the table
@group(0) @binding(10) var mainTable: texture_2d<f32>;
const MAIN_TABLE_N: i32 = 2048;
const MAIN_TABLE_STEP: f32 = 0.8;

fn mainCoastAt(t: f32) -> vec4f {
  let f = t / MAIN_TABLE_STEP + f32(MAIN_TABLE_N - 1) * 0.5;
  let fc = clamp(f, 0.0, f32(MAIN_TABLE_N - 1));
  let j0 = min(i32(floor(fc)), MAIN_TABLE_N - 2);
  let a = fc - f32(j0);
  var c = mix(textureLoad(mainTable, vec2i(j0, 0), 0), textureLoad(mainTable, vec2i(j0 + 1, 0), 0), a);
  let n = normalize(c.zw);
  let over = (f - fc) * MAIN_TABLE_STEP;
  // tangent = landward normal rotated -90 degrees
  return vec4f(c.xy + vec2f(-n.y, n.x) * over, n);
}

@vertex
fn vs(in: VSIn) -> VSOut {
  // rows follow the camera's coast arclength, snapped to the fine row
  // pitch; columns map through the film window's moving center
  let t = u.simTCam + in.pos.y;
  let b = in.pos.x * (SIM_SPAN + SIM_BAND + 2.0 * SKIRT_W) - SIM_BAND - 2.0 * SKIRT_W;
  let col = clamp(((t - u.simZBase) / 160.0 + 0.5) * f32(MAIN_COLS - 1), 0.0, f32(MAIN_COLS - 1));
  let c = mainCoastAt(t);
  return ribbonVertex(b, col, c.xy, c.zw, in.cell);
}

@vertex
fn vs_island(in: VSIn) -> VSOut {
  let b = in.pos.x * (SIM_SPAN + SIM_BAND + 2.0 * SKIRT_W) - SIM_BAND - 2.0 * SKIRT_W;
  let col = in.pos.y;
  let c = coastAt(col);
  return ribbonVertex(b, col, c.xy, normalize(c.zw), in.cell);
}

struct NormalSample {
  n: vec3f,
  // horizontal displacement Jacobian — the generator's breaking measure,
  // band-attenuated exactly like the rendered waves
  jac: f32,
  // standard deviation of jac - 1, summed in closed form over the layers
  // (the noise height channel has unit variance): with the rendered
  // attenuation, and without it (the physical value)
  sigma: f32,
  sigmaP: f32,
}

fn surfaceNormal(xz: vec2f, rippleXZ: vec2f, dist: f32, eta: f32, hScale: f32) -> NormalSample {
  var dPx = vec3f(1.0, 0.0, 0.0);
  var dPz = vec3f(0.0, 0.0, 1.0);
  var varC = 0.0;
  var varP = 0.0;
  // per-pixel sampling footprint in meters; the same band-limited-noise
  // argument as the vertex shader replaces mip filtering with a smooth
  // per-layer attenuation (texels per pixel against the band's wavelength)
  let mpp = length(fwidth(xz));
  for (var i = 0; i < i32(u.numLayers); i++) {
    let l = u.layers[i];
    let dir = l.dirScaleAmp.xy;
    let invL = l.dirScaleAmp.z;
    let amp = l.dirScaleAmp.w * (1.0 - smoothstep(5.0, 14.0, mpp * l.dirScaleAmp.z * u.hGrad));
    let s = textureSample(waveTex, samp, layerUV(xz, i));
    let duvdx = vec2f(dir.x, -dir.y) * invL;
    let duvdz = vec2f(dir.y, dir.x) * invL;
    let grad = vec2f(s.z, s.w) * (u.hGrad * hScale);
    // D is the x-cumsum of h with sign and scale baked into dGrad, so ∂D/∂u = h * dGrad
    let dDdu = u.choppiness * amp * s.x * u.dGrad;
    dPx += vec3f(dir.x * dDdu * duvdx.x, amp * dot(grad, duvdx), dir.y * dDdu * duvdx.x);
    dPz += vec3f(dir.x * dDdu * duvdz.x, amp * dot(grad, duvdz), dir.y * dDdu * duvdz.x);
    let cAmp = u.choppiness * amp * u.dGrad * invL;
    let cAmpP = u.choppiness * l.dirScaleAmp.w * u.dGrad * invL;
    varC += cAmp * cAmp;
    varP += cAmpP * cAmpP;
  }
  let leanSlope = (eta * eta + 2.0 * eta) / ((1.0 + eta) * (1.0 + eta));
  dPx += vec3f(u.leanX * leanSlope * dPx.y, 0.0, u.leanY * leanSlope * dPx.y);
  dPz += vec3f(u.leanX * leanSlope * dPz.y, 0.0, u.leanY * leanSlope * dPz.y);
  let jac = dPx.x * dPz.z - dPz.x * dPx.z;
  let sigma = sqrt(varC);
  let sigmaP = sqrt(varP);
  // Ripples concentrate where the long waves strain the surface: orbital
  // convergence (compression, near crests) with the peak shifted toward the
  // front face. Layers 0-2 are isotropic wind ripples (weak bias); layers 3-5
  // are anisotropic parasitic-capillary ripples following the gravity waves.
  let front = smoothstep(0.0, 0.15, -dPx.y);
  let squeeze = smoothstep(0.0, 0.3, 2.0 - dPx.x - dPz.z);
  let conc = front + squeeze;
  let fade = clamp(1.0 - dist / (150.0 * u.lodScale), 0.0, 1.0);
  let isoScale = mix(1.0, conc, u.rippleBias * 0.4) * fade;
  let anisoScale = mix(1.0, conc, u.rippleBias) * fade;
  for (var i = 0; i < 6; i++) {
    let l = u.capLayers[i];
    let dir = l.dirScaleAmp.xy;
    let invL = l.dirScaleAmp.z;
    let uvc = vec2f(dot(rippleXZ, dir), dot(rippleXZ, vec2f(-dir.y, dir.x))) * invL + l.scroll.xy;
    var s: vec4f;
    var amp: f32;
    if (i < 3) {
      s = textureSample(capTex, samp, uvc);
      amp = isoScale * l.dirScaleAmp.w * u.capHGrad * (1.0 - smoothstep(5.0, 14.0, mpp * invL * u.capHGrad));
    } else {
      s = textureSample(waveTex, samp, uvc);
      amp = anisoScale * l.dirScaleAmp.w * u.hGrad * (1.0 - smoothstep(5.0, 14.0, mpp * invL * u.hGrad));
    }
    let grad = vec2f(s.z, s.w) * amp;
    dPx.y += dot(grad, vec2f(dir.x, -dir.y) * invL);
    dPz.y += dot(grad, vec2f(dir.y, dir.x) * invL);
  }
  return NormalSample(normalize(cross(dPz, dPx)), jac, sigma, sigmaP);
}

// far-foam calibration: extra crest passages counted per foam lifetime
const SWEEP_K: f32 = 1.2;

@fragment
fn fs(in: VSOut) -> @location(0) vec4f {
  if (in.cut > 0.0) {
    discard;
  }
  let dist = distance(u.cameraPos, in.world);
  let sbF = simBlend(in.st.x);
  // Gravity-wave normal detail follows the geometry, whose height dies
  // across the handover band; sampled in the film's compressed material it
  // would otherwise keep painting shading bumps onto the flat sheet.
  // Ripples instead switch to WORLD-space sampling over the film — wind
  // ripples propagate on their own rather than riding the swash flow, and
  // material sampling would compress them onto the wedge with a step at
  // the junction — so they survive onto the film without artifacts
  let rippleXZ = mix(in.gridXZ, in.world.xz, sbF);
  let ns = surfaceNormal(in.waveXZ, rippleXZ, dist, max(in.world.y * u.ampInv, 0.0), 1.0 - sbF);
  var n = ns.n;
  let ty = terrainHeight(in.world.xz);
  // The lower edge sits above the residual softmax offset left on dry sand,
  // which otherwise keeps fresnel and ripple glints alive landward of the film
  let column = max(in.world.y - ty, 0.0);
  let waterM = smoothstep(0.025, 0.09, column);
  // thin columns blend toward the actual terrain normal — a fixed proxy
  // would tilt the wrong way on beaches not facing the mainland's
  // direction, and fresnel at grazing angles amplifies that into a dark
  // sky-reflection rim around the waterline
  let eT = 0.5;
  let hx = terrainHeight(in.world.xz + vec2f(eT, 0.0)) - terrainHeight(in.world.xz - vec2f(eT, 0.0));
  let hz = terrainHeight(in.world.xz + vec2f(0.0, eT)) - terrainHeight(in.world.xz - vec2f(0.0, eT));
  let nTerr = normalize(vec3f(-hx / (2.0 * eT), 1.0, -hz / (2.0 * eT)));
  n = normalize(mix(nTerr, n, waterM));
  let v = normalize(u.cameraPos - in.world);
  let SKY = skyState(u.sunDir, u.skyTurbidity, u.skyRayleigh, u.skyIntensity);
  if (dot(n, v) < 0.0) { n = -n; }
  let fresnel = 0.02 + 0.98 * pow(1.0 - max(dot(n, v), 0.0), 5.0);
  let r = reflect(-v, n);
  let spec = sunTint(SKY) * (mix(8.0, 4.5, sunWarmth(SKY)) * pow(max(dot(r, u.sunDir), 0.0), 600.0));
  let fCenter = vec2f(u.foamCX, u.foamCZ);
  let fuv = (in.gridXZ - fCenter) / (2.0 * u.foamRegion) + 0.5;
  let edgeFade = 1.0 - smoothstep(0.85, 1.0, length(in.gridXZ - fCenter) / u.foamRegion);
  let foamRaw = textureSample(foamTex, samp, fuv).rgb;
  let foamAcc = foamRaw * edgeFade;
  // Beyond the window the buffer fades out; an instantaneous stand-in from
  // the actual wave field keeps whitecaps alive at any distance. Its
  // coverage is matched to the buffer's steady state analytically instead
  // of by eye: jac - 1 is ~Gaussian, so the probability of crossing the
  // generator's threshold follows from sigmaP, the number of independent
  // crest passages within the foam lifetime amplifies it into the swept
  // coverage, and the current field is then cut at its own quantile of
  // that coverage — the area matches by construction, placed on the most
  // compressed spots. The quantile is taken in the rendered field's sigma,
  // so band attenuation moves the cut instead of thinning distant foam.
  let sigmaR = max(ns.sigma, 1e-4);
  let pGen = 1.0 / (1.0 + exp(-1.702 * (u.foamThreshold - 1.0) / max(ns.sigmaP, 1e-4)));
  let period = 6.2832 / sqrt(9.81 * u.waveK);
  let cover = clamp(1.0 - pow(1.0 - pGen, 1.0 + SWEEP_K * u.foamLife / period), 1e-4, 0.6);
  let zQ = -log(1.0 / cover - 1.0) / 1.702;
  let zNow = (ns.jac - 1.0) / sigmaR;
  let waterGateF = smoothstep(0.0, 0.3, in.world.y - ty);
  let depthF = max(-ty, 0.05);
  let genSurfF = smoothstep(0.55, 0.9, in.world.y / depthF) * smoothstep(0.0, 0.5, depthF) * waterGateF;
  // The stand-in value must keep VARYING across its covered region: the
  // erosion reads it as a pattern threshold, so any plateau (a saturating
  // ramp) paints textureless flat white — the buffer's wakes always carry
  // their decay gradient. A soft exponential saturation only approaches
  // its ceiling, so a gradient survives everywhere; its width follows the
  // z-distribution's conditional tail beyond the quantile, and the
  // visibility level (~0.48) still crosses at the calibrated point, so the
  // matched coverage is untouched.
  let tailW = 1.0 / (1.0 + abs(zQ));
  let farJ = clamp(0.48 + 0.45 * (1.0 - exp((zNow - zQ + 0.28) / tailW)), 0.0, 1.0);
  let farR = max(farJ * waterGateF, genSurfF);
  // From high above, fragments inside the window are still far from the
  // camera: the buffer's wake shapes resolve to speckle while its
  // character differs from the stand-in outside, so the boundary shows.
  // Blend by camera distance as well as by the window edge.
  let bufBlend = edgeFade * (1.0 - smoothstep(u.foamRegion, 2.0 * u.foamRegion, dist));
  let accR = mix(farR, foamRaw.r, bufBlend);
  // Bubble clouds scatter multiply and emerge nearly isotropic (white water);
  // a mild forward lobe remains for thin backlit crests. The film is a sheet
  // too thin to hold a submerged bubble cloud, so the glow fades out there
  // and its foam reads as surface foam only
  let towardSun = max(0.0, -dot(v, u.sunDir));
  let sss = u.sssStrength * (0.55 + 0.45 * towardSun * towardSun) * foamAcc.g * (1.0 - sbF);
  // Flat sand bottom seen through the refracted view ray with per-channel
  // Beer-Lambert extinction; +1.4 is the sun-side path per meter of column
  // (refracted solar zenith ~44°)
  let refr = refract(-v, n, 0.752);
  // The grazing-path factor assumes laterally endless water; a grazing ray
  // through the film (e.g. down a bore front) exits the thin sheet almost
  // immediately, so cap its underwater path or the film's edge renders as
  // if seen through meters of water
  let lateral = mix(1.0 / max(-refr.y, 0.05), min(1.0 / max(-refr.y, 0.05), 2.0), sbF);
  let pathLen = column * (lateral + 1.4);
  let trans = exp(-waterSigma(u.sTurbidity, u.chlorophyll * u.sChlorophyll) * pathLen);
  // Sampled at the refracted bottom point so the web swims with the surface;
  // defocus fades it with column depth
  let tHit = column * lateral * u.distortionScale;
  let bottomXZ = in.world.xz + refr.xz * tHit;
  let web = causticWeb(bottomXZ);
  // Caustics need some water column to focus in; a centimeters-thin film
  // (or the residual softmax offset on dry sand) must not carry the web
  let focus = u.causticStrength * exp(-column * 0.12) * clamp(1.0 - dist / (120.0 * u.lodScale), 0.0, 1.0) * smoothstep(0.04, 0.25, column);
  // Analytic flat bottom, kept as the fallback wherever the screen-space tap
  // has nothing valid to say
  let sandA = vec3f(0.86, 0.78, 0.58) * (0.85 + focus * (1.6 * web - 0.18));
  // Screen-space refraction. The offset is the parallax between where the
  // refracted ray actually lands on the floor and this fragment's own pixel —
  // view-consistent by construction, shrinking with distance through the
  // perspective divide, and scaling with the water column for free because
  // tHit is proportional to it: a centimetres-thin film barely offsets, an 8 m
  // column offsets a lot, with no tuning constant.
  let dims = vec2f(textureDimensions(refrTex, 0));
  let uvSelf = in.clip.xy / dims;
  let hitClip = u.viewProj * vec4f(bottomXZ.x, in.world.y - column, bottomXZ.y, 1.0);
  var uvTap = uvSelf;
  if (hitClip.w > 1e-4) {
    let uvHit = (hitClip.xy / hitClip.w) * vec2f(0.5, -0.5) + 0.5;
    // Near the horizon tHit grows faster than the perspective divide shrinks
    // it, so cap the offset or a grazing tap flies across the frame
    var d = (uvHit - uvSelf) * u.distortionStrength;
    let dl = length(d);
    if (dl > REFR_CLAMP) { d *= REFR_CLAMP / dl; }
    uvTap = uvSelf + d;
  }
  // Explicit LOD: the screen-space uv is discontinuous across ripple detail,
  // and there is a discard earlier in this shader
  let refrBase = textureSampleLevel(refrTex, refrSamp, uvSelf, 0.0);
  let refrTap = textureSampleLevel(refrTex, refrSamp, clamp(uvTap, vec2f(0.0), vec2f(1.0)), 0.0);
  // Retreat to the unoffset tap where the offset one landed above water, then
  // to the analytic bottom where even that is invalid
  let sand = mix(mix(sandA, refrBase.rgb, refrBase.a), refrTap.rgb, refrTap.a);
  // -v is the ray from the eye toward this fragment
  let uw = underwaterAt(-v, u.camDepth, u.lensR);
  let lightTint = mix(vec3f(1.0), sunTint(SKY), 0.6);
  // Direct sunlight in the water column fades as the sun drops; the floor
  // stands in for diffuse sky light
  let sunLev = sunLevel(SKY);
  var water = mix(waterHue(u.sTurbidity, u.chlorophyll * u.sChlorophyll) * 0.16, sand, trans) * lightTint;
  water += vec3f(0.05, 0.45, 0.38) * sss;
  water *= sunLev;
  var color = mix(water, skyColor(r, SKY), fresnel) + spec;
  // Dry sand above the runup line: matte, no fresnel reflection or caustics
  let sandMatte = vec3f(0.86, 0.78, 0.58) * lightTint * sunLev * (0.55 + 0.45 * max(n.y, 0.0));
  color = mix(sandMatte, color, waterM);
  // Seen from underneath, the whole sky squeezes into Snell's window — the
  // ~97 degree cone about vertical set by the critical angle — and outside it
  // the surface is a perfect mirror mailing the view back down into the
  // basin. n was already flipped to face the camera above, so it points down
  // into the water here: exactly the inward normal refract() wants.
  // Small ripples the surface carries break the window's rim up; this is the
  // (uw) ripple control, acting only on the submerged view.
  // Guarded on camDepth/lensR, which are UNIFORMS: when the eye is clear of
  // the surface no fragment can be submerged, so this whole branch is skipped
  // rather than paid for on every above-water pixel. Being uniform, the branch
  // may legally contain texture samples with implicit derivatives.
  var nUW = n;
  if (u.camDepth > -u.lensR) {
    let chop = textureSample(capTex, samp, in.world.xz * 0.8 + vec2f(0.031, -0.019) * u.time).zw
             * (0.28 * u.rippleStrength);
    nUW = normalize(n + vec3f(chop.x, 0.0, chop.y));
  }
  let tRay = refract(-v, nUW, WATER_TO_AIR);
  let tir = dot(tRay, tRay) < 1e-6;
  // Both the mirrored ray and the sky ray need a floor hit for the mirror
  // term: march the reflection down to the flat basin and tint by the path
  let mRay = reflect(-v, nUW);
  let mPath = min((in.world.y + u.seaDepth) / max(-mRay.y, 0.05), 400.0);
  let mirror = mix(waterFogAlong(mRay, u.camDepth, SKY, u.uwTurbidity, u.chlorophyll), sand * lightTint * sunLev,
                   exp(-waterSigma(u.uwTurbidity, u.chlorophyll) * mPath));
  // Fresnel on the air side; at and past the critical angle it saturates to a
  // full mirror, which is what closes the rim of the window
  let cosAir = clamp(dot(tRay, -nUW), 0.0, 1.0);
  let fresUp = select(0.02 + 0.98 * pow(1.0 - cosAir, 5.0), 1.0, tir);
  // skyColor already carries the sun disc, so the window shows the sun on its
  // own; the above-water spec lobe belongs to the other side of the interface
  let underside = mix(skyColor(tRay, SKY), mirror, fresUp);
  color = mix(color, underside, uw);
  // The foam pattern rides the water (material coords); as the accumulated
  // foam decays the threshold rises, eroding the pattern from its thin parts
  // so patches fragment into clumps before vanishing
  // In the film the fresh channel (rise-smoothed generation = the film's
  // instantaneous compression) drives the front at full strength and the
  // accumulated trail follows at reduced weight; offshore the accumulated
  // foam renders as before
  // The two foam systems live in different material frames — world for
  // wave foam, (band, alongshore) for film foam — so each erodes its own
  // pattern lookup in its own frame and only the resulting MASKS blend.
  // Eroding a single pattern at blended coordinates smears it into
  // streaks across the handover where the frames diverge.
  let filmAcc = filmFoamAt(in.st.x, in.st.y).rgb;
  // The dominant wave direction orients the streak shear
  let waveFlow = u.layers[0].dirScaleAmp.xy;
  // Level from the pixel's world footprint against the plate's tile size, so
  // the film's compressed material frame cannot pick a level that shimmers
  let plateLod = max(log2(max(length(fwidth(in.world.xz)) / (5.0 * u.noiseScale), 1e-6) * 1024.0), 0.0);
  let patWave = foamPlate(in.gridXZ, waveFlow, foamAcc.r, plateLod);
  // The film's material band compresses onto the still-water wedge about
  // 14:1, so sampling the plate at raw band coordinates squashes every bubble
  // by that factor in the rendered image. Converting the band coordinate to
  // world metres with the LOCAL stretch makes the plate isotropic on screen
  // and keeps it anchored to the water rather than to the ground.
  // Clamped because an unclamped factor lets over-compressed filaments thin
  // without bound, which is what the old fixed coarsening was guarding.
  let restStretch = REST_DEPTH / u.slope / SIM_SPAN;
  let bToWorld = clamp(in.stretch, restStretch * 0.6, restStretch * 3.0);
  let filmCover = filmAcc.b + filmAcc.r * 0.8;
  let patFilm = foamPlate(vec2f(in.st.x * bToWorld, colT(in.st.y)), vec2f(1.0, 0.0), filmCover, plateLod);
  let maskWave = smoothstep(0.0, 0.15, patWave - (1.05 - 1.15 * foamAcc.r));
  let maskFilm = smoothstep(0.0, 0.15, patFilm - (1.05 - 1.15 * filmCover));
  // The unbroken lip at the swash front: a thin water column carries a bright
  // rim independently of accumulation, gated so a dead-calm film grows none
  let rim = smoothstep(0.25 * u.contactWidth, 0.0, column) * smoothstep(0.02, 0.15, filmAcc.b);
  // the masks are thresholded 0/1 fields, so foam is present when either
  // system says so — a blend would half-fade both across the handover
  let foamMask = min(maskWave + maskFilm + rim, 1.0) * u.opacity;
  // The mask is a near-binary cut, so at full coverage the dense plate passes
  // everywhere and the raft renders as flat white paint. Feed the plate's own
  // density back in as shading so its bubble structure survives where the
  // threshold has stopped discriminating.
  let patLit = max(patWave * maskWave, patFilm * maskFilm);
  let foamColor = lightTint * mix(0.45, 1.0, sunLev) * (0.72 + 0.22 * max(n.y, 0.0))
                * (0.60 + 0.55 * patLit);
  color = mix(color, foamColor, foamMask);
  let fog = 1.0 - exp(-dist * 3e-5);
  let air = mix(color, skyColor(normalize(vec3f(-v.x, 0.02, -v.z)), SKY), fog);
  // Underwater the aerial perspective is the water column itself, and it
  // closes in orders of magnitude faster
  let ext = exp(-waterSigma(u.uwTurbidity, u.chlorophyll) * u.uwFog * dist);
  var murk = waterFogAlong(-v, u.camDepth, SKY, u.uwTurbidity, u.chlorophyll);
  if (u.camDepth > -u.lensR) {
    murk += vec3f(0.9, 1.0, 0.95) * suspended(in.world.xz, 1.0 - ext.g);
  }
  color = mix(air, color * ext + murk * (1.0 - ext), uw);
  color = 1.0 - exp(-1.8 * color);
  return vec4f(pow(color, vec3f(1.0 / 2.2)), 1.0);
}

// Land: the same warped grid as the ocean, lifted to the terrain. Where
// the film thins to nothing at the waterline tip the two surfaces meet;
// the shading below matches the water shader's dry-sand branch exactly,
// so the coincident fragments are indistinguishable. Underwater parts
// are simply occluded by the opaque sea surface.
@vertex
fn vs_land(in: VSIn) -> VSOut {
  let xz = warpVertex(in.pos).xy;
  var out: VSOut;
  out.world = vec3f(xz.x, terrainHeight(xz), xz.y);
  out.gridXZ = xz;
  out.cut = -1.0;
  out.st = vec2f(-1000.0, 0.0);
  out.waveXZ = xz;
  out.stretch = 1.0;
  out.clip = u.viewProj * vec4f(out.world, 1.0);
  return out;
}

@fragment
fn fs_land(in: VSOut) -> @location(0) vec4f {
  let SKY = skyState(u.sunDir, u.skyTurbidity, u.skyRayleigh, u.skyIntensity);
  let e = 0.5;
  let hx = terrainHeight(in.gridXZ + vec2f(e, 0.0)) - terrainHeight(in.gridXZ - vec2f(e, 0.0));
  let hz = terrainHeight(in.gridXZ + vec2f(0.0, e)) - terrainHeight(in.gridXZ - vec2f(0.0, e));
  let n = normalize(vec3f(-hx / (2.0 * e), 1.0, -hz / (2.0 * e)));
  let lightTint = mix(vec3f(1.0), sunTint(SKY), 0.6);
  let sunLev = sunLevel(SKY);
  var color = vec3f(0.86, 0.78, 0.58) * lightTint * sunLev * (0.55 + 0.45 * max(n.y, 0.0));
  let dist = distance(u.cameraPos, in.world);
  let v = normalize(u.cameraPos - in.world);
  // Submerged, this same mesh IS the basin floor: daylight reaches it filtered
  // by the column above, carrying the caustic web the surface focuses. Above
  // the waterline the column is zero and it stays plain lit sand.
  let uw = underwaterAt(-v, u.camDepth, u.lensR);
  let column = max(-in.world.y, 0.0);
  let focus = u.uwCaustics * exp(-column * 0.12) * clamp(1.0 - dist / (120.0 * u.lodScale), 0.0, 1.0);
  let floorLit = 0.85 + focus * (1.6 * causticWeb(in.gridXZ) - 0.18);
  let floor = vec3f(0.86, 0.78, 0.58) * floorLit
            * waterAmbient(column, SKY, waterSigma(u.uwTurbidity, u.chlorophyll)) * (0.55 + 0.45 * max(n.y, 0.0));
  color = mix(color, floor, uw);
  let fog = 1.0 - exp(-dist * 3e-5);
  let air = mix(color, skyColor(normalize(vec3f(-v.x, 0.02, -v.z)), SKY), fog);
  let ext = exp(-waterSigma(u.uwTurbidity, u.chlorophyll) * u.uwFog * dist);
  var murk = waterFogAlong(-v, u.camDepth, SKY, u.uwTurbidity, u.chlorophyll);
  if (u.camDepth > -u.lensR) {
    murk += vec3f(0.9, 1.0, 0.95) * suspended(in.world.xz, 1.0 - ext.g);
  }
  color = mix(air, color * ext + murk * (1.0 - ext), uw);
  color = 1.0 - exp(-1.8 * color);
  return vec4f(pow(color, vec3f(1.0 / 2.2)), 1.0);
}

// The submerged scene, in linear radiance with no fog, tonemap or gamma, so
// the surface can attenuate it by Beer-Lambert before its own tonemap. It
// reproduces exactly the expression the surface calls `sand`, so at zero
// offset the swap is a visual no-op.
//
// fs_land cannot be reused: its submerged branch is gated on the camera being
// underwater, which is false exactly when refraction is visible.
@fragment
fn fs_land_refract(in: VSOut) -> @location(0) vec4f {
  let column = max(-in.world.y, 0.0);
  let dist = distance(u.cameraPos, in.world);
  let focus = u.causticStrength * exp(-column * 0.12) * clamp(1.0 - dist / (120.0 * u.lodScale), 0.0, 1.0);
  let lit = 0.85 + focus * (1.6 * causticWeb(in.gridXZ) - 0.18);
  // Soft over ~15 cm of terrain height so the mask ramps instead of popping
  let submerged = smoothstep(0.02, 0.15, column);
  return vec4f(vec3f(0.86, 0.78, 0.58) * lit, submerged);
}

@fragment
fn fs_wire(in: VSOut) -> @location(0) vec4f {
  if (in.cut > 0.0) {
    discard;
  }
  return vec4f(0.15, 0.85, 0.5, 1.0);
}
