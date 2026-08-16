@group(0) @binding(3) var capTex: texture_2d<f32>;
@group(0) @binding(4) var foamTex: texture_2d<f32>;
@group(0) @binding(5) var foamPatTex: texture_2d<f32>;
// Coast table: per column (P.x, P.z, N.x, N.z), written by chain.js
@group(0) @binding(8) var coastTex: texture_2d<f32>;
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
// around the camera with a radial warp — identity out to WARP_LINEAR, then
// exponentially growing cells. The center snaps to the lattice pitch so
// near vertices sit on a fixed world lattice (no swimming); far vertices
// slide with the camera, which is invisible because every field they
// sample is a function of world position.
const WARP_CELL: f32 = 0.4;
const WARP_LINEAR: f32 = 64.0;
const WARP_GROWTH: f32 = 1.08;

fn warpVertex(p: vec2f) -> vec3f {
  let snap = floor(u.cameraPos.xz / WARP_CELL + 0.5) * WARP_CELL;
  let r = length(p);
  if (r <= WARP_LINEAR) {
    return vec3f(snap + p, WARP_CELL);
  }
  let k = min((r - WARP_LINEAR) / WARP_CELL, 98.0);
  let g = pow(WARP_GROWTH, k);
  let rw = WARP_LINEAR + WARP_CELL * (g - 1.0) / (WARP_GROWTH - 1.0);
  return vec3f(snap + p * (rw / r), WARP_CELL * g);
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

// one ribbon row (28m band / 140 cells) folded down below the grid
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
  let w = sampleWaves(matWorld, cell);
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
  // Skirt: the row beyond the band's seaward edge sinks below the grid.
  // At the edge itself the two tessellations disagree by interpolation
  // error, and a bare ribbon edge shows see-through slivers wherever the
  // grid interpolates below it; with both meshes sinking into each
  // other's territory every crack has geometry behind it.
  y -= SKIRT_DROP * clamp((-SIM_BAND - b) / SKIRT_W, 0.0, 1.0);
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
  let b = in.pos.x * (SIM_SPAN + SIM_BAND + SKIRT_W) - SIM_BAND - SKIRT_W;
  let col = clamp(((t - u.simZBase) / 160.0 + 0.5) * f32(MAIN_COLS - 1), 0.0, f32(MAIN_COLS - 1));
  let c = mainCoastAt(t);
  return ribbonVertex(b, col, c.xy, c.zw, in.cell);
}

@vertex
fn vs_island(in: VSIn) -> VSOut {
  let b = in.pos.x * (SIM_SPAN + SIM_BAND + SKIRT_W) - SIM_BAND - SKIRT_W;
  let col = in.pos.y;
  let c = coastAt(col);
  return ribbonVertex(b, col, c.xy, normalize(c.zw), in.cell);
}

fn surfaceNormal(xz: vec2f, rippleXZ: vec2f, dist: f32, eta: f32, hScale: f32) -> vec3f {
  var dPx = vec3f(1.0, 0.0, 0.0);
  var dPz = vec3f(0.0, 0.0, 1.0);
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
  }
  let leanSlope = (eta * eta + 2.0 * eta) / ((1.0 + eta) * (1.0 + eta));
  dPx += vec3f(u.leanX * leanSlope * dPx.y, 0.0, u.leanY * leanSlope * dPx.y);
  dPz += vec3f(u.leanX * leanSlope * dPz.y, 0.0, u.leanY * leanSlope * dPz.y);
  // Ripples concentrate where the long waves strain the surface: orbital
  // convergence (compression, near crests) with the peak shifted toward the
  // front face. Layers 0-2 are isotropic wind ripples (weak bias); layers 3-5
  // are anisotropic parasitic-capillary ripples following the gravity waves.
  let front = smoothstep(0.0, 0.15, -dPx.y);
  let squeeze = smoothstep(0.0, 0.3, 2.0 - dPx.x - dPz.z);
  let conc = front + squeeze;
  let fade = clamp(1.0 - dist / 150.0, 0.0, 1.0);
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
  return normalize(cross(dPz, dPx));
}

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
  var n = surfaceNormal(in.waveXZ, rippleXZ, dist, max(in.world.y * u.ampInv, 0.0), 1.0 - sbF);
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
  if (dot(n, v) < 0.0) { n = -n; }
  let fresnel = 0.02 + 0.98 * pow(1.0 - max(dot(n, v), 0.0), 5.0);
  let r = reflect(-v, n);
  let spec = sunTint(u.sunDir) * (mix(8.0, 4.5, sunWarmth(u.sunDir)) * pow(max(dot(r, u.sunDir), 0.0), 600.0));
  let fCenter = vec2f(u.foamCX, u.foamCZ);
  let fuv = (in.gridXZ - fCenter) / (2.0 * u.foamRegion) + 0.5;
  let edgeFade = 1.0 - smoothstep(0.85, 1.0, length(in.gridXZ - fCenter) / u.foamRegion);
  let foamAcc = textureSample(foamTex, samp, fuv).rgb * edgeFade;
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
  let trans = exp(-vec3f(0.25, 0.04, 0.02) * pathLen);
  // Caustic web on the sand: bright filaments along the zero-crossing lines of
  // two drifting noise fields, sampled at the refracted bottom point so the
  // pattern swims with the surface; defocus fades it with column depth
  let bottomXZ = in.world.xz + refr.xz * (column * lateral);
  let cs = textureSample(capTex, samp, bottomXZ / (13.0 * u.causticScale) + vec2f(0.023, 0.011) * u.time).x
         + textureSample(capTex, samp, bottomXZ / (8.7 * u.causticScale) + vec2f(-0.017, 0.019) * u.time).x;
  let web = pow(max(0.0, 1.0 - 0.6 * abs(cs)), 4.0);
  // Caustics need some water column to focus in; a centimeters-thin film
  // (or the residual softmax offset on dry sand) must not carry the web
  let focus = u.causticStrength * exp(-column * 0.12) * clamp(1.0 - dist / 120.0, 0.0, 1.0) * smoothstep(0.04, 0.25, column);
  let sand = vec3f(0.86, 0.78, 0.58) * (0.85 + focus * (1.6 * web - 0.18));
  let lightTint = mix(vec3f(1.0), sunTint(u.sunDir), 0.6);
  // Direct sunlight in the water column fades as the sun drops; the floor
  // stands in for diffuse sky light
  let sunLevel = mix(0.18, 1.0, smoothstep(0.0, 0.5, clamp(u.sunDir.y, 0.0, 1.0)));
  var water = mix(vec3f(0.004, 0.02, 0.05), sand, trans) * lightTint;
  water += vec3f(0.05, 0.45, 0.38) * sss;
  water *= sunLevel;
  var color = mix(water, skyColor(r, u.sunDir), fresnel) + spec;
  // Dry sand above the runup line: matte, no fresnel reflection or caustics
  let sandMatte = vec3f(0.86, 0.78, 0.58) * lightTint * sunLevel * (0.55 + 0.45 * max(n.y, 0.0));
  color = mix(sandMatte, color, waterM);
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
  let patWave = textureSample(foamPatTex, samp, in.gridXZ / (5.0 * u.foamScale)).r;
  // Over-compressed foam filaments merge instead of thinning forever: as
  // the film compresses, blend toward a pattern that is coarser in the
  // cross-shore direction only, so the rendered streaks stop shrinking
  let patFilmUV = vec2f(in.st.x, colT(in.st.y)) / (5.0 * u.foamScale);
  let patFine = textureSample(foamPatTex, samp, vec2f(patFilmUV.x / 3.0, patFilmUV.y)).r;
  let patCoarse = textureSample(foamPatTex, samp, vec2f(patFilmUV.x / 9.0, patFilmUV.y)).r;
  let patFilm = mix(patFine, patCoarse, 1.0 - smoothstep(0.07, 0.4, in.stretch));
  let maskWave = smoothstep(0.0, 0.15, patWave - (1.05 - 1.15 * foamAcc.r));
  let maskFilm = smoothstep(0.0, 0.15, patFilm - (1.05 - 1.15 * (filmAcc.b + filmAcc.r * 0.8)));
  // the masks are thresholded 0/1 fields, so foam is present when either
  // system says so — a blend would half-fade both across the handover
  let foamMask = min(maskWave + maskFilm, 1.0);
  let foamColor = lightTint * mix(0.45, 1.0, sunLevel) * (0.72 + 0.22 * max(n.y, 0.0));
  color = mix(color, foamColor, foamMask);
  let fog = 1.0 - exp(-dist * 3e-5);
  color = mix(color, skyColor(normalize(vec3f(-v.x, 0.02, -v.z)), u.sunDir), fog);
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
  let e = 0.5;
  let hx = terrainHeight(in.gridXZ + vec2f(e, 0.0)) - terrainHeight(in.gridXZ - vec2f(e, 0.0));
  let hz = terrainHeight(in.gridXZ + vec2f(0.0, e)) - terrainHeight(in.gridXZ - vec2f(0.0, e));
  let n = normalize(vec3f(-hx / (2.0 * e), 1.0, -hz / (2.0 * e)));
  let lightTint = mix(vec3f(1.0), sunTint(u.sunDir), 0.6);
  let sunLevel = mix(0.18, 1.0, smoothstep(0.0, 0.5, clamp(u.sunDir.y, 0.0, 1.0)));
  var color = vec3f(0.86, 0.78, 0.58) * lightTint * sunLevel * (0.55 + 0.45 * max(n.y, 0.0));
  let dist = distance(u.cameraPos, in.world);
  let v = normalize(u.cameraPos - in.world);
  let fog = 1.0 - exp(-dist * 3e-5);
  color = mix(color, skyColor(normalize(vec3f(-v.x, 0.02, -v.z)), u.sunDir), fog);
  color = 1.0 - exp(-1.8 * color);
  return vec4f(pow(color, vec3f(1.0 / 2.2)), 1.0);
}

@fragment
fn fs_wire(in: VSOut) -> @location(0) vec4f {
  if (in.cut > 0.0) {
    discard;
  }
  return vec4f(0.15, 0.85, 0.5, 1.0);
}
