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
// 9 and 10 belong to the coastline SDF and the mainland table
@group(0) @binding(11) var refrTex: texture_2d<f32>;
@group(0) @binding(12) var refrSamp: sampler;
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


// ===========================================================================
// SUN VISIBILITY — analytic, no shadow map, no depth bias
// ===========================================================================
//
// One occluder set, one test, evaluated identically by every consumer: the sea
// floor drawn directly (fs_land), the sea floor drawn into the refraction
// target (fs_land_refract), the floor the surface shader falls back to
// (sandA in fs), the bodies themselves (fs_body / fs_body_refract), the dry
// beach, and the mid-water in-scatter. There is no shadow map, so there is
// nothing to keep in sync between passes and NO BIAS TO TUNE: self-shadowing is
// removed by instance index, exactly, rather than pushed away by an epsilon.

// Solar angular radius (0.2665 deg) as a slope: penumbra half-width grows by
// this per metre of occluder-to-receiver distance. In air this is the only
// penumbra source — 2.6 cm at 6 m, i.e. essentially a hard edge.
const SUN_TAN_RADIUS: f32 = 0.00465;
// Multiple scattering in the column smears the beam far more than the disc
// does: a boulder 2 m off the sand throws an edge tens of centimetres wide, not
// 1 cm. Also a slope (extra penumbra per metre of in-water shadow ray).
const SUN_SPREAD_WATER: f32 = 0.09;
// The bed's unoccludable share: skylight and multiply-scattered light, which no
// boulder can block. A shadow that reaches black is this constant set to 0.
const BED_SHADOW_FLOOR: f32 = 0.34;
// Sentinel for "exclude no body" (u32 max)
const NO_SKIP: u32 = 0xffffffffu;

// Sun direction INSIDE the water, refracted across the mean surface.
//
// This is not a nicety. The bed is metres down; using the AIR direction slides
// every shadow along the sand by depth * (tan(theta_air) - tan(theta_water)) —
// 1.7 m at 6 m depth with a 30 deg sun, a whole boulder-width, and as a
// consistent bias rather than noise it is the kind of error the eye reads
// immediately. Snell across the flat mean plane is the right approximation for
// exactly the reason the surface shader's own "+1.4" sun-side path factor is:
// wave-scale surface tilt randomises the beam WITHIN the penumbra the water
// already imposes, it does not bias it.
//
// This is also the term a shadow map cannot have: an ortho depth render along
// u.sunDir produces air-angle shadows by construction.
fn sunDirWater() -> vec3f {
  let s = normalize(u.sunDir);
  // sin(theta_w) = sin(theta_a) / n  ->  the horizontal component shrinks by n
  let h = vec2f(s.x, s.z) / WATER_TO_AIR;
  return normalize(vec3f(h.x, sqrt(max(1.0 - dot(h, h), 1e-4)), h.y));
}

// Closest approach of the ray p + t*dir (t >= 0, dir unit, world metres) to one
// body, in the body's own normalized space where the ellipsoid is the unit
// sphere. Returns (gapInNormalizedSpace, tAtClosestInMetres); gap < 0 means the
// ray passes through the body.
//
// Normalizing the transformed direction makes the gap a distance in THAT space,
// i.e. in units of local semi-axis; the caller scales by centerR.w (the mean
// semi-axis) to get approximate metres. For a strongly flattened boulder the
// resulting PENUMBRA is therefore mildly anisotropic. That is a shading nicety,
// not a visibility error: the hard test (gap < 0) is exact for any ellipsoid.
fn bodyRayGap(b: Body, p: vec3f, dir: vec3f) -> vec2f {
  let inv = b.invRadiusSoft.xyz;
  let o = (p - b.centerR.xyz) * inv;
  let eRaw = dir * inv;
  let dl = max(length(eRaw), 1e-6);
  let d = eRaw / dl;
  // normalized-space parameter of closest approach, clamped forward: an
  // occluder behind the receiver cannot shadow it
  let sc = max(-dot(o, d), 0.0);
  return vec2f(length(o + d * sc) - 1.0, sc / dl);
}

// Sun visibility at a world point: 0 fully shadowed .. 1 fully lit.
//   lightDir  unit vector TOWARD the light (u.sunDir in air, sunDirWater()
//             below the surface)
//   spread    extra penumbra SLOPE (per metre of shadow ray); pass
//             SUN_SPREAD_WATER * submergedFraction underwater, 0 in air
//   minPen    penumbra floor in metres — antialiasing for the razor-sharp
//             contact edge; pass length(fwidth(p.xz))
//   skip      instance index to exclude (a body shading itself), or NO_SKIP
fn bodyShadowSkip(p: vec3f, lightDir: vec3f, spread: f32, minPen: f32, skip: u32) -> f32 {
  // No direct beam to occlude. Both conditions are UNIFORMS, so the nights and
  // the no-bodies case cost one scalar branch for the whole draw rather than a
  // loop per pixel.
  if (u.numBodies < 0.5 || u.sunDir.y <= 0.02) { return 1.0; }
  let n = i32(u.numBodies);
  var vis = 1.0;
  for (var i = 0; i < n; i++) {
    if (u32(i) == skip) { continue; }
    let b = u.bodies[i];
    let g = bodyRayGap(b, p, lightDir);
    let gapM = g.x * b.centerR.w;
    // Penumbra half-width at the receiver: solar disc plus in-water beam
    // spread, both proportional to occluder-to-receiver distance. It goes to
    // zero at contact, which is correct (contact shadows are sharp) and is why
    // minPen exists.
    let pen = max((SUN_TAN_RADIUS + spread) * g.y, minPen);
    let s = smoothstep(-pen, pen, gapM);
    // min, NOT a product. Two boulders are a UNION of shadow volumes; min is
    // the union's indicator in the hard limit, whereas multiplying drives the
    // seam between two touching rocks darker than either one alone — which is
    // precisely where the eye looks. (Clouds DO multiply: see cloudShadow.)
    vis = min(vis, mix(1.0, s, b.invRadiusSoft.w));
  }
  return vis;
}

// There is no cloud layer in this renderer: nothing in shaders/ or src/
// references clouds, so this is the identity and costs nothing.
//
// It exists as the SITE. When a cloud layer lands it belongs here and nowhere
// else, and it must combine as a PRODUCT with the body term — clouds and
// boulders are independent occluders of the same beam, unlike two boulders.
// Because every caustic term below is already gated on this same scalar, a
// cloud shadow will kill caustics for free, which is the behaviour you want.
// (One thing it will additionally need, which is NOT here: the surface shader's
// `spec` glitter and the `sunLev` multiply in fs are direct-beam terms too.)
fn cloudShadow(xz: vec2f) -> f32 { return 1.0; }

// ---------------------------------------------------------------------------
// The sea bed's lit factor — shadow AND caustics, in one place
// ---------------------------------------------------------------------------
//
// fs_land and fs_land_refract MUST call this rather than each spelling the
// expression out. They shade the SAME surface: fs_land_refract's own comment
// says it "reproduces exactly the expression the surface calls `sand`, so at
// zero offset the swap is a visual no-op", and at the waterline their fragments
// are coincident. Any divergence is a visible seam. (fs_land cannot be reused
// wholesale — its submerged branch is gated on the camera being underwater,
// which is false exactly when refraction is visible — but this factor can be,
// and must be.)
//
// `focus` stays a parameter because the three callers legitimately differ:
// fs_land uses u.uwCaustics, fs_land_refract uses u.causticStrength, and fs
// additionally gates on smoothstep(0.04, 0.25, column) so a centimetres-thin
// film grows no web. The SHADOW composition is what is shared.
//
// Note the shape: the ambient share is NOT multiplied by visibility, and the
// caustic term is multiplied by it INCLUDING its negative inter-filament lobe.
// Caustics are the focused direct beam; with no beam there are no filaments and
// no inter-filament darkening either — the sand just sits at its shadow level.
fn bedLit(p: vec3f, focus: f32, footprint: f32) -> f32 {
  let column = max(-p.y, 0.0);
  let submerged = smoothstep(0.02, 0.15, column);
  let vis = bodyShadowSkip(p, sunDirWater(), SUN_SPREAD_WATER * submerged,
                           max(footprint, 0.03), NO_SKIP) * cloudShadow(p.xz);
  return 0.85 * mix(BED_SHADOW_FLOOR, 1.0, vis)
       + focus * vis * (1.6 * causticWeb(p.xz) - 0.18);
}

// Dry sand's lit factor. fs (its sandMatte branch) and fs_land (its above-water
// branch) both draw the beach and are coincident at the waterline tip, so this
// goes in one place too. 0.55 is the sky's share and does not respond to an
// occluder; only the N.L term does — which is why no floor constant is needed
// here. The air sun direction is correct: nothing above water is refracted.
fn dryLit(p: vec3f, n: vec3f, footprint: f32) -> f32 {
  let vis = bodyShadowSkip(p, normalize(u.sunDir), 0.0, max(footprint, 0.03), NO_SKIP)
          * cloudShadow(p.xz);
  return 0.55 + 0.45 * max(n.y, 0.0) * vis;
}

// ===========================================================================
// MID-WATER SHADOW — the dark cone, in CLOSED FORM. No volumetric march.
// ===========================================================================
//
// Asked honestly: does a visible dark cone in mid-water need a volumetric
// march? With a shadow map, yes, unavoidably. With analytic occluders, NO — and
// this is the single biggest dividend of that choice.
//
// Under a DIRECTIONAL light the shadow volume of an ellipsoid is the ellipsoid
// swept along the light direction: an oblique semi-infinite cylinder. In the
// body's own normalized space (divide by the semi-axes) the ellipsoid is the
// unit sphere, so the shadow volume is a plain CIRCULAR CYLINDER OF RADIUS 1
// about the light axis, cut off at the terminator plane through the centre.
// Intersecting a view ray with that is one quadratic plus one linear
// inequality, so the shadowed SEGMENT [t0, t1] of the view ray is closed-form.
//
// The renderer's underwater in-scatter is already the homogeneous
// single-scatter integral: I = J * (1 - exp(-sigma*len)), written in fs and
// fs_land as `murk * (1 - ext)`. Removing a segment from a homogeneous
// exponential is also closed-form: the share of that integral falling inside
// [t0, t1] is exp(-sigma*t0) - exp(-sigma*t1). So the cone costs ONE QUADRATIC
// AND TWO exp() PER BODY — no steps, no jitter, no banding, no dependence on
// screen resolution, and no shadow-map fetch per step (the incoherent memory
// traffic that actually makes volumetric shadows expensive).
//
// WHAT THIS CANNOT DO, stated plainly:
//  * GOD-RAY WIGGLE. Real shafts shimmer because the surface focuses the beam.
//    That is a volumetric caustic and it genuinely needs the caustic field
//    evaluated at points along the ray, i.e. a real march with a TEXTURE FETCH
//    per step — about 8 fetches per submerged pixel, roughly the cost of one
//    more wave layer. The cone below a rock does not need it and does not get
//    it here. If you want wiggle, that march is the price, and it is separate
//    from and additional to everything below.
//  * A source term that varies along the ray. J is constant, exactly as
//    waterFogAlong() already assumes, so the cone does not itself darken with
//    depth.
//  * Exact union of two bodies' cones along one ray: the masses are summed and
//    clamped to the total. Exact when the segments are disjoint (the normal
//    case for scattered boulders), slightly over-dark when they are not.
//  * A physical penumbra: it is faked by averaging two cylinder radii (below),
//    which is still O(bodies) rather than O(steps).
//  * Points strictly INSIDE a body read as lit by the terminator test. The
//    depth buffer has already rejected those fragments, so it never shows.

// Occludable share of the in-scatter: the direct beam. The rest is
// multiply-scattered skylight arriving from every direction, which a boulder
// cannot block. A cone that reaches black is this set to 1.
const SHAFT_DIRECT: f32 = 0.55;
// Wide cylinder for the faked penumbra. A single hard cylinder gives a
// razor-edged cone that reads as a cardboard cutout underwater.
const BEAM_PEN_WIDE: f32 = 1.35;
// Henyey-Greenstein asymmetry. Sea water is strongly forward-scattering, which
// is WHY shafts are obvious looking toward the sun and nearly invisible looking
// away — and therefore why the cone is too. An art knob; 0.6 with the clamps in
// beamGain() gives a usable range without a hot spot.
const SHAFT_G: f32 = 0.6;

fn hg(cosT: f32, g: f32) -> f32 {
  let d = 1.0 + g * g - 2.0 * g * cosT;
  return (1.0 - g * g) / (4.0 * SKY_PI * max(d * sqrt(max(d, 1e-4)), 1e-4));
}

// Forward-scattering gain relative to isotropic (the 1/4pi is undone, so 1.0
// means "as bright as isotropic"). `dir` is the view direction FROM the eye;
// the scattering cosine is dot(dir, sunW) because the beam propagates along
// -sunW and reaches the eye along -dir. Clamped so looking straight into the
// sun cannot push the occludable share past 1.
fn beamGain(dir: vec3f, sunW: vec3f) -> f32 {
  return clamp(4.0 * SKY_PI * hg(dot(dir, sunW), SHAFT_G), 0.45, 3.0);
}

// Shadowed share of the homogeneous in-scatter integral along p0 + t*dir for
// t in [0, len], for ONE body: exp(-sigma*t0) - exp(-sigma*t1).
// `radius` widens the shadow cylinder; see beamShadowMass.
fn bodyShadowMass(b: Body, sw: vec3f, p0: vec3f, dir: vec3f, len: f32,
                  sigma: f32, radius: f32) -> f32 {
  let inv = b.invRadiusSoft.xyz;
  let o = (p0 - b.centerR.xyz) * inv;
  let e = dir * inv;
  let lRaw = sw * inv;
  let L = lRaw / max(length(lRaw), 1e-6);
  // project the light axis out: the cylinder condition is |P*(o + t*e)| <= r
  let A = o - L * dot(o, L);
  let B = e - L * dot(e, L);
  let a = dot(B, B);
  if (a < 1e-12) { return 0.0; }          // view ray parallel to the beam
  let hb = dot(A, B);
  let c = dot(A, A) - radius * radius;
  let disc = hb * hb - a * c;
  if (disc <= 0.0) { return 0.0; }        // ray misses the shadow cylinder
  let sq = sqrt(disc);
  var t0 = (-hb - sq) / a;
  var t1 = (-hb + sq) / a;
  // Terminator half-space: only the side of the body AWAY from the sun is dark,
  // i.e. (o + t*e).L <= 0 with L pointing toward the sun. Without this the
  // cylinder extends upward through the rock and out the sunlit side.
  let dL = dot(e, L);
  let oL = dot(o, L);
  if (abs(dL) < 1e-9) {
    if (oL > 0.0) { return 0.0; }
  } else if (dL > 0.0) {
    t1 = min(t1, -oL / dL);
  } else {
    t0 = max(t0, -oL / dL);
  }
  t0 = max(t0, 0.0);
  t1 = min(t1, len);
  if (t1 <= t0) { return 0.0; }
  return exp(-sigma * t0) - exp(-sigma * t1);
}

// Penumbra: two cylinder radii, mass averaged. Still closed-form — O(bodies),
// not O(steps).
fn beamShadowMass(sw: vec3f, p0: vec3f, dir: vec3f, len: f32, sigma: f32) -> f32 {
  var m = 0.0;
  let n = i32(u.numBodies);
  for (var i = 0; i < n; i++) {
    let b = u.bodies[i];
    m += b.invRadiusSoft.w * 0.5
       * (bodyShadowMass(b, sw, p0, dir, len, sigma, 1.0)
        + bodyShadowMass(b, sw, p0, dir, len, sigma, BEAM_PEN_WIDE));
  }
  return m;
}

// Multiplier on the underwater in-scatter along a view ray: 1 in open water,
// dropping toward (1 - SHAFT_DIRECT * gain) inside a body's cone. Apply it to
// BOTH the murk and the suspended motes — the motes going dark inside the cone
// is what actually makes it read as an occluded shaft rather than a smudge.
//
// `sigma` is scalar (use the green channel). The geometric shadowed FRACTION is
// wavelength-independent; the chromatic part comes from `murk`'s own tint, and
// the second-order shift (less direct beam, relatively more ambient) is not
// worth three quadratics.
fn beamVis(p0: vec3f, dir: vec3f, len: f32, sigma: f32) -> f32 {
  if (u.numBodies < 0.5 || u.sunDir.y <= 0.02) { return 1.0; }
  let total = 1.0 - exp(-sigma * max(len, 0.0));
  if (total <= 1e-5) { return 1.0; }
  let sw = sunDirWater();
  let shaded = min(beamShadowMass(sw, p0, dir, len, sigma), total);
  let direct = clamp(SHAFT_DIRECT * beamGain(dir, sw), 0.0, 0.92);
  return 1.0 - direct * (shaded / total);
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
  return u.warpCell + max((WARP_GROWTH - 1.0) * (dist - u.warpLinear), 0.0);
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
  let SKY = skyState(u.sunDir, u.moonDir, u.skyTurbidity, u.skyRayleigh, u.skyIntensity);
  if (dot(n, v) < 0.0) { n = -n; }
  let fresnel = 0.02 + 0.98 * pow(1.0 - max(dot(n, v), 0.0), 5.0);
  let r = reflect(-v, n);
  // Two glitter paths rather than one blended direction: the sun and moon sit
  // on opposite sides of the sky, so interpolating between them would sweep a
  // highlight across the water at dusk that belongs to neither.
  let specSun = mix(8.0, 4.5, sunWarmth(SKY)) * pow(max(dot(r, SKY.sunDir), 0.0), 600.0) * SKY.day;
  let specMoon = 5.0 * pow(max(dot(r, SKY.moonDir), 0.0), 600.0) * SKY.nightMix * SKY.moonLit;
  let spec = sunTint(SKY) * (specSun * cloudShade(in.world, SKY) + specMoon);
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

  // Caustics need some water column to focus in; a centimeters-thin film
  // (or the residual softmax offset on dry sand) must not carry the web
  let focus = u.causticStrength * exp(-column * 0.12) * clamp(1.0 - dist / (120.0 * u.lodScale), 0.0, 1.0) * smoothstep(0.04, 0.25, column);
  // Analytic flat bottom, kept as the fallback wherever the screen-space tap
  // has nothing valid to say
  // The refracted taps below already carry the bed's shadow — they ARE the
  // refraction pass's output. This ANALYTIC FALLBACK does not, and it is what
  // shows wherever a tap is rejected: frame edges, grazing rays, taps landing
  // above water. Left unshadowed, a boulder's shadow flickers back to full
  // brightness along every one of those silhouettes.
  // The footprint comes from in.world.xz rather than bottomXZ, which inherits
  // the ripple normals' wobble and would turn into noise on the shadow edge.
  let fwBed = max(length(fwidth(in.world.xz)), 0.03);
  let bedP = vec3f(bottomXZ.x, in.world.y - column, bottomXZ.y);
  let sandA = vec3f(0.86, 0.78, 0.58) * bedLit(bedP, focus, fwBed);
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
  // one shadow lookup per fragment, applied per term below
  let shade = cloudShade(in.world, SKY);
  // Seen from ABOVE, the cone shows as a dimming of the water's own in-scatter
  // along the refracted ray, not of the bed — the bed's share arrives through
  // refrTex and through sandA. Closed form, so this is a handful of ALU on water
  // pixels and one uniform branch when there are no bodies.
  let bedT = column * min(lateral, 8.0);
  let beamS = beamVis(in.world, refr, bedT,
                      waterSigma(u.sTurbidity, u.chlorophyll * u.sChlorophyll).g);
  var water = mix(waterHue(u.sTurbidity, u.chlorophyll * u.sChlorophyll) * 0.16 * beamS,
                  sand, trans) * lightTint;
  water += vec3f(0.05, 0.45, 0.38) * sss;
  water *= sunLev * mix(1.0, shade, CLOUD_DIRECT_WATER);
  var color = mix(water, skyColorRough(r, SKY, 16.0), fresnel) + spec;
  // Dry sand above the runup line: matte, no fresnel reflection or caustics
  let sandMatte = vec3f(0.86, 0.78, 0.58) * lightTint * sunLev
                * mix(1.0, shade, CLOUD_DIRECT_SAND) * dryLit(in.world, n, fwBed);
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
  // The whole window is inside ONE uniform branch on camDepth/lensR: with the
  // eye clear of the surface no fragment can be submerged, so above water this
  // skips a chop sample AND a full cloud march through skyColorRough — the
  // window's sky lookup costs the same as the mirror reflection's, and paying
  // for it on every above-water pixel was most of the deck's cost.
  if (u.camDepth > -u.lensR) {
    let chop = textureSample(capTex, samp, in.world.xz * 0.8 + vec2f(0.031, -0.019) * u.time).zw
             * (0.28 * u.rippleStrength);
    let nUW = normalize(n + vec3f(chop.x, 0.0, chop.y));
    let tRay = refract(-v, nUW, WATER_TO_AIR);
    let tir = dot(tRay, tRay) < 1e-6;
    // The mirrored ray needs a floor hit: march the reflection down to the
    // flat basin and tint it by the path
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
    let underside = mix(skyColorRough(tRay, SKY, 3.0), mirror, fresUp);
    color = mix(color, underside, uw);
  }
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
  let patWave = foamPlate(in.gridXZ, waveFlow, accR, plateLod);
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
  let maskWave = smoothstep(0.0, 0.15, patWave - (1.05 - 1.15 * accR));
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
  let foamColor = lightTint * mix(0.45, 1.0, sunLev) * mix(1.0, shade, CLOUD_DIRECT_FOAM)
                * (0.72 + 0.22 * max(n.y, 0.0))
                * (0.60 + 0.55 * patLit);
  color = mix(color, foamColor, foamMask);
  let fog = 1.0 - exp(-dist * 3e-5);
  let air = mix(color, skyColorRough(normalize(vec3f(-v.x, 0.02, -v.z)), SKY, 40.0), fog);
  // Underwater the aerial perspective is the water column itself, and it
  // closes in orders of magnitude faster
  let ext = exp(-waterSigma(u.uwTurbidity, u.chlorophyll) * u.uwFog * dist);
  // Submerged camera: the same cone, now along the eye ray
  let beamU = beamVis(u.cameraPos, -v, dist,
                      waterSigma(u.uwTurbidity, u.chlorophyll).g * u.uwFog);
  var murk = waterFogAlong(-v, u.camDepth, SKY, u.uwTurbidity, u.chlorophyll) * beamU;
  if (u.camDepth > -u.lensR) {
    murk += vec3f(0.9, 1.0, 0.95) * suspended(in.world.xz, 1.0 - ext.g) * beamU;
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
  let SKY = skyState(u.sunDir, u.moonDir, u.skyTurbidity, u.skyRayleigh, u.skyIntensity);
  let e = 0.5;
  let hx = terrainHeight(in.gridXZ + vec2f(e, 0.0)) - terrainHeight(in.gridXZ - vec2f(e, 0.0));
  let hz = terrainHeight(in.gridXZ + vec2f(0.0, e)) - terrainHeight(in.gridXZ - vec2f(0.0, e));
  let n = normalize(vec3f(-hx / (2.0 * e), 1.0, -hz / (2.0 * e)));
  let lightTint = mix(vec3f(1.0), sunTint(SKY), 0.6);
  let sunLev = sunLevel(SKY);
  let shade = cloudShade(in.world, SKY);
  let fwL = max(length(fwidth(in.world.xz)), 0.03);
  var color = vec3f(0.86, 0.78, 0.58) * lightTint * sunLev
            * mix(1.0, shade, CLOUD_DIRECT_SAND) * dryLit(in.world, n, fwL);
  let dist = distance(u.cameraPos, in.world);
  let v = normalize(u.cameraPos - in.world);
  // Submerged, this same mesh IS the basin floor: daylight reaches it filtered
  // by the column above, carrying the caustic web the surface focuses. Above
  // the waterline the column is zero and it stays plain lit sand.
  let uw = underwaterAt(-v, u.camDepth, u.lensR);
  let column = max(-in.world.y, 0.0);
  let focus = u.uwCaustics * exp(-column * 0.12) * clamp(1.0 - dist / (120.0 * u.lodScale), 0.0, 1.0);
  // vs_land sets out.gridXZ = xz and out.world.xz = xz, so in.world.xz IS
  // in.gridXZ here and bedLit's web sample matches the previous code exactly.
  let floorLit = bedLit(in.world, focus, fwL);
  let floor = vec3f(0.86, 0.78, 0.58) * floorLit
            * waterAmbient(column, SKY, waterSigma(u.uwTurbidity, u.chlorophyll)) * (0.55 + 0.45 * max(n.y, 0.0));
  color = mix(color, floor, uw);
  let fog = 1.0 - exp(-dist * 3e-5);
  let air = mix(color, skyColorRough(normalize(vec3f(-v.x, 0.02, -v.z)), SKY, 40.0), fog);
  let ext = exp(-waterSigma(u.uwTurbidity, u.chlorophyll) * u.uwFog * dist);
  let beamU = beamVis(u.cameraPos, -v, dist,
                      waterSigma(u.uwTurbidity, u.chlorophyll).g * u.uwFog);
  var murk = waterFogAlong(-v, u.camDepth, SKY, u.uwTurbidity, u.chlorophyll) * beamU;
  if (u.camDepth > -u.lensR) {
    murk += vec3f(0.9, 1.0, 0.95) * suspended(in.world.xz, 1.0 - ext.g) * beamU;
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
  let fw = max(length(fwidth(in.world.xz)), 0.03);
  // Both bed paths go through bedLit, so "the refracted floor equals the direct
  // floor at zero offset" is guaranteed by construction, not maintained by hand
  let lit = bedLit(in.world, focus, fw);
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

// ===========================================================================
// DRAWING THE BODIES
// ===========================================================================
//
// One shared unit-ball mesh, instanced; the transform comes from
// u.bodies[instance_index]. There is deliberately NO per-body vertex data and
// no per-body instance buffer: the array the shadow test reads is the array the
// vertex shader reads, so there is no second copy and nothing to desync.
//
// Two entry points share this vertex shader:
//   fs_body_refract -> the refraction target (linear radiance, no envelope),
//                      sharing the bed's depth buffer so the rock occludes the
//                      sand. This is what makes the rock visible THROUGH the
//                      surface with ZERO changes to the surface shader: its tap
//                      machinery already handles whatever is in refrTex.
//   fs_body         -> the main pass, for a submerged or breaching camera, with
//                      exactly fs_land's fog tail.
// Nothing is blended and nothing is sorted: everything is opaque and
// depth-tested. That is WHY the from-above case resolves to the refracted image
// automatically — the ocean surface is opaque and nearer, so it wins the depth
// test and what you see is its refracted tap.

const ROCK_ALBEDO: vec3f = vec3f(0.30, 0.29, 0.27);

struct BodyIn {
  // unit sphere direction; the mesh is one shared unit ball
  @location(0) dir: vec3f,
}

struct BodyOut {
  @builtin(position) clip: vec4f,
  @location(0) world: vec3f,
  @location(1) nrm: vec3f,
  // FLAT: the index the shadow test excludes must be exact, not 2.9999
  @location(2) @interpolate(flat) bi: u32,
}

// Low-frequency silhouette roughening. Deliberately low-frequency: the shadow
// test uses the SMOOTH ellipsoid, so a high-frequency silhouette would visibly
// disagree with its own shadow outline. At +-9% over three lobes the
// disagreement is under 20 cm on a 2 m boulder — inside the penumbra the water
// already imposes (SUN_SPREAD_WATER * distance is ~35 cm at 4 m). This is the
// one approximation the analytic route makes; if it ever matters, either drop
// the bump or fold the same hash into bodyRayGap and give up the closed-form
// cone.
fn bodyBump(dir: vec3f, seed: f32) -> f32 {
  let a = sin(dir.x * 3.1 + seed * 1.7) * sin(dir.y * 2.3 - seed * 2.9) * sin(dir.z * 2.7 + seed * 0.7);
  let b = sin(dir.x * 5.7 - seed * 3.3 + dir.z * 4.1);
  return 1.0 + 0.09 * a + 0.05 * b;
}

// Surface point in world METRES relative to the body centre. Because this
// multiplies by the semi-axes, a normal built from finite differences of THIS
// function is already the world normal — no inverse-transpose needed.
fn bodyLocal(dir: vec3f, r: vec3f, seed: f32) -> vec3f {
  return dir * bodyBump(dir, seed) * r;
}

@vertex
fn vs_body(in: BodyIn, @builtin(instance_index) inst: u32) -> BodyOut {
  let b = u.bodies[i32(inst)];
  // semi-axes back out of the reciprocals the shadow test wants
  let r = vec3f(1.0) / max(b.invRadiusSoft.xyz, vec3f(1e-4));
  let seed = f32(inst) * 3.7 + 1.3;
  let d = normalize(in.dir);
  // Tangent frame, robust at the poles. (t1, t2, d) is right-handed
  // (cross(t1, t2) = d), so the cross product below points outward.
  let t0 = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(d.y) > 0.9);
  let t1 = normalize(cross(t0, d));
  let t2 = cross(d, t1);
  let eps = 0.03;
  let p0 = bodyLocal(d, r, seed);
  let pu = bodyLocal(normalize(d + t1 * eps), r, seed);
  let pv = bodyLocal(normalize(d + t2 * eps), r, seed);
  var nrm = normalize(cross(pu - p0, pv - p0));
  if (dot(nrm, d) < 0.0) { nrm = -nrm; }
  var out: BodyOut;
  out.world = b.centerR.xyz + p0;
  out.nrm = nrm;
  out.bi = inst;
  out.clip = u.viewProj * vec4f(out.world, 1.0);
  return out;
}

// The body's lit FACTOR — same shape as bedLit's: a share that survives shadow
// (sky fill) plus one that does not (N.L beam, and caustics).
//
// NO LIGHTING ENVELOPE HERE — no sunLevel, no sunTint, no column extinction.
// This is not an oversight, it is the contract the refraction target already
// has: the surface shader multiplies its refrTex tap by `lightTint`, `sunLev`
// and Beer-Lambert AFTER sampling (exactly as it does for fs_land_refract's
// sand), so including them here would apply them twice. fs_body, which has no
// surface above it, applies its own.
//
// `skip` is this body's own instance index. Excluding self BY INDEX is why this
// whole feature has no depth bias: the lit hemisphere is handled exactly by
// N.L, and there is no epsilon anywhere to tune, leak, or peter-pan.
fn bodyLitFactor(p: vec3f, n: vec3f, skip: u32, dist: f32,
                 causticGain: f32, footprint: f32) -> f32 {
  let column = max(-p.y, 0.0);
  let submerged = smoothstep(0.02, 0.15, column);
  // A body breaking the surface transitions from the air beam to the refracted
  // one over the same 13 cm the sand's submerged mask uses.
  let ldir = normalize(mix(normalize(u.sunDir), sunDirWater(), submerged));
  let vis = bodyShadowSkip(p, ldir, SUN_SPREAD_WATER * submerged,
                           max(footprint, 0.03), skip) * cloudShadow(p.xz);
  let nl = max(dot(n, ldir), 0.0);
  // sky fill: a boulder sees roughly the upper hemisphere. Unoccludable, so it
  // is the body's equivalent of BED_SHADOW_FLOOR — same 0.34 base, so a rock
  // and the sand it sits on go into shadow to the same depth.
  let fill = 0.34 + 0.26 * max(n.y, 0.0);
  // Caustics land on the rock's up-facing surfaces exactly as on the sand, and
  // die in shadow with everything else.
  let focus = causticGain * exp(-column * 0.12)
            * clamp(1.0 - dist / (120.0 * u.lodScale), 0.0, 1.0) * submerged;
  return 0.9 * (fill + nl * vis)
       + focus * vis * max(n.y, 0.0) * (1.6 * causticWeb(p.xz) - 0.18);
}

@fragment
fn fs_body_refract(in: BodyOut) -> @location(0) vec4f {
  let dist = distance(u.cameraPos, in.world);
  let v = normalize(u.cameraPos - in.world);
  var n = normalize(in.nrm);
  if (dot(n, v) < 0.0) { n = -n; }
  let fw = max(length(fwidth(in.world.xz)), 0.03);
  let lit = bodyLitFactor(in.world, n, in.bi, dist, u.causticStrength, fw);
  // `a` is the submerged mask the surface uses to reject taps that land above
  // water — the SAME ramp fs_land_refract uses, so a rock breaching the surface
  // hands back to the unoffset tap over the same 13 cm the sand does.
  let column = max(-in.world.y, 0.0);
  return vec4f(ROCK_ALBEDO * lit, smoothstep(0.02, 0.15, column));
}

@fragment
fn fs_body(in: BodyOut) -> @location(0) vec4f {
  let SKY = skyState(u.sunDir, u.moonDir, u.skyTurbidity, u.skyRayleigh, u.skyIntensity);
  let dist = distance(u.cameraPos, in.world);
  let v = normalize(u.cameraPos - in.world);
  var n = normalize(in.nrm);
  if (dot(n, v) < 0.0) { n = -n; }
  let fw = max(length(fwidth(in.world.xz)), 0.03);
  let column = max(-in.world.y, 0.0);
  let submerged = smoothstep(0.02, 0.15, column);
  let lit = bodyLitFactor(in.world, n, in.bi, dist, u.uwCaustics, fw);
  // This path has no surface above it, so it supplies its own envelope: the
  // column's own extinction of the downwelling beam below water, the plain
  // solar level above.
  let lightTint = mix(vec3f(1.0), sunTint(SKY), 0.6);
  let envelope = mix(vec3f(sunLevel(SKY)) * lightTint,
                     waterAmbient(column, SKY, waterSigma(u.uwTurbidity, u.chlorophyll)),
                     submerged);
  var color = ROCK_ALBEDO * lit * envelope;
  // From here on this is fs_land's tail verbatim. Copied rather than shared
  // because fs_land's tail is inlined; if it is ever factored out, factor this
  // with it.
  let uw = underwaterAt(-v, u.camDepth, u.lensR);
  let fog = 1.0 - exp(-dist * 3e-5);
  let air = mix(color, skyColor(normalize(vec3f(-v.x, 0.02, -v.z)), SKY), fog);
  let ext = exp(-waterSigma(u.uwTurbidity, u.chlorophyll) * u.uwFog * dist);
  let beamU = beamVis(u.cameraPos, -v, dist,
                      waterSigma(u.uwTurbidity, u.chlorophyll).g * u.uwFog);
  var murk = waterFogAlong(-v, u.camDepth, SKY, u.uwTurbidity, u.chlorophyll) * beamU;
  if (u.camDepth > -u.lensR) {
    murk += vec3f(0.9, 1.0, 0.95) * suspended(in.world.xz, 1.0 - ext.g) * beamU;
  }
  color = mix(air, color * ext + murk * (1.0 - ext), uw);
  color = 1.0 - exp(-1.8 * color);
  return vec4f(pow(color, vec3f(1.0 / 2.2)), 1.0);
}
