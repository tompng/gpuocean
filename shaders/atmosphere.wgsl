// Shared by the sky dome and the ocean's reflections so both match.

// Per-channel extinction of clear sea water [1/m]: red dies within a couple of
// meters, blue-green carries.
const WATER_SIGMA: vec3f = vec3f(0.25, 0.04, 0.02);
// Refraction: entering air from water, and the cosine of the critical angle
// (48.6 deg from vertical) that bounds Snell's window
const WATER_TO_AIR: f32 = 1.333;

// --- Preetham / Perez analytic sky ------------------------------------------
// Preetham, Shirley & Smits, "A Practical Analytic Model for Daylight",
// SIGGRAPH 99. Defined on the upper hemisphere and returning ABSOLUTE
// luminance, so it needs a reference to land in this renderer's units.
const SKY_PI: f32 = 3.141592653589793;

// Preetham's zenith luminance is in kcd/m^2. Dividing by 30 (30000 cd/m^2)
// puts a clear zenith with the sun at 60 deg near 0.19/0.29/0.57 scene-linear,
// which the existing tonemap lands close to the sky this replaces; raw cd/m^2
// would be ~6000 and clip every channel to white. The reference is ABSOLUTE on
// purpose — normalising per frame by Yz would make a 2 deg sun as bright as a
// 60 deg one and discard the whole point of the model.
const SKY_Y_REF_KCD: f32 = 30.0;
// D65: the white point `rayleigh` scales the sky's excursion away from
const SKY_WHITE: vec2f = vec2f(0.3127, 0.3290);
// Rayleigh optical depth at sea level, tau = 0.008735 * lambda^-4.08 sampled
// at the sRGB primaries, and the Angstrom lambda^-1.3 aerosol basis
const TAU_RAYLEIGH: vec3f = vec3f(0.0635, 0.1121, 0.1986);
const TAU_AEROSOL_BASIS: vec3f = vec3f(1.8814, 2.2550, 2.7059);
// Straddles the sun's true 0.2665 deg angular radius with an antialiasing band
const SUN_COS_OUTER: f32 = 0.99998629;
const SUN_COS_INNER: f32 = 0.99999123;
// The disc's real luminance is ~1.6e9 cd/m^2, which normalised would clip to
// white at every elevation; 6.0 keeps noon white and sunset orange
const SUN_DISC_PEAK: f32 = 6.0;
const NIGHT_SKY: vec3f = vec3f(0.006, 0.009, 0.018);
// Moonlight is scattered sunlight, reddened once by the lunar surface and then
// perceived through scotopic vision, which reads it as cool. The moon's angular
// radius (0.259 deg) is within a hair of the sun's, so it reuses the disc
// cosines above.
const MOON_TINT: vec3f = vec3f(0.62, 0.74, 1.0);
const MOON_DISC: f32 = 1.5;
// Physically moonlight is ~1e-6 of daylight; the eye dark-adapts, so it is
// rendered as a dim cool key rather than as darkness
const MOON_LEVEL: f32 = 0.13;

// Everything depending on the sun and turbidity but not the view ray.
// skyColor() runs up to three times per ocean fragment (reflection, Snell's
// window, aerial perspective), so this is built once and threaded through.
struct SkyState {
  sunDir: vec3f,
  // Perez coefficients, packed (Y, x, y) per coefficient
  A: vec3f, B: vec3f, C: vec3f, D: vec3f, E: vec3f,
  // (Yz [kcd/m^2], xz, yz)
  zenith: vec3f,
  normF: vec3f,
  sunTrans: vec3f,
  airMass: f32,
  rayleigh: f32,
  intensity: f32,
  day: f32,
  moonDir: vec3f,
  // 1 - day: how far the sun is past setting
  nightMix: f32,
  // how much the moon is actually up to light anything
  moonLit: f32,
  // Cloud lighting, built once per fragment because skyColor() runs up to four
  // times: the direct beam a fully lit white cloud face returns, and the
  // skylight reaching the deck. Both derive from the sky itself, which is what
  // makes the deck redden with the sun inside the same absolute normalisation.
  cloudSun: vec3f,
  cloudAmb: vec3f,
}

// F(theta, gamma) = (1 + A e^(B/cos theta)) (1 + C e^(D gamma) + E cos^2 gamma)
// for luminance and both chromaticities at once. cos theta is floored: it
// reaches 0 at the horizon, the model's one real singularity.
fn perez(A: vec3f, B: vec3f, C: vec3f, D: vec3f, E: vec3f, cosTheta: f32, gamma: f32) -> vec3f {
  let ct = max(cosTheta, 0.01);
  let cg = cos(gamma);
  return (vec3f(1.0) + A * exp(B / ct)) * (vec3f(1.0) + C * exp(D * gamma) + E * (cg * cg));
}

// --- clouds ------------------------------------------------------------------
//
// One analytic deck at altitude, intersected by the view ray. It lives HERE, in
// the file both the sky pass and the ocean surface share, for the same reason
// the Perez model does: skyColor() is the single door every sky lookup goes
// through — the dome, the mirror reflection, Snell's window, and the aerial
// perspective — so a deck composited inside it is reflected, refracted and
// fogged with no extra code and no way for the reflection to disagree with the
// sky above it.
//
// Cloud parameters live in their OWN uniform block, declared here rather than
// added to each pass's Uniforms struct. atmosphere.wgsl is textually prepended
// to two shader modules whose `u` disagree — sky.wgsl's Uniforms has no `time`
// at all — so shared code cannot read `u`. Declaring the block in the shared
// text makes the binding number and the layout identical in both modules by
// construction, and both bind groups can point at ONE buffer, so sky and ocean
// cannot drift apart even by a mistake in the JS.
struct CloudUniforms {
  camPos: vec3f,
  pad0: f32,
  // Accumulated wind advection in metres, integrated in f64 on the CPU rather
  // than as wind * time in here. Two reasons: sky.wgsl's Uniforms has no clock
  // at all, so a shared block that needs one would have to invent it; and an
  // integrated offset means moving the windSpeed slider changes the deck's
  // VELOCITY, where wind * time would teleport it.
  drift: vec2f,
  coverage: f32,
  // base altitude and vertical extent of the deck [m]
  altitude: f32,
  thickness: f32,
  // extinction of a fully dense sample [1/m]
  density: f32,
  // horizontal size of the largest cloud feature [m]
  scale: f32,
  // width of the coverage threshold: 0 is a hard cut, 0.4 is stratus haze
  softness: f32,
  // strength of the Worley erosion that turns blobs into cauliflower
  detail: f32,
  // 0 fluffy cumulus with thinning tops, 1 flat-topped anvil
  anvil: f32,
  // 0 white cumulus, 1 storm grey
  absorption: f32,
  // gain on the skylight term
  ambient: f32,
  // radians per pixel: 2 * tan(fovY / 2) / heightPx. Stands in for the mip
  // chain analytic noise does not have.
  pixelAngle: f32,
  // 0 full march, 1 half, 2 single sample
  lod: f32,
  // how much faster the top of the deck runs downwind than its base
  shear: f32,
  // strength of the deck's shadow on the water (used by cloudShade)
  shadow: f32,
}
@group(0) @binding(15) var<uniform> cld: CloudUniforms;

// Earth radius. The ONLY reason a planet appears in a flat-water renderer is to
// bound the horizon. A flat layer at y = altitude is hit at
// t = (altitude - eyeY) / dir.y, which diverges as the ray levels out, so every
// cloud feature compresses without bound into the horizon LINE — the one place
// an ocean renderer cannot afford to alias, because the horizon is the picture.
// On a shell the levelling ray leaves tangentially and still reaches the deck at
// a finite t ~ sqrt(2 R altitude) (159 km for a 2 km deck), so the deck closes
// onto the horizon at a bounded scale; and the ray that grazes sea level is
// exactly the ray with dir.y = 0, which makes the below-horizon test in
// cloudLayer() exact instead of an epsilon.
const CL_EARTH_R: f32 = 6371000.0;
// Radiance of a fully lit white cloud face, in the same absolute units as the
// sky. A Lambertian patch of albedo 0.85 under the direct beam returns
//   L = 0.85 * E0 * trans / pi = 0.85 * 127500 lx * trans / pi = 34500 * trans
// cd/m^2, and the same division by SKY_Y_REF_KCD that turns Preetham's kcd into
// scene-linear turns that into 1.15 * sunTrans. So this constant is DERIVED
// from the sky's normalisation rather than tuned against it, and because it
// multiplies sunTrans the deck reddens channel-for-channel with the sun disc.
const CL_SUN_GAIN: f32 = 1.15;
// Moonlight is a look level, not an absolute, exactly like MOON_LEVEL: physical
// moonlit cloud is ~1e-6 of this, and the eye dark-adapts.
const CL_MOON_GAIN: f32 = 0.07;
// Fraction of the up-sky's radiance a cloud sees as skylight. The zenith is the
// brightest part of the hemisphere a cloud top looks at, so a fraction of it
// stands in for the whole irradiance integral.
const CL_AMB_GAIN: f32 = 0.55;
// Mean extinction along the sun ray as a fraction of the local sample's, used
// by the shadow estimate below.
const CL_SHADOW_MEAN: f32 = 0.6;
// Blend toward the Beer-Powder curve; 0 is plain Beer (flat, edges as bright as
// centres), 1 the full powder-sugar dip.
const CL_POWDER: f32 = 0.5;
// Multiple scattering. A cloud droplet's single-scattering albedo is ~0.9999 and
// its phase function is strongly forward (g ~ 0.85), so single-scatter Beer's law
// underestimates the interior by an order of magnitude and paints black cloud
// bases — the classic failure of a one-term march. Similarity theory gives the
// diffuse transmittance of a conservative slab as 1 / (1 + 0.75 (1 - g) tau),
// so CL_MS_K = 0.75 * (1 - 0.85) = 0.11 is derived, not tuned; CL_MS is the
// diffuse fraction at the lit surface, which with the beam term below sums to
// just under the Lambertian value CL_SUN_GAIN already carries.
const CL_MS: f32 = 0.55;
const CL_MS_K: f32 = 0.11;
const CL_OCTAVES: i32 = 5;
const CL_STEPS: i32 = 8;
// Angular width of the fade below the horizon, in sin(elevation). 0.01 is
// 0.57 deg — well over a pixel at 60 deg vertical FOV, so it antialiases, and
// well under the compressed horizon band, so nothing lands underfoot.
const CL_DIP_FADE: f32 = 0.01;
const CL_MAX_DIST: f32 = 4.0e5;
// Irrational-ish rotation between octaves: without it value noise stacks its
// grid on itself and the deck shows axis-aligned creases.
const CL_ROT: mat2x2f = mat2x2f(0.8, 0.6, -0.6, 0.8);

// --- analytic noise ----------------------------------------------------------
// There is no 3D noise texture in this project and this code deliberately does
// not add one: see the notes. Everything below is hash noise, so atmosphere.wgsl
// keeps needing exactly one binding, which is what lets the sky pass and the
// ocean pass share it without their very different bind groups having to agree
// on a texture.

// lowbias32 (Chris Wellons). Four ops of avalanche, no visible lattice
// structure at the frequencies used here.
fn cl_mix32(x0: u32) -> u32 {
  var x = x0;
  x = x ^ (x >> 16u);
  x = x * 0x7feb352du;
  x = x ^ (x >> 15u);
  x = x * 0x846ca68bu;
  x = x ^ (x >> 16u);
  return x;
}

fn cl_seed(c: vec2i) -> u32 {
  return bitcast<u32>(c.x) * 0x9e3779b9u + bitcast<u32>(c.y) * 0x85ebca6bu;
}

fn cl_hash1(c: vec2i) -> f32 {
  return f32(cl_mix32(cl_seed(c))) * (1.0 / 4294967296.0);
}

// Two decorrelated values from one mix, for Worley feature points
fn cl_hash2(c: vec2i) -> vec2f {
  let h = cl_mix32(cl_seed(c));
  return vec2f(f32(h >> 16u), f32(h & 0xffffu)) * (1.0 / 65536.0);
}

fn cl_vnoise(p: vec2f) -> f32 {
  let i = floor(p);
  let f = p - i;
  let w = f * f * (3.0 - 2.0 * f);
  let b = vec2i(i32(i.x), i32(i.y));
  let a00 = cl_hash1(b);
  let a10 = cl_hash1(b + vec2i(1, 0));
  let a01 = cl_hash1(b + vec2i(0, 1));
  let a11 = cl_hash1(b + vec2i(1, 1));
  return mix(mix(a00, a10, w.x), mix(a01, a11, w.x), w.y);
}

// 2D value-noise fBm with every octave faded out as its wavelength falls to the
// sample's world footprint. This is mip filtering done in the amplitude domain —
// exactly the substitution sampleWaves() and surfaceNormal() already make on the
// band-limited wave textures — and it is what makes analytic noise viable here:
// there is no mip chain to fall back on, and the ray-marched sample point jumps
// far enough between steps that no screen-space derivative would give the right
// level anyway.
//
// Returns (value in 0..1, fraction of the field's variance actually resolved).
// The second component is load-bearing: it is how the deck knows to stop
// thresholding and converge on its mean.
fn cl_fbm(p0: vec2f, foot: f32, oct: i32) -> vec2f {
  var p = p0;
  var amp = 0.5;
  var freq = 1.0;
  var sum = 0.0;
  var wsum = 0.0;
  var full = 0.0;
  for (var i = 0; i < oct; i++) {
    let w = 1.0 - smoothstep(0.5, 1.2, foot * freq);
    if (w <= 0.0) {
      // The fade is monotonic in frequency, so every remaining octave is gone
      // too. Their amplitudes still count toward `full` — that is the record of
      // how much of the field went missing — and they sum to 2 * amp.
      full = full + 2.0 * amp;
      break;
    }
    full = full + amp;
    sum = sum + amp * w * (cl_vnoise(p) - 0.5);
    wsum = wsum + amp * w;
    p = CL_ROT * p * 2.03 + vec2f(19.7, 4.3);
    freq = freq * 2.03;
    amp = amp * 0.5;
  }
  return vec2f(clamp(0.5 + sum, 0.0, 1.0), wsum / max(full, 1e-5));
}

// Worley F1. fBm alone gives soft blobs; the billowy cauliflower edge of a
// cumulus is the inverse of a distance field, and one erosion pass against
// 1 - F1 is what turns a blob into a cloud. Nine hashes, so it is gated hard.
fn cl_worley(p: vec2f) -> f32 {
  let ip = floor(p);
  let fp = p - ip;
  let b = vec2i(i32(ip.x), i32(ip.y));
  var m = 8.0;
  for (var j = -1; j <= 1; j++) {
    for (var i = -1; i <= 1; i++) {
      let r = vec2f(f32(i), f32(j)) + cl_hash2(b + vec2i(i, j)) - fp;
      m = min(m, dot(r, r));
    }
  }
  return clamp(sqrt(m), 0.0, 1.0);
}

fn cl_remap(x: f32, lo: f32, hi: f32) -> f32 {
  return clamp((x - lo) / max(hi - lo, 1e-4), 0.0, 1.0);
}

// Extinction [1/m] at a world point, with pos.y the point's ALTITUDE above sea
// level (the caller computes it radially, not as a flat y — see cloudLayer).
// `foot` is the world-space width of the pixel at this point, in metres.
fn cl_density(pos: vec3f, foot: f32) -> f32 {
  let thick = max(cld.thickness, 1.0);
  let hN = (pos.y - cld.altitude) / thick;
  if (hN < 0.0 || hN > 1.0) { return 0.0; }
  // Rounded base, flat top; anvil pushes the flattening upward. Evaluated
  // before any noise so the slab's own boundaries cost nothing.
  let prof = smoothstep(0.0, 0.18, hN) * (1.0 - smoothstep(mix(0.55, 0.9, cld.anvil), 1.0, hN));
  if (prof <= 0.0) { return 0.0; }
  let s = max(cld.scale, 1.0);
  // Advection, with a mild shear so the top of the deck outruns its base and
  // towers lean downwind instead of standing to attention
  let q = pos.xz - cld.drift * (1.0 + cld.shear * hN);
  let base = cl_fbm(q / s, foot / s, CL_OCTAVES);
  // Coverage is a threshold on the field, softened by `softness`. Towers thin
  // upward unless anvil spreads them back out.
  let cover = clamp(cld.coverage * (1.0 - 0.45 * hN * (1.0 - cld.anvil)), 0.0, 1.0);
  var d = cl_remap(base.x, 1.0 - cover, 1.0 - cover + max(cld.softness, 0.02));
  // Where the field is unresolved — the horizon band, and every ray the ocean
  // sends in with a widened footprint — there is no detail left to threshold,
  // so converge on the MEAN cover rather than on the threshold of the mean.
  // Without this the deck fades to clear exactly where it should be closing
  // into a flat band, because the mean of the field sits at 0.5 and a coverage
  // of 0.5 thresholds it to nothing.
  d = mix(cover, d, base.y);
  if (d <= 0.0) { return 0.0; }
  // Erosion, gated twice: skipped in the interior where the threshold has
  // stopped discriminating, and faded out (smoothly, so no ring appears) once
  // its cell size drops under the footprint. Both are pure early-outs on the
  // expensive half of this function.
  let dScale = max(s * 0.11, 1.0);
  let ew = cld.detail * (1.0 - smoothstep(0.15, 0.4, foot / dScale));
  if (ew > 0.0 && d < 0.92) {
    let billow = 1.0 - cl_worley(q / dScale + vec2f(11.0, 37.0));
    // the top of the deck erodes hardest: that is where the wisps are
    d = cl_remap(d, billow * ew * mix(0.35, 1.0, hN), 1.0);
  }
  return d * prof * cld.density;
}

// --- ray / shell -------------------------------------------------------------

// Far and near intersections of a ray with the sphere of radius rs, from radius
// r0 with vertical cosine dy. Negative when the sphere is not on the ray.
fn cl_far(dy: f32, r0: f32, rs: f32) -> f32 {
  let b = r0 * dy;
  let disc = b * b - (r0 * r0 - rs * rs);
  return select(-1.0, -b + sqrt(max(disc, 0.0)), disc >= 0.0);
}

fn cl_near(dy: f32, r0: f32, rs: f32) -> f32 {
  let b = r0 * dy;
  let disc = b * b - (r0 * r0 - rs * rs);
  return select(-1.0, -b - sqrt(max(disc, 0.0)), disc >= 0.0);
}

// Energy-normalised two-lobe phase: mean 1 over the sphere, so an isotropic
// medium reproduces exactly the Lambertian face value CL_SUN_GAIN carries and
// the phase function cannot inject or destroy energy. The forward lobe is the
// silver lining; the small backward lobe is the glow around the antisolar point.
fn cl_phase(c: f32) -> f32 {
  let g1 = 0.72;
  let d1 = 1.0 + g1 * g1 - 2.0 * g1 * c;
  let g2 = -0.4;
  let d2 = 1.0 + g2 * g2 - 2.0 * g2 * c;
  return 0.6
       + 0.35 * (1.0 - g1 * g1) / max(d1 * sqrt(d1), 1e-4)
       + 0.05 * (1.0 - g2 * g2) / max(d2 * sqrt(d2), 1e-4);
}

struct CloudHit {
  // radiance the deck ADDS along the ray, and the transmittance it leaves for
  // whatever is behind it: Perez, the night floor, the moon halo, both discs
  scatter: vec3f,
  trans: f32,
}

// `spread` widens the sampling footprint for rays that are not one pixel wide:
// the ocean's mirror reflection is smeared by wave curvature, Snell's window by
// refraction, and the aerial-perspective ray stands for a whole hemisphere of
// haze. Widening it both antialiases those lookups and makes them cheap, because
// cl_fbm's octave fade and the single-sample path below key off the same number.
fn cloudLayer(dir: vec3f, s: SkyState, spread: f32) -> CloudHit {
  var hit = CloudHit(vec3f(0.0), 1.0);
  if (cld.coverage <= 0.0 || cld.density <= 0.0 || cld.thickness <= 0.0) { return hit; }

  let eyeY = max(cld.camPos.y, 0.0);
  let r0 = CL_EARTH_R + eyeY;
  let rBot = CL_EARTH_R + max(cld.altitude, 1.0);
  let rTop = rBot + cld.thickness;

  // Rays pointing BELOW the horizon. sin of the horizon depression for this eye
  // height (0 at sea level, -0.0056 at 100 m): a ray steeper than this hits the
  // water, so there is nothing above it to see.
  let sinDip = -sqrt(max(2.0 * CL_EARTH_R * eyeY + eyeY * eyeY, 0.0)) / r0;
  let below = smoothstep(sinDip, sinDip - CL_DIP_FADE, dir.y);
  if (below >= 1.0) { return hit; }
  // Inside the fade the ray is CLAMPED to the horizon rather than rejected, for
  // the same reason perez() clamps cos(theta) at the horizon: the ocean's
  // reflection ray dips below zero on the back of a wave, and if clouds
  // vanished there while the blue stayed, neighbouring fragments would pop. The
  // clamped sample lands 159 km away, where cl_fbm has already faded every
  // octave, so what it returns is the smooth mean of the deck — a horizon band,
  // never cloud shapes underfoot.
  let dy = max(dir.y, sinDip + 1e-6);
  let dc = normalize(vec3f(dir.x, dy, dir.z));

  var t0: f32;
  var t1: f32;
  if (r0 < rBot) {
    // the normal case: eye under the deck, both crossings are outgoing
    t0 = cl_far(dy, r0, rBot);
    t1 = cl_far(dy, r0, rTop);
  } else if (r0 > rTop) {
    // above the deck, looking down onto the tops
    t0 = cl_near(dy, r0, rTop);
    t1 = cl_near(dy, r0, rBot);
  } else {
    // inside it: leaves through the base if the ray descends far enough,
    // otherwise out through the top
    t0 = 0.0;
    let tb = cl_near(dy, r0, rBot);
    t1 = select(cl_far(dy, r0, rTop), tb, tb > 0.0);
  }
  t0 = max(t0, 0.0);
  t1 = min(t1, CL_MAX_DIST);
  if (t1 <= t0) { return hit; }

  var steps = CL_STEPS;
  if (cld.lod > 0.5) { steps = 4; }
  if (cld.lod > 1.5) { steps = 1; }
  // A footprint wider than the largest cloud feature has already flattened the
  // field to its mean, and a constant field integrates exactly in one step. This
  // is what makes the aerial-perspective lookup essentially free.
  let footNear = t0 * cld.pixelAngle * spread;
  if (footNear > cld.scale) { steps = 1; }
  let dt = (t1 - t0) / f32(steps);

  // Empty-sky pre-test. Two octaves bound the remaining three to +-0.12 (each
  // contributes at most amp/2), so a probe that misses the coverage threshold by
  // more than that cannot be rescued by finer detail — and erosion only ever
  // removes density. Sound only while the traversal is shorter than the field's
  // horizontal correlation length, since one probe cannot speak for a 20 km
  // horizon path, and while the footprint is small enough that cl_density is
  // still thresholding rather than converging on its mean cover.
  let sc0 = max(cld.scale, 1.0);
  let tMid = 0.5 * (t0 + t1);
  if (t1 - t0 < 0.6 * sc0 && tMid * cld.pixelAngle * spread < 0.05 * sc0) {
    let altM = sqrt(r0 * r0 + 2.0 * tMid * r0 * dy + tMid * tMid) - CL_EARTH_R;
    let pM = cld.camPos + dc * tMid;
    let hM = clamp((altM - cld.altitude) / max(cld.thickness, 1.0), 0.0, 1.0);
    let qM = vec2f(pM.x, pM.z) - cld.drift * (1.0 + cld.shear * hM);
    if (cl_fbm(qM / sc0, 0.0, 2).x + 0.12 < 1.0 - cld.coverage) { return hit; }
  }

  let ph = cl_phase(dot(dc, s.sunDir));
  let albedo = 1.0 - 0.8 * clamp(cld.absorption, 0.0, 1.0);
  var T = 1.0;
  var sc = vec3f(0.0);
  var t = t0 + 0.5 * dt;
  for (var i = 0; i < steps; i++) {
    // Altitude is RADIAL, not the flat y: at 159 km the two differ by 2 km,
    // which is the whole deck. Horizontal position is the chord, which differs
    // from the arc by 0.2% at that range.
    let alt = sqrt(r0 * r0 + 2.0 * t * r0 * dy + t * t) - CL_EARTH_R;
    let p = cld.camPos + dc * t;
    let foot = t * cld.pixelAngle * spread;
    let dens = cl_density(vec3f(p.x, alt, p.z), foot);
    if (dens > 1e-5) {
      let hN = clamp((alt - cld.altitude) / max(cld.thickness, 1.0), 0.0, 1.0);
      // Optical depth from here to the top of the deck along the sun ray, with
      // the local extinction standing in for the mean along it. No secondary
      // march and no extra noise taps, and it still carries what reads as cloud
      // lighting: bright tops, dark bases, and undersides that go black at
      // sunset because the slant path through the deck blows up as sunDir.y
      // falls. The 0.10 floor is what stops it going infinite at sunset.
      let tauSun = dens * CL_SHADOW_MEAN * (1.0 - hN) * cld.thickness / max(s.sunDir.y, 0.10);
      // Beer-Powder (Guerrilla): plain Beer makes thin edges as bright as thick
      // centres, which reads as fog rather than cloud.
      let powder = 2.0 * exp(-tauSun) * (1.0 - exp(-2.0 * tauSun));
      // Two terms: the phase-peaked single-scattered beam, which carries the
      // silver lining and the bright tops, and an isotropic diffusion floor,
      // which keeps interiors and undersides luminous instead of black and
      // still reddens with the sun because it scales the same cloudSun.
      let beam = ph * mix(exp(-tauSun), powder, CL_POWDER);
      let ms = CL_MS / (1.0 + CL_MS_K * tauSun);
      let lit = s.cloudSun * (beam + ms);
      // A cloud base sees less sky than a cloud top does
      let amb = s.cloudAmb * mix(0.3, 1.0, hN);
      let a = 1.0 - exp(-dens * dt);
      sc += T * a * albedo * (lit + amb);
      T *= 1.0 - a;
      if (T < 0.012) { break; }
    }
    t += dt;
  }
  return CloudHit(sc * (1.0 - below), mix(T, 1.0, below));
}

// Optical depth of the deck along the sun ray from a point on the water: the
// cloud shadow that sweeps the sea. One density tap at the mid-plane, at a
// deliberately wide footprint so the shadow is soft, its edges do not alias
// across a moving surface, and cl_fbm drops to two octaves. Multiply into
// sunLevel()/spec at the call site.
// Fraction of each term that is DIRECT beam, and so can be shadowed. The rest
// is dome light, which a cloud does not remove: an overcast day is dim and flat,
// not dark. Folding the shadow into sunLevel() instead would scale the 0.10
// daylight floor, the moon handover, and every dome-lit term riding on it at
// once — which is the "shadowed water goes black" failure.
const CLOUD_DIRECT_WATER: f32 = 0.72;
const CLOUD_DIRECT_FOAM: f32 = 0.5;
const CLOUD_DIRECT_SAND: f32 = 0.7;

fn cloudShade(p: vec3f, s: SkyState) -> f32 {
  if (cld.shadow <= 0.0 || cld.coverage <= 0.0) { return 1.0; }
  let mid = cld.altitude + 0.5 * cld.thickness;
  let t = (mid - p.y) / max(s.sunDir.y, 0.08);
  if (t <= 0.0) { return 1.0; }
  let hit = p + s.sunDir * t;
  let dens = cl_density(vec3f(hit.x, mid, hit.z), 0.25 * cld.scale);
  return mix(1.0, exp(-dens * cld.thickness * 0.7), clamp(cld.shadow, 0.0, 1.0) * s.day);
}

// Everything the clear sky is except the sun disc: Perez, the night floor, the
// moon. Split out because the deck's skylight term needs a sky value that cannot
// contain the disc — sampled straight up with the sun near the zenith it would
// otherwise hand the clouds the disc's 6.0 peak as ambient.
fn skyDome(dir: vec3f, s: SkyState) -> vec3f {
  let d = normalize(dir);
  let cosGamma = clamp(dot(d, s.sunDir), -1.0, 1.0);
  // Rays dipping below the horizon (grazing reflections, the aerial
  // perspective ray) clamp to the horizon value rather than hitting the pole
  let F = perez(s.A, s.B, s.C, s.D, s.E, max(d.y, 0.0), acos(cosGamma)) * s.normF;

  let Y = s.zenith.x * F.x / SKY_Y_REF_KCD;
  var xy = s.zenith.yz * F.yz;
  // 'rayleigh' as a chromaticity gain about D65: 1 is exact Preetham, 0 a
  // neutral grey sky, >1 deepens the blue the way a larger molecular
  // scattering coefficient does. Y is untouched, so luminance is unaffected.
  xy = SKY_WHITE + (xy - SKY_WHITE) * s.rayleigh;

  let yy = max(xy.y, 1e-3);
  let XYZ = vec3f(xy.x * Y / yy, Y, (1.0 - xy.x - xy.y) * Y / yy);
  var rgb = vec3f(
    dot(vec3f( 3.2404542, -1.5371385, -0.4985314), XYZ),
    dot(vec3f(-0.9692660,  1.8760108,  0.0415560), XYZ),
    dot(vec3f( 0.0556434, -0.2040259,  1.0572252), XYZ));

  // A low sun or rayleigh > 1 can land outside sRGB, and a negative channel
  // survives the tonemap and makes pow() return NaN. Desaturating toward grey
  // of the same luminance keeps Y exact.
  let lum = dot(rgb, vec3f(0.2126, 0.7152, 0.0722));
  let lo = min(min(rgb.r, rgb.g), rgb.b);
  if (lo < 0.0 && lum > 0.0) { rgb = mix(vec3f(lum), rgb, lum / (lum - lo)); }
  rgb = mix(NIGHT_SKY * (1.0 + 2.6 * s.moonLit), max(rgb, vec3f(0.0)), s.day);
  // Moon disc and halo, gated to the night side and to above the horizon
  let cosM = dot(d, s.moonDir);
  rgb += MOON_TINT * (s.nightMix * s.moonLit * smoothstep(-0.004, 0.004, d.y)
       * (MOON_DISC * smoothstep(SUN_COS_OUTER, SUN_COS_INNER, cosM)
        + 0.05 * pow(max(cosM, 0.0), 110.0)));
  return rgb;
}

fn skyState(sunDir: vec3f, moonDir: vec3f, turbidity: f32, rayleigh: f32, intensity: f32) -> SkyState {
  var s: SkyState;
  let sd = normalize(sunDir);
  s.sunDir = sd;
  s.rayleigh = max(rayleigh, 0.0);
  s.intensity = max(intensity, 0.0);
  // Below T = 1.203 the luminance coefficient B_Y turns positive and the
  // horizon exponent blows up instead of decaying; 2.0 is also the physical
  // floor for "exceptionally clear"
  let T = clamp(turbidity, 2.0, 20.0);
  let T2 = T * T;
  let cs = clamp(sd.y, 0.0, 1.0);
  let ts = acos(cs);

  // Table 2, Preetham et al. Rows are (Y, x, y).
  s.A = vec3f( 0.1787 * T - 1.4630, -0.0193 * T - 0.2592, -0.0167 * T - 0.2608);
  s.B = vec3f(-0.3554 * T + 0.4275, -0.0665 * T + 0.0008, -0.0950 * T + 0.0092);
  s.C = vec3f(-0.0227 * T + 5.3251, -0.0004 * T + 0.2125, -0.0079 * T + 0.2102);
  s.D = vec3f( 0.1206 * T - 2.5771, -0.0641 * T - 0.8989, -0.0441 * T - 1.6537);
  s.E = vec3f(-0.0670 * T + 0.3703, -0.0033 * T + 0.0452, -0.0109 * T + 0.0529);

  let p = vec4f(ts * ts * ts, ts * ts, ts, 1.0);
  let chi = (4.0 / 9.0 - T / 120.0) * (SKY_PI - 2.0 * ts);
  s.zenith = vec3f(
    max((4.0453 * T - 4.9710) * tan(chi) - 0.2155 * T + 2.4192, 0.0),
    T2 * dot(vec4f( 0.00166, -0.00375,  0.00209, 0.0),        p)
      + T * dot(vec4f(-0.02903,  0.06377, -0.03202, 0.00394), p)
      +     dot(vec4f( 0.11693, -0.21196,  0.06052, 0.25886), p),
    T2 * dot(vec4f( 0.00275, -0.00610,  0.00317, 0.0),        p)
      + T * dot(vec4f(-0.04214,  0.08970, -0.04153, 0.00516), p)
      +     dot(vec4f( 0.15346, -0.26756,  0.06670, 0.26688), p));

  s.normF = vec3f(1.0) / max(perez(s.A, s.B, s.C, s.D, s.E, 1.0, ts), vec3f(1e-4));

  // Relative air mass, Kasten & Young 1989 — ~38 at the horizon, which is what
  // reddens a low sun; capped so a sun just below cannot underflow to zero
  let elevDeg = degrees(asin(clamp(sd.y, -1.0, 1.0)));
  let m = min(1.0 / (max(sd.y, 0.0) + 0.15 * pow(max(3.885 + elevDeg, 0.02), -1.253)), 40.0);
  s.airMass = m;
  s.sunTrans = exp(-(TAU_RAYLEIGH + max(0.04608 * T - 0.04586, 0.0) * TAU_AEROSOL_BASIS) * m);
  // The analytic model has nothing to say once the sun is down
  s.day = smoothstep(-0.25, 0.0, sd.y);
  s.nightMix = 1.0 - s.day;
  s.moonDir = normalize(moonDir);
  s.moonLit = smoothstep(-0.06, 0.22, s.moonDir.y);
  // Set last: skyDome() reads every field above, and none below.
  s.cloudSun = CL_SUN_GAIN * s.sunTrans * s.day
             + MOON_TINT * (CL_MOON_GAIN * s.moonLit * s.nightMix);
  // Only the deck consumes this, and skyDome is not cheap, so an absent deck
  // does not pay for its skylight probe
  s.cloudAmb = vec3f(0.0);
  if (cld.coverage > 0.0 && cld.density > 0.0) {
    s.cloudAmb = (CL_AMB_GAIN * max(cld.ambient, 0.0)) * skyDome(vec3f(0.0, 1.0, 0.0), s);
  }
  return s;
}


// The clear sky exactly as this renderer has always drawn it, sun disc included,
// short of the final intensity scale — skyColor applies that once, to the deck
// as well, so the two cannot come out of step.
fn skyClear(dir: vec3f, s: SkyState) -> vec3f {
  let d = normalize(dir);
  let cosGamma = clamp(dot(d, s.sunDir), -1.0, 1.0);
  // Perez carries the aureole but not the disc, and the disc is what the
  // specular highlight and Snell's window need to see
  return skyDome(d, s) + SUN_DISC_PEAK * s.sunTrans
       * smoothstep(SUN_COS_OUTER, SUN_COS_INNER, cosGamma)
       * smoothstep(-0.004, 0.004, d.y);
}

// The sky with its cloud deck. `spread` is the footprint multiplier for rays
// that are wider than a pixel; 1 is a pixel-tight sky ray.
//
// The deck OCCLUDES rather than being blended over: everything behind it —
// Perez, the night floor, the moon halo, both discs — is multiplied by its
// transmittance. That is what dims and reddens the sun as it goes behind a
// cloud, and what stops the disc punching through the deck at its full 6.0
// peak. With coverage 0 the deck returns (0, 1) and this reduces exactly to the
// old skyColor, term for term.
fn skyColorRough(dir: vec3f, s: SkyState, spread: f32) -> vec3f {
  let d = normalize(dir);
  let clear = skyClear(d, s);
  let c = cloudLayer(d, s, spread);
  return max((clear * c.trans + c.scatter) * s.intensity, vec3f(0.0));
}

fn skyColor(dir: vec3f, s: SkyState) -> vec3f {
  return skyColorRough(dir, s, 1.0);
}

// Hue of the direct beam. tau is monotonic in wavelength so red is always the
// largest channel; dividing by it leaves a pure tint with max component 1.
// Hue of whatever is lighting the water. Dividing the solar beam by its own red
// leaves a tint with max component 1, but once the sun is down every channel
// has underflowed and that ratio blows up to pure red — hence the clamp, and
// the handover to the moon.
fn sunTint(s: SkyState) -> vec3f {
  let solar = clamp(s.sunTrans / max(s.sunTrans.r, 1e-4), vec3f(0.0), vec3f(1.0));
  return mix(solar, MOON_TINT, s.nightMix);
}

// How low and reddened the sun is, 0..1 — now driven by air mass rather than a
// raw sin(elevation) ramp
fn sunWarmth(s: SkyState) -> f32 { return smoothstep(1.6, 8.0, s.airMass); }

// Direct sunlight reaching the water. 0.72 is the beam luminance with the sun
// high, so this saturates above roughly 55 degrees.
fn sunLevel(s: SkyState) -> f32 {
  let solar = mix(0.10, 1.0, clamp(dot(s.sunTrans, vec3f(0.2126, 0.7152, 0.0722)) / 0.72, 0.0, 1.0));
  // With the moon down there is nothing to key off, so the night goes properly
  // dark instead of resting on the daylight floor
  return mix(solar, MOON_LEVEL * s.moonLit, s.nightMix) * s.intensity;
}

// --- water volume -----------------------------------------------------------

// Chlorophyll absorbs blue and red and leaves green, turning coastal water from
// blue toward jade; turbidity is suspended sediment, which scatters every
// channel and shortens sight lines without much changing the hue.
fn waterSigma(turbidity: f32, chlorophyll: f32) -> vec3f {
  return (WATER_SIGMA + vec3f(0.020, 0.004, 0.085) * chlorophyll) * (1.0 + 3.5 * turbidity);
}

fn waterHue(turbidity: f32, chlorophyll: f32) -> vec3f {
  let hue = mix(vec3f(0.09, 0.34, 0.46), vec3f(0.17, 0.38, 0.19), clamp(chlorophyll / 1.5, 0.0, 1.0));
  // sediment is lit from every direction at once, so murk reads paler
  return mix(hue, vec3f(0.34, 0.38, 0.32), 0.5 * clamp(turbidity, 0.0, 1.0));
}

// Daylight surviving to a given depth, tinted by the sun's own colour
fn waterAmbient(depth: f32, s: SkyState, sigma: vec3f) -> vec3f {
  return exp(-sigma * max(depth, 0.0)) * sunLevel(s) * mix(vec3f(1.0), sunTint(s), 0.6);
}

// Colour a long path through the volume converges to: the murk that closes in
// around a submerged camera
fn waterVolume(depth: f32, s: SkyState, turbidity: f32, chlorophyll: f32) -> vec3f {
  return waterHue(turbidity, chlorophyll) * waterAmbient(depth, s, waterSigma(turbidity, chlorophyll));
}

// The same murk, seen along a given ray. A ray aimed up scatters toward the eye
// from water nearer the surface, where more daylight survives, so it converges
// brighter than one aimed down into the dark.
// Every submerged view — surface, sea floor, and the open background where
// neither is drawn — converges through THIS function, so distant geometry and
// the gap beside it reach the same colour and no step appears at the silhouette.
fn waterFogAlong(dir: vec3f, depth: f32, s: SkyState, turbidity: f32, chlorophyll: f32) -> vec3f {
  return waterVolume(depth, s, turbidity, chlorophyll) * mix(0.55, 1.4, clamp(dir.y * 0.5 + 0.5, 0.0, 1.0));
}

// --- waterline --------------------------------------------------------------

// Whether THIS view ray is in water, per pixel.
//
// A point camera is either under or over, so blending the whole frame by one
// scalar washes the sky into murk the moment the eye touches the surface. A
// real port has a radius: the ray leaving in direction `dir` crosses the
// housing at about `eye + lensR * dir`, so that ray is submerged when
//   eye.y + lensR * dir.y < surfaceY   <=>   dir.y < camDepth / lensR
// which splits the screen at a waterline instead of fading everything. It
// saturates on its own — once |camDepth| exceeds lensR the threshold leaves the
// range of dir.y and the whole frame is uniformly under or over.
fn underwaterAt(dir: vec3f, camDepth: f32, lensR: f32) -> f32 {
  let thr = camDepth / max(lensR, 1e-3);
  // the band is the meniscus: the wetted lip of water clinging to the port
  let band = 0.035 + 0.5 * clamp(lensR, 0.0, 1.0);
  return smoothstep(thr + band, thr - band, dir.y);
}

// Brightness of that meniscus: water bunched at the lip refracts a bright
// sliver, so the boundary reads as a lens rather than a cut
fn waterlineLip(dir: vec3f, camDepth: f32, lensR: f32) -> f32 {
  let thr = camDepth / max(lensR, 1e-3);
  let band = 0.035 + 0.5 * clamp(lensR, 0.0, 1.0);
  return 1.0 - smoothstep(0.0, band, abs(dir.y - thr));
}
