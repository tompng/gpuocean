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
}

// F(theta, gamma) = (1 + A e^(B/cos theta)) (1 + C e^(D gamma) + E cos^2 gamma)
// for luminance and both chromaticities at once. cos theta is floored: it
// reaches 0 at the horizon, the model's one real singularity.
fn perez(A: vec3f, B: vec3f, C: vec3f, D: vec3f, E: vec3f, cosTheta: f32, gamma: f32) -> vec3f {
  let ct = max(cosTheta, 0.01);
  let cg = cos(gamma);
  return (vec3f(1.0) + A * exp(B / ct)) * (vec3f(1.0) + C * exp(D * gamma) + E * (cg * cg));
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
  return s;
}

fn skyColor(dir: vec3f, s: SkyState) -> vec3f {
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

  // Perez carries the aureole but not the disc, and the disc is what the
  // specular highlight and Snell's window need to see
  rgb += SUN_DISC_PEAK * s.sunTrans
       * smoothstep(SUN_COS_OUTER, SUN_COS_INNER, cosGamma)
       * smoothstep(-0.004, 0.004, d.y);

  return max(rgb * s.intensity, vec3f(0.0));
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
