// Shared by the sky dome and the ocean's reflections so both match.

// Per-channel extinction of sea water [1/m]: red dies within a couple of
// meters, blue-green carries. The surface shader's through-the-water tint
// uses the same coefficients, so a submerged view and the view from above
// agree on the water's colour.
const WATER_SIGMA: vec3f = vec3f(0.25, 0.04, 0.02);
// Refraction: entering air from water, and the cosine of the critical angle
// (48.6 deg from vertical) that bounds Snell's window
const WATER_TO_AIR: f32 = 1.333;

fn sunWarmth(sunDir: vec3f) -> f32 {
  return 1.0 - smoothstep(0.03, 0.5, clamp(sunDir.y, 0.0, 1.0));
}

fn sunTint(sunDir: vec3f) -> vec3f {
  let w = sunWarmth(sunDir);
  return mix(vec3f(1.0, 0.97, 0.9), vec3f(1.25, 0.5, 0.18), w * w);
}

fn skyColor(dir: vec3f, sunDir: vec3f) -> vec3f {
  let w = sunWarmth(sunDir);
  let t = pow(clamp(dir.y, 0.0, 1.0), mix(0.5, 0.65, w));
  let facing = pow(0.5 + 0.5 * dot(normalize(dir.xz + vec2f(1e-5, 0.0)), normalize(sunDir.xz)), 3.0);
  let zenith = mix(vec3f(0.11, 0.30, 0.60), vec3f(0.08, 0.12, 0.30), w);
  let horizon = mix(vec3f(0.62, 0.72, 0.83), mix(vec3f(0.42, 0.36, 0.52), vec3f(1.1, 0.45, 0.16), facing), w);
  var c = mix(horizon, zenith, t);
  let g = max(dot(dir, sunDir), 0.0);
  c += sunTint(sunDir) * (pow(g, mix(40.0, 10.0, w)) * mix(0.25, 0.6, w) + pow(g, 4000.0) * 3.0);
  return c;
}

// Whether THIS view ray is in water, per pixel.
//
// A point camera is either under or over, so blending the whole frame by one
// scalar washes the sky into murk the moment the eye touches the surface. A
// real port has a radius: the ray leaving in direction `dir` crosses the
// housing at about `eye + lensR * dir`, so that ray is submerged when
//   eye.y + lensR * dir.y < surfaceY   <=>   dir.y < camDepth / lensR
// which splits the screen at a waterline instead of fading everything. It
// saturates on its own — once |camDepth| exceeds lensR the threshold leaves
// the range of dir.y and the whole frame is uniformly under or over.
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

// Extinction of the water body. Chlorophyll absorbs blue and red and leaves
// green, which is what turns coastal water from blue toward jade; turbidity is
// suspended sediment, which scatters every channel and shortens sight lines
// without much changing the hue.
fn waterSigma(turbidity: f32, chlorophyll: f32) -> vec3f {
  return (WATER_SIGMA + vec3f(0.020, 0.004, 0.085) * chlorophyll) * (1.0 + 3.5 * turbidity);
}

// Hue a thick column settles on, swinging blue -> jade with chlorophyll
fn waterHue(turbidity: f32, chlorophyll: f32) -> vec3f {
  let hue = mix(vec3f(0.09, 0.34, 0.46), vec3f(0.17, 0.38, 0.19), clamp(chlorophyll / 1.5, 0.0, 1.0));
  // sediment is lit by every direction at once, so murk reads paler
  return mix(hue, vec3f(0.34, 0.38, 0.32), 0.5 * clamp(turbidity, 0.0, 1.0));
}

// Daylight surviving to a given depth, tinted by the sun's own colour
fn waterAmbient(depth: f32, sunDir: vec3f, sigma: vec3f) -> vec3f {
  let level = mix(0.18, 1.0, smoothstep(0.0, 0.5, clamp(sunDir.y, 0.0, 1.0)));
  return exp(-sigma * max(depth, 0.0)) * level * mix(vec3f(1.0), sunTint(sunDir), 0.6);
}

// Colour a long path through the volume converges to: the murk that closes in
// around a submerged camera
fn waterVolume(depth: f32, sunDir: vec3f, turbidity: f32, chlorophyll: f32) -> vec3f {
  return waterHue(turbidity, chlorophyll) * waterAmbient(depth, sunDir, waterSigma(turbidity, chlorophyll));
}

// The same murk, seen along a given ray. A ray aimed up scatters toward the
// eye from water nearer the surface, where more daylight survives, so it
// converges brighter than one aimed down into the dark.
// Every submerged view — the surface, the sea floor, and the open background
// where neither is drawn — converges through THIS function, so distant
// geometry and the gap beside it reach the same colour and no brightness step
// appears along the silhouette.
fn waterFogAlong(dir: vec3f, depth: f32, sunDir: vec3f, turbidity: f32, chlorophyll: f32) -> vec3f {
  return waterVolume(depth, sunDir, turbidity, chlorophyll) * mix(0.55, 1.4, clamp(dir.y * 0.5 + 0.5, 0.0, 1.0));
}
