// Shared by the ocean surface shader and the foam simulation pass, which bind
// the same uniform buffer and wave-field texture.

struct Layer {
  // xy: direction, z: 1 / tile size, w: amplitude
  dirScaleAmp: vec4f,
  // xy: uv scroll offset
  scroll: vec4f,
}

struct Uniforms {
  viewProj: mat4x4f,
  cameraPos: vec3f,
  time: f32,
  sunDir: vec3f,
  padA: f32,
  numLayers: f32,
  choppiness: f32,
  dGrad: f32,
  hGrad: f32,
  padB: f32,
  pad0: f32,
  pad1: f32,
  pad2: f32,
  layers: array<Layer, 8>,
  capLayers: array<Layer, 6>,
  capHGrad: f32,
  rippleBias: f32,
  sssStrength: f32,
  ampInv: f32,
  seaDepth: f32,
  causticStrength: f32,
  causticScale: f32,
  leanX: f32,
  leanY: f32,
  foamThreshold: f32,
  foamRegion: f32,
  foamDecay: f32,
  foamDecayG: f32,
  foamRise: f32,
  shoreX: f32,
  slope: f32,
  foamDecaySwallow: f32,
  simDt: f32,
  waveK: f32,
  shoreCurve: f32,
  foamScale: f32,
  cPad1: f32,
  cPad2: f32,
  cPad3: f32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var waveTex: texture_2d<f32>;

fn layerUV(xz: vec2f, i: i32) -> vec2f {
  let l = u.layers[i];
  let dir = l.dirScaleAmp.xy;
  return vec2f(dot(xz, dir), dot(xz, vec2f(-dir.y, dir.x))) * l.dirScaleAmp.z + l.scroll.xy;
}

// The scene's fixed coastline: a gently curved mainland (function graph
// x = shoreX(z)) plus a wobbly-circle island offshore. chain.js carries
// matching replicas of every shape constant.
fn shoreX(z: f32) -> f32 {
  return u.shoreX + u.shoreCurve * (6.0 * sin(z * 0.041) + 3.5 * sin(z * 0.093 + 1.7));
}

const ISLAND_C: vec2f = vec2f(-45.0, 15.0);
const ISLAND_R: f32 = 20.0;

// Land positive inside; near-SDF (the boundary wobble skews it slightly)
fn islandSDF(xz: vec2f) -> f32 {
  let d = xz - ISLAND_C;
  let th = atan2(d.y, d.x);
  return ISLAND_R + 3.0 * sin(3.0 * th + 1.0) - length(d);
}

fn dShoreX(z: f32) -> f32 {
  return u.shoreCurve * (6.0 * 0.041 * cos(z * 0.041) + 3.5 * 0.093 * cos(z * 0.093 + 1.7));
}

// Landward unit normal of the coastline
fn coastNormal(z: f32) -> vec2f {
  let d = dShoreX(z);
  return vec2f(1.0, -d) / sqrt(1.0 + d * d);
}

// Signed distance to the nearest coastline (negative in the sea): the
// mainland graph foreshortened by its obliquity, unioned with the island.
// Analytic near-SDFs; swap for a baked SDF for freeform coasts.
fn coastSDF(xz: vec2f) -> f32 {
  let d = dShoreX(xz.y);
  return max((xz.x - shoreX(xz.y)) / sqrt(1.0 + d * d), islandSDF(xz));
}

// Beach rising along the landward normal, flat sea floor offshore, capped
// at a flat berm above the waterline
fn terrainHeight(xz: vec2f) -> f32 {
  return min(max(u.slope * coastSDF(xz), -u.seaDepth), 3.0);
}

// Material x where the film's band starts (the junction isobath) at a given
// alongshore position
fn simX0At(z: f32) -> f32 {
  return shoreX(z) - REST_DEPTH / u.slope;
}

// Heightless film chain state, indexed by (band coordinate b, column):
// x = s-displacement relative to the REST state, y = velocity,
// z = column tip s, w = unused. Columns 0..MAIN_COLS-1 run along the
// mainland (z in ±80); the rest loop around the island (wrapping).
@group(0) @binding(7) var simTex: texture_2d<f32>;

const SIM_NODES: i32 = 64;
const SIM_COLS: i32 = 256;
const MAIN_COLS: i32 = 160;
const ISLAND_COLS: i32 = 96;
const SIM_SPAN: f32 = 24.0;
// Seaward width of the wave-to-film handover band (also the shore ribbon's
// seaward margin and the open-ocean grid's dive-under ramp)
const SIM_BAND: f32 = 4.0;
// Junction depth: waves hand over to the film at this isobath, and the film
// thickness runs from this value at the junction to zero at the tip, exactly
// canceling the terrain rise so the resting film is the flat sea surface
const REST_DEPTH: f32 = 0.25;

// Band coordinate b (0 at the first node, SIM_SPAN at the tip node) ->
// rest s (normal distance from the static shoreline): the material band
// compresses onto the still-water wedge [-REST_DEPTH/slope, 0]; identity
// outside the band
fn simRestS(b: f32) -> f32 {
  let m = clamp(b, 0.0, SIM_SPAN);
  return -REST_DEPTH / u.slope + b + m * (REST_DEPTH / u.slope / SIM_SPAN - 1.0);
}

// Island columns wrap; mainland columns clamp at their segment
fn wrapCol(col: f32) -> f32 {
  if (col >= f32(MAIN_COLS)) {
    return f32(MAIN_COLS) + fract((col - f32(MAIN_COLS)) / f32(ISLAND_COLS)) * f32(ISLAND_COLS);
  }
  return clamp(col, 0.0, f32(MAIN_COLS - 1));
}

// Approximate alongshore arclength of a column, for material-anchored
// pattern coordinates
fn colT(col: f32) -> f32 {
  if (col < f32(MAIN_COLS)) {
    return (col / f32(MAIN_COLS - 1) - 0.5) * 160.0;
  }
  return (col - f32(MAIN_COLS)) / f32(ISLAND_COLS) * 6.2832 * ISLAND_R;
}

// Waves flatten approaching the waterline: horizontal displacement over the
// sloped terrain already reads as vertical motion there, and keeping the true
// height out of the softmax floor stops wave volume sinking into the sand.
// chain.js applies the same attenuation when marching for the wave edge.
fn shoreHeightScale(xz: vec2f) -> f32 {
  return mix(1.0, 0.35, smoothstep(-1.2, -0.15, terrainHeight(xz)));
}

fn simState(b: f32, col: f32) -> vec4f {
  let fx = clamp(b / (SIM_SPAN / f32(SIM_NODES - 1)), 0.0, f32(SIM_NODES - 1));
  let c = wrapCol(col);
  let i0 = i32(floor(fx));
  let i1 = min(i0 + 1, SIM_NODES - 1);
  var j0 = i32(floor(c));
  var j1 = j0 + 1;
  if (j0 >= MAIN_COLS) {
    if (j1 >= SIM_COLS) { j1 = MAIN_COLS; }
  } else {
    j1 = min(j1, MAIN_COLS - 1);
  }
  let a = fx - floor(fx);
  let fb = c - floor(c);
  return mix(
    mix(textureLoad(simTex, vec2i(i0, j0), 0), textureLoad(simTex, vec2i(i1, j0), 0), a),
    mix(textureLoad(simTex, vec2i(i0, j1), 0), textureLoad(simTex, vec2i(i1, j1), 0), a), fb);
}

// Where the chain owns the surface, fading at the alongshore edges. The
// ramp sits SEAWARD of the junction, where the film mapping extrapolates
// as identity plus the driven displacement and so nearly agrees with the
// wave mapping — ramping inside the film would mix the identity mapping
// with the rest-compressed one and fold the mesh over itself. No landward
// fade: the last node keeps its full displacement so the discarded-region
// boundary is exactly the tip polyline.
// No alongshore fade: rows beyond the simulated columns reuse the clamped
// edge column's state instead. A fade would create a zone where the film's
// world position is partially rest-compressed while the wave side still
// renders — world-anchored foam seen through that mixed mapping smears
// into streaks, and the normals kink the same way.
fn simBlend(b: f32) -> f32 {
  return smoothstep(-SIM_BAND, 0.0, b);
}
