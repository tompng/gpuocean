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
  islandArcStep: f32,
  numLayers: f32,
  choppiness: f32,
  dGrad: f32,
  hGrad: f32,
  // world foam window: center follows the camera (snapped to texels);
  // delta is the center's move since last frame in uv units
  foamCX: f32,
  foamCZ: f32,
  foamDX: f32,
  foamDZ: f32,
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
  foamLife: f32,
  slope: f32,
  foamDecaySwallow: f32,
  simDt: f32,
  waveK: f32,
  padC: f32,
  foamScale: f32,
  // mainland film window: center z (follows the camera in whole-column
  // steps) and this frame's shift in film-foam buffer rows
  simZBase: f32,
  simZShift: f32,
  // camera's coast arclength, snapped to the ribbon row pitch
  simTCam: f32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var waveTex: texture_2d<f32>;

fn layerUV(xz: vec2f, i: i32) -> vec2f {
  let l = u.layers[i];
  let dir = l.dirScaleAmp.xy;
  return vec2f(dot(xz, dir), dot(xz, vec2f(-dir.y, dir.x))) * l.dirScaleAmp.z + l.scroll.xy;
}

// The coastline is authored data (src/coast.js) baked at load into a
// signed-distance texture (negative in the sea) and arclength tables.
// Beyond the baked region the coast is a straight line at BASE_SHORE_X.
@group(0) @binding(9) var sdfTex: texture_2d<f32>;

const SDF_EXTENT: f32 = 384.0;
const BASE_SHORE_X: f32 = 10.0;

fn coastSDF(xz: vec2f) -> f32 {
  let baked = textureSampleLevel(sdfTex, samp, xz / (2.0 * SDF_EXTENT) + 0.5, 0.0).r;
  let far = smoothstep(SDF_EXTENT - 48.0, SDF_EXTENT - 8.0, max(abs(xz.x), abs(xz.y)));
  return mix(baked, xz.x - BASE_SHORE_X, far);
}

// Beach rising along the landward normal, flat sea floor offshore, capped
// at a flat berm above the waterline
fn terrainHeight(xz: vec2f) -> f32 {
  return min(max(u.slope * coastSDF(xz), -u.seaDepth), 3.0);
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
    return u.simZBase + (col / f32(MAIN_COLS - 1) - 0.5) * 160.0;
  }
  return (col - f32(MAIN_COLS)) * u.islandArcStep;
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
