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
  simX0: f32,
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

// Beach rising along +x (the mean wave direction), flat sea floor offshore,
// capped at a flat berm above the waterline
fn terrainHeight(xz: vec2f) -> f32 {
  return min(max(u.slope * (xz.x - u.shoreX), -u.seaDepth), 3.0);
}

// Heightless film chain state, indexed by MATERIAL position: x = horizontal
// displacement relative to the REST state (not the material grid),
// y = velocity, z = column tip x (world), w = unused.
// The junction's world x is recoverable as simX0 + simState(vec2f(simX0, z)).x
// since the first node is pinned to it.
@group(0) @binding(7) var simTex: texture_2d<f32>;

const SIM_NODES: i32 = 64;
const SIM_COLS: i32 = 256;
const SIM_SPAN: f32 = 24.0;
// Seaward width of the wave-to-film handover band (also the shore ribbon's
// seaward margin and the open-ocean grid's dive-under ramp)
const SIM_BAND: f32 = 4.0;
// Junction depth: waves hand over to the film at this isobath, and the film
// thickness runs from this value at the junction to zero at the tip, exactly
// canceling the terrain rise so the resting film is the flat sea surface
const REST_DEPTH: f32 = 0.25;

// Material -> rest world x: the chain's material band compresses onto the
// still-water wedge between the junction isobath and the static shoreline;
// identity outside the band
fn simRestX(mx: f32) -> f32 {
  let m = clamp(mx - u.simX0, 0.0, SIM_SPAN);
  return mx + m * (REST_DEPTH / u.slope / SIM_SPAN - 1.0);
}

// Waves flatten approaching the waterline: horizontal displacement over the
// sloped terrain already reads as vertical motion there, and keeping the true
// height out of the softmax floor stops wave volume sinking into the sand.
// chain.js applies the same attenuation when marching for the wave edge.
fn shoreHeightScale(xz: vec2f) -> f32 {
  return mix(1.0, 0.35, smoothstep(-1.2, -0.15, terrainHeight(xz)));
}

fn simState(xz: vec2f) -> vec4f {
  let fx = clamp((xz.x - u.simX0) / (SIM_SPAN / f32(SIM_NODES - 1)), 0.0, f32(SIM_NODES - 1));
  let fz = clamp((xz.y / (2.0 * u.foamRegion) + 0.5) * f32(SIM_COLS - 1), 0.0, f32(SIM_COLS - 1));
  let i0 = i32(floor(fx));
  let i1 = min(i0 + 1, SIM_NODES - 1);
  let j0 = i32(floor(fz));
  let j1 = min(j0 + 1, SIM_COLS - 1);
  let a = fx - floor(fx);
  let b = fz - floor(fz);
  return mix(
    mix(textureLoad(simTex, vec2i(i0, j0), 0), textureLoad(simTex, vec2i(i1, j0), 0), a),
    mix(textureLoad(simTex, vec2i(i0, j1), 0), textureLoad(simTex, vec2i(i1, j1), 0), a), b);
}

// Where the chain owns the surface, fading at the alongshore edges. The
// ramp sits SEAWARD of the junction, where the film mapping extrapolates
// as identity plus the driven displacement and so nearly agrees with the
// wave mapping — ramping inside the film would mix the identity mapping
// with the rest-compressed one and fold the mesh over itself. No landward
// fade: the last node keeps its full displacement so the discarded-region
// boundary is exactly the tip polyline.
fn simBlend(xz: vec2f) -> f32 {
  return smoothstep(u.simX0 - SIM_BAND, u.simX0, xz.x)
    * (1.0 - smoothstep(u.foamRegion - 8.0, u.foamRegion - 1.0, abs(xz.y)));
}
