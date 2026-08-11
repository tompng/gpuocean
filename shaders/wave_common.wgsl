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
