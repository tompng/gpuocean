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
  patchSize: f32,
  numLayers: f32,
  choppiness: f32,
  dGrad: f32,
  hGrad: f32,
  gridN: f32,
  pad0: f32,
  pad1: f32,
  pad2: f32,
  layers: array<Layer, 8>,
  capLayers: array<Layer, 3>,
  capHGrad: f32,
  rippleBias: f32,
  capPad0: f32,
  capPad1: f32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var waveTex: texture_2d<f32>;
@group(0) @binding(3) var capTex: texture_2d<f32>;

struct VSOut {
  @builtin(position) clip: vec4f,
  @location(0) gridXZ: vec2f,
  @location(1) world: vec3f,
}

fn layerUV(xz: vec2f, i: i32) -> vec2f {
  let l = u.layers[i];
  let dir = l.dirScaleAmp.xy;
  return vec2f(dot(xz, dir), dot(xz, vec2f(-dir.y, dir.x))) * l.dirScaleAmp.z + l.scroll.xy;
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
  let cols = u32(u.gridN) + 1u;
  let ix = vi % cols;
  let iz = vi / cols;
  let xz = (vec2f(f32(ix), f32(iz)) / u.gridN - 0.5) * u.patchSize;
  var height = 0.0;
  var disp = vec2f(0.0);
  for (var i = 0; i < i32(u.numLayers); i++) {
    let l = u.layers[i];
    let s = textureSampleLevel(waveTex, samp, layerUV(xz, i), 0.0);
    height += l.dirScaleAmp.w * s.x;
    disp += (u.choppiness * l.dirScaleAmp.w * s.y) * l.dirScaleAmp.xy;
  }
  var out: VSOut;
  out.world = vec3f(xz.x + disp.x, height, xz.y + disp.y);
  out.gridXZ = xz;
  out.clip = u.viewProj * vec4f(out.world, 1.0);
  return out;
}

fn surfaceNormal(xz: vec2f, dist: f32) -> vec3f {
  var dPx = vec3f(1.0, 0.0, 0.0);
  var dPz = vec3f(0.0, 0.0, 1.0);
  for (var i = 0; i < i32(u.numLayers); i++) {
    let l = u.layers[i];
    let dir = l.dirScaleAmp.xy;
    let invL = l.dirScaleAmp.z;
    let amp = l.dirScaleAmp.w;
    let s = textureSample(waveTex, samp, layerUV(xz, i));
    let duvdx = vec2f(dir.x, -dir.y) * invL;
    let duvdz = vec2f(dir.y, dir.x) * invL;
    let grad = vec2f(s.z, s.w) * u.hGrad;
    // D is the x-cumsum of h with sign and scale baked into dGrad, so ∂D/∂u = h * dGrad
    let dDdu = u.choppiness * amp * s.x * u.dGrad;
    dPx += vec3f(dir.x * dDdu * duvdx.x, amp * dot(grad, duvdx), dir.y * dDdu * duvdx.x);
    dPz += vec3f(dir.x * dDdu * duvdz.x, amp * dot(grad, duvdz), dir.y * dDdu * duvdz.x);
  }
  // Parasitic capillaries concentrate on the front face of gravity waves,
  // where the slope along the travel direction (+x) is negative
  let front = smoothstep(0.0, 0.15, -dPx.y);
  let capScale = mix(1.0, 2.0 * front, u.rippleBias) * clamp(1.0 - dist / 150.0, 0.0, 1.0);
  for (var i = 0; i < 3; i++) {
    let l = u.capLayers[i];
    let dir = l.dirScaleAmp.xy;
    let invL = l.dirScaleAmp.z;
    let amp = capScale * l.dirScaleAmp.w;
    let uvc = vec2f(dot(xz, dir), dot(xz, vec2f(-dir.y, dir.x))) * invL + l.scroll.xy;
    let s = textureSample(capTex, samp, uvc);
    let grad = vec2f(s.z, s.w) * u.capHGrad;
    dPx.y += amp * dot(grad, vec2f(dir.x, -dir.y) * invL);
    dPz.y += amp * dot(grad, vec2f(dir.y, dir.x) * invL);
  }
  return normalize(cross(dPz, dPx));
}

fn skyColor(dir: vec3f) -> vec3f {
  let t = pow(clamp(dir.y, 0.0, 1.0), 0.5);
  var c = mix(vec3f(0.65, 0.75, 0.85), vec3f(0.12, 0.32, 0.6), t);
  c += vec3f(1.0, 0.9, 0.7) * (0.3 * pow(max(dot(dir, u.sunDir), 0.0), 30.0));
  return c;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4f {
  var n = surfaceNormal(in.gridXZ, distance(u.cameraPos, in.world));
  let v = normalize(u.cameraPos - in.world);
  if (dot(n, v) < 0.0) { n = -n; }
  let fresnel = 0.02 + 0.98 * pow(1.0 - max(dot(n, v), 0.0), 5.0);
  let r = reflect(-v, n);
  let spec = vec3f(8.0, 7.5, 6.5) * pow(max(dot(r, u.sunDir), 0.0), 600.0);
  var color = mix(vec3f(0.02, 0.08, 0.10), skyColor(r), fresnel) + spec;
  color = 1.0 - exp(-1.8 * color);
  return vec4f(pow(color, vec3f(1.0 / 2.2)), 1.0);
}

@fragment
fn fs_wire(in: VSOut) -> @location(0) vec4f {
  return vec4f(0.15, 0.85, 0.5, 1.0);
}
