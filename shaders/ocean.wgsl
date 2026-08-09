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
  depthPad0: f32,
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

struct VSIn {
  @location(0) pos: vec2f,
  @location(1) cell: f32,
}

@vertex
fn vs(in: VSIn) -> VSOut {
  let xz = in.pos;
  var height = 0.0;
  var disp = vec2f(0.0);
  for (var i = 0; i < i32(u.numLayers); i++) {
    let l = u.layers[i];
    // Coarse cells sample a mip matching their footprint, so distant waves
    // average toward zero instead of aliasing vertex heights
    let lod = clamp(log2(max(in.cell * l.dirScaleAmp.z * u.hGrad, 1.0)), 0.0, 9.0);
    let s = textureSampleLevel(waveTex, samp, layerUV(xz, i), lod);
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
    let uvc = vec2f(dot(xz, dir), dot(xz, vec2f(-dir.y, dir.x))) * invL + l.scroll.xy;
    var s: vec4f;
    var amp: f32;
    if (i < 3) {
      s = textureSample(capTex, samp, uvc);
      amp = isoScale * l.dirScaleAmp.w * u.capHGrad;
    } else {
      s = textureSample(waveTex, samp, uvc);
      amp = anisoScale * l.dirScaleAmp.w * u.hGrad;
    }
    let grad = vec2f(s.z, s.w) * amp;
    dPx.y += dot(grad, vec2f(dir.x, -dir.y) * invL);
    dPz.y += dot(grad, vec2f(dir.y, dir.x) * invL);
  }
  return normalize(cross(dPz, dPx));
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4f {
  let dist = distance(u.cameraPos, in.world);
  var n = surfaceNormal(in.gridXZ, dist);
  let v = normalize(u.cameraPos - in.world);
  if (dot(n, v) < 0.0) { n = -n; }
  let fresnel = 0.02 + 0.98 * pow(1.0 - max(dot(n, v), 0.0), 5.0);
  let r = reflect(-v, n);
  let spec = sunTint(u.sunDir) * (mix(8.0, 4.5, sunWarmth(u.sunDir)) * pow(max(dot(r, u.sunDir), 0.0), 600.0));
  // Sunlight transmitted through thin wave crests toward a viewer facing the sun
  let towardSun = max(0.0, -dot(v, u.sunDir));
  let thin = max(0.0, in.world.y * u.ampInv);
  let sss = u.sssStrength * pow(towardSun, 3.0) * thin * thin;
  // Flat sand bottom seen through the refracted view ray with per-channel
  // Beer-Lambert extinction; +1.4 is the sun-side path per meter of column
  // (refracted solar zenith ~44°)
  let refr = refract(-v, n, 0.752);
  let column = max(in.world.y + u.seaDepth, 0.0);
  let pathLen = column * (1.0 / max(-refr.y, 0.05) + 1.4);
  let trans = exp(-vec3f(0.25, 0.04, 0.02) * pathLen);
  // Caustic web on the sand: bright filaments along the zero-crossing lines of
  // two drifting noise fields, sampled at the refracted bottom point so the
  // pattern swims with the surface; defocus fades it with column depth
  let bottomXZ = in.world.xz + refr.xz * (column / max(-refr.y, 0.05));
  let cs = textureSample(capTex, samp, bottomXZ / (13.0 * u.causticScale) + vec2f(0.023, 0.011) * u.time).x
         + textureSample(capTex, samp, bottomXZ / (8.7 * u.causticScale) + vec2f(-0.017, 0.019) * u.time).x;
  let web = pow(max(0.0, 1.0 - 0.6 * abs(cs)), 4.0);
  let focus = u.causticStrength * exp(-column * 0.12) * clamp(1.0 - dist / 120.0, 0.0, 1.0);
  let sand = vec3f(0.86, 0.78, 0.58) * (0.85 + focus * (1.6 * web - 0.18));
  let lightTint = mix(vec3f(1.0), sunTint(u.sunDir), 0.6);
  var water = mix(vec3f(0.02, 0.08, 0.10), sand, trans) * lightTint;
  water += vec3f(0.05, 0.45, 0.38) * sss;
  var color = mix(water, skyColor(r, u.sunDir), fresnel) + spec;
  let fog = 1.0 - exp(-dist * 3e-5);
  color = mix(color, skyColor(normalize(vec3f(-v.x, 0.02, -v.z)), u.sunDir), fog);
  color = 1.0 - exp(-1.8 * color);
  return vec4f(pow(color, vec3f(1.0 / 2.2)), 1.0);
}

@fragment
fn fs_wire(in: VSOut) -> @location(0) vec4f {
  return vec4f(0.15, 0.85, 0.5, 1.0);
}
