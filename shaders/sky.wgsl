struct Uniforms {
  invViewProj: mat4x4f,
  cameraPos: vec3f,
  // Camera port radius: the width of the waterline split
  lensR: f32,
  sunDir: vec3f,
  // Meters the camera sits below the local water surface; negative in air
  camDepth: f32,
  turbidity: f32,
  chlorophyll: f32,
  skyTurbidity: f32,
  skyRayleigh: f32,
  skyIntensity: f32,
  // vec3f aligns to 16 B, so this starts at 128 and the struct ends at 140,
  // rounding to the 144 the buffer in sky.js is sized to
  moonDir: vec3f,
}

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) ndc: vec2f,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
  let xy = vec2f(vec2u((vi << 1u) & 2u, vi & 2u)) * 2.0 - 1.0;
  var out: VSOut;
  out.pos = vec4f(xy, 0.0, 1.0);
  out.ndc = xy;
  return out;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4f {
  let far = u.invViewProj * vec4f(in.ndc, 1.0, 1.0);
  let dir = normalize(far.xyz / far.w - u.cameraPos);
  let SKY = skyState(u.sunDir, u.moonDir, u.skyTurbidity, u.skyRayleigh, u.skyIntensity);
  var c = skyColor(dir, SKY);
  // Submerged there is no sky to see: this pass only fills what the surface
  // and floor meshes leave open, and what fills it is the murk, brightening
  // toward the daylight above
  let volume = waterFogAlong(dir, u.camDepth, SKY, u.turbidity, u.chlorophyll);
  // Per-pixel, so a camera at the surface keeps real sky above the waterline
  // instead of fogging the whole frame
  c = mix(c, volume, underwaterAt(dir, u.camDepth, u.lensR));
  c = 1.0 - exp(-1.8 * c);
  return vec4f(pow(c, vec3f(1.0 / 2.2)), 1.0);
}
