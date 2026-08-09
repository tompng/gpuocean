struct Uniforms {
  invViewProj: mat4x4f,
  cameraPos: vec3f,
  pad0: f32,
  sunDir: vec3f,
  pad1: f32,
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
  var c = skyColor(dir, u.sunDir);
  c = 1.0 - exp(-1.8 * c);
  return vec4f(pow(c, vec3f(1.0 / 2.2)), 1.0);
}
