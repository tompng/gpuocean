struct Uniforms {
  // xy: uv offset, z: weight
  copies: array<vec4f, 3>,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var noiseTex: texture_2d<f32>;

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
  let xy = vec2f(vec2u((vi << 1u) & 2u, vi & 2u));
  var out: VSOut;
  out.pos = vec4f(xy * 2.0 - 1.0, 0.0, 1.0);
  out.uv = vec2f(xy.x, 1.0 - xy.y);
  return out;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4f {
  var acc = vec4f(0.0);
  for (var i = 0; i < 3; i++) {
    acc += u.copies[i].z * textureSampleLevel(noiseTex, samp, in.uv + u.copies[i].xy, 0.0);
  }
  return acc;
}
