// Film foam accumulation in the film's own (band, column) material space,
// so it works identically for the open mainland segment and the island
// loop. One texel row per chain column; x spans the ribbon band. The foam
// source is the film's compression relative to rest (see the ocean shader);
// channels mirror the world foam buffer: R decayed, G glow, B/A fresh.
@group(0) @binding(3) var prevFoam: texture_2d<f32>;

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
  let b = in.uv.x * (SIM_BAND + SIM_SPAN) - SIM_BAND;
  let col = in.uv.y * f32(SIM_COLS) - 0.5;
  let sim = simState(b, col);
  let e = 0.8;
  let restScale = REST_DEPTH / u.slope / SIM_SPAN;
  let compress = (simState(b - e, col).x - simState(b + e, col).x) / (2.0 * e * restScale);
  let sNow = simRestS(b) + sim.x;
  let sb = simBlend(b, col);
  let inFilm = sb * (1.0 - smoothstep(sim.z - 0.3, sim.z + 0.1, sNow));
  let gen = inFilm * smoothstep(0.25, 0.7, compress);
  // Foam on the beach face is swallowed where the waves flood over it again
  let sJ = -REST_DEPTH / u.slope + simState(0.0, col).x;
  let tyM = u.slope * simRestS(b);
  let swallowed = sb * smoothstep(0.3, -0.7, sNow - sJ) * smoothstep(-1.2, -0.3, tyM);
  let decayR = mix(u.foamDecay, u.foamDecaySwallow, swallowed);
  let prev = textureSampleLevel(prevFoam, samp, in.uv, 0.0);
  let smoothR = mix(gen, prev.b, u.foamRise);
  let smoothG = mix(gen, prev.a, u.foamRise);
  return vec4f(max(prev.r * decayR, smoothR), max(prev.g * u.foamDecayG, smoothG), smoothR, smoothG);
}
