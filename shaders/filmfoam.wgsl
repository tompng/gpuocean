// Film foam accumulation in the film's own (band, column) material space,
// so it works identically for the open mainland segment and the island
// loop. One texel row per chain column; x spans the ribbon band. The foam
// source is the film's compression relative to rest (see the ocean shader);
// channels mirror the world foam buffer: R decayed, G glow, B/A fresh.
@group(0) @binding(3) var prevFoam: texture_2d<f32>;

// Band metres of generating shore belt at shoreWidth = 1
const SHORE_BELT: f32 = 14.0;

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
  let sb = simBlend(b);
  // lapOvershoot lets a receding lap keep depositing past the instantaneous
  // swash tip, so its lace strands at the high-water mark instead of being cut
  // off exactly at the moving edge — that stationary tide line is what the
  // sparse plate is for
  let tip = sim.z + u.lapOvershoot;
  let inFilm = sb * (1.0 - smoothstep(tip - 0.3, tip + 0.1, sNow));
  // shoreWidth is the cross-shore extent of the generating belt, rolled off
  // over its landward 40% so the upper beach face stays clean
  let belt = 1.0 - smoothstep(0.6 * SHORE_BELT * u.shoreWidth, SHORE_BELT * u.shoreWidth, b);
  // surgeRate is the gain from swash convergence to foam birth
  let gen = inFilm * belt * smoothstep(0.25, 0.7, compress * u.surgeRate);
  // Foam on the beach face is swallowed where the waves flood over it again
  let sJ = -REST_DEPTH / u.slope + simState(0.0, col).x;
  let tyM = u.slope * simRestS(b);
  let swallowed = sb * smoothstep(0.3, -0.7, sNow - sJ) * smoothstep(-1.2, -0.3, tyM);
  let decayR = mix(u.foamDecay, u.foamDecaySwallow, swallowed);
  // the mainland rows scroll with the film window; rows shifted in from
  // outside the segment have no history (island rows never move)
  var prevUV = in.uv;
  if (col < f32(MAIN_COLS)) {
    prevUV.y += u.simZShift / f32(SIM_COLS);
  }
  var prev = textureSampleLevel(prevFoam, samp, prevUV, 0.0);
  let pCol = prevUV.y * f32(SIM_COLS) - 0.5;
  if (col < f32(MAIN_COLS) && (pCol < -0.5 || pCol >= f32(MAIN_COLS) - 0.5)) {
    prev = vec4f(0.0);
  }
  let smoothR = mix(gen, prev.b, u.foamRise);
  let smoothG = mix(gen, prev.a, u.foamRise);
  return vec4f(max(prev.r * decayR, smoothR), max(prev.g * u.foamDecayG, smoothG), smoothR, smoothG);
}
