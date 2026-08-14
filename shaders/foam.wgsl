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

// Horizontal-map Jacobian and height of the displaced surface, the Jacobian
// terms matching the ocean shader's surfaceNormal
fn waveSurface(xz: vec2f) -> vec2f {
  var dxx = 1.0;
  var dxz = 0.0;
  var dzx = 0.0;
  var dzz = 1.0;
  var height = 0.0;
  var gradH = vec2f(0.0);
  for (var i = 0; i < i32(u.numLayers); i++) {
    let l = u.layers[i];
    let dir = l.dirScaleAmp.xy;
    let invL = l.dirScaleAmp.z;
    let amp = l.dirScaleAmp.w;
    let s = textureSampleLevel(waveTex, samp, layerUV(xz, i), 0.0);
    let duvdx = vec2f(dir.x, -dir.y) * invL;
    let duvdz = vec2f(dir.y, dir.x) * invL;
    let grad = vec2f(s.z, s.w) * u.hGrad;
    let dDdu = u.choppiness * amp * s.x * u.dGrad;
    height += amp * s.x;
    gradH += amp * vec2f(dot(grad, duvdx), dot(grad, duvdz));
    dxx += dir.x * dDdu * duvdx.x;
    dxz += dir.y * dDdu * duvdx.x;
    dzx += dir.x * dDdu * duvdz.x;
    dzz += dir.y * dDdu * duvdz.x;
  }
  let hs = shoreHeightScale(xz);
  height *= hs;
  gradH *= hs;
  let eta = max(height * u.ampInv, 0.0);
  let ls = (eta * eta + 2.0 * eta) / ((1.0 + eta) * (1.0 + eta));
  dxx += u.leanX * ls * gradH.x;
  dxz += u.leanY * ls * gradH.x;
  dzx += u.leanX * ls * gradH.y;
  dzz += u.leanY * ls * gradH.y;
  return vec2f(dxx * dzz - dxz * dzx, height);
}

// R: surface foam (long-lived). G: entrained bubbles just under fresh
// breakers (stricter threshold, short-lived) driving the scattering glow.
// B/A: one-pole smoothed generation feeding R/G, so the breaking front
// rises over ~0.1s instead of popping in frame- and texel-sized steps.
@fragment
fn fs(in: VSOut) -> @location(0) vec4f {
  let xz = (in.uv - 0.5) * (2.0 * u.foamRegion);
  let s = waveSurface(xz);
  let jac = s.x;
  let ty = terrainHeight(xz);
  let sb = simBlend(xz);
  // The ghost wave field extends over dry land; only actually submerged
  // spots may generate breaking foam
  let waterGate = smoothstep(0.0, 0.3, s.y - ty);
  let dNow = max(-ty, 0.05);
  let genSurf = smoothstep(0.55, 0.9, s.y / dNow) * smoothstep(0.0, 0.5, dNow) * waterGate;
  // Film chain: the foam source is the film's compression relative to
  // rest — packed material at a bore front or piled against the incoming
  // wave. The same quantity the surface shader once computed per fragment
  // now accumulates here, so the buffer records exactly what the front
  // generates and the fresh channel doubles as the rendered instant foam.
  // No advection needed — foam is material-anchored and rides the mesh.
  let sim = simState(xz);
  let e = 0.8;
  let restScale = REST_DEPTH / u.slope / SIM_SPAN;
  let compress = (simState(xz - vec2f(e, 0.0)).x - simState(xz + vec2f(e, 0.0)).x) / (2.0 * e * restScale);
  let worldX = simRestX(xz.x) + sim.x;
  let inFilm = sb * (1.0 - smoothstep(sim.z - 0.3, sim.z + 0.1, worldX));
  let genSim = inFilm * smoothstep(0.25, 0.7, compress);
  let genR = max(max(smoothstep(u.foamThreshold, u.foamThreshold - 0.25, jac) * waterGate, genSurf), genSim);
  let genG = max(smoothstep(u.foamThreshold - 0.15, u.foamThreshold - 0.45, jac) * waterGate, max(genSurf, genSim));
  // Foam on the beach face is swallowed where the waves flood over it again
  let junc = u.simX0 + simState(vec2f(u.simX0, xz.y)).x;
  let swallowed = sb * smoothstep(0.3, -0.7, worldX - junc) * smoothstep(-1.2, -0.3, ty);
  let decayR = mix(u.foamDecay, u.foamDecaySwallow, swallowed);
  let prev = textureSampleLevel(prevFoam, samp, in.uv, 0.0);
  let smoothR = mix(genR, prev.b, u.foamRise);
  let smoothG = mix(genG, prev.a, u.foamRise);
  return vec4f(max(prev.r * decayR, smoothR), max(prev.g * u.foamDecayG, smoothG), smoothR, smoothG);
}
