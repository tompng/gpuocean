@group(0) @binding(3) var capTex: texture_2d<f32>;
@group(0) @binding(4) var foamTex: texture_2d<f32>;
@group(0) @binding(5) var foamPatTex: texture_2d<f32>;

struct VSOut {
  @builtin(position) clip: vec4f,
  @location(0) gridXZ: vec2f,
  @location(1) world: vec3f,
  // fragments with cut > 0 are discarded: the grid mesh ends just past the
  // junction, hidden under the shore ribbon's overlap
  @location(2) cut: f32,
}

struct VSIn {
  @location(0) pos: vec2f,
  @location(1) cell: f32,
}

struct WaveSample {
  height: f32,
  disp: vec2f,
}

fn sampleWaves(xz: vec2f, cell: f32) -> WaveSample {
  var height = 0.0;
  var disp = vec2f(0.0);
  for (var i = 0; i < i32(u.numLayers); i++) {
    let l = u.layers[i];
    // Coarse cells sample a mip matching their footprint, so distant waves
    // average toward zero instead of aliasing vertex heights
    let lod = clamp(log2(max(cell * l.dirScaleAmp.z * u.hGrad, 1.0)), 0.0, 9.0);
    let s = textureSampleLevel(waveTex, samp, layerUV(xz, i), lod);
    height += l.dirScaleAmp.w * s.x;
    disp += (u.choppiness * l.dirScaleAmp.w * s.y) * l.dirScaleAmp.xy;
  }
  height *= shoreHeightScale(xz);
  // Forward displacement through a convex ramp of crest-relative height:
  // only tall crests lean (a linear ramp would shear every scale by the same
  // angle, reading as wind-carved dunes), and f' saturates to bound the
  // front-face compression
  let eta = max(height * u.ampInv, 0.0);
  disp += vec2f(u.leanX, u.leanY) * (eta * eta / (1.0 + eta) / u.ampInv);
  // Horizontal orbital displacement with shallow amplification (≈ 1/tanh(kd)),
  // fading through the waterline band
  let ty0 = terrainHeight(xz);
  let wSea = 1.0 - smoothstep(-0.6, 0.1, ty0);
  let shallowAmp = clamp(1.0 / tanh(u.waveK * max(-ty0, 0.05)), 1.0, 2.5);
  return WaveSample(height, disp * shallowAmp * wSea);
}

fn softClamp(height: f32, ty: f32) -> f32 {
  // full wave motion, but the surface never sinks below the sand (kept a
  // hair above so the wetted film stays visible)
  let dy = height - (ty + 0.1);
  return ty + 0.1 + 0.5 * (dy + sqrt(dy * dy + 0.0225));
}

// Open-ocean grid: pure scroll waves. Across the shore ribbon's seaward
// band it dives below the sand and is cut just past the junction, so the
// ribbon always covers it; at the band's seaward edge both meshes evaluate
// the same surface, so the overlap seam has matching shape and color.
@vertex
fn vs_grid(in: VSIn) -> VSOut {
  let xz = in.pos;
  let w = sampleWaves(xz, in.cell);
  let dispXZ = xz + w.disp;
  let ty = terrainHeight(dispXZ);
  // The same height ramp as the ribbon's wave side, so the two surfaces
  // agree inside the overlap band — otherwise a tall crest on the grid
  // can outrun the dive-under margin and poke through the ribbon
  var y = softClamp(w.height * (1.0 - simBlend(xz)), ty);
  y -= 1.5 * smoothstep(u.simX0 - SIM_BAND, u.simX0, xz.x);
  var out: VSOut;
  out.world = vec3f(dispXZ.x, y, dispXZ.y);
  out.gridXZ = xz;
  out.cut = xz.x - u.simX0;
  out.clip = u.viewProj * vec4f(out.world, 1.0);
  return out;
}

// Shore ribbon: covers the junction band and the film, ending exactly at
// the chain's material domain end (the waterline tip), so nothing renders
// landward of the tip. Vertex x is normalized over the ribbon's band.
@vertex
fn vs(in: VSIn) -> VSOut {
  let xz = vec2f(u.simX0 - SIM_BAND + in.pos.x * (SIM_SPAN + SIM_BAND), in.pos.y);
  let w = sampleWaves(xz, in.cell);
  let sb = simBlend(xz);
  // In the chain strip the material displacement comes from the simulated
  // nodes (rest-state compression plus the stored deviation), so foam
  // anchored to material coordinates rides the flow
  let chain = simState(xz);
  let chainDx = simRestX(xz.x) - xz.x + chain.x;
  let disp = w.disp * (1.0 - sb) + vec2f(chainDx, 0.0) * sb;
  let dispXZ = xz + disp;
  let ty = terrainHeight(dispXZ);
  // The film carries no wave height: the vertical displacement ramps out
  // across the handover band and is zero from the junction on, so water
  // never wells up out of the beach — the surge shows through horizontal
  // motion over the slope instead
  let yWave = softClamp(w.height * (1.0 - sb), ty);
  // Film thickness tapers from the junction's still-water column to zero
  // at the tip, so the junction sits exactly at sea level and at rest
  // terrain + thickness cancels to the flat sea. Seaward of the junction
  // (the blend ramp) the terrain keeps dropping while the column stays at
  // the junction value, so clamp the film's terrain at the junction's —
  // the extrapolation is then flat at sea level instead of sagging below
  let junc = u.simX0 + simState(vec2f(u.simX0, xz.y)).x;
  let tyJ = terrainHeight(vec2f(junc, dispXZ.y));
  let tyF = max(ty, tyJ);
  let tTip = clamp((xz.x - u.simX0) / SIM_SPAN, 0.0, 1.0);
  let y = mix(yWave, tyF - tyJ * (1.0 - tTip), sb);
  var out: VSOut;
  out.world = vec3f(dispXZ.x, y, dispXZ.y);
  out.gridXZ = xz;
  out.cut = -1.0;
  out.clip = u.viewProj * vec4f(out.world, 1.0);
  return out;
}

fn surfaceNormal(xz: vec2f, dist: f32, eta: f32, hScale: f32) -> vec3f {
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
    let grad = vec2f(s.z, s.w) * (u.hGrad * hScale);
    // D is the x-cumsum of h with sign and scale baked into dGrad, so ∂D/∂u = h * dGrad
    let dDdu = u.choppiness * amp * s.x * u.dGrad;
    dPx += vec3f(dir.x * dDdu * duvdx.x, amp * dot(grad, duvdx), dir.y * dDdu * duvdx.x);
    dPz += vec3f(dir.x * dDdu * duvdz.x, amp * dot(grad, duvdz), dir.y * dDdu * duvdz.x);
  }
  let leanSlope = (eta * eta + 2.0 * eta) / ((1.0 + eta) * (1.0 + eta));
  dPx += vec3f(u.leanX * leanSlope * dPx.y, 0.0, u.leanY * leanSlope * dPx.y);
  dPz += vec3f(u.leanX * leanSlope * dPz.y, 0.0, u.leanY * leanSlope * dPz.y);
  // Ripples concentrate where the long waves strain the surface: orbital
  // convergence (compression, near crests) with the peak shifted toward the
  // front face. Layers 0-2 are isotropic wind ripples (weak bias); layers 3-5
  // are anisotropic parasitic-capillary ripples following the gravity waves.
  let front = smoothstep(0.0, 0.15, -dPx.y);
  let squeeze = smoothstep(0.0, 0.3, 2.0 - dPx.x - dPz.z);
  let conc = front + squeeze;
  // Ripples are sampled in material space, and the film's material is
  // strongly compressed onto the swash zone — they would render as a dense
  // shimmer with a hard step at the junction, so fade them out with the
  // same ramp that hands the surface to the film
  let fade = clamp(1.0 - dist / 150.0, 0.0, 1.0) * (1.0 - smoothstep(u.simX0 - 2.0, u.simX0, xz.x));
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
  if (in.cut > 0.0) {
    discard;
  }
  let dist = distance(u.cameraPos, in.world);
  let sbF = simBlend(in.gridXZ);
  // Gravity-wave normal detail follows the geometry, whose height dies
  // across the handover band; sampled in the film's compressed material it
  // would otherwise keep painting shading bumps onto the flat sheet
  var n = surfaceNormal(in.gridXZ, dist, max(in.world.y * u.ampInv, 0.0), shoreHeightScale(in.gridXZ) * (1.0 - sbF));
  let ty = terrainHeight(in.world.xz);
  // The lower edge sits above the residual softmax offset left on dry sand,
  // which otherwise keeps fresnel and ripple glints alive landward of the film
  let column = max(in.world.y - ty, 0.0);
  let waterM = smoothstep(0.025, 0.09, column);
  n = normalize(mix(normalize(vec3f(-u.slope, 1.0, 0.0)), n, waterM));
  let v = normalize(u.cameraPos - in.world);
  if (dot(n, v) < 0.0) { n = -n; }
  let fresnel = 0.02 + 0.98 * pow(1.0 - max(dot(n, v), 0.0), 5.0);
  let r = reflect(-v, n);
  let spec = sunTint(u.sunDir) * (mix(8.0, 4.5, sunWarmth(u.sunDir)) * pow(max(dot(r, u.sunDir), 0.0), 600.0));
  let fuv = in.gridXZ / (2.0 * u.foamRegion) + 0.5;
  let edgeFade = 1.0 - smoothstep(0.85, 1.0, max(abs(in.gridXZ.x), abs(in.gridXZ.y)) / u.foamRegion);
  let foamAcc = textureSample(foamTex, samp, fuv).rg * edgeFade;
  // Bubble clouds scatter multiply and emerge nearly isotropic (white water);
  // a mild forward lobe remains for thin backlit crests. The film is a sheet
  // too thin to hold a submerged bubble cloud, so the glow fades out there
  // and its foam reads as surface foam only
  let towardSun = max(0.0, -dot(v, u.sunDir));
  let sss = u.sssStrength * (0.55 + 0.45 * towardSun * towardSun) * foamAcc.g * (1.0 - sbF);
  // Flat sand bottom seen through the refracted view ray with per-channel
  // Beer-Lambert extinction; +1.4 is the sun-side path per meter of column
  // (refracted solar zenith ~44°)
  let refr = refract(-v, n, 0.752);
  // The grazing-path factor assumes laterally endless water; a grazing ray
  // through the film (e.g. down a bore front) exits the thin sheet almost
  // immediately, so cap its underwater path or the film's edge renders as
  // if seen through meters of water
  let lateral = mix(1.0 / max(-refr.y, 0.05), min(1.0 / max(-refr.y, 0.05), 2.0), sbF);
  let pathLen = column * (lateral + 1.4);
  let trans = exp(-vec3f(0.25, 0.04, 0.02) * pathLen);
  // Caustic web on the sand: bright filaments along the zero-crossing lines of
  // two drifting noise fields, sampled at the refracted bottom point so the
  // pattern swims with the surface; defocus fades it with column depth
  let bottomXZ = in.world.xz + refr.xz * (column * lateral);
  let cs = textureSample(capTex, samp, bottomXZ / (13.0 * u.causticScale) + vec2f(0.023, 0.011) * u.time).x
         + textureSample(capTex, samp, bottomXZ / (8.7 * u.causticScale) + vec2f(-0.017, 0.019) * u.time).x;
  let web = pow(max(0.0, 1.0 - 0.6 * abs(cs)), 4.0);
  // Caustics need some water column to focus in; a centimeters-thin film
  // (or the residual softmax offset on dry sand) must not carry the web
  let focus = u.causticStrength * exp(-column * 0.12) * clamp(1.0 - dist / 120.0, 0.0, 1.0) * smoothstep(0.04, 0.25, column);
  let sand = vec3f(0.86, 0.78, 0.58) * (0.85 + focus * (1.6 * web - 0.18));
  let lightTint = mix(vec3f(1.0), sunTint(u.sunDir), 0.6);
  // Direct sunlight in the water column fades as the sun drops; the floor
  // stands in for diffuse sky light
  let sunLevel = mix(0.18, 1.0, smoothstep(0.0, 0.5, clamp(u.sunDir.y, 0.0, 1.0)));
  var water = mix(vec3f(0.004, 0.02, 0.05), sand, trans) * lightTint;
  water += vec3f(0.05, 0.45, 0.38) * sss;
  water *= sunLevel;
  var color = mix(water, skyColor(r, u.sunDir), fresnel) + spec;
  // Dry sand above the runup line: matte, no fresnel reflection or caustics
  let sandMatte = vec3f(0.86, 0.78, 0.58) * lightTint * sunLevel * (0.55 + 0.45 * max(n.y, 0.0));
  color = mix(sandMatte, color, waterM);
  // The foam pattern rides the water (material coords); as the accumulated
  // foam decays the threshold rises, eroding the pattern from its thin parts
  // so patches fragment into clumps before vanishing
  let pat = textureSample(foamPatTex, samp, in.gridXZ / 5.0).r;
  let foamMask = smoothstep(0.0, 0.15, pat - (1.05 - 1.15 * foamAcc.r));
  let foamColor = lightTint * mix(0.45, 1.0, sunLevel) * (0.72 + 0.22 * max(n.y, 0.0));
  color = mix(color, foamColor, foamMask);
  let fog = 1.0 - exp(-dist * 3e-5);
  color = mix(color, skyColor(normalize(vec3f(-v.x, 0.02, -v.z)), u.sunDir), fog);
  color = 1.0 - exp(-1.8 * color);
  return vec4f(pow(color, vec3f(1.0 / 2.2)), 1.0);
}

@fragment
fn fs_wire(in: VSOut) -> @location(0) vec4f {
  if (in.cut > 0.0) {
    discard;
  }
  return vec4f(0.15, 0.85, 0.5, 1.0);
}
