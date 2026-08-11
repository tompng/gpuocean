@group(0) @binding(3) var capTex: texture_2d<f32>;
@group(0) @binding(4) var foamTex: texture_2d<f32>;
@group(0) @binding(5) var foamPatTex: texture_2d<f32>;

struct VSOut {
  @builtin(position) clip: vec4f,
  @location(0) gridXZ: vec2f,
  @location(1) world: vec3f,
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
  // Forward displacement through a convex ramp of crest-relative height:
  // only tall crests lean (a linear ramp would shear every scale by the same
  // angle, reading as wind-carved dunes), and f' saturates to bound the
  // front-face compression
  let eta = max(height * u.ampInv, 0.0);
  disp += vec2f(u.leanX, u.leanY) * (eta * eta / (1.0 + eta) / u.ampInv);
  var out: VSOut;
  out.world = vec3f(xz.x + disp.x, height, xz.y + disp.y);
  out.gridXZ = xz;
  out.clip = u.viewProj * vec4f(out.world, 1.0);
  return out;
}

fn surfaceNormal(xz: vec2f, dist: f32, eta: f32) -> vec3f {
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
  var n = surfaceNormal(in.gridXZ, dist, max(in.world.y * u.ampInv, 0.0));
  let v = normalize(u.cameraPos - in.world);
  if (dot(n, v) < 0.0) { n = -n; }
  let fresnel = 0.02 + 0.98 * pow(1.0 - max(dot(n, v), 0.0), 5.0);
  let r = reflect(-v, n);
  let spec = sunTint(u.sunDir) * (mix(8.0, 4.5, sunWarmth(u.sunDir)) * pow(max(dot(r, u.sunDir), 0.0), 600.0));
  let fuv = in.gridXZ / (2.0 * u.foamRegion) + 0.5;
  let edgeFade = 1.0 - smoothstep(0.85, 1.0, max(abs(in.gridXZ.x), abs(in.gridXZ.y)) / u.foamRegion);
  let foamAcc = textureSample(foamTex, samp, fuv).rg * edgeFade;
  // Bubble clouds scatter multiply and emerge nearly isotropic (white water);
  // a mild forward lobe remains for thin backlit crests
  let towardSun = max(0.0, -dot(v, u.sunDir));
  let sss = u.sssStrength * (0.55 + 0.45 * towardSun * towardSun) * foamAcc.g;
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
  // Direct sunlight in the water column fades as the sun drops; the floor
  // stands in for diffuse sky light
  let sunLevel = mix(0.18, 1.0, smoothstep(0.0, 0.5, clamp(u.sunDir.y, 0.0, 1.0)));
  var water = mix(vec3f(0.004, 0.02, 0.05), sand, trans) * lightTint;
  water += vec3f(0.05, 0.45, 0.38) * sss;
  water *= sunLevel;
  var color = mix(water, skyColor(r, u.sunDir), fresnel) + spec;
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
  return vec4f(0.15, 0.85, 0.5, 1.0);
}
