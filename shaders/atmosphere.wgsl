// Shared by the sky dome and the ocean's reflections so both match.

fn sunWarmth(sunDir: vec3f) -> f32 {
  return 1.0 - smoothstep(0.03, 0.5, clamp(sunDir.y, 0.0, 1.0));
}

fn sunTint(sunDir: vec3f) -> vec3f {
  let w = sunWarmth(sunDir);
  return mix(vec3f(1.0, 0.97, 0.9), vec3f(1.25, 0.5, 0.18), w * w);
}

fn skyColor(dir: vec3f, sunDir: vec3f) -> vec3f {
  let w = sunWarmth(sunDir);
  let t = pow(clamp(dir.y, 0.0, 1.0), mix(0.5, 0.65, w));
  let facing = pow(0.5 + 0.5 * dot(normalize(dir.xz + vec2f(1e-5, 0.0)), normalize(sunDir.xz)), 3.0);
  let zenith = mix(vec3f(0.11, 0.30, 0.60), vec3f(0.08, 0.12, 0.30), w);
  let horizon = mix(vec3f(0.62, 0.72, 0.83), mix(vec3f(0.42, 0.36, 0.52), vec3f(1.1, 0.45, 0.16), facing), w);
  var c = mix(horizon, zenith, t);
  let g = max(dot(dir, sunDir), 0.0);
  c += sunTint(sunDir) * (pow(g, mix(40.0, 10.0, w)) * mix(0.25, 0.6, w) + pow(g, 4000.0) * 3.0);
  return c;
}
