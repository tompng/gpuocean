// Sun visibility at a world point, in [0, 1]. The open sea has nothing to
// cast a shadow, so this returns 1 and the multiplications below fold away;
// a scene that embeds this renderer replaces the whole file with its own
// implementation (a shadow map projection, a cloud cover lookup, ...).
//
// n is the receiver's surface normal, a hint for implementations that need
// to bias a depth-map lookup off the surface. The occlusion itself is a
// function of position alone.
fn sunOcclusion(wp: vec3f, n: vec3f) -> f32 {
  return 1.0;
}
