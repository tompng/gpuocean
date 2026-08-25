// Sun visibility at a world point, in [0, 1]. The open sea has nothing to
// cast a shadow, so these return 1 and the multiplications below fold away;
// a scene that embeds this renderer replaces the whole file with its own
// implementation (a shadow map projection, a cloud cover lookup, ...).
//
// n is the receiver's surface normal, a hint for implementations that need
// to bias a depth-map lookup off the surface. The occlusion itself is a
// function of position alone.

// The water needs two answers, not one. What lights the surface (sun glint,
// foam) is the sun arriving at the surface; what lights the water column and
// the sand under it is the sun arriving at the SEABED. Collapsing them to a
// single lookup at the surface paints the shadow of anything floating there
// through the whole column, so a bird sitting on the water darkens the water
// around it as if it were a stain rather than a shadow.
struct SunVis {
  surface: f32,
  bed: f32,
}

fn sunOcclusion(wp: vec3f, n: vec3f) -> f32 {
  return 1.0;
}

fn sunOcclusionPair(surf: vec3f, bed: vec3f, n: vec3f) -> SunVis {
  return SunVis(1.0, 1.0);
}
