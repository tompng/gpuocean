# ocean

WebGPU ocean waves using scrolling noise textures (an FFT-ocean approximation),
with a shoreline swash simulation, a physically-based sky, and a submersible
camera.

## Run

Serve statically (WebGPU requires a secure context, and shaders are fetched):

```sh
python3 -m http.server 8000
# open http://localhost:8000
```

## Controls

| Input | Action |
| --- | --- |
| Left-drag | Orbit. Dragging up past level carries the camera under the surface |
| Right-drag, middle-drag, shift+left-drag | Pan |
| Wheel | Dolly |
| One finger | Orbit |
| Two fingers | Pinch dolly and pan |
| `WASD`, `shift` | Move the orbit target, faster with shift |

The camera floats off the sea floor rather than burrowing through it, and the
near plane tightens as the surface approaches the eye so the water right at the
lens still renders.

## Above and below the surface

Submersion is decided per pixel, not per frame. A point camera is either under
or over, so blending the whole frame by one scalar washes the sky into murk the
moment the eye touches the surface; instead the camera is treated as a port of
finite radius, and a ray leaving in direction `d` is submerged when
`d.y < camDepth / lensR`. That splits the screen at a waterline and saturates on
its own once the eye is clear of the surface either way. `waterlineThickness`
is that radius.

Underneath, the surface refracts properly: the sky compresses into Snell's
window (the ~97° cone set by the 48.6° critical angle) and total internal
reflection mirrors the basin outside it. The terrain mesh, which is drawn every
frame and was simply hidden under the opaque surface, shades as a caustic-lit
sea floor fading into murk.

Seen from above, the floor is refracted in screen space: the refracted ray is
marched to the bottom and the sampling offset is the screen-space parallax
between that hit and the fragment's own pixel. That is view-consistent by
construction and scales with the water column for free — a centimetres-thin
swash film barely offsets, an eight-metre column offsets a lot.

## Sky and time of day

`timeOfDay` and `latitude` set the sun's elevation, `azimuth` its bearing. The
sky is the Preetham analytic model (Perez distribution over luminance and both
chromaticities), normalised by 30 kcd/m² so a clear zenith lands where the
tonemap expects it; the reference is absolute, so a low sun really is dimmer
than a high one. `rayleigh` folds in as a chromaticity gain about D65 — 1 is
exact Preetham, 0 a neutral grey sky — which leaves luminance untouched.

The moon rides the sun's path half a day out of phase on the opposite bearing,
so it rises as the sun sets. Once the sun is down it becomes the key light,
cool-tinted and gated on actually being above the horizon, so a moonless night
goes dark rather than resting on a floor. The water carries a separate glitter
lobe for it.

## Foam

Foam coverage accumulates from wave-field folding offshore and from swash
compression at the shoreline. Coverage drives an erosion threshold that sweeps
down through a density plate, so patches fragment from their thin parts first
and dissolve rather than fading uniformly.

Three photographic plates span the coverage ramp — sparse lace, mid sheets with
flow streaks, dense bubble raft. Their densities are blended and the result is
eroded **once**: eroding each and cross-fading the masks would leave half-grey
ghost foam wherever two overlap, because the masks are near-binary. The `plate`
control forces a single plate for inspection, with `procedural` kept as an A/B
against the pattern the plates replaced.

## Parameters

The panel is generated from a spec in `src/ui.js`, grouped into sections that
collapse independently (state persists). Groups tagged **(s)** act on the
above-water view, **(uw)** on the submerged view, and **(b)** on both.
`preset` switches between sea, river, lake and lagoon, which change the water's
appearance rather than the coastline geometry.

## Quality

`quality` scales the near-field **cell size**, not the vertex count directly.
Scaling `gridN` alone is the wrong lever: the warp grows cells exponentially
past `linearCells`, so halving it does not coarsen the ocean, it shrinks it from
177 km to a 51 m puddle. Each level pairs a cell size with the `gridN` and
`linearCells` that hold the extent near 175 km, so only detail changes.

The panel reports per-pass GPU milliseconds beside the frame rate, measured with
timestamp queries. Wall-clock frame timing only measures how long the browser
waited for vsync — below the refresh interval every frame reports the same
number, and above it the value snaps to multiples of it.

## Structure

- `src/noise.js` — CPU-generated gravity-wave and capillary noise textures, the
  procedural foam density plate, and the photographic foam plate loader
- `src/waveField.js` — per-frame blend of scrolled copies (dispersion
  approximation) + mipmaps
- `src/ocean.js`, `shaders/ocean.wgsl` — layered directional sampling, vertex
  displacement, surface and sea-floor shading, screen-space refraction
- `shaders/atmosphere.wgsl` — Preetham sky, moon, and the water volume model,
  shared by the sky pass and the ocean's reflections so both agree
- `src/chain.js`, `shaders/filmfoam.wgsl` — shoreline swash chain and its foam
- `src/foam.js`, `shaders/foam.wgsl` — world foam accumulation buffer
- `src/camera.js` — orbit camera, including the submersible path
- `src/ui.js` — parameter spec, presets, quality levels, panel generation
- `src/gputimer.js` — per-pass GPU timing via timestamp queries

### Uniform layout

`src/ocean.js` writes its uniform buffer through a named layout derived from the
same declaration order as `Uniforms` in `shaders/wave_common.wgsl`, applying
WGSL's alignment rules. The two are checked against each other at startup: a
field added to one side and not the other throws with both names rather than
silently shifting every subsequent offset.
