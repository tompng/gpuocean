# ocean

WebGPU ocean waves using scrolling noise textures (an FFT-ocean approximation).

## Run

Serve statically (WebGPU requires a secure context, and shaders are fetched):

```sh
python3 -m http.server 8000
# open http://localhost:8000
```

## Structure

- `src/noise.js` — CPU-generated gravity-wave noise texture (height, horizontal displacement, gradients)
- `src/waveField.js` — per-frame blend of scrolled copies (dispersion approximation) + mipmaps
- `src/ocean.js`, `shaders/ocean.wgsl` — layered directional sampling, vertex displacement, shading
