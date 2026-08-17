import { initWebGPU, fetchText } from './gpu.js'
import { generateGravityNoiseSet, generateCapillaryNoiseTexture, generateFoamPatternTexture } from './noise.js'
import { WaveField } from './waveField.js'
import { Ocean } from './ocean.js'
import { FoamSim } from './foam.js'
import { ChainSim, sampleWaveLevel, sampleWaveHeight } from './chain.js'
import { buildCoast } from './coast.js'
import { Sky } from './sky.js'
import { OrbitCamera } from './camera.js'
import { invert } from './mat4.js'
import { setupUI, setupFPS, QUALITY } from './ui.js'
import { setupNoiseDebug } from './debug.js'

const canvas = document.getElementById('canvas')
const GRAVITY = 9.81
const CAPILLARY_SIGMA_RHO = 7.4e-5
const CAP_DISPERSION = 1.5

// Solar elevation from clock time and latitude, at equinox declination: the
// hour angle runs 15 deg per hour either side of local noon. Azimuth stays a
// direct compass bearing so the sun can be aimed independently of the clock.
function sunDirection(timeOfDay, latitude, azimuth) {
  const hourAngle = (timeOfDay - 12) * 15 * Math.PI / 180
  const lat = latitude * Math.PI / 180
  const elevation = Math.asin(Math.cos(lat) * Math.cos(hourAngle))
  const bearing = azimuth * Math.PI / 180
  const ground = Math.cos(elevation)
  return [ground * Math.sin(bearing), Math.sin(elevation), ground * Math.cos(bearing)]
}

async function main() {
  const { device, context, format } = await initWebGPU(canvas)
  const [waveFieldCode, waveCommonCode, oceanCode, atmosphereCode, skyCode, foamCode, filmFoamCode] = await Promise.all(
    ['wave_field', 'wave_common', 'ocean', 'atmosphere', 'sky', 'foam', 'filmfoam'].map(name => fetchText(new URL(`../shaders/${name}.wgsl`, import.meta.url)))
  )
  const noise = generateGravityNoiseSet(device)
  const capNoise = generateCapillaryNoiseTexture(device)
  const waveField = new WaveField(device, waveFieldCode, noise)
  const capField = new WaveField(device, waveFieldCode, capNoise)
  const foamPattern = generateFoamPatternTexture(device)
  const foam = new FoamSim(device, waveCommonCode + foamCode)
  const coast = buildCoast(device)
  const chain = new ChainSim(device, coast)
  const filmFoam = new FoamSim(device, waveCommonCode + filmFoamCode, [128, 256])
  const ocean = new Ocean(device, atmosphereCode + waveCommonCode + oceanCode, waveField.texture, capField.texture, foam.views, filmFoam.views, foamPattern, chain.view, chain.coastView, coast.sdfView, coast.mainTableView, format)
  foam.bind(ocean.uniform, waveField.texture, null, coast.sdfView)
  filmFoam.bind(ocean.uniform, null, chain.view, null)
  ocean.chain = chain
  const sky = new Sky(device, atmosphereCode + skyCode, format)
  const camera = new OrbitCamera(canvas)
  const params = setupUI()
  const reportFPS = setupFPS()
  camera.floor = (x, z) => terrainHeightAt(x, z, params.depth)
  setupNoiseDebug([
    ...noise.variants.map(v => ({ name: v.name, size: noise.size, channels: v.channels })),
    { name: 'capillary', size: capNoise.size, channels: capNoise.channels },
    { name: 'foamPattern', size: foamPattern.size, channels: foamPattern.channels },
  ])

  let depth = null
  let msaa = null
  let width = 0
  let height = 0
  let lastTime = performance.now()

  function frame(now) {
    const dt = Math.min((now - lastTime) / 1000, 0.1)
    lastTime = now
    reportFPS(dt)

    const w = Math.max(1, Math.floor(canvas.clientWidth * devicePixelRatio))
    const h = Math.max(1, Math.floor(canvas.clientHeight * devicePixelRatio))
    if (w !== width || h !== height) {
      width = w
      height = h
      canvas.width = w
      canvas.height = h
      depth?.destroy()
      msaa?.destroy()
      depth = device.createTexture({
        size: [w, h],
        format: 'depth24plus',
        sampleCount: 4,
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      })
      msaa = device.createTexture({
        size: [w, h],
        format,
        sampleCount: 4,
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      })
    }

    const waveDt = params.pause ? 0 : dt
    const lambda = params.wavelength
    waveField.update(waveDt, Math.sqrt(GRAVITY * lambda / (2 * Math.PI)) / (lambda * noise.wavesPerTile), params.dispersion)
    const capK = 2 * Math.PI / params.rippleScale
    const capSpeed = Math.sqrt(GRAVITY / capK + CAPILLARY_SIGMA_RHO * capK)
    capField.update(waveDt, capSpeed / (params.rippleScale * capNoise.wavesPerTile), CAP_DISPERSION)
    chain.update(waveDt, params, (x, z) =>
      sampleWaveLevel(x, z, noise, waveField, ocean.layerCache),
      camera.target[0], camera.target[2])
    const encoder = device.createCommandEncoder()
    waveField.render(encoder)
    capField.render(encoder)
    // While paused the decay factors are exp(0) = 1 but generation would
    // keep running against the frozen sim state and pump the accumulation
    // to saturation, so freeze the foam buffer entirely
    if (waveDt > 0) {
      foam.render(encoder)
      filmFoam.render(encoder)
    }
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: msaa.createView(),
        resolveTarget: context.getCurrentTexture().createView(),
        loadOp: 'clear',
        storeOp: 'discard',
        clearValue: { r: 0.84, g: 0.87, b: 0.9, a: 1 },
      }],
      depthStencilAttachment: {
        view: depth.createView(),
        depthLoadOp: 'clear',
        depthStoreOp: 'discard',
        depthClearValue: 1,
      },
    })
    const sunDir = sunDirection(params.timeOfDay, params.latitude, params.azimuth)
    camera.update(dt)
    const eye = camera.eye
    // Which side of the surface the camera is on, in meters. The waterline
    // rides the waves, so this is sampled against the live wave height rather
    // than mean sea level — ducking under a passing crest counts as submerged
    const camDepth = sampleWaveHeight(eye[0], eye[2], noise, waveField, ocean.layerCache) - eye[1]
    // Pull the near plane in as the surface closes on the eye, so the water
    // right at the lens still renders instead of being clipped to a hole
    const near = Math.min(Math.max(0.35 * Math.abs(camDepth), 0.02), 0.5)
    const viewProj = camera.viewProj(w / h, near)
    const lensR = 0.02 + params.waterlineThickness * 0.5
    const lodScale = QUALITY[params.quality].lodScale
    sky.render(pass, invert(viewProj), eye, sunDir, camDepth, lensR, params.uwTurbidity, params.chlorophyll)
    ocean.render(pass, waveDt, params, noise, capNoise, viewProj, eye, sunDir, foam.index, filmFoam.index, camDepth, lodScale)
    pass.end()
    device.queue.submit([encoder.finish()])
    requestAnimationFrame(frame)
  }
  requestAnimationFrame(frame)
}

main().catch(e => {
  const el = document.getElementById('error')
  el.style.display = 'grid'
  el.textContent = String(e?.message ?? e)
  throw e
})
