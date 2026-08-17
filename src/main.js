import { initWebGPU, fetchText } from './gpu.js'
import { generateGravityNoiseSet, generateCapillaryNoiseTexture, generateFoamPatternTexture, loadFoamPlates } from './noise.js'
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
  // Coverage ramp order — sparse lace, mid sheets, dense raft. NOT filename
  // order: foam4 is the mid plate and foam3 the dense one.
  const foamPlates = await loadFoamPlates(device, [
    new URL('../foam2-B6vBNkcW.jpg', import.meta.url),
    new URL('../foam4-XPoWFsfC.jpg', import.meta.url),
    new URL('../foam3-CyaGqCrv.jpg', import.meta.url),
  ])
  const foam = new FoamSim(device, waveCommonCode + foamCode)
  const coast = buildCoast(device)
  const chain = new ChainSim(device, coast)
  const filmFoam = new FoamSim(device, waveCommonCode + filmFoamCode, [128, 256])
  const oceanCodeFull = atmosphereCode + waveCommonCode + oceanCode
  const buildOcean = q => new Ocean(device, oceanCodeFull, waveField.texture, capField.texture,
    foam.views, filmFoam.views, foamPattern, foamPlates, chain.view, chain.coastView, format,
    { gridN: QUALITY[q].gridN, ribbonCells: QUALITY[q].ribbonCells,
      cell: QUALITY[q].cell, linearCells: QUALITY[q].linearCells })
  let ocean = buildOcean('high')
  foam.bind(ocean.uniform, waveField.texture, null)
  filmFoam.bind(ocean.uniform, null, chain.view)
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
  let refrColor = null
  let refrDepth = null
  let width = 0
  let height = 0
  let lastTime = performance.now()
  let builtQuality = 'high'

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
      refrColor?.destroy()
      refrDepth?.destroy()
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
      // Single-sampled: this image exists to be sampled at a wobbling offset
      // and then multiplied by extinction, so 4x coverage would be thrown away
      refrColor = device.createTexture({
        size: [w, h],
        format: 'rgba16float',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      })
      refrDepth = device.createTexture({
        size: [w, h],
        format: 'depth24plus',
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      })
      ocean.setRefractionTarget(refrColor.createView())
    }

    // Quality bakes the grid density into vertex and index buffers, so a
    // change means rebuilding the Ocean and rebinding what points at it
    if (params.quality !== builtQuality) {
      builtQuality = params.quality
      ocean = buildOcean(builtQuality)
      ocean.chain = chain
      foam.bind(ocean.uniform, waveField.texture, null)
      filmFoam.bind(ocean.uniform, null, chain.view)
      if (refrColor) ocean.setRefractionTarget(refrColor.createView())
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
    const { lodScale, maxLayers } = QUALITY[params.quality]
    // Exactly once per frame: this advances the wave phases
    ocean.update(waveDt, params, noise, capNoise, viewProj, eye, sunDir, camDepth, lodScale, maxLayers)

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
    // The submerged scene, drawn first into its own linear target so the
    // surface can refract it. storeOp must be 'store' — the other passes here
    // all discard, and copying that habit yields a garbage texture silently.
    const refrPass = encoder.beginRenderPass({
      colorAttachments: [{
        view: refrColor.createView(),
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
      }],
      depthStencilAttachment: {
        view: refrDepth.createView(),
        depthLoadOp: 'clear',
        depthStoreOp: 'discard',
        depthClearValue: 1,
      },
    })
    ocean.drawRefraction(refrPass, foam.index, filmFoam.index)
    refrPass.end()

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
    sky.render(pass, invert(viewProj), eye, sunDir, camDepth, lensR, params.uwTurbidity, params.chlorophyll, params)
    ocean.draw(pass, params, foam.index, filmFoam.index)
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
