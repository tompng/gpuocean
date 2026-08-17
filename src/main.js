import { initWebGPU, fetchText } from './gpu.js'
import { generateGravityNoiseSet, generateCapillaryNoiseTexture, generateFoamPatternTexture, loadFoamPlates } from './noise.js'
import { WaveField } from './waveField.js'
import { Ocean } from './ocean.js'
import { FoamSim } from './foam.js'
import { ChainSim, sampleWaveLevel, sampleWaveHeight, SLOPE } from './chain.js'
import { buildCoast } from './coast.js'
import { Sky } from './sky.js'
import { OrbitCamera } from './camera.js'
import { invert } from './mat4.js'
import { setupUI, setupFPS, QUALITY } from './ui.js'
import { GPUTimer } from './gputimer.js'
import { Clouds } from './clouds.js'
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

// Everything that has to be rebuilt when the canvas resizes. The refraction
// target is single-sampled on purpose: it exists to be sampled at a wobbling
// offset and then multiplied by extinction, so 4x coverage would be discarded.
function createTargets(device, format, w, h) {
  const attachment = GPUTextureUsage.RENDER_ATTACHMENT
  const make = (fmt, usage, sampleCount) =>
    device.createTexture({ size: [w, h], format: fmt, sampleCount, usage })
  const t = {
    depth: make('depth24plus', attachment, 4),
    msaa: make(format, attachment, 4),
    refrColor: make('rgba16float', attachment | GPUTextureUsage.TEXTURE_BINDING, 1),
    refrDepth: make('depth24plus', attachment, 1),
  }
  t.destroy = () => { for (const k of ['depth', 'msaa', 'refrColor', 'refrDepth']) t[k].destroy() }
  return t
}

// Full-moon geometry: the moon rides the sun's path half a day out of phase and
// on the opposite bearing, so it rises as the sun sets and is highest at
// midnight. Enough to key a night scene without a full lunar ephemeris.
function moonDirection(timeOfDay, latitude, azimuth) {
  return sunDirection(timeOfDay + 12, latitude, azimuth + 180)
}

async function main() {
  const { device, context, format, timestamps } = await initWebGPU(canvas)
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
  const clouds = new Clouds(device)
  const oceanCodeFull = atmosphereCode + waveCommonCode + oceanCode
  const buildOcean = q => new Ocean(device, oceanCodeFull, waveField.texture, capField.texture,
    foam.views, filmFoam.views, foamPattern, foamPlates, chain.view, chain.coastView,
    coast.sdfView, coast.mainTableView, clouds.buffer, format,
    { gridN: QUALITY[q].gridN, ribbonCells: QUALITY[q].ribbonCells,
      cell: QUALITY[q].cell, linearCells: QUALITY[q].linearCells })
  let ocean = buildOcean('high')
  foam.bind(ocean.uniform, waveField.texture, null, coast.sdfView)
  filmFoam.bind(ocean.uniform, null, chain.view, null)
  ocean.chain = chain
  const sky = new Sky(device, atmosphereCode + skyCode, format, clouds.buffer)
  const camera = new OrbitCamera(canvas)
  const params = setupUI()
  const reportFPS = setupFPS()
  const timer = new GPUTimer(device, timestamps, ['refract', 'scene'])
  // Same expression as terrainHeight() in wave_common.wgsl, over the same
  // baked coastline the shader samples, so the floor the camera stops at is
  // the floor that gets drawn
  camera.floor = (x, z) => Math.min(Math.max(SLOPE * coast.sdfAt(x, z), -params.depth), 3)
  setupNoiseDebug([
    ...noise.variants.map(v => ({ name: v.name, size: noise.size, channels: v.channels })),
    { name: 'capillary', size: capNoise.size, channels: capNoise.channels },
    { name: 'foamPattern', size: foamPattern.size, channels: foamPattern.channels },
  ])

  let targets = null
  let width = 0
  let height = 0
  let lastTime = performance.now()
  let builtQuality = 'high'

  function frame(now) {
    const dt = Math.min((now - lastTime) / 1000, 0.1)
    lastTime = now
    reportFPS(dt, timer.label())

    const w = Math.max(1, Math.floor(canvas.clientWidth * devicePixelRatio))
    const h = Math.max(1, Math.floor(canvas.clientHeight * devicePixelRatio))
    if (w !== width || h !== height) {
      width = w
      height = h
      canvas.width = w
      canvas.height = h
      targets?.destroy()
      targets = createTargets(device, format, w, h)
      ocean.setRefractionTarget(targets.refrColor.createView())
    }

    // Quality bakes the grid density into vertex and index buffers, so a
    // change means rebuilding the Ocean and rebinding what points at it
    if (params.quality !== builtQuality) {
      builtQuality = params.quality
      ocean = buildOcean(builtQuality)
      ocean.chain = chain
      foam.bind(ocean.uniform, waveField.texture, null, coast.sdfView)
      filmFoam.bind(ocean.uniform, null, chain.view, null)
      if (targets) ocean.setRefractionTarget(targets.refrColor.createView())
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
    const moonDir = moonDirection(params.timeOfDay, params.latitude, params.azimuth)
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
    clouds.update(dt, params, eye, h, Math.PI / 3)
    // Exactly once per frame: this advances the wave phases
    ocean.update(waveDt, params, noise, capNoise, viewProj, eye, sunDir, moonDir, camDepth, lodScale, maxLayers)

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
        view: targets.refrColor.createView(),
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
      }],
      depthStencilAttachment: {
        view: targets.refrDepth.createView(),
        depthLoadOp: 'clear',
        depthStoreOp: 'discard',
        depthClearValue: 1,
      },
      timestampWrites: timer.writes(0),
    })
    ocean.drawRefraction(refrPass, foam.index, filmFoam.index)
    refrPass.end()

    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: targets.msaa.createView(),
        resolveTarget: context.getCurrentTexture().createView(),
        loadOp: 'clear',
        storeOp: 'discard',
        clearValue: { r: 0.84, g: 0.87, b: 0.9, a: 1 },
      }],
      depthStencilAttachment: {
        view: targets.depth.createView(),
        depthLoadOp: 'clear',
        depthStoreOp: 'discard',
        depthClearValue: 1,
      },
      timestampWrites: timer.writes(1),
    })
    sky.render(pass, invert(viewProj), eye, sunDir, moonDir, camDepth, lensR, params.uwTurbidity, params.chlorophyll, params)
    ocean.draw(pass, params, foam.index, filmFoam.index)
    pass.end()
    timer.resolveInto(encoder)
    device.queue.submit([encoder.finish()])
    timer.read()
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
