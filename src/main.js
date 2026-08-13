import { initWebGPU, fetchText } from './gpu.js'
import { generateGravityNoiseTexture, generateCapillaryNoiseTexture, generateFoamPatternTexture } from './noise.js'
import { WaveField } from './waveField.js'
import { Ocean } from './ocean.js'
import { FoamSim } from './foam.js'
import { ChainSim, sampleWaveDispX } from './chain.js'
import { Sky } from './sky.js'
import { OrbitCamera } from './camera.js'
import { invert } from './mat4.js'
import { setupUI } from './ui.js'
import { setupNoiseDebug } from './debug.js'

const canvas = document.getElementById('canvas')
const GRAVITY = 9.81
const CAPILLARY_SIGMA_RHO = 7.4e-5
const CAP_DISPERSION = 1.5
const SUN_AZIMUTH = [0.65, -0.76]

async function main() {
  const { device, context, format } = await initWebGPU(canvas)
  const [waveFieldCode, mipCode, waveCommonCode, oceanCode, atmosphereCode, skyCode, foamCode] = await Promise.all(
    ['wave_field', 'mip', 'wave_common', 'ocean', 'atmosphere', 'sky', 'foam'].map(name => fetchText(new URL(`../shaders/${name}.wgsl`, import.meta.url)))
  )
  const noise = generateGravityNoiseTexture(device)
  const capNoise = generateCapillaryNoiseTexture(device)
  const waveField = new WaveField(device, waveFieldCode, mipCode, noise)
  const capField = new WaveField(device, waveFieldCode, mipCode, capNoise)
  const foamPattern = generateFoamPatternTexture(device)
  const foam = new FoamSim(device, waveCommonCode + foamCode)
  const chain = new ChainSim(device)
  const ocean = new Ocean(device, atmosphereCode + waveCommonCode + oceanCode, waveField.texture, capField.texture, foam.views, foamPattern, chain.view, format)
  foam.bind(ocean.uniform, waveField.texture, chain.view)
  const sky = new Sky(device, atmosphereCode + skyCode, format)
  const camera = new OrbitCamera(canvas)
  const params = setupUI()
  setupNoiseDebug([
    { name: 'gravity', size: noise.size, channels: noise.channels },
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
    chain.update(waveDt, params, 80, (x, z) =>
      sampleWaveDispX(x, z, noise, waveField, ocean.layerCache, params.choppiness, 2 * Math.PI / params.wavelength, params))
    const encoder = device.createCommandEncoder()
    waveField.render(encoder)
    capField.render(encoder)
    foam.render(encoder)
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
    const elevation = params.sun * Math.PI / 180
    const sunDir = [
      SUN_AZIMUTH[0] * Math.cos(elevation),
      Math.sin(elevation),
      SUN_AZIMUTH[1] * Math.cos(elevation),
    ]
    const viewProj = camera.viewProj(w / h)
    sky.render(pass, invert(viewProj), camera.eye, sunDir)
    ocean.render(pass, waveDt, params, noise, capNoise, viewProj, camera.eye, sunDir, foam.index)
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
