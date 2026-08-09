import { initWebGPU, fetchText } from './gpu.js'
import { generateGravityNoiseTexture, generateCapillaryNoiseTexture } from './noise.js'
import { WaveField } from './waveField.js'
import { Ocean } from './ocean.js'
import { OrbitCamera } from './camera.js'
import { setupUI } from './ui.js'

const canvas = document.getElementById('canvas')
const GRAVITY = 9.81
const CAPILLARY_SIGMA_RHO = 7.4e-5
const CAP_DISPERSION = 1.5

async function main() {
  const { device, context, format } = await initWebGPU(canvas)
  const [waveFieldCode, mipCode, oceanCode] = await Promise.all(
    ['wave_field', 'mip', 'ocean'].map(name => fetchText(new URL(`../shaders/${name}.wgsl`, import.meta.url)))
  )
  const noise = generateGravityNoiseTexture(device)
  const capNoise = generateCapillaryNoiseTexture(device)
  const waveField = new WaveField(device, waveFieldCode, mipCode, noise)
  const capField = new WaveField(device, waveFieldCode, mipCode, capNoise)
  const ocean = new Ocean(device, oceanCode, waveField.texture, capField.texture, format)
  const camera = new OrbitCamera(canvas)
  const params = setupUI()

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

    const lambda = params.wavelength
    waveField.update(dt, Math.sqrt(GRAVITY * lambda / (2 * Math.PI)) / (lambda * noise.wavesPerTile), params.dispersion)
    const capK = 2 * Math.PI / params.rippleScale
    const capSpeed = Math.sqrt(GRAVITY / capK + CAPILLARY_SIGMA_RHO * capK)
    capField.update(dt, capSpeed / (params.rippleScale * capNoise.wavesPerTile), CAP_DISPERSION)
    const encoder = device.createCommandEncoder()
    waveField.render(encoder)
    capField.render(encoder)
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
    ocean.render(pass, dt, params, noise, capNoise, camera.viewProj(w / h), camera.eye)
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
