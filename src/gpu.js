export async function initWebGPU(canvas) {
  if (!navigator.gpu) throw new Error('WebGPU is not available in this browser')
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
  if (!adapter) throw new Error('No WebGPU adapter found')
  // Wall-clock frame timing is vsync-capped and cannot see below the refresh
  // interval, so ask for GPU timestamps when the adapter has them
  const timestamps = adapter.features.has('timestamp-query')
  const device = await adapter.requestDevice({
    requiredFeatures: timestamps ? ['timestamp-query'] : [],
  })
  // A validation error otherwise only reaches the console, where a per-frame
  // pass failure scrolls past as an unattributed warning and the canvas just
  // goes black. Surface the first one in the error panel instead.
  device.addEventListener('uncapturederror', e => {
    const el = document.getElementById('error')
    if (el.style.display === 'grid') return
    el.style.display = 'grid'
    el.textContent = `WebGPU: ${e.error.message}`
  })
  const context = canvas.getContext('webgpu')
  const format = navigator.gpu.getPreferredCanvasFormat()
  context.configure({ device, format, alphaMode: 'opaque' })
  return { device, context, format, timestamps }
}

export async function fetchText(url) {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`Failed to fetch ${url}: ${res.status}`)
  return res.text()
}

const f32buf = new Float32Array(1)
const u32buf = new Uint32Array(f32buf.buffer)

// Subnormals flush to zero, values beyond f16 range clamp to max normal
export function floatToHalf(value) {
  f32buf[0] = value
  const bits = u32buf[0]
  const sign = (bits >>> 16) & 0x8000
  const e = ((bits >>> 23) & 0xff) - 112
  if (e <= 0) return sign
  if (e >= 31) return sign | 0x7bff
  return sign | (e << 10) | ((bits & 0x7fffff) >>> 13)
}
