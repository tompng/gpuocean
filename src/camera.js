import { lookAt, multiply, perspective } from './mat4.js'

export class OrbitCamera {
  constructor(canvas) {
    this.yaw = Math.PI - 0.6
    this.pitch = 0.18
    this.distance = 20
    let drag = null
    canvas.addEventListener('pointerdown', e => {
      drag = [e.clientX, e.clientY]
      canvas.setPointerCapture(e.pointerId)
    })
    canvas.addEventListener('pointermove', e => {
      if (!drag) return
      this.yaw += (e.clientX - drag[0]) * 0.005
      this.pitch = clamp(this.pitch + (e.clientY - drag[1]) * 0.005, 0.05, 1.5)
      drag = [e.clientX, e.clientY]
    })
    canvas.addEventListener('pointerup', () => { drag = null })
    canvas.addEventListener('pointercancel', () => { drag = null })
    canvas.addEventListener('wheel', e => {
      e.preventDefault()
      this.distance = clamp(this.distance * Math.exp(e.deltaY * 0.0015), 1, 1000)
    }, { passive: false })
  }

  get eye() {
    const ground = Math.cos(this.pitch) * this.distance
    return [
      Math.cos(this.yaw) * ground,
      Math.sin(this.pitch) * this.distance,
      Math.sin(this.yaw) * ground,
    ]
  }

  viewProj(aspect) {
    const view = lookAt(this.eye, [0, 0, 0], [0, 1, 0])
    return multiply(perspective(Math.PI / 3, aspect, 0.5, 300000), view)
  }
}

function clamp(v, min, max) {
  return Math.min(Math.max(v, min), max)
}
