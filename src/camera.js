import { lookAt, multiply, perspective } from './mat4.js'

export class OrbitCamera {
  constructor(canvas) {
    this.yaw = Math.PI - 0.6
    this.pitch = 0.18
    this.distance = 20
    this.target = [0, 0, 0]
    this.keys = new Set()
    window.addEventListener('keydown', e => { this.keys.add(e.code) })
    window.addEventListener('keyup', e => { this.keys.delete(e.code) })
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
      this.target[0] + Math.cos(this.yaw) * ground,
      this.target[1] + Math.sin(this.pitch) * this.distance,
      this.target[2] + Math.sin(this.yaw) * ground,
    ]
  }

  // WASD pans the orbit target horizontally, camera-relative; shift is fast
  move(dt) {
    const speed = (this.keys.has('ShiftLeft') || this.keys.has('ShiftRight') ? 75 : 15) * dt
    const fx = -Math.cos(this.yaw)
    const fz = -Math.sin(this.yaw)
    let mx = 0
    let mz = 0
    if (this.keys.has('KeyW')) { mx += fx; mz += fz }
    if (this.keys.has('KeyS')) { mx -= fx; mz -= fz }
    if (this.keys.has('KeyA')) { mx += fz; mz -= fx }
    if (this.keys.has('KeyD')) { mx -= fz; mz += fx }
    this.target[0] += mx * speed
    this.target[2] += mz * speed
  }

  viewProj(aspect) {
    const view = lookAt(this.eye, this.target, [0, 1, 0])
    return multiply(perspective(Math.PI / 3, aspect, 0.5, 300000), view)
  }
}

function clamp(v, min, max) {
  return Math.min(Math.max(v, min), max)
}
