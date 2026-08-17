import { lookAt, multiply, perspective } from './mat4.js'

const FOV = Math.PI / 3
// Pitch runs negative so the orbit can pass under the surface and look back
// up at it; the limits stop just short of the poles, where the lookAt up
// vector degenerates
const MIN_PITCH = -1.55
const MAX_PITCH = 1.55
const MIN_DISTANCE = 1
const MAX_DISTANCE = 1000
// Deeper than the deepest sea floor the depth slider can reach (40 m), so the
// target can be dropped to the basin floor anywhere
const MIN_TARGET_Y = -60
const MAX_TARGET_Y = 500
const ROTATE_SPEED = 0.005
// How far above the sand the eye is held when the orbit dives into the basin.
// Must clear the 0.5 m near plane, or the sand directly below the camera is
// clipped away and the basin opens into a hole.
const GROUND_CLEARANCE = 1.2
const ZOOM_SPEED = 0.0015
// Exponential approach rate for the damping; higher is snappier
const DAMPING = 18

export class OrbitCamera {
  constructor(canvas) {
    this.canvas = canvas
    this.yaw = Math.PI - 0.6
    this.pitch = 0.18
    this.distance = 20
    this.target = [0, 0, 0]
    // Input writes the goal state; update() eases the live state toward it
    this.goalYaw = this.yaw
    this.goalPitch = this.pitch
    this.goalDistance = this.distance
    this.goalTarget = [...this.target]

    this.keys = new Set()
    window.addEventListener('keydown', e => { this.keys.add(e.code) })
    window.addEventListener('keyup', e => { this.keys.delete(e.code) })
    window.addEventListener('blur', () => { this.keys.clear() })

    this.pointers = new Map()
    this.pinch = null
    canvas.addEventListener('pointerdown', e => this.onDown(e))
    canvas.addEventListener('pointermove', e => this.onMove(e))
    canvas.addEventListener('pointerup', e => this.onUp(e))
    canvas.addEventListener('pointercancel', e => this.onUp(e))
    // Right-drag pans, so the browser menu has to stay out of the way
    canvas.addEventListener('contextmenu', e => e.preventDefault())
    canvas.addEventListener('wheel', e => {
      e.preventDefault()
      this.dolly(e.deltaY * lineScale(e.deltaMode) * ZOOM_SPEED)
    }, { passive: false })
  }

  onDown(e) {
    e.preventDefault()
    this.canvas.setPointerCapture(e.pointerId)
    // Middle/right button, or shift with the left one, pans instead of orbiting
    const pan = e.pointerType === 'mouse' && (e.button === 1 || e.button === 2 || e.shiftKey)
    this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY, pan })
    // Seed the two-finger baseline now so the first move already applies
    if (this.pointers.size === 2) {
      this.pinch = null
      this.gesture()
    }
  }

  onMove(e) {
    const p = this.pointers.get(e.pointerId)
    if (!p) return
    const dx = e.clientX - p.x
    const dy = e.clientY - p.y
    p.x = e.clientX
    p.y = e.clientY
    if (this.pointers.size >= 2) this.gesture()
    else if (p.pan) this.pan(dx, dy)
    else this.rotate(dx, dy)
  }

  onUp(e) {
    if (!this.pointers.delete(e.pointerId)) return
    if (this.canvas.hasPointerCapture(e.pointerId)) this.canvas.releasePointerCapture(e.pointerId)
    if (this.pointers.size < 2) this.pinch = null
  }

  // Two fingers: spread/squeeze dollies, and the midpoint drags the target
  gesture() {
    const [a, b] = [...this.pointers.values()]
    const spread = Math.max(Math.hypot(a.x - b.x, a.y - b.y), 1)
    const cx = (a.x + b.x) / 2
    const cy = (a.y + b.y) / 2
    if (this.pinch) {
      this.dolly(Math.log(this.pinch.spread / spread))
      this.pan(cx - this.pinch.cx, cy - this.pinch.cy)
    }
    this.pinch = { spread, cx, cy }
  }

  rotate(dx, dy) {
    this.goalYaw += dx * ROTATE_SPEED
    this.goalPitch = clamp(this.goalPitch + dy * ROTATE_SPEED, MIN_PITCH, MAX_PITCH)
  }

  dolly(amount) {
    this.goalDistance = clamp(this.goalDistance * Math.exp(amount), MIN_DISTANCE, MAX_DISTANCE)
  }

  // Slide the target across the view plane so the scene tracks the cursor
  pan(dx, dy) {
    const height = this.canvas.clientHeight || 1
    const scale = 2 * this.distance * Math.tan(FOV / 2) / height
    const cy = Math.cos(this.yaw)
    const sy = Math.sin(this.yaw)
    const cp = Math.cos(this.pitch)
    const sp = Math.sin(this.pitch)
    // right = (sy, 0, -cy), up = (-sp * cy, cp, -sp * sy)
    this.goalTarget[0] += (-sy * dx - sp * cy * dy) * scale
    this.goalTarget[1] += cp * dy * scale
    this.goalTarget[2] += (cy * dx - sp * sy * dy) * scale
    this.goalTarget[1] = clamp(this.goalTarget[1], MIN_TARGET_Y, MAX_TARGET_Y)
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
  update(dt) {
    const speed = (this.keys.has('ShiftLeft') || this.keys.has('ShiftRight') ? 75 : 15) * dt
    const fx = -Math.cos(this.goalYaw)
    const fz = -Math.sin(this.goalYaw)
    let mx = 0
    let mz = 0
    if (this.keys.has('KeyW')) { mx += fx; mz += fz }
    if (this.keys.has('KeyS')) { mx -= fx; mz -= fz }
    if (this.keys.has('KeyA')) { mx += fz; mz -= fx }
    if (this.keys.has('KeyD')) { mx -= fz; mz += fx }
    this.goalTarget[0] += mx * speed
    this.goalTarget[2] += mz * speed

    const k = 1 - Math.exp(-DAMPING * dt)
    this.yaw += (this.goalYaw - this.yaw) * k
    this.pitch += (this.goalPitch - this.pitch) * k
    // Geometric so a dolly reads the same at every scale
    this.distance *= Math.exp(Math.log(this.goalDistance / this.distance) * k)
    for (let i = 0; i < 3; i++) this.target[i] += (this.goalTarget[i] - this.target[i]) * k

    // Float the whole rig off the sea floor rather than clamping the orbit
    // angles: diving under the sand renders the basin from beneath, and
    // raising the target keeps yaw/pitch/distance exactly as the user set
    // them. Solved for the target height that clears the floor rather than
    // nudged by the shortfall — the eye's ground position does not depend on
    // target height, so this is exact, and clamping (rather than accumulating
    // a lift) leaves a target already above the floor untouched.
    if (this.floor) {
      const e = this.eye
      const minY = this.floor(e[0], e[2]) + GROUND_CLEARANCE - Math.sin(this.pitch) * this.distance
      this.target[1] = Math.max(this.target[1], minY)
      this.goalTarget[1] = Math.max(this.goalTarget[1], minY)
    }
  }

  // `near` is a parameter because sitting at the waterline puts the surface
  // centimeters from the eye, and the default 0.5 m plane slices it away —
  // leaving a hole that shows only the sea floor beneath. main.js tightens it
  // as the camera approaches the surface and relaxes it again for the depth
  // precision the 300 km far plane needs.
  viewProj(aspect, near = 0.5) {
    const view = lookAt(this.eye, this.target, [0, 1, 0])
    return multiply(perspective(FOV, aspect, near, 300000), view)
  }
}

// Wheel deltas arrive in pixels, lines, or pages depending on the device
function lineScale(mode) {
  return mode === 1 ? 16 : mode === 2 ? 100 : 1
}

function clamp(v, min, max) {
  return Math.min(Math.max(v, min), max)
}
