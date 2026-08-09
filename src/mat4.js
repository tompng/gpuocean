// Column-major, WebGPU clip space (z in [0, 1])

export function perspective(fovY, aspect, near, far) {
  const f = 1 / Math.tan(fovY / 2)
  const m = new Float32Array(16)
  m[0] = f / aspect
  m[5] = f
  m[10] = far / (near - far)
  m[11] = -1
  m[14] = (near * far) / (near - far)
  return m
}

export function lookAt(eye, target, up) {
  const z = normalize(sub(eye, target))
  const x = normalize(cross(up, z))
  const y = cross(z, x)
  return new Float32Array([
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -dot(x, eye), -dot(y, eye), -dot(z, eye), 1,
  ])
}

export function multiply(a, b) {
  const m = new Float32Array(16)
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      let s = 0
      for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k]
      m[c * 4 + r] = s
    }
  }
  return m
}

export function invert(m) {
  const inv = new Float32Array(16)
  const [
    a00, a01, a02, a03, a10, a11, a12, a13,
    a20, a21, a22, a23, a30, a31, a32, a33,
  ] = m
  const b00 = a00 * a11 - a01 * a10
  const b01 = a00 * a12 - a02 * a10
  const b02 = a00 * a13 - a03 * a10
  const b03 = a01 * a12 - a02 * a11
  const b04 = a01 * a13 - a03 * a11
  const b05 = a02 * a13 - a03 * a12
  const b06 = a20 * a31 - a21 * a30
  const b07 = a20 * a32 - a22 * a30
  const b08 = a20 * a33 - a23 * a30
  const b09 = a21 * a32 - a22 * a31
  const b10 = a21 * a33 - a23 * a31
  const b11 = a22 * a33 - a23 * a32
  const det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06
  const d = 1 / det
  inv[0] = (a11 * b11 - a12 * b10 + a13 * b09) * d
  inv[1] = (a02 * b10 - a01 * b11 - a03 * b09) * d
  inv[2] = (a31 * b05 - a32 * b04 + a33 * b03) * d
  inv[3] = (a22 * b04 - a21 * b05 - a23 * b03) * d
  inv[4] = (a12 * b08 - a10 * b11 - a13 * b07) * d
  inv[5] = (a00 * b11 - a02 * b08 + a03 * b07) * d
  inv[6] = (a32 * b02 - a30 * b05 - a33 * b01) * d
  inv[7] = (a20 * b05 - a22 * b02 + a23 * b01) * d
  inv[8] = (a10 * b10 - a11 * b08 + a13 * b06) * d
  inv[9] = (a01 * b08 - a00 * b10 - a03 * b06) * d
  inv[10] = (a30 * b04 - a31 * b02 + a33 * b00) * d
  inv[11] = (a21 * b02 - a20 * b04 - a23 * b00) * d
  inv[12] = (a11 * b07 - a10 * b09 - a12 * b06) * d
  inv[13] = (a00 * b09 - a01 * b07 + a02 * b06) * d
  inv[14] = (a31 * b01 - a30 * b03 - a32 * b00) * d
  inv[15] = (a20 * b03 - a21 * b01 + a22 * b00) * d
  return inv
}

export function normalize(v) {
  const l = Math.hypot(v[0], v[1], v[2])
  return [v[0] / l, v[1] / l, v[2] / l]
}

function sub(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ]
}

function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
