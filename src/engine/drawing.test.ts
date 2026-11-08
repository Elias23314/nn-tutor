import { describe, expect, it } from 'vitest'
import { hasInk, prepareDrawing, rotate180 } from './drawing'

/** A 280 × 280 canvas with a filled rectangle. */
function canvasWith(x0: number, y0: number, x1: number, y1: number, size = 280): Float32Array {
  const px = new Float32Array(size * size)
  for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) px[y * size + x] = 1
  return px
}

function inkBox(img: Float32Array) {
  let x0 = 28, y0 = 28, x1 = -1, y1 = -1, mass = 0, mx = 0, my = 0
  img.forEach((v, k) => {
    if (v <= 0) return
    const x = k % 28, y = Math.floor(k / 28)
    x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y)
    mass += v; mx += v * (x + 0.5); my += v * (y + 0.5)
  })
  return { w: x1 - x0 + 1, h: y1 - y0 + 1, cx: mx / mass, cy: my / mass }
}

describe('prepareDrawing', () => {
  it('returns a blank image for a blank canvas', () => {
    const img = prepareDrawing(new Float32Array(280 * 280), 280, 280)
    expect(img.length).toBe(784)
    expect(hasInk(img)).toBe(false)
  })

  it('shrinks the ink to fit 20 × 20, keeping its shape', () => {
    // A tall stroke in the top-left corner: 40 wide, 200 tall.
    const box = inkBox(prepareDrawing(canvasWith(10, 5, 49, 204), 280, 280))
    expect(box.h).toBe(20)
    expect(box.w).toBe(4)
  })

  it('centers the ink by its center of mass, wherever it was drawn', () => {
    const box = inkBox(prepareDrawing(canvasWith(200, 200, 279, 279), 280, 280))
    expect(Math.abs(box.cx - 14)).toBeLessThanOrEqual(0.5)
    expect(Math.abs(box.cy - 14)).toBeLessThanOrEqual(0.5)
  })

  it('keeps values in [0, 1]', () => {
    for (const v of prepareDrawing(canvasWith(30, 30, 250, 60), 280, 280)) {
      expect(v).toBeGreaterThanOrEqual(0)
      expect(v).toBeLessThanOrEqual(1)
    }
  })
})

it('rotate180 turns the image upside down', () => {
  const img = new Float32Array(784)
  img[0] = 1
  const turned = rotate180(img)
  expect(turned[783]).toBe(1)
  expect(turned[0]).toBe(0)
  expect(img[0]).toBe(1)
})
