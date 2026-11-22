// Turning a hand-drawn digit into the network's input, the way MNIST digits were prepared:
// the ink is shrunk to fit a 20 × 20 box (keeping its shape), then placed in the 28 × 28 image
// so that its center of mass sits at the center.
import { IMAGE_SIDE, PIXELS } from './network'

const BOX = 20
const INK = 0.05

/**
 * @param src grayscale values in [0, 1], row by row, `width × height`
 * @returns 784 pixel values in [0, 1]
 */
export function prepareDrawing(src: ArrayLike<number>, width: number, height: number): Float32Array {
  let x0 = width, y0 = height, x1 = -1, y1 = -1
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (src[y * width + x] > INK) {
        if (x < x0) x0 = x
        if (x > x1) x1 = x
        if (y < y0) y0 = y
        if (y > y1) y1 = y
      }
    }
  }
  const out = new Float32Array(PIXELS)
  if (x1 < 0) return out

  const bw = x1 - x0 + 1
  const bh = y1 - y0 + 1
  const scale = BOX / Math.max(bw, bh)
  const tw = Math.max(1, Math.round(bw * scale))
  const th = Math.max(1, Math.round(bh * scale))

  // Area-average the ink box down to tw × th.
  const box = new Float32Array(tw * th)
  for (let ty = 0; ty < th; ty++) {
    const sy0 = y0 + (ty * bh) / th
    const sy1 = y0 + ((ty + 1) * bh) / th
    for (let tx = 0; tx < tw; tx++) {
      const sx0 = x0 + (tx * bw) / tw
      const sx1 = x0 + ((tx + 1) * bw) / tw
      let sum = 0
      let area = 0
      for (let y = Math.floor(sy0); y < Math.ceil(sy1); y++) {
        const hy = Math.min(y + 1, sy1) - Math.max(y, sy0)
        for (let x = Math.floor(sx0); x < Math.ceil(sx1); x++) {
          const hx = Math.min(x + 1, sx1) - Math.max(x, sx0)
          sum += src[y * width + x] * hx * hy
          area += hx * hy
        }
      }
      box[ty * tw + tx] = area > 0 ? sum / area : 0
    }
  }

  // Center of mass of the shrunk ink, then shift it to the image center.
  let mass = 0, mx = 0, my = 0
  for (let ty = 0; ty < th; ty++) {
    for (let tx = 0; tx < tw; tx++) {
      const v = box[ty * tw + tx]
      mass += v
      mx += v * (tx + 0.5)
      my += v * (ty + 0.5)
    }
  }
  const ox = Math.round(IMAGE_SIDE / 2 - mx / mass)
  const oy = Math.round(IMAGE_SIDE / 2 - my / mass)
  for (let ty = 0; ty < th; ty++) {
    for (let tx = 0; tx < tw; tx++) {
      const x = tx + ox
      const y = ty + oy
      if (x >= 0 && x < IMAGE_SIDE && y >= 0 && y < IMAGE_SIDE) {
        out[y * IMAGE_SIDE + x] = Math.min(1, box[ty * tw + tx])
      }
    }
  }
  return out
}

/** The same image turned upside down (rotated 180°). */
export const rotate180 = (image: ArrayLike<number>): Float32Array => Float32Array.from(image).reverse()

export const hasInk = (image: ArrayLike<number>): boolean => {
  for (let k = 0; k < image.length; k++) if (image[k] > INK) return true
  return false
}
