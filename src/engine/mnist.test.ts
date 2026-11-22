import { describe, expect, it } from 'vitest'
import { PIXELS, createNetwork } from './network'
import {
  batchOrder,
  createDataset,
  evaluate,
  filterDataset,
  getSample,
  labelCounts,
  parseImages,
  parseLabels,
} from './mnist'

function idxImages(pixels: number[][]): ArrayBuffer {
  const buf = new ArrayBuffer(16 + pixels.length * PIXELS)
  const view = new DataView(buf)
  view.setUint32(0, 0x803)
  view.setUint32(4, pixels.length)
  view.setUint32(8, 28)
  view.setUint32(12, 28)
  pixels.forEach((img, n) => new Uint8Array(buf, 16 + n * PIXELS, PIXELS).set(img))
  return buf
}

function idxLabels(labels: number[]): ArrayBuffer {
  const buf = new ArrayBuffer(8 + labels.length)
  const view = new DataView(buf)
  view.setUint32(0, 0x801)
  view.setUint32(4, labels.length)
  new Uint8Array(buf, 8).set(labels)
  return buf
}

const blank = () => new Array<number>(PIXELS).fill(0)

describe('IDX parsing', () => {
  it('reads images scaled to [0, 1] and labels', () => {
    const a = blank()
    a[0] = 255
    a[783] = 51
    const images = parseImages(idxImages([a, blank()]))
    expect(images.length).toBe(2 * PIXELS)
    expect(images[0]).toBe(1)
    expect(images[783]).toBeCloseTo(0.2, 6)
    expect(Array.from(parseLabels(idxLabels([7, 2])))).toEqual([7, 2])
  })

  it('rejects the wrong file', () => {
    expect(() => parseImages(idxLabels([1]))).toThrow()
    expect(() => parseLabels(idxImages([blank()]))).toThrow()
  })
})

describe('datasets', () => {
  const labels = [0, 1, 7, 7, 3, 7]
  const data = createDataset(new Float32Array(labels.length * PIXELS), Uint8Array.from(labels))

  it('counts and filters examples', () => {
    expect(labelCounts(data)).toEqual([1, 1, 0, 1, 0, 0, 0, 3, 0, 0])
    const no7 = filterDataset(data, (label) => label !== 7)
    expect(Array.from(no7.labels)).toEqual([0, 1, 3])
    expect(getSample(no7, 2).x.length).toBe(PIXELS)
  })

  it('covers every example once per pass, in batches', () => {
    const batches = batchOrder(60_000, 32, 1)
    expect(batches.length).toBe(1_875)
    expect(new Set(batches.flat()).size).toBe(60_000)
  })

  it('builds a confusion table', () => {
    const result = evaluate(createNetwork(), data)
    expect(result.confusion.flat().reduce((a, b) => a + b, 0)).toBe(labels.length)
    expect(result.accuracy).toBeGreaterThanOrEqual(0)
    expect(result.perDigit.length).toBe(10)
  })
})
