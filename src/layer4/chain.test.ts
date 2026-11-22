import { describe, expect, it } from 'vitest'
import { createNetwork, createRng, forward, PIXELS } from '../engine/network'
import { focusConnection } from '../layer3/slope'
import { deepConnection, gradientOf, outputChain, pathsFromPixel } from './chain'

/** A digit-like image: a block of ink, the rest blank. */
function image(seed: number): Float64Array {
  const rand = createRng(seed)
  const x = new Float64Array(PIXELS)
  for (let r = 6; r < 22; r++) for (let c = 10; c < 18; c++) if (rand() < 0.8) x[r * 28 + c] = 0.3 + 0.7 * rand()
  return x
}

describe('chain rule', () => {
  const net = createNetwork()
  const x = image(3)

  it('multiplies three local slopes into the gradient backpropagation finds', () => {
    for (const c of [focusConnection(net, x, 7), { layer: 2, from: 3, to: 0 }, { layer: 2, from: 9, to: 4 }]) {
      const ch = outputChain(net, x, 7, c)
      expect(ch.dLdy * ch.dydz * ch.dzdw).toBe(ch.gradient)
      expect(ch.gradient).toBeCloseTo(gradientOf(net, x, 7, c), 12)
    }
  })

  it('adds up 160 paths from a pixel into the gradient backpropagation finds', () => {
    const c = deepConnection(net, x, 7)
    const p = pathsFromPixel(net, forward(net, x), 7, c.from, c.to)
    expect(x[c.from]).toBeGreaterThan(0)
    expect(p.paths).toHaveLength(160)
    expect(p.paths.filter((q) => q.value === 0).length).toBeGreaterThanOrEqual(p.silentPaths)
    expect(p.total).toBeCloseTo(gradientOf(net, x, 7, c), 12)
    expect(p.total).not.toBe(0)
  })

  it('gives 0 for a blank pixel', () => {
    const p = pathsFromPixel(net, forward(net, x), 7, 0, 0)
    expect(p.total).toBe(0)
  })
})
