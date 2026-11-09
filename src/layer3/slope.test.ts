import { describe, expect, it } from 'vitest'
import { createNetwork, createRng, forward, PIXELS } from '../engine/network'
import { strength } from '../playground/connection'
import { exactSlope, focusConnection, mistakeAt, nudgeSlope, stepFor, withStrength } from './slope'

/** A digit-like image: a block of ink, the rest blank. */
function image(seed: number): Float64Array {
  const rand = createRng(seed)
  const x = new Float64Array(PIXELS)
  for (let r = 6; r < 22; r++) for (let c = 10; c < 18; c++) if (rand() < 0.8) x[r * 28 + c] = 0.3 + 0.7 * rand()
  return x
}

describe('slope', () => {
  const net = createNetwork()
  const x = image(3)

  it('focuses on the hardest-firing hidden-layer-2 neuron into the correct output', () => {
    const c = focusConnection(net, x, 7)
    const a = forward(net, x).layers[1].a
    expect(c).toMatchObject({ layer: 2, to: 7 })
    expect(a[c.from]).toBe(Math.max(...a))
    expect(a[c.from]).toBeGreaterThan(0)
  })

  it('changes one connection and leaves the original alone', () => {
    const c = { layer: 1, from: 2, to: 5 }
    const copy = withStrength(net, c, 0.5)
    expect(strength(copy, c)).toBe(0.5)
    expect(net).toEqual(createNetwork())
  })

  it('finds the same steepness by nudging as backpropagation does, in every layer', () => {
    for (const c of [
      { layer: 0, from: 14 * 28 + 13, to: 3 },
      { layer: 1, from: 4, to: 9 },
      { layer: 2, from: 0, to: 7 },
      { layer: 2, from: 5, to: 2 },
    ]) {
      const w = strength(net, c)
      expect(nudgeSlope(net, x, 7, c, w, 1e-5)).toBeCloseTo(exactSlope(net, x, 7, c, w), 5)
    }
  })

  it('steps downhill', () => {
    const c = focusConnection(net, x, 7)
    const w = strength(net, c)
    const step = stepFor(2, exactSlope(net, x, 7, c, w))
    expect(step).toBeGreaterThan(0)
    expect(mistakeAt(net, x, 7, c, w + step)).toBeLessThan(mistakeAt(net, x, 7, c, w))
  })
})
