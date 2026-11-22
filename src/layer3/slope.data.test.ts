/// <reference types="node" />
// Layer 3's story on the real first training 7, before any practice.
import { readFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'
import { createNetwork, forward, neuronName } from '../engine/network'
import { getSample, loadMnist } from '../engine/mnist'
import { sourceActivity, strength } from '../playground/connection'
import { exactSlope, focusConnection, mistakeAt, nudgeSlope, slice, stepFor } from './slope'

const dir = fileURLToPath(new URL('../../public/mnist/', import.meta.url))

describe('the focus connection on the first training 7', () => {
  it('fires its source, and a stronger connection lowers the mistake', async () => {
    const mnist = await loadMnist(async (name) => {
      const buf = await readFile(dir + name)
      return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer
    })
    let index = 0
    while (mnist.train.labels[index] !== 7) index++
    const x = getSample(mnist.train, index).x
    const net = createNetwork()
    const c = focusConnection(net, x, 7)
    const fwd = forward(net, x)
    const w = strength(net, c)
    const exact = exactSlope(net, x, 7, c, w)
    console.log({
      focus: `${neuronName(2, c.from)} → 7`,
      firing: sourceActivity(fwd, c),
      w,
      y7: fwd.yHat[7],
      mistake: mistakeAt(net, x, 7, c, w),
      nudged: mistakeAt(net, x, 7, c, w + 0.01),
      slopes: [0.1, 0.01, 0.001].map((h) => nudgeSlope(net, x, 7, c, w, h)),
      exact,
      step: stepFor(2, exact),
    })
    expect(sourceActivity(fwd, c)).toBeGreaterThan(0)
    expect(exact).toBeLessThan(0)
    expect(nudgeSlope(net, x, 7, c, w, 0.001)).toBeCloseTo(exact, 4)
    // Downhill all the way: a stronger connection always lifts the 7 output toward 1.
    const curve = slice(net, x, 7, c, w - 2, w + 4)
    for (let i = 1; i < curve.length; i++) expect(curve[i].mistake).toBeLessThan(curve[i - 1].mistake)
    // One step at learning rate 2 lowers the mistake.
    expect(mistakeAt(net, x, 7, c, w + stepFor(2, exact))).toBeLessThan(mistakeAt(net, x, 7, c, w))
  })
})
