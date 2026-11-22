/// <reference types="node" />
// Layer 4's story on the real first training 7, before any practice.
import { readFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'
import { createNetwork, forward, neuronName } from '../engine/network'
import { getSample, loadMnist } from '../engine/mnist'
import { focusConnection, nudgeSlope } from '../layer3/slope'
import { deepConnection, gradientOf, outputChain, pathsFromPixel } from './chain'

const dir = fileURLToPath(new URL('../../public/mnist/', import.meta.url))

describe('the chain rule on the first training 7', () => {
  it('matches nudging and backpropagation', async () => {
    const mnist = await loadMnist(async (name) => {
      const buf = await readFile(dir + name)
      return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer
    })
    let index = 0
    while (mnist.train.labels[index] !== 7) index++
    const x = getSample(mnist.train, index).x
    const net = createNetwork()
    const c = focusConnection(net, x, 7)
    const ch = outputChain(net, x, 7, c)
    expect(ch.gradient).toBeCloseTo(gradientOf(net, x, 7, c), 12)
    expect(nudgeSlope(net, x, 7, c, ch.w, 0.001)).toBeCloseTo(ch.gradient, 4)

    const d = deepConnection(net, x, 7)
    const p = pathsFromPixel(net, forward(net, x), 7, d.from, d.to)
    console.log({
      chain: [ch.dLdy, ch.dydz, ch.dzdw, ch.gradient],
      deep: `pixel (${Math.floor(d.from / 28) + 1}, ${(d.from % 28) + 1}) → ${neuronName(1, d.to)}`,
      silentPaths: p.silentPaths,
      total: p.total,
    })
    expect(p.total).toBeCloseTo(gradientOf(net, x, 7, d), 12)
    expect(p.silentPaths).toBeGreaterThan(0)
  })
})
