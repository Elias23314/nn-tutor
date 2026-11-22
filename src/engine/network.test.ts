import { describe, expect, it } from 'vitest'
import {
  PIXELS,
  backward,
  countParams,
  createNetwork,
  createRng,
  forward,
  loss,
  neuronName,
  pixelIndex,
  predict,
  trainStep,
  type Network,
  type Sample,
} from './network'

const randomImage = (seed: number, size = PIXELS): Float64Array => {
  const rand = createRng(seed)
  return Float64Array.from({ length: size }, () => (rand() < 0.2 ? rand() : 0))
}

describe('shape', () => {
  it('has the 13,002 numbers from the mockups', () => {
    const net = createNetwork()
    expect(net.layers.map((l) => [l.inSize, l.outSize])).toEqual([[784, 16], [16, 16], [16, 10]])
    expect(countParams(net)).toBe(13_002)
    expect(net.layers[0].weights.length).toBe(12_544)
  })

  it('is the same network for the same seed', () => {
    expect(createNetwork().layers[0].weights).toEqual(createNetwork().layers[0].weights)
    expect(createNetwork(undefined, 1).layers[0].weights).not.toEqual(createNetwork(undefined, 2).layers[0].weights)
  })

  it('names neurons and pixels the way the mockups do', () => {
    expect(neuronName(1, 1)).toBe('b')
    expect(neuronName(2, 6)).toBe('G')
    expect(neuronName(1, 15)).toBe('p')
    expect(pixelIndex(1, 1)).toBe(0)
    expect(pixelIndex(13, 10)).toBe(12 * 28 + 9)
  })
})

describe('forward', () => {
  it('gives ten outputs between 0 and 1', () => {
    const { yHat, layers } = forward(createNetwork(), randomImage(1))
    expect(yHat.length).toBe(10)
    for (const v of yHat) {
      expect(v).toBeGreaterThan(0)
      expect(v).toBeLessThan(1)
    }
    for (const a of layers[0].a) expect(a).toBeGreaterThanOrEqual(0)
  })

  it('averages ten squared gaps for the loss', () => {
    const yHat = [0.1, 0, 0, 0, 0, 0, 0.67, 0.31, 0, 0]
    // gaps: 0.1, 0.67, 0.69 (7 is the goal)
    expect(loss(yHat, 7)).toBeCloseTo((0.01 + 0.67 ** 2 + 0.69 ** 2) / 10, 12)
    expect(predict(yHat)).toBe(6)
  })
})

describe('backward', () => {
  const lossAt = (net: Network, x: ArrayLike<number>, label: number) => loss(forward(net, x).yHat, label)

  function checkGradients(net: Network, x: ArrayLike<number>, label: number, pick: (l: number, n: number) => number[]) {
    const grads = backward(net, forward(net, x), label)
    const eps = 1e-6
    net.layers.forEach((layer, l) => {
      for (const kind of ['weights', 'biases'] as const) {
        const params = layer[kind]
        for (const k of pick(l, params.length)) {
          const original = params[k]
          params[k] = original + eps
          const up = lossAt(net, x, label)
          params[k] = original - eps
          const down = lossAt(net, x, label)
          params[k] = original
          expect(grads.layers[l][kind][k], `layer ${l} ${kind}[${k}]`).toBeCloseTo((up - down) / (2 * eps), 7)
        }
      }
    })
  }

  it('agrees with numerical gradients on a small network, every parameter', () => {
    const net = createNetwork([6, 5, 4, 3], 3)
    checkGradients(net, randomImage(4, 6).map((v) => v + 0.3), 2, (_, n) => [...Array(n).keys()])
  })

  it('agrees with numerical gradients on the full network, sampled parameters', () => {
    const net = createNetwork()
    const x = randomImage(5)
    const rand = createRng(9)
    checkGradients(net, x, 7, (_, n) => Array.from({ length: 40 }, () => Math.floor(rand() * n)))
  })

  it('gives 0 to connections from blank pixels', () => {
    const net = createNetwork()
    const x = randomImage(6)
    const grads = backward(net, forward(net, x), 3)
    const blank = x.findIndex((v) => v === 0)
    for (let j = 0; j < 16; j++) expect(grads.layers[0].weights[j * PIXELS + blank]).toBe(0)
  })
})

describe('trainStep', () => {
  const batch: Sample[] = Array.from({ length: 32 }, (_, k) => ({ x: randomImage(100 + k), label: k % 10 }))

  it('lowers the loss on the batch it trained on', () => {
    const step = trainStep(createNetwork(), batch)
    const after = batch.reduce((s, b) => s + loss(forward(step.network, b.x).yHat, b.label), 0) / batch.length
    expect(after).toBeLessThan(step.loss)
  })

  it('does not mutate the input network', () => {
    const net = createNetwork()
    trainStep(net, batch)
    expect(net).toEqual(createNetwork())
  })

  it('learns ten separable patterns', () => {
    const patterns: Sample[] = Array.from({ length: 10 }, (_, d) => {
      const x = new Float64Array(PIXELS)
      for (let p = d * 70; p < d * 70 + 70; p++) x[p] = 1
      return { x, label: d }
    })
    let net = createNetwork()
    for (let round = 0; round < 300; round++) net = trainStep(net, patterns).network
    for (const p of patterns) expect(predict(forward(net, p.x).yHat)).toBe(p.label)
  })
})
