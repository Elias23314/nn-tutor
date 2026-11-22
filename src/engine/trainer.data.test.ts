/// <reference types="node" />
// The playground's "Things to try", checked against the real MNIST data.
import { readFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { beforeAll, describe, expect, it } from 'vitest'
import { loadMnist, type Mnist } from './mnist'
import { Trainer, applyChanges, firstReached } from './trainer'

const dir = fileURLToPath(new URL('../../public/mnist/', import.meta.url))
let mnist: Mnist

beforeAll(async () => {
  mnist = await loadMnist(async (name) => {
    const buf = await readFile(dir + name)
    return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer
  })
})

describe('Trainer', () => {
  it('starts at round 0 with a test score, and practices one batch per round', () => {
    const t = new Trainer(mnist)
    expect(t.round).toBe(0)
    expect(t.roundsPerPass).toBe(1_875)
    expect(t.accuracyHistory).toHaveLength(1)
    t.practiceOnce()
    t.practiceOnce()
    expect(t.round).toBe(2)
    expect(t.recentLoss).toBeGreaterThan(0)
  })

  it('tracks each digit, and finds when each first reaches 80%', () => {
    const t = new Trainer(mnist)
    for (let r = 0; r < 600; r++) {
      t.practiceOnce()
      if (t.round % 100 === 0) t.test()
    }
    expect(t.digitHistory).toHaveLength(10)
    expect(t.digitHistory[0].map((p) => p.round)).toEqual([0, 100, 200, 300, 400, 500, 600])
    const reached = firstReached(t.digitHistory)
    expect(reached.filter((r) => r !== null).length).toBeGreaterThan(5)
    for (const r of reached) if (r !== null) expect(r % 100).toBe(0)
  })

  it('finds its most common mistake and its most confident example', () => {
    const t = new Trainer(mnist)
    for (let r = 0; r < 600; r++) t.practiceOnce()
    const { topMistake, confusion } = t.test()
    expect(topMistake).not.toBeNull()
    const m = topMistake!
    expect(m.actual).not.toBe(m.said)
    expect(m.count).toBe(confusion[m.actual][m.said])
    const offDiagonal = confusion.flatMap((row, a) => row.filter((_, s) => s !== a))
    expect(m.count).toBe(Math.max(...offDiagonal))
    expect(m.image).toHaveLength(784)
    expect(m.confidence).toBeGreaterThan(m.outputForActual)
  })

  it('collapses at learning rate 40, as the mockup describes', () => {
    const t = new Trainer(mnist)
    t.eta = 40
    for (let r = 0; r < 100; r++) t.practiceOnce()
    t.test()
    const c = t.collapse()
    expect(c).not.toBeNull()
    expect(c!.eta).toBe(40)
    expect(c!.lossTo).toBeLessThan(c!.lossFrom)
    expect(c!.accuracy).toBeLessThan(0.3)
    expect(c!.silentHidden2).toBeGreaterThan(0)
  })

  it('does not report a collapse at learning rate 2', () => {
    const t = new Trainer(mnist)
    for (let r = 0; r < 200; r++) t.practiceOnce()
    t.test()
    expect(t.collapse()).toBeNull()
    expect(t.lastTest!.accuracy).toBeGreaterThan(0.5)
  })

  it('removing a digit takes it out of the examples and starts over', () => {
    const t = new Trainer(mnist)
    t.practiceOnce()
    t.setExamples({ removed: [7], relabel: null })
    expect(t.round).toBe(0)
    expect(t.labelCounts()[7]).toBe(0)
    expect(t.train.count).toBe(60_000 - 6_265)
  })

  it('setWeight changes one connection without touching the rest', () => {
    const t = new Trainer(mnist)
    const before = t.network
    t.setWeight(2, 6, 7, 0.5)
    expect(t.network.layers[2].weights[7 * 16 + 6]).toBe(0.5)
    expect(before.layers[2].weights[7 * 16 + 6]).not.toBe(0.5)
  })
})

it('relabels about the requested fraction, the same examples each time', () => {
  const a = applyChanges(mnist.train, { removed: [], relabel: { from: 3, to: 8, fraction: 0.5 } })
  const b = applyChanges(mnist.train, { removed: [], relabel: { from: 3, to: 8, fraction: 0.5 } })
  const threes = a.labels.filter((l) => l === 3).length
  expect(threes / 6_131).toBeGreaterThan(0.4)
  expect(threes / 6_131).toBeLessThan(0.6)
  expect(a.labels).toEqual(b.labels)
  expect(a.labels.length).toBe(60_000)
})
