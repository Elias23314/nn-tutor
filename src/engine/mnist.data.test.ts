/// <reference types="node" />
// Checks against the real MNIST files in public/mnist.
import { readFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'
import { BATCH_SIZE, createNetwork, trainStep } from './network'
import { batchOrder, evaluate, labelCounts, loadMnist, toBatch, type Mnist } from './mnist'

const dir = fileURLToPath(new URL('../../public/mnist/', import.meta.url))
const load = () => loadMnist(async (name) => {
  const buf = await readFile(dir + name)
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer
})

describe('real MNIST', () => {
  let mnist: Mnist

  it('loads 60,000 training and 10,000 test digits', async () => {
    mnist = await load()
    expect(mnist.train.count).toBe(60_000)
    expect(mnist.test.count).toBe(10_000)
    // The playground's "Its whole experience" counts.
    expect(labelCounts(mnist.train)).toEqual([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949])
  })

  it('reads most test digits after one pass of 1,875 rounds', { timeout: 120_000 }, () => {
    let net = createNetwork()
    const before = evaluate(net, mnist.test).accuracy
    for (const indices of batchOrder(mnist.train.count, BATCH_SIZE, 1)) {
      net = trainStep(net, toBatch(mnist.train, indices)).network
    }
    const after = evaluate(net, mnist.test)
    console.log(`test accuracy: ${(before * 100).toFixed(1)}% → ${(after.accuracy * 100).toFixed(1)}%`,
      'per digit:', after.perDigit.map((p) => p.toFixed(2)).join(' '))
    // The mockups show 92.7% at this point.
    expect(after.accuracy).toBeGreaterThan(0.9)
    // No output stuck silent: every digit is learned.
    for (const p of after.perDigit) expect(p).toBeGreaterThan(0.75)
  })
})
