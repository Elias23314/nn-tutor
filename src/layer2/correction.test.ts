import { describe, expect, it } from 'vitest'
import { createNetwork, createRng, forward, loss, PIXELS } from '../engine/network'
import { correct } from './correction'

/** A digit-like image: a block of ink, the rest blank. */
function image(seed: number): Float64Array {
  const rand = createRng(seed)
  const x = new Float64Array(PIXELS)
  for (let r = 6; r < 22; r++) for (let c = 10; c < 18; c++) if (rand() < 0.8) x[r * 28 + c] = 0.3 + 0.7 * rand()
  return x
}

describe('correct', () => {
  const net = createNetwork()
  const x = image(1)
  const c = correct(net, x, 7)

  it('never changes a connection from a blank pixel', () => {
    expect(c.changedFromBlank).toBe(0)
  })

  it('changes exactly the connections from inked pixels into firing neurons', () => {
    expect(c.changedFromPixels).toBe(c.inkPixels * c.firing1)
  })

  it('changes only some of the 13,002 numbers', () => {
    expect(c.total).toBe(13_002)
    expect(c.changed).toBeGreaterThan(c.changedFromPixels)
    expect(c.changed).toBeLessThan(c.total)
  })

  it('lowers the mistake on that example', () => {
    expect(c.lossBefore).toBeCloseTo(loss(forward(net, x).yHat, 7), 12)
    expect(c.lossAfter).toBeLessThan(c.lossBefore)
    expect(c.after.yHat[7]).toBeGreaterThan(c.before.yHat[7])
  })

  it('leaves the original network alone', () => {
    expect(net).toEqual(createNetwork())
  })
})
