import { describe, expect, it } from 'vitest'
import type { Point } from '../engine/trainer'
import { confusedWith, finishOrder, mostCommonGuess } from './story'

/** Builds a digit history from accuracies at rounds 0, 100, 200, ... */
const history = (...rows: number[][]): Point[][] => rows.map((vals) => vals.map((value, i) => ({ round: i * 100, value })))

describe('finishOrder', () => {
  it('names the digit that reached 80% last, once all have', () => {
    const digits = history(
      [0.1, 0.85, 0.9],
      [0.1, 0.5, 0.82], // last, at round 200
      ...Array.from({ length: 8 }, () => [0.1, 0.81, 0.9]),
    )
    const o = finishOrder(digits, 200, 1875)
    expect(o.known).toBe(true)
    expect(o.last).toBe(1)
    expect(o.reached[1]).toBe(200)
    expect(o.behind).toEqual([])
  })

  it('counts a digit as reached the first time it gets to 80%, even if it dips after', () => {
    const digits = history([0.1, 0.8, 0.7], ...Array.from({ length: 9 }, () => [0.9, 0.9, 0.9]))
    expect(finishOrder(digits, 200, 1875).reached[0]).toBe(100)
  })

  it('is not known yet while a digit is behind, before one pass', () => {
    const digits = history([0.1, 0.6], [0.1, 0.4], ...Array.from({ length: 8 }, () => [0.1, 0.9]))
    const o = finishOrder(digits, 100, 1875)
    expect(o.known).toBe(false)
    expect(o.behind).toEqual([0, 1])
    expect(o.last).toBe(1) // furthest behind
  })

  it('is known after one pass even if a digit never made it', () => {
    const digits = history([0.1, 0.6], ...Array.from({ length: 9 }, () => [0.1, 0.9]))
    const o = finishOrder(digits, 1875, 1875)
    expect(o.known).toBe(true)
    expect(o.last).toBe(0)
  })
})

it('confusedWith lists the most common mix-ups for a digit', () => {
  const confusion = Array.from({ length: 10 }, () => new Array<number>(10).fill(0))
  confusion[5] = [1, 0, 2, 33, 0, 800, 9, 0, 36, 4]
  expect(confusedWith(confusion, 5)).toEqual([
    { digit: 8, count: 36 },
    { digit: 3, count: 33 },
  ])
})

it('mostCommonGuess finds the digit it says most', () => {
  expect(mostCommonGuess([0, 0, 0, 5, 0, 4, 0, 0, 4, 0])).toEqual({ digit: 0, count: 7 })
})
