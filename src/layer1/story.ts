// Layer 1's question: which digit does it get right last?
// "Right" means read correctly at least 80% of the time on that digit's test examples.
import { firstReached, type Point } from '../engine/trainer'

export const THRESHOLD = 0.8

export interface FinishOrder {
  /** Whether we know the answer yet: every digit has reached 80%, or one pass is done. */
  known: boolean
  /** The digit it got right last (or is furthest behind, if not every digit has made it). */
  last: number
  /** Round when each digit first reached 80%, or null. */
  reached: (number | null)[]
  /** Digits that haven't reached 80% yet. */
  behind: number[]
}

export function finishOrder(digitHistory: Point[][], round: number, roundsPerPass: number): FinishOrder {
  const reached = firstReached(digitHistory, THRESHOLD)
  const behind = reached.flatMap((r, d) => (r === null ? [d] : []))
  const latest = (d: number) => digitHistory[d].at(-1)?.value ?? 0
  let last: number
  if (behind.length > 0) {
    // Furthest behind: the lowest current accuracy.
    last = behind.reduce((a, b) => (latest(b) < latest(a) ? b : a))
  } else {
    // Latest to arrive; ties go to the one with lower accuracy now.
    last = reached.reduce<number>((a, r, d) => {
      const ra = reached[a]!
      if (r! > ra || (r === ra && latest(d) < latest(a))) return d
      return a
    }, 0)
  }
  const known = behind.length === 0 || round >= roundsPerPass
  return { known, last, reached, behind }
}

/** The digits it most often mistakes `digit` for, most common first. */
export function confusedWith(confusion: number[][], digit: number, top = 2): { digit: number; count: number }[] {
  return confusion[digit]
    .map((count, d) => ({ digit: d, count }))
    .filter((c) => c.digit !== digit && c.count > 0)
    .sort((a, b) => b.count - a.count)
    .slice(0, top)
}

/** The digit it called most of these, and how many. */
export function mostCommonGuess(guesses: number[]): { digit: number; count: number } {
  const counts = new Array<number>(10).fill(0)
  for (const g of guesses) counts[g]++
  const digit = counts.indexOf(Math.max(...counts))
  return { digit, count: counts[digit] }
}
