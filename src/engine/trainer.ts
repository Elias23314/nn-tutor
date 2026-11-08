// The playground's training session: the network, its examples, and everything it has done so far.
// Plain TypeScript so it runs the same in a Web Worker and in tests.
import {
  BATCH_SIZE,
  DEFAULT_ETA,
  DEFAULT_SEED,
  DIGITS,
  cloneNetwork,
  createNetwork,
  forward,
  predict,
  trainStep,
  type Network,
} from './network'
import { batchOrder, getSample, labelCounts, toBatch, type Dataset, type Mnist } from './mnist'

export interface Relabel {
  from: number
  to: number
  /** Fraction of `from` examples that get the wrong label. */
  fraction: number
}

export interface ExampleChanges {
  /** Digits taken out of the training examples entirely. */
  removed: number[]
  relabel: Relabel | null
}

export const NO_CHANGES: ExampleChanges = { removed: [], relabel: null }

export interface Point {
  round: number
  value: number
}

export interface TestResult {
  round: number
  accuracy: number
  perDigit: number[]
  /** confusion[actual][predicted] */
  confusion: number[][]
  /** Neurons in hidden layer 2 that never fire on any test digit. */
  silentHidden2: number
  /** Average of the largest output across test digits. Near 0 means it has stopped firing. */
  meanTopOutput: number
  /** Its most common mistake, shown by the test digit it got wrong most confidently. */
  topMistake: Mistake | null
}

export interface Mistake {
  actual: number
  said: number
  /** How many test digits it made this same mistake on. */
  count: number
  /** The most confident example of it. */
  image: Float32Array
  confidence: number
  outputForActual: number
}

export interface Collapse {
  eta: number
  lossFrom: number
  lossTo: number
  accuracy: number
  silentHidden2: number
}

/** Rounds of recent mistakes averaged for the "recent average". */
const RECENT = 50

export class Trainer {
  readonly mnist: Mnist
  readonly seed: number
  eta = DEFAULT_ETA
  changes: ExampleChanges = NO_CHANGES
  train: Dataset
  network!: Network
  initialNetwork!: Network
  round = 0
  losses: number[] = []
  accuracyHistory: Point[] = []
  /** Accuracy on each digit's test examples over time: digitHistory[d]. */
  digitHistory: Point[][] = []
  lastTest: TestResult | null = null
  /** Goes up on every reset. */
  session = 0
  /** η in effect for each round, so a collapse can be blamed on the right learning rate. */
  private etaAtRound: number[] = []
  private order: number[][] = []
  private epoch = 0

  constructor(mnist: Mnist, seed = DEFAULT_SEED) {
    this.mnist = mnist
    this.seed = seed
    this.train = mnist.train
    this.reset()
  }

  reset(): void {
    this.network = createNetwork(undefined, this.seed)
    this.initialNetwork = cloneNetwork(this.network)
    this.session++
    this.round = 0
    this.epoch = 0
    this.losses = []
    this.etaAtRound = []
    this.order = batchOrder(this.train.count, BATCH_SIZE, this.seed)
    this.accuracyHistory = []
    this.digitHistory = Array.from({ length: DIGITS }, () => [])
    this.lastTest = null
    this.test()
  }

  /** Changes the training examples and starts practice over, since its experience is different. */
  setExamples(changes: ExampleChanges): void {
    this.changes = changes
    this.train = applyChanges(this.mnist.train, changes, this.seed)
    this.reset()
  }

  /** Rounds per pass through the current examples. */
  get roundsPerPass(): number {
    return Math.ceil(this.train.count / BATCH_SIZE)
  }

  /** One round of practice: one batch of 32 examples. */
  practiceOnce(): number {
    const indexInPass = this.round % this.roundsPerPass
    if (indexInPass === 0 && this.round > 0) {
      this.epoch++
      this.order = batchOrder(this.train.count, BATCH_SIZE, this.seed + this.epoch)
    }
    const step = trainStep(this.network, toBatch(this.train, this.order[indexInPass]), this.eta)
    this.network = step.network
    this.losses.push(step.loss)
    this.etaAtRound.push(this.eta)
    this.round++
    return step.loss
  }

  /** Average mistake over the last 50 rounds. */
  get recentLoss(): number | null {
    if (this.losses.length === 0) return null
    const recent = this.losses.slice(-RECENT)
    return recent.reduce((a, b) => a + b, 0) / recent.length
  }

  /** The mistake curve, averaged over windows so it stays readable. */
  lossHistory(maxPoints = 300): Point[] {
    const n = this.losses.length
    const window = Math.max(1, Math.ceil(n / maxPoints))
    const points: Point[] = []
    for (let start = 0; start < n; start += window) {
      const slice = this.losses.slice(start, start + window)
      points.push({ round: start + slice.length, value: slice.reduce((a, b) => a + b, 0) / slice.length })
    }
    return points
  }

  /** Scores the network on all 10,000 test digits. */
  test(): TestResult {
    const { test } = this.mnist
    const confusion = Array.from({ length: DIGITS }, () => new Array<number>(DIGITS).fill(0))
    const hidden2Fired = new Array<boolean>(this.network.layers[1].outSize).fill(false)
    // For each (actual, said) mistake, the test digit it was most confident about.
    const surest = new Array<{ k: number; confidence: number; outputForActual: number } | null>(DIGITS * DIGITS).fill(null)
    let correct = 0
    let topSum = 0
    for (let k = 0; k < test.count; k++) {
      const { x, label } = getSample(test, k)
      const fwd = forward(this.network, x)
      const guess = predict(fwd.yHat)
      confusion[label][guess]++
      if (guess === label) correct++
      else {
        const slot = label * DIGITS + guess
        const best = surest[slot]
        if (!best || fwd.yHat[guess] > best.confidence) {
          surest[slot] = { k, confidence: fwd.yHat[guess], outputForActual: fwd.yHat[label] }
        }
      }
      topSum += fwd.yHat[guess]
      fwd.layers[1].a.forEach((a, j) => {
        if (a > 0) hidden2Fired[j] = true
      })
    }
    const perDigit = confusion.map((row, d) => row[d] / Math.max(1, row.reduce((a, b) => a + b, 0)))

    let topMistake: Mistake | null = null
    confusion.forEach((row, actual) =>
      row.forEach((count, said) => {
        const best = surest[actual * DIGITS + said]
        if (actual === said || !best || (topMistake && count <= topMistake.count)) return
        const image = Float32Array.from(getSample(test, best.k).x)
        topMistake = { actual, said, count, image, confidence: best.confidence, outputForActual: best.outputForActual }
      }),
    )

    const result: TestResult = {
      round: this.round,
      accuracy: correct / test.count,
      perDigit,
      confusion,
      silentHidden2: hidden2Fired.filter((f) => !f).length,
      meanTopOutput: topSum / test.count,
      topMistake,
    }
    this.lastTest = result
    record(this.accuracyHistory, this.round, result.accuracy)
    perDigit.forEach((p, d) => record(this.digitHistory[d], this.round, p))
    return result
  }

  /**
   * Training has collapsed when the outputs have all gone quiet: the mistake is low-ish
   * (every goal of 0 is met) but it reads almost nothing.
   */
  collapse(): Collapse | null {
    const t = this.lastTest
    if (!t || this.round < 5 || t.accuracy > 0.3 || t.meanTopOutput > 0.35) return null
    return {
      eta: Math.max(...this.etaAtRound),
      lossFrom: this.losses[0],
      lossTo: this.recentLoss ?? this.losses[0],
      accuracy: t.accuracy,
      silentHidden2: t.silentHidden2,
    }
  }

  setWeight(layer: number, from: number, to: number, value: number): void {
    const next = cloneNetwork(this.network)
    const l = next.layers[layer]
    l.weights[to * l.inSize + from] = value
    this.network = next
  }

  labelCounts(): number[] {
    return labelCounts(this.train)
  }
}

/** Adds a point, replacing the last one if it's for the same round. */
function record(points: Point[], round: number, value: number) {
  if (points.at(-1)?.round === round) points[points.length - 1] = { round, value }
  else points.push({ round, value })
}

/**
 * The first round at which each digit was read correctly at least `threshold` of the time,
 * or null if it hasn't been yet.
 */
export function firstReached(digitHistory: Point[][], threshold = 0.8): (number | null)[] {
  return digitHistory.map((points) => points.find((p) => p.value >= threshold)?.round ?? null)
}

/** Removes digits and mislabels a fraction of one digit (the same examples every time for a seed). */
export function applyChanges(train: Dataset, changes: ExampleChanges, seed = DEFAULT_SEED): Dataset {
  const removed = new Set(changes.removed)
  const keep: number[] = []
  for (let k = 0; k < train.count; k++) if (!removed.has(train.labels[k])) keep.push(k)
  if (keep.length === train.count && !changes.relabel) return train

  const images = new Float32Array(keep.length * 784)
  const labels = new Uint8Array(keep.length)
  keep.forEach((src, dst) => {
    images.set(train.images.subarray(src * 784, (src + 1) * 784), dst * 784)
    labels[dst] = train.labels[src]
  })

  const relabel = changes.relabel
  if (relabel && !removed.has(relabel.from)) {
    // Deterministic choice: hash each example's position with the seed.
    let state = seed * 2654435761
    for (let k = 0; k < labels.length; k++) {
      if (labels[k] !== relabel.from) continue
      state = (Math.imul(state ^ (state >>> 15), 2246822519) + k) >>> 0
      if (state / 4294967296 < relabel.fraction) labels[k] = relabel.to
    }
  }
  return { count: keep.length, images, labels }
}
