// Messages between the playground page and its training worker.
import type { Network } from '../engine/network'
import type { Collapse, ExampleChanges, Point, TestResult } from '../engine/trainer'

export type Speed = 'slow' | 'normal' | 'fast'

export type Command =
  | { type: 'practice'; rounds: number }
  | { type: 'play' }
  | { type: 'pause' }
  | { type: 'reset' }
  | { type: 'setEta'; eta: number }
  | { type: 'setSpeed'; speed: Speed }
  | { type: 'setExamples'; changes: ExampleChanges }
  | { type: 'setWeight'; layer: number; from: number; to: number; value: number }
  /**
   * Reset with these settings, then practice `rounds` rounds (for "Things to try").
   * If `probe` is given, the result includes what the trained network says for it.
   */
  | { type: 'experiment'; id: number; eta: number; changes: ExampleChanges; rounds: number; probe?: Float32Array }

export interface ExperimentResult {
  id: number
  accuracy: number
  collapsed: boolean
  probe: { digit: number; value: number } | null
}

export interface Snapshot {
  /** Goes up on every reset, so the page can tell one run of practice from the next. */
  session: number
  /** Results of recently finished experiments, newest last. */
  experiments: ExperimentResult[]
  round: number
  roundsPerPass: number
  eta: number
  speed: Speed
  playing: boolean
  /** Rounds still to run for a queued practice or experiment. */
  pending: number
  recentLoss: number | null
  lossHistory: Point[]
  accuracyHistory: Point[]
  lastTest: TestResult | null
  collapse: Collapse | null
  labelCounts: number[]
  trainCount: number
  changes: ExampleChanges
  network: Network
  initialNetwork: Network
}

export type Report =
  | { type: 'loading' }
  | { type: 'ready'; sample7: Float32Array }
  | { type: 'state'; state: Snapshot }
  | { type: 'error'; message: string }
