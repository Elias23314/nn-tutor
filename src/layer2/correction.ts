// Layer 2's subject: one correction, slowed down. A single example, one backward pass, one step.
import {
  DEFAULT_ETA,
  applyGradients,
  backward,
  countParams,
  forward,
  loss,
  predict,
  type ForwardResult,
  type Gradients,
  type Network,
} from '../engine/network'

export interface Correction {
  label: number
  /** The network before the step. */
  start: Network
  before: ForwardResult
  grads: Gradients
  network: Network
  after: ForwardResult
  lossBefore: number
  lossAfter: number
  guessBefore: number
  guessAfter: number
  /** How many of the network's numbers (weights and biases) this step changes. */
  changed: number
  total: number
  /** Pixels with any ink. */
  inkPixels: number
  /** Neurons in hidden layer 1 that fire for this example. */
  firing1: number
  /** Pixel → hidden layer 1 connections that change. */
  changedFromPixels: number
  /** Of those, how many start at a blank pixel. Always 0: nothing flowed from them. */
  changedFromBlank: number
}

const nonzero = (values: ArrayLike<number>) => {
  let n = 0
  for (let k = 0; k < values.length; k++) if (values[k] !== 0) n++
  return n
}

/** One gradient-descent step on one example, with everything layer 2 shows about it. */
export function correct(net: Network, x: ArrayLike<number>, label: number, eta = DEFAULT_ETA): Correction {
  const before = forward(net, x)
  const grads = backward(net, before, label)
  const network = applyGradients(net, grads, eta)
  const after = forward(network, x)

  const g0 = grads.layers[0]
  let changedFromPixels = 0
  let changedFromBlank = 0
  for (let j = 0; j < g0.outSize; j++) {
    for (let i = 0; i < g0.inSize; i++) {
      if (g0.weights[j * g0.inSize + i] === 0) continue
      changedFromPixels++
      if (x[i] === 0) changedFromBlank++
    }
  }

  return {
    label,
    start: net,
    before,
    grads,
    network,
    after,
    lossBefore: loss(before.yHat, label),
    lossAfter: loss(after.yHat, label),
    guessBefore: predict(before.yHat),
    guessAfter: predict(after.yHat),
    changed: grads.layers.reduce((n, l) => n + nonzero(l.weights) + nonzero(l.biases), 0),
    total: countParams(net),
    inkPixels: nonzero(x),
    firing1: nonzero(before.layers[0].a),
    changedFromPixels,
    changedFromBlank,
  }
}
