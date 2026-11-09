// Layer 3's subject: hold every connection still but one, and find how the mistake changes with it.
import { backward, cloneNetwork, forward, loss, type Network } from '../engine/network'
import type { Connection } from '../playground/connection'

/**
 * The connection layer 3 zooms in on: from the hidden-layer-2 neuron that fires hardest for this
 * example to the correct output. A silent source would make the connection irrelevant (steepness 0).
 */
export function focusConnection(net: Network, x: ArrayLike<number>, label: number): Connection {
  const a = forward(net, x).layers[1].a
  let from = 0
  for (let j = 1; j < a.length; j++) if (a[j] > a[from]) from = j
  return { layer: 2, from, to: label }
}

/** A copy of the network with one connection set to `w`. */
export function withStrength(net: Network, c: Connection, w: number): Network {
  const copy = cloneNetwork(net)
  const l = copy.layers[c.layer]
  l.weights[c.to * l.inSize + c.from] = w
  return copy
}

/** The mistake on one example when connection `c` has strength `w` and everything else is held still. */
export const mistakeAt = (net: Network, x: ArrayLike<number>, label: number, c: Connection, w: number) =>
  loss(forward(withStrength(net, c, w), x).yHat, label)

/**
 * The steepness found by nudging: (mistake after − mistake before) ÷ nudge.
 * This is what a person would do by hand, one connection at a time.
 */
export function nudgeSlope(net: Network, x: ArrayLike<number>, label: number, c: Connection, w: number, nudge: number): number {
  return (mistakeAt(net, x, label, c, w + nudge) - mistakeAt(net, x, label, c, w)) / nudge
}

/** The exact steepness, from backpropagation (layer 4's subject). */
export function exactSlope(net: Network, x: ArrayLike<number>, label: number, c: Connection, w: number): number {
  const at = withStrength(net, c, w)
  const g = backward(at, forward(at, x), label).layers[c.layer]
  return g.weights[c.to * g.inSize + c.from]
}

/** The step gradient descent takes: −(learning rate) × steepness. */
export const stepFor = (eta: number, steepness: number) => -eta * steepness

/** The mistake for `n` evenly spaced strengths from `from` to `to`. */
export function slice(net: Network, x: ArrayLike<number>, label: number, c: Connection, from: number, to: number, n = 121) {
  return Array.from({ length: n }, (_, i) => {
    const w = from + ((to - from) * i) / (n - 1)
    return { w, mistake: mistakeAt(net, x, label, c, w) }
  })
}
