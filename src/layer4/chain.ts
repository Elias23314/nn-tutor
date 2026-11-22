// Layer 4's subject: the chain rule, written out for one connection, and the sum over paths for a deep one.
import { backward, forward, loss, type ForwardResult, type Network } from '../engine/network'
import type { Connection } from '../playground/connection'

/** Every local slope on the single path from a hidden-layer-2 → output connection to the mistake. */
export interface OutputChain {
  /** The source neuron's firing: ∂z/∂w. */
  a: number
  w: number
  /** The weighted sum into the output. */
  z: number
  yHat: number
  goal: number
  loss: number
  /** ∂L/∂ŷ = (2/10)(ŷ − goal). */
  dLdy: number
  /** ∂ŷ/∂z = ŷ(1 − ŷ). */
  dydz: number
  /** ∂z/∂w = a. */
  dzdw: number
  /** Their product, ∂L/∂w. */
  gradient: number
}

export function outputChain(net: Network, x: ArrayLike<number>, label: number, c: Connection): OutputChain {
  if (c.layer !== net.layers.length - 1) throw new Error('outputChain needs a connection into an output')
  const fwd = forward(net, x)
  const last = fwd.layers[c.layer]
  const outputs = net.layers[c.layer].outSize
  const a = fwd.layers[c.layer - 1].a[c.from]
  const yHat = fwd.yHat[c.to]
  const goal = c.to === label ? 1 : 0
  const dLdy = (2 / outputs) * (yHat - goal)
  const dydz = yHat * (1 - yHat)
  const l = net.layers[c.layer]
  return {
    a,
    w: l.weights[c.to * l.inSize + c.from],
    z: last.z[c.to],
    yHat,
    goal,
    loss: loss(fwd.yHat, label),
    dLdy,
    dydz,
    dzdw: a,
    gradient: dLdy * dydz * a,
  }
}

/** The gradient of every weight, from backpropagation. */
export const gradients = (net: Network, x: ArrayLike<number>, label: number) => backward(net, forward(net, x), label)

export const gradientOf = (net: Network, x: ArrayLike<number>, label: number, c: Connection) => {
  const g = gradients(net, x, label).layers[c.layer]
  return g.weights[c.to * g.inSize + c.from]
}

/** One route from a pixel → hidden-layer-1 connection to the mistake: through neuron m in layer 2, then output k. */
export interface Path {
  m: number
  k: number
  /** The product of the local slopes along it. 0 when m is silent. */
  value: number
}

/**
 * Every path from the connection pixel → neuron j (hidden layer 1) to the mistake, and their sum:
 * x · r₁ · Σₘ w_jm r_m Σₖ w_mk ŷₖ(1 − ŷₖ) ∂L/∂ŷₖ. The sum is the gradient backpropagation finds.
 */
export function pathsFromPixel(net: Network, fwd: ForwardResult, label: number, pixel: number, j: number) {
  const [l1, l2, l3] = net.layers
  const x = fwd.x[pixel]
  const r1 = fwd.layers[0].a[j] > 0 ? 1 : 0
  const paths: Path[] = []
  for (let m = 0; m < l2.outSize; m++) {
    const r2 = fwd.layers[1].a[m] > 0 ? 1 : 0
    for (let k = 0; k < l3.outSize; k++) {
      const y = fwd.yHat[k]
      const dLdy = (2 / l3.outSize) * (y - (k === label ? 1 : 0))
      const value = x * r1 * l2.weights[m * l2.inSize + j] * r2 * l3.weights[k * l3.inSize + m] * y * (1 - y) * dLdy
      paths.push({ m, k, value })
    }
  }
  const silentHidden2 = Array.from(fwd.layers[1].a).filter((v) => v === 0).length
  return {
    x,
    r1,
    paths,
    silentPaths: silentHidden2 * l3.outSize,
    total: paths.reduce((s, p) => s + p.value, 0),
    inSize: l1.inSize,
  }
}

/** The hidden-layer-1 neuron that fires hardest, and the inked pixel whose connection into it has the largest gradient. */
export function deepConnection(net: Network, x: ArrayLike<number>, label: number): Connection {
  const fwd = forward(net, x)
  const a = fwd.layers[0].a
  let j = 0
  for (let n = 1; n < a.length; n++) if (a[n] > a[j]) j = n
  const g = backward(net, fwd, label).layers[0]
  let from = -1
  for (let i = 0; i < g.inSize; i++) {
    if (x[i] === 0) continue
    if (from < 0 || Math.abs(g.weights[j * g.inSize + i]) > Math.abs(g.weights[j * g.inSize + from])) from = i
  }
  return { layer: 0, from, to: j }
}
