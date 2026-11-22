import { IMAGE_SIDE, neuronName, weight, type ForwardResult, type Network } from '../engine/network'

/** One connection: `layer` 0 is pixels → hidden 1, 1 is hidden 1 → hidden 2, 2 is hidden 2 → outputs. */
export interface Connection {
  layer: number
  from: number
  to: number
}

/** G → 7, the playground's starting selection. Layer 3 picks its own focus from the example. */
export const G_TO_7: Connection = { layer: 2, from: 6, to: 7 }

export function sourceName(c: Connection): string {
  if (c.layer === 0) return `pixel at row ${Math.floor(c.from / IMAGE_SIDE) + 1}, column ${(c.from % IMAGE_SIDE) + 1}`
  return neuronName(c.layer === 1 ? 1 : 2, c.from)
}

export function targetName(c: Connection): string {
  if (c.layer === 0) return neuronName(1, c.to)
  if (c.layer === 1) return neuronName(2, c.to)
  return String(c.to)
}

export const connectionLabel = (c: Connection) => `${sourceName(c)} → ${targetName(c)}`

export const strength = (net: Network, c: Connection) => weight(net, c.layer, c.from, c.to)

/** What flows into the connection: the source neuron's firing, or the pixel value. */
export function sourceActivity(fwd: ForwardResult, c: Connection): number {
  return c.layer === 0 ? fwd.x[c.from] : fwd.layers[c.layer - 1].a[c.from]
}

export const fmt = (v: number, digits = 3) => {
  const s = v.toFixed(digits)
  return s.startsWith('-') ? `−${s.slice(1)}` : s
}

export const pct = (v: number) => `${(v * 100).toFixed(1)}%`

export const thousands = (v: number) => v.toLocaleString('en-US')

/**
 * A color scale for weight pictures that isn't washed out by a few extreme weights:
 * the 99th percentile of |w|, but at least `floor` so untrained noise stays faint.
 */
export function weightScale(network: Network, floor = 0.3): number {
  const sorted = Float64Array.from(network.layers[0].weights, Math.abs).sort()
  return Math.max(floor, sorted[Math.floor(sorted.length * 0.99)])
}
