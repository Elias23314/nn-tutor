import { useEffect, useRef } from 'react'
import { IMAGE_SIDE, PIXELS, neuronName, type Network } from '../engine/network'
import { thousands } from './connection'

const BLUE = [0x25, 0x63, 0xeb]
const ORANGE = [0xc2, 0x41, 0x0c]

/** One neuron's 784 incoming weights (or changes to them) as a picture: blue excites, orange inhibits. */
export function WeightCanvas({ weights, max, name }: { weights: Float64Array; max: number; name: string }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const ctx = ref.current?.getContext('2d')
    if (!ctx) return
    const data = ctx.createImageData(IMAGE_SIDE, IMAGE_SIDE)
    for (let k = 0; k < PIXELS; k++) {
      const w = weights[k]
      const t = Math.min(1, Math.abs(w) / max)
      const color = w >= 0 ? BLUE : ORANGE
      data.data.set([...color.map((c) => Math.round(255 + (c - 255) * t)), 255], k * 4)
    }
    ctx.putImageData(data, 0, 0)
  }, [weights, max])
  return (
    <canvas ref={ref} width={IMAGE_SIDE} height={IMAGE_SIDE} role="img" aria-label={`Connections into neuron ${name}`} />
  )
}

/** The 16 hidden-layer-1 neurons' connections from the pixels, one picture each. */
export function WeightGrid({ network, captions = true, scale }: { network: Network; captions?: boolean; scale?: number }) {
  const layer = network.layers[0]
  const rows = Array.from({ length: layer.outSize }, (_, j) => layer.weights.subarray(j * PIXELS, (j + 1) * PIXELS))
  // One shared scale, so pictures can be compared with each other.
  let max = 1e-9
  if (scale) max = scale
  else for (const w of layer.weights) max = Math.max(max, Math.abs(w))
  return (
    <div className="wgrid">
      {rows.map((w, j) => (
        <div key={j}>
          <WeightCanvas weights={w} max={max} name={neuronName(1, j)} />
          {captions && <div className="cap">{neuronName(1, j)}</div>}
        </div>
      ))}
    </div>
  )
}

export function WeightPictures({ network, round }: { network: Network; round: number }) {
  return (
    <div className="panel">
      <div className="ptitle">
        <span className="h3">What hidden layer 1 responds to</span>
        <span className="small">{round === 0 ? 'Before practice' : `After ${thousands(round)} rounds`}</span>
      </div>
      <WeightGrid network={network} />
      <p className="small" style={{ marginTop: 10 }}>
        Each square is one neuron&rsquo;s 784 connections from the pixels. Blue pixels push it up, orange pixels push it down.
      </p>
    </div>
  )
}
