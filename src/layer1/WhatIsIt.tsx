// Layer 1's opening: what a neural network is, and how a handwritten digit becomes the numbers it reads.
import { useMemo, useState, type KeyboardEvent } from 'react'
import { IMAGE_SIDE, PIXELS, countParams, forward, predict, type Network } from '../engine/network'
import { fmt, thousands } from '../playground/connection'
import './layer1.css'
import { NetworkDiagram } from '../playground/NetworkDiagram'
import { OutputBars } from '../playground/OutputBars'
import { InputLayer } from './InputLayer'

const CELL = 10
/** Cells shown around the picked pixel: a 5 × 5 window. */
const R = 2

const ink = (v: number) => {
  const s = 255 - Math.round(v * 255)
  return `rgb(${s},${s},${s})`
}

/** The inked pixel nearest the image's middle: a good first pick. */
function firstPick(x: ArrayLike<number>): number {
  let best = 0
  let bestD = Infinity
  for (let k = 0; k < PIXELS; k++) {
    if (x[k] < 0.9) continue
    const r = Math.floor(k / IMAGE_SIDE) - 13.5
    const c = (k % IMAGE_SIDE) - 13.5
    if (r * r + c * c < bestD) {
      bestD = r * r + c * c
      best = k
    }
  }
  return best
}

export function WhatIsIt({ image, network, digit }: { image: Float32Array; network: Network; digit: number }) {
  const fwd = useMemo(() => forward(network, image), [network, image])
  const guess = predict(fwd.yHat)
  const weights = network.layers.reduce((n, l) => n + l.weights.length, 0)
  const total = countParams(network)
  const [pick, setPick] = useState(() => firstPick(image))
  return (
    <section className="intro" aria-labelledby="intro-h">
      <div className="intro-head">
        <h2 id="intro-h">First: what is a neural network?</h2>
      </div>
      <p className="body intro-lede">
        A neural network is a program that turns numbers into other numbers. Nobody writes its rules. Instead it has
        thousands of adjustable settings, and practice tunes them. This one takes in a picture of a handwritten digit and
        gives back ten numbers, one for each digit it could be.
      </p>

      <div className="panel">
        <div className="ptitle" style={{ flexWrap: 'wrap' }}>
          <span className="h3">
            <span className="fnum-lg">1</span> A digit becomes 784 numbers
          </span>
          <span className="small">Pick any pixel</span>
        </div>
        <PixelNumbers image={image} digit={digit} pick={pick} setPick={setPick} />
      </div>

      <div className="panel sec">
        <div className="ptitle" style={{ flexWrap: 'wrap' }}>
          <span className="h3">
            <span className="fnum-lg">2</span> The 784 numbers become its 784 inputs
          </span>
          <span className="small">The pixel you picked above is highlighted</span>
        </div>
        <InputLayer image={image} network={network} fwd={fwd} pick={pick} />
      </div>

      <div className="grid2 sec">
        <NetworkDiagram
          title={null}
          network={network}
          fwd={fwd}
          caption={`The ${digit} above, flowing through the whole network: inputs on the left, outputs on the right. Thicker lines push harder; only the strongest few into each neuron are drawn.`}
        />
        <div className="stack">
          <div className="panel">
            <div className="h3" style={{ marginBottom: 10 }}>
              <span className="fnum-lg">3</span> The numbers flow through layers of neurons
            </div>
            <p className="body">
              A neuron takes in numbers, weighs each one by the strength of its connection, and adds them up. If the total
              is positive it <b>fires</b> and passes the total on; if not it stays <b>silent</b> and passes on 0.
            </p>
            <ul className="intro-list body">
              <li>
                <b>Hidden layer 1:</b> 16 neurons, each connected to all 784 pixels.
              </li>
              <li>
                <b>Hidden layer 2:</b> 16 neurons, each connected to all of layer 1.
              </li>
              <li>
                <b>Outputs:</b> 10 neurons, one per digit, each squeezed to between 0 and 1.
              </li>
            </ul>
            <p className="body" style={{ marginTop: 10 }}>
              &ldquo;Hidden&rdquo; only means you never see those layers from outside: just pixels in, ten numbers out.
            </p>
          </div>
          <div className="panel">
            <div className="h3" style={{ marginBottom: 10 }}>
              <span className="fnum-lg">4</span> The biggest output is its guess
            </div>
            <OutputBars yHat={fwd.yHat} />
            <p className="body" style={{ marginTop: 12 }}>
              Right now it says {guess} for this {digit}
              {guess === digit ? '.' : ', because it hasn’t practiced yet: its connection strengths start out random.'}{' '}
              Learning means adjusting its {thousands(total)} numbers: {thousands(weights)} connection strengths, plus{' '}
              {total - weights} offsets, one per neuron, that make it easier or harder to fire.
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}

function PixelNumbers({
  image,
  digit,
  pick,
  setPick,
}: {
  image: Float32Array
  digit: number
  pick: number
  setPick: (k: number) => void
}) {
  const row = Math.floor(pick / IMAGE_SIDE)
  const col = pick % IMAGE_SIDE
  const size = IMAGE_SIDE * CELL

  const onKey = (e: KeyboardEvent) => {
    const moves: Record<string, [number, number]> = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] }
    const m = moves[e.key]
    if (!m) return
    e.preventDefault()
    const r = Math.min(IMAGE_SIDE - 1, Math.max(0, row + m[0]))
    const c = Math.min(IMAGE_SIDE - 1, Math.max(0, col + m[1]))
    setPick(r * IMAGE_SIDE + c)
  }

  // The 5 × 5 window of numbers around the picked pixel, clamped to the image.
  const r0 = Math.min(IMAGE_SIDE - 1 - 2 * R, Math.max(0, row - R))
  const c0 = Math.min(IMAGE_SIDE - 1 - 2 * R, Math.max(0, col - R))
  const zoom = Array.from({ length: 2 * R + 1 }, (_, i) => Array.from({ length: 2 * R + 1 }, (_, j) => (r0 + i) * IMAGE_SIDE + c0 + j))
  // The picked pixel's row, as the start of the long list the network reads.
  const listStart = Math.max(0, pick - 3)
  const list = Array.from({ length: 7 }, (_, i) => listStart + i).filter((k) => k < PIXELS)

  return (
    <div className="pixnum">
      <div>
        <svg
          className="pixgrid"
          viewBox={`0 0 ${size} ${size}`}
          width={size}
          height={size}
          tabIndex={0}
          role="application"
          aria-label={`The ${digit}, 28 by 28 pixels. Picked: row ${row + 1}, column ${col + 1}, value ${fmt(image[pick], 2)}. Use the arrow keys to move.`}
          onKeyDown={onKey}
        >
          {Array.from(image, (v, k) => (
            <rect
              key={k}
              x={(k % IMAGE_SIDE) * CELL}
              y={Math.floor(k / IMAGE_SIDE) * CELL}
              width={CELL}
              height={CELL}
              fill={ink(v)}
              onClick={() => setPick(k)}
              onMouseEnter={(e) => e.buttons === 1 && setPick(k)}
            />
          ))}
          <rect x={c0 * CELL} y={r0 * CELL} width={(2 * R + 1) * CELL} height={(2 * R + 1) * CELL} fill="none" stroke="var(--excite)" strokeWidth={1.5} strokeDasharray="3 2" />
          <rect x={col * CELL} y={row * CELL} width={CELL} height={CELL} fill="none" stroke="var(--excite)" strokeWidth={2.5} />
        </svg>
        <p className="small" style={{ marginTop: 6 }}>
          28 × 28 pixels. Click a pixel, or use the arrow keys.
        </p>
      </div>

      <div>
        <div className="sub" style={{ marginBottom: 8 }}>
          Zoomed in, as numbers
        </div>
        <table className="zoom" aria-label="The numbers around the picked pixel">
          <tbody>
            {zoom.map((cells, i) => (
              <tr key={i}>
                {cells.map((k) => (
                  <td
                    key={k}
                    className={k === pick ? 'on' : undefined}
                    style={{ background: ink(image[k]), color: image[k] > 0.5 ? '#fff' : 'var(--ink)' }}
                    onClick={() => setPick(k)}
                  >
                    {fmt(image[k], 2)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="pixtext">
        <p className="body">
          Each pixel is one number: <span className="mono">0.00</span> for blank paper, <span className="mono">1.00</span>{' '}
          for full ink, and in-between for the soft edges of the pen stroke.
        </p>
        <p className="body">
          The picked pixel, row {row + 1}, column {col + 1}, is <span className="mono">{fmt(image[pick], 2)}</span>.
        </p>
        <p className="body">
          Read row by row, the picture becomes one long list of {thousands(PIXELS)} numbers. Your pixel is number{' '}
          <b>{thousands(pick + 1)}</b>:
        </p>
        <div className="numlist mono" aria-hidden="true">
          {list[0] > 0 && <span className="dots">…</span>}
          {list.map((k) => (
            <span key={k} className={k === pick ? 'on' : undefined}>
              {fmt(image[k], 2)}
            </span>
          ))}
          {list.at(-1)! < PIXELS - 1 && <span className="dots">…</span>}
        </div>
        <p className="body">
          That list is all the network ever sees. It has no idea the numbers came from a picture, or which ones were next
          to each other.
        </p>
      </div>
    </div>
  )
}
