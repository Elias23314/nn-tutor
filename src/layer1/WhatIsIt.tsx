// Layer 1: what a neural network is, and how a handwritten digit becomes the numbers it reads.
import { useMemo, useState, type KeyboardEvent } from 'react'
import { IMAGE_SIDE, PIXELS, forward, neuronName, predict, type ForwardResult, type Network } from '../engine/network'
import { fmt, thousands } from '../playground/connection'
import { NetworkDiagram } from '../playground/NetworkDiagram'
import { OutputBars } from '../playground/OutputBars'
import { InputLayer } from './InputLayer'
import './layer1.css'

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

/** Step 4's switch between the network before any practice and after one pass over its examples. */
export interface Practice {
  view: 'before' | 'after'
  setView: (view: 'before' | 'after') => void
  round: number
  total: number
}

/** The whole route, shown before the steps fill it in. */
const ROUTE: [string, string][] = [
  ['A picture', 'a handwritten digit'],
  ['784 numbers', 'step 1'],
  ['784 inputs', 'step 2'],
  ['Layers of neurons', 'step 3'],
  ['10 outputs', 'step 4'],
  ['Its guess', 'the biggest output'],
]

export function WhatIsIt({
  image,
  network,
  digit,
  practice,
}: {
  image: Float32Array
  network: Network
  digit: number
  practice: Practice
}) {
  const fwd = useMemo(() => forward(network, image), [network, image])
  const [pick, setPick] = useState(() => firstPick(image))
  const [neuron, setNeuron] = useState(() => {
    const a = fwd.layers[0].a
    let best = 0
    for (let j = 1; j < a.length; j++) if (a[j] > a[best]) best = j
    return best
  })
  return (
    <section className="intro">
      <p className="body intro-lede">
        A neural network is a program that turns numbers into other numbers. Nobody writes its rules. Instead it has
        thousands of adjustable <b>connection strengths</b>, and practice tunes them. This one takes in a picture of a
        handwritten digit and gives back ten numbers, one for each digit it could be.
      </p>

      <ol className="route" aria-label="The route from a picture to a guess">
        {ROUTE.map(([what, where], i) => (
          <li key={what}>
            {i > 0 && (
              <span className="route-arrow" aria-hidden="true">
                →
              </span>
            )}
            <span className="route-box">
              <b>{what}</b>
              <span>{where}</span>
            </span>
          </li>
        ))}
      </ol>

      <div className="panel">
        <div className="ptitle" style={{ flexWrap: 'wrap' }}>
          <span className="h3">
            <span className="fnum-lg">1</span> A digit becomes 784 numbers
          </span>
          <span className="pick-hint">Click any pixel to pick it</span>
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
          top={1}
          selected={{ layer: 0, from: pick, to: neuron }}
          marks={[{ layer: 1, index: neuron }]}
          caption={`The ${digit} flowing through the whole network. To keep it readable, only the strongest connection into each neuron is drawn. The outlined line runs from your pixel to neuron ${neuronName(1, neuron)}.`}
        />
        <div className="stack">
          <div className="panel">
            <div className="h3" style={{ marginBottom: 10 }}>
              <span className="fnum-lg">3</span> The numbers flow through layers of neurons
            </div>
            <p className="body">
              A <b>neuron</b> takes in numbers, multiplies each one by the strength of its connection, and adds them up. If
              the total is positive it <b>fires</b> and passes the total on. If not, it stays <b>silent</b> and passes on 0.
            </p>
            <NeuronSum image={image} network={network} fwd={fwd} pick={pick} neuron={neuron} setNeuron={setNeuron} />
            <ul className="intro-list body">
              <li>
                <b>Hidden layer 1:</b> 16 neurons, each connected to all 784 inputs.
              </li>
              <li>
                <b>Hidden layer 2:</b> 16 neurons, each connected to all of hidden layer 1.
              </li>
              <li>
                <b>Outputs:</b> 10 neurons, one per digit. Each gives a number between 0 and 1.
              </li>
            </ul>
            <p className="body" style={{ marginTop: 10 }}>
              &ldquo;Hidden&rdquo; only means you never see those layers from outside: just pixels in, ten numbers out.
            </p>
          </div>
          <Outputs fwd={fwd} digit={digit} practice={practice} />
        </div>
      </div>
    </section>
  )
}

/** One neuron's sum, worked through for the picked pixel. */
function NeuronSum({
  image,
  network,
  fwd,
  pick,
  neuron,
  setNeuron,
}: {
  image: Float32Array
  network: Network
  fwd: ForwardResult
  pick: number
  neuron: number
  setNeuron: (j: number) => void
}) {
  const layer = network.layers[0]
  const name = neuronName(1, neuron)
  const v = image[pick]
  const w = layer.weights[neuron * layer.inSize + pick]
  const a = fwd.layers[0].a[neuron]
  const z = fwd.layers[0].z[neuron]
  const inked = Array.from(image).filter((x) => x > 0).length
  return (
    <div className="neuron-sum">
      <div className="sub" id="neuron-q">
        See it for one neuron in hidden layer 1
      </div>
      <div className="chips neuron-chips" role="group" aria-labelledby="neuron-q">
        {Array.from({ length: layer.outSize }, (_, j) => (
          <button key={j} className="chip" aria-pressed={j === neuron} aria-label={`Neuron ${neuronName(1, j)}`} onClick={() => setNeuron(j)}>
            {neuronName(1, j)}
          </button>
        ))}
      </div>
      <p className="body" aria-live="polite">
        Neuron {name} receives all 784 inputs. From your pixel (input #{thousands(pick + 1)}) it gets{' '}
        <span className="mono">
          {fmt(v, 2)} × {fmt(w, 3)} = {fmt(v * w, 3)}
        </span>
        . Only the {inked} inputs with ink add anything; the other {PIXELS - inked} are 0. Adding it all up, along with a
        small starting amount of the neuron&rsquo;s own, the total is <span className="mono">{fmt(z, 2)}</span>, so it{' '}
        {a > 0 ? (
          <>
            <b>fires</b> and passes <span className="mono">{fmt(a, 2)}</span> on to hidden layer 2.
          </>
        ) : (
          <>
            stays <b>silent</b> and passes on <span className="mono">0</span>.
          </>
        )}
      </p>
    </div>
  )
}

function Outputs({ fwd, digit, practice }: { fwd: ForwardResult; digit: number; practice: Practice }) {
  const guess = predict(fwd.yHat)
  const { view, setView, round, total } = practice
  const practicing = view === 'after' && round < total
  return (
    <div className="panel">
      <div className="ptitle" style={{ flexWrap: 'wrap' }}>
        <span className="h3">
          <span className="fnum-lg">4</span> The biggest output is its guess
        </span>
        <div className="chips" role="group" aria-label="Which network to show">
          <button className="chip" aria-pressed={view === 'before'} onClick={() => setView('before')}>
            Before practice
          </button>
          <button className="chip" aria-pressed={view === 'after'} onClick={() => setView('after')}>
            After practice
          </button>
        </div>
      </div>
      <p className="body" style={{ marginBottom: 12 }}>
        Each bar is one output. Close to 1 means &ldquo;I think it&rsquo;s this digit&rdquo;; close to 0 means &ldquo;I
        don&rsquo;t&rdquo;.
      </p>
      <OutputBars yHat={fwd.yHat} />
      <p className="body" style={{ marginTop: 12 }} aria-live="polite">
        {view === 'before' ? (
          <>
            Before any practice, its connection strengths are random. Every output sits near 0.5, and its guess for this{' '}
            {digit} is {guess}
            {guess === digit ? ', which is only luck.' : ': wrong.'} Switch to <b>After practice</b> to see the same network
            once it has learned.
          </>
        ) : practicing ? (
          <>
            Practicing: round {thousands(round)} of {thousands(total)}. Right now it says {guess} (
            <span className="mono">{fmt(fwd.yHat[guess], 2)}</span>).
          </>
        ) : (
          <>
            After practicing on 60,000 labeled examples, it says {guess} (
            <span className="mono">{fmt(fwd.yHat[guess], 2)}</span>) for this {digit}. Nobody rewrote it: the only things
            that changed are its connection strengths. Layer 2 shows how.
          </>
        )}
      </p>
    </div>
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
  // The picked pixel's neighbors in the long list the network reads.
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
        <p className="callout">
          That list is all the network ever sees. It has no idea the numbers came from a picture, or which ones were next
          to each other.
        </p>
      </div>
    </div>
  )
}
