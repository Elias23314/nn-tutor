// Layer 1: what a neural network is, and how a handwritten digit becomes the numbers it reads.
import { useMemo, useState, type KeyboardEvent } from 'react'
import { Bridge } from '../components/LayerParts'
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
  ['784 inputs', 'step 1'],
  ['Weights', 'step 2'],
  ['Neurons', 'step 3'],
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
        A neural network is a computer model inspired by the human brain. Through a series of interconnected pieces it
        learns patterns from data to map inputs to outputs. It is the foundation of what we think of as modern AI
        systems. In the example we will look at, it is given a large number of images of handwritten digits and learns
        patterns to correctly identify which digit it is being given.
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
            <span className="fnum-lg">1</span> Input: the picture becomes 784 numbers
          </span>
          <span className="pick-hint">Click any pixel to pick it</span>
        </div>
        <PixelNumbers image={image} digit={digit} pick={pick} setPick={setPick} />
        <Bridge>
          Those 784 inputs now travel into the network along connections, and every connection has a <b>weight</b>.
          That&rsquo;s step 2.
        </Bridge>
      </div>

      <div className="panel sec">
        <div className="ptitle" style={{ flexWrap: 'wrap' }}>
          <span className="h3">
            <span className="fnum-lg">2</span> Weights: how much each input counts
          </span>
          <span className="small">The pixel you picked above is highlighted</span>
        </div>
        <InputLayer image={image} network={network} fwd={fwd} pick={pick} />
        <Bridge>
          Each of those 16 neurons now has 784 weighted inputs arriving at once. What it does with them is step 3.
        </Bridge>
      </div>

      <div className="grid2 sec">
        <div className="panel">
          <div className="h3" style={{ marginBottom: 10 }}>
            <span className="fnum-lg">3</span> Neurons: add it all up, then fire or stay silent
          </div>
          <p className="body">
            A <b>neuron</b> collects every weighted input arriving at it and adds them up, along with a starting number of
            its own (called its <b>bias</b>, which practice adjusts too). If the total is positive, it <b>fires</b> and
            passes the total on. If not, it stays <b>silent</b> and passes on 0.
          </p>
          <NeuronSum image={image} network={network} fwd={fwd} pick={pick} neuron={neuron} setNeuron={setNeuron} />
          <p className="body" style={{ marginTop: 12 }}>
            The neurons sit in layers, and each layer&rsquo;s outputs are the next layer&rsquo;s inputs:
          </p>
          <ul className="intro-list body">
            <li>
              <b>Hidden layer 1:</b> 16 neurons, named a to p, each with a weighted connection from all 784 inputs.
            </li>
            <li>
              <b>Hidden layer 2:</b> 16 neurons, named A to P, each with a weighted connection from all 16 neurons of
              hidden layer 1.
            </li>
          </ul>
          <p className="body" style={{ marginTop: 10 }}>
            &ldquo;Hidden&rdquo; only means you never see those layers from outside: just pixels in, ten numbers out. The
            diagram shows the whole route.
          </p>
          <Bridge>After hidden layer 2 comes the last layer of neurons: the outputs. That&rsquo;s step 4.</Bridge>
        </div>
        <NetworkDiagram
          title={null}
          network={network}
          fwd={fwd}
          top={1}
          selected={{ layer: 0, from: pick, to: neuron }}
          marks={[{ layer: 1, index: neuron }]}
          caption={`The ${digit} flowing through the whole network. To keep it readable, only the strongest connection into each neuron is drawn. The outlined line runs from your pixel to neuron ${neuronName(1, neuron)}.`}
        />
      </div>

      <div className="sec">
        <Outputs fwd={fwd} digit={digit} practice={practice} />
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
        Neuron {name} receives all 784 inputs. From your pixel (input #{thousands(pick + 1)}) it gets the pixel&rsquo;s
        value times the weight:{' '}
        <span className="mono">
          {fmt(v, 2)} × {fmt(w, 4)} = {fmt(v * w, 4)}
        </span>
        {v === 0 ? ' (blank paper sends nothing, whatever the weight)' : w < 0 ? ' (a negative weight pushes down)' : ' (a positive weight pushes up)'}
        . Only the {inked} inputs with ink add anything; the other {PIXELS - inked} are 0. Adding it all up, along with its
        bias, the total is <span className="mono">{fmt(z, 2)}</span>, so it{' '}
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
  // When the top two scores round to the same value, show enough digits to see which is ahead.
  let second = guess === 0 ? 1 : 0
  for (let d = 0; d < 10; d++) if (d !== guess && fwd.yHat[d] > fwd.yHat[second]) second = d
  const close = fwd.yHat[guess] - fwd.yHat[second] < 0.01
  return (
    <div className="panel">
      <div className="ptitle" style={{ flexWrap: 'wrap' }}>
        <span className="h3">
          <span className="fnum-lg">4</span> Output: the biggest score is its guess
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
        The last layer has 10 neurons, one for each digit from 0 to 9. They work like the others with one difference:
        instead of firing or staying silent, each one squeezes its total into a <b>score</b> between 0 and 1. Close to 1
        means &ldquo;I think it&rsquo;s this digit&rdquo;; close to 0 means &ldquo;I don&rsquo;t&rdquo;. Each bar is one
        score, and the biggest score is its guess.
      </p>
      <OutputBars yHat={fwd.yHat} />
      {practicing && <progress className="practice-progress" max={total} value={round} aria-label="Practice progress" />}
      <p className="body" style={{ marginTop: 12 }} aria-live="polite">
        {view === 'before' ? (
          <>
            Before any practice, its weights are random, so every score sits near 0.5. Its guess for this {digit} is{' '}
            {guess}
            {close && (
              <>
                {' '}
                (<span className="mono">{fmt(fwd.yHat[guess], 3)}</span>, just ahead of {second} at{' '}
                <span className="mono">{fmt(fwd.yHat[second], 3)}</span>)
              </>
            )}
            {guess === digit ? ', which is only luck.' : ': wrong.'} <b>Practice</b> is the learning from the top of the
            page: show it handwritten digits with the right answer attached, and adjust its weights a little after each
            guess. Switch to <b>After practice</b> to watch it happen.
          </>
        ) : practicing ? (
          <>
            Practicing: round {thousands(round)} of {thousands(total)}. Each round shows it 32 digits, so{' '}
            {thousands(total)} rounds covers all 60,000. Right now it says {guess} (
            <span className="mono">{fmt(fwd.yHat[guess], 2)}</span>).
          </>
        ) : (
          <>
            After practicing on 60,000 handwritten digits with the right answer attached, it says {guess} (
            <span className="mono">{fmt(fwd.yHat[guess], 2)}</span>) for this {digit}. Nobody rewrote it: the only things
            that changed are its weights and biases. Part 2 shows how.
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
          Read row by row, the picture becomes one long list of {thousands(PIXELS)} numbers. Your pixel, row {row + 1}{' '}
          column {col + 1}, is number <b>{thousands(pick + 1)}</b> (the {row} full {row === 1 ? 'row' : 'rows'} of 28 before it, then{' '}
          {col + 1} more):
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
          Each number in the list is one <b>input</b> to the network: 784 numbers, 784 inputs.
        </p>
        <p className="callout">
          That list is all the network ever sees. It has no idea the numbers came from a picture, or which ones were next
          to each other.
        </p>
      </div>
    </div>
  )
}
