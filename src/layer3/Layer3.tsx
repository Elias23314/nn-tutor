import { useMemo, useState, type ReactNode } from 'react'
import { LayerHead, LockIcon, QuickCheck, SayItCard } from '../components/LayerParts'
import { DEFAULT_ETA, DEFAULT_SEED, createNetwork, forward, neuronName, zeroGradients, type Network } from '../engine/network'
import { connectionLabel, fmt, sourceActivity, strength, type Connection } from '../playground/connection'
import { NetworkDiagram } from '../playground/NetworkDiagram'
import { useTrainer } from '../playground/useTrainer'
import { href } from '../routes'
import { exactSlope, focusConnection, mistakeAt, nudgeSlope, slice, stepFor, withStrength } from './slope'
import '../playground/playground.css'
import '../components/layer.css'
import './layer3.css'

const LABEL = 7
/** The nudge you'd try by hand. */
const NUDGE = 0.01
const RATES = [0.5, 2, 10, 40]
/** "Your turn" asks about this learning rate. */
const TURN_RATE = 0.5

/** A signed number: +0.078, −0.039. */
const signed = (v: number, digits = 3) => (v >= 0 ? `+${fmt(v, digits)}` : fmt(v, digits))

export function Layer3() {
  const { status } = useTrainer()
  return (
    <main className="layer">
      <LayerHead
        n={3}
        name="With numbers"
        tag="Arithmetic"
        title="How much should one connection change?"
        lede={`Hold every connection still except one, from the neuron in hidden layer 2 that fires hardest to the ${LABEL} output. Same training ${LABEL}, before any practice.`}
        loopOn={['Find the slope']}
        loopCaption="This layer is mostly the “find the slope” stage"
      />
      <QuickCheck
        before="Learning means changing "
        after="."
        options={['the examples', 'connection strengths', 'the rules']}
        answer="connection strengths"
      />
      {status.kind === 'loading' && <p className="notice">Loading its training examples…</p>}
      {status.kind === 'error' && (
        <p className="notice" role="alert">
          Couldn&rsquo;t start the network: {status.message}
        </p>
      )}
      {status.kind === 'ready' && <Ready example={status.trainSamples[LABEL]} />}
    </main>
  )
}

type Direction = 'stronger' | 'weaker' | 'same'

/** Everything the page shows about one strength of the focus connection. */
function measure(net: Network, x: Float32Array, c: Connection, w: number) {
  const fwd = forward(withStrength(net, c, w), x)
  const mistake = mistakeAt(net, x, LABEL, c, w)
  const nudged = mistakeAt(net, x, LABEL, c, w + NUDGE)
  return {
    fwd,
    mistake,
    nudged,
    steepness: (nudged - mistake) / NUDGE,
    nudges: [0.1, 0.01, 0.001].map((h) => [h, nudgeSlope(net, x, LABEL, c, w, h)] as const),
    exact: exactSlope(net, x, LABEL, c, w),
  }
}

function Ready({ example }: { example: Float32Array }) {
  const net = useMemo(() => createNetwork(undefined, DEFAULT_SEED), [])
  const c = useMemo(() => focusConnection(net, example, LABEL), [net, example])
  const w0 = strength(net, c)
  const start = useMemo(() => measure(net, example, c, w0), [net, example, c, w0])
  const source = neuronName(2, c.from)
  const name = connectionLabel(c)
  const firing = sourceActivity(start.fwd, c)

  const [w, setW] = useState(w0)
  const [eta, setEta] = useState(DEFAULT_ETA)
  const [prediction, setPrediction] = useState<Direction | null>(null)
  const [locked, setLocked] = useState(false)
  const [steps, setSteps] = useState(0)

  const now = useMemo(() => measure(net, example, c, w), [net, example, c, w])
  const shown = useMemo(() => withStrength(net, c, w), [net, c, w])
  // Only the focus connection is colored; every other one is held still.
  const heldStill = useMemo(() => {
    const g = zeroGradients(net)
    const l = g.layers[c.layer]
    l.weights[c.to * l.inSize + c.from] = 1
    return g
  }, [net, c])
  const right: Direction = start.exact < 0 ? 'stronger' : start.exact > 0 ? 'weaker' : 'same'

  return (
    <>
      <div className="grid2">
        <NetworkDiagram
          title={`${name} in focus`}
          network={shown}
          fwd={now.fwd}
          selected={c}
          changed={heldStill}
          unchangedLabel="held still"
          changedDescription={`Only ${name} is free to change; every other connection is held still.`}
          caption="Gray connections are held still so you can see one connection’s effect. Real training changes them all at once."
        />

        <div className="stack">
          <section className="predict" aria-labelledby="dir-q">
            <div className="ptop">
              <span className="pk">Predict first</span>
              <span className="locked">
                <LockIcon /> {locked ? 'Locked in' : 'Lock it in before you try it'}
              </span>
            </div>
            <div className="q" id="dir-q">
              Should {name} get stronger or weaker?
            </div>
            <div className="chips" role="group" aria-labelledby="dir-q">
              {(
                [
                  ['stronger', 'Stronger'],
                  ['weaker', 'Weaker'],
                  ['same', 'Stay the same'],
                ] as const
              ).map(([d, label]) => (
                <button key={d} className="chip" aria-pressed={prediction === d} disabled={locked} onClick={() => setPrediction(d)}>
                  {label}
                  {locked && prediction === d && <LockIcon />}
                </button>
              ))}
            </div>
            {!locked ? (
              <div>
                <button className="btn" disabled={prediction === null} onClick={() => setLocked(true)}>
                  Lock in my prediction
                </button>
              </div>
            ) : (
              <p className="feedback" aria-live="polite">
                {prediction === right ? `Right: ${right}. ` : `It should get ${right}. `}
                {source} is firing (<span className="mono">{fmt(firing, 2)}</span>), so a stronger connection lifts the{' '}
                {LABEL} output toward its goal of 1.
                {w0 < 0 &&
                  ` Right now it barely inhibits the ${LABEL} (strength ${fmt(w0)}), which pulls the wrong way.`}
              </p>
            )}
          </section>

          {locked ? (
            <ByHand name={name} w={w} w0={w0} setW={setW} now={now} start={start} />
          ) : (
            <div className="panel">
              <div className="h3">Try it by hand</div>
              <p className="body" style={{ marginTop: 8 }}>
                Lock in a prediction first, then nudge {name} yourself.
              </p>
            </div>
          )}
        </div>
      </div>

      {locked && (
        <>
          <div className="grid2 sec">
            <Downhill net={net} x={example} c={c} name={name} w0={w0} w={w} now={now} eta={eta} />
            <Arithmetic
              w={w}
              now={now}
              eta={eta}
              setEta={setEta}
              steps={steps}
              takeStep={() => {
                setW(w + stepFor(eta, now.steepness))
                setSteps(steps + 1)
              }}
              startOver={() => {
                setW(w0)
                setSteps(0)
              }}
              atStart={w === w0}
            />
          </div>
          <div className="sec">
            <YourTurn steepness={start.steepness} />
          </div>
        </>
      )}

      <div className="sec">
        <SayItCard
          before="It measures which direction makes the mistake smaller and takes a small "
          answer="step"
          after=" that way."
          accepts={(t) => /step|move|change|nudge|jump|adjust/i.test(t)}
        />
      </div>

      <div className="exits">
        <a className="btn sec" href={href('layer-2')}>
          Back to layer 2
        </a>
        <a className="btn" href={href('layer-4')}>
          Go deeper: layer 4 finds it without nudging
        </a>
      </div>
    </>
  )
}

type Measured = ReturnType<typeof measure>

function ByHand({
  name,
  w,
  w0,
  setW,
  now,
  start,
}: {
  name: string
  w: number
  w0: number
  setW: (w: number) => void
  now: Measured
  start: Measured
}) {
  const y7 = now.fwd.yHat[LABEL]
  const round = (v: number) => Math.round(v * 1000) / 1000
  return (
    <div className="panel">
      <div className="ptitle">
        <span className="h3">Try it by hand first</span>
        {w !== w0 && (
          <button className="lnk" style={{ fontSize: 14 }} onClick={() => setW(w0)}>
            Put it back
          </button>
        )}
      </div>
      <div className="hand">
        <button className="btn sec sm" aria-label={`Weaken ${name} by ${NUDGE}`} onClick={() => setW(round(w - NUDGE))}>
          −
        </button>
        <div className="hand-val">
          <span className="mono">{fmt(w)}</span>
          <span className="small">Strength of {name}</span>
        </div>
        <button className="btn sec sm" aria-label={`Strengthen ${name} by ${NUDGE}`} onClick={() => setW(round(w + NUDGE))}>
          +
        </button>
      </div>
      <input
        className="hand-range"
        type="range"
        min={round(w0 - 2)}
        max={round(w0 + 4)}
        step={NUDGE}
        value={w}
        aria-label={`Strength of ${name}`}
        onChange={(e) => setW(Number(e.target.value))}
      />
      <div className="kv">
        <span>{LABEL} output</span>
        <span>
          <span className="m">{fmt(y7, 3)}</span> <span className="small">goal 1</span>
        </span>
      </div>
      <div className="kv">
        <span>Mistake</span>
        <span>
          <span className="m">{fmt(now.mistake, 5)}</span> <span className="small">average of ten squared gaps</span>
        </span>
      </div>
      <p className="body" style={{ marginTop: 12 }} aria-live="polite">
        {w === w0 ? (
          <>
            Nudge it from <span className="mono">{fmt(w0)}</span> to <span className="mono">{fmt(w0 + NUDGE)}</span> and
            the mistake goes from <span className="mono">{fmt(start.mistake, 5)}</span> to{' '}
            <span className="mono">{fmt(start.nudged, 5)}</span>.
          </>
        ) : (
          <>
            You moved it from <span className="mono">{fmt(w0)}</span> to <span className="mono">{fmt(w)}</span>, and the
            mistake went from <span className="mono">{fmt(start.mistake, 5)}</span> to{' '}
            <span className="mono">{fmt(now.mistake, 5)}</span>
            {now.mistake < start.mistake ? ': smaller.' : now.mistake > start.mistake ? ': bigger.' : '.'}
          </>
        )}
      </p>
    </div>
  )
}

const CW = 560
const CH = 340
const PAD = { l: 58, r: 20, t: 20, b: 52 }

function Downhill({
  net,
  x,
  c,
  name,
  w0,
  w,
  now,
  eta,
}: {
  net: Network
  x: Float32Array
  c: Connection
  name: string
  w0: number
  w: number
  now: Measured
  eta: number
}) {
  const lo = w0 - 2
  const hi = w0 + 3
  const curve = useMemo(() => slice(net, x, LABEL, c, lo, hi), [net, x, c, lo, hi])
  const ys = curve.map((p) => p.mistake)
  const yMin = Math.floor(Math.min(...ys) * 100) / 100
  const yMax = Math.ceil(Math.max(...ys) * 100) / 100
  const sx = (v: number) => PAD.l + ((CW - PAD.l - PAD.r) * (v - lo)) / (hi - lo)
  const sy = (v: number) => CH - PAD.b - ((CH - PAD.t - PAD.b) * (v - yMin)) / (yMax - yMin)
  const xTicks = Array.from({ length: 7 }, (_, i) => Math.round(lo) + i).filter((t) => t >= lo && t <= hi)
  const yTicks = Array.from({ length: Math.round((yMax - yMin) * 100) + 1 }, (_, i) => yMin + i / 100).filter(
    (_, i, all) => all.length <= 6 || i % 2 === 0,
  )
  const inView = w >= lo && w <= hi
  // The tangent: the steepness drawn as a short straight line through the current point.
  const half = 0.5
  const step = stepFor(eta, now.steepness)
  const to = Math.min(hi, Math.max(lo, w + step))
  // A key in the empty top-right corner, so labels never sit on the curve.
  const keyX = CW - PAD.r - 230
  const keyY = PAD.t + 6
  return (
    <div className="panel chart">
      <div className="ptitle">
        <span className="h3">Downhill: one slice through the mistake</span>
      </div>
      <p className="small" style={{ marginBottom: 10 }}>
        The mistake for every strength of {name}, with everything else held still. It keeps falling as the strength grows,
        more and more slowly.
      </p>
      <svg
        viewBox={`0 0 ${CW} ${CH}`}
        role="img"
        aria-label={`The mistake falls from ${fmt(ys[0], 3)} to ${fmt(ys.at(-1)!, 3)} as ${name} goes from ${fmt(lo, 1)} to ${fmt(hi, 1)}. Now at ${fmt(w)}, steepness about ${fmt(now.steepness, 3)}, step ${signed(step)}.`}
      >
        <defs>
          <marker id="step-head" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">
            <path d="M0 0L8 4L0 8z" fill="var(--ink)" />
          </marker>
        </defs>
        {yTicks.map((t) => (
          <g key={t}>
            <line x1={PAD.l} x2={CW - PAD.r} y1={sy(t)} y2={sy(t)} stroke="var(--track)" />
            <text x={PAD.l - 8} y={sy(t) + 4} fontSize={12} fill="var(--muted)" textAnchor="end">
              {t.toFixed(2)}
            </text>
          </g>
        ))}
        {xTicks.map((t) => (
          <text key={t} x={sx(t)} y={CH - PAD.b + 18} fontSize={12} fill="var(--muted)" textAnchor="middle">
            {fmt(t, 0)}
          </text>
        ))}
        <polyline
          points={curve.map((p) => `${sx(p.w).toFixed(1)},${sy(p.mistake).toFixed(1)}`).join(' ')}
          fill="none"
          stroke="var(--ink)"
          strokeWidth={2.3}
          strokeLinejoin="round"
        />
        <text x={sx(hi) - 4} y={sy(ys.at(-1)!) - 10} fontSize={12} fill="var(--muted)" textAnchor="end">
          flattens: the {LABEL} output nears 1
        </text>
        {inView && (
          <>
            <line
              x1={sx(w - half)}
              y1={sy(now.mistake - half * now.exact)}
              x2={sx(w + half)}
              y2={sy(now.mistake + half * now.exact)}
              stroke="var(--excite)"
              strokeWidth={2}
              strokeDasharray="5 3"
            />
            {Math.abs(sx(to) - sx(w)) > 2 && (
              <line
                x1={sx(w)}
                y1={sy(now.mistake) - 14}
                x2={sx(to)}
                y2={sy(now.mistake) - 14}
                stroke="var(--ink)"
                strokeWidth={1.6}
                markerEnd="url(#step-head)"
              />
            )}
            <circle cx={sx(w)} cy={sy(now.mistake)} r={5.5} fill="var(--ink)" stroke="#fff" strokeWidth={1.5} />
          </>
        )}
        <g fontSize={12.5}>
          <circle cx={keyX + 8} cy={keyY + 6} r={5} fill="var(--ink)" />
          <text x={keyX + 22} y={keyY + 10} fontWeight={600} fill="var(--ink)">
            now: {fmt(w)}
            {!inView && ' (off this slice)'}
          </text>
          <line x1={keyX} y1={keyY + 28} x2={keyX + 16} y2={keyY + 28} stroke="var(--excite)" strokeWidth={2} strokeDasharray="5 3" />
          <text x={keyX + 22} y={keyY + 32} fill="var(--excite)">
            steepness ≈ {fmt(now.steepness, 4)}
          </text>
          <line x1={keyX} y1={keyY + 50} x2={keyX + 16} y2={keyY + 50} stroke="var(--ink)" strokeWidth={1.6} markerEnd="url(#step-head)" />
          <text x={keyX + 22} y={keyY + 54} fill="var(--ink)">
            step {signed(step)} (learning rate {eta})
          </text>
        </g>
        <text x={(PAD.l + CW - PAD.r) / 2} y={CH - 10} fontSize={12} fill="var(--muted)" textAnchor="middle">
          Strength of {name}
        </text>
        <text x={14} y={(PAD.t + CH - PAD.b) / 2} fontSize={12} fill="var(--muted)" textAnchor="middle" transform={`rotate(-90 14 ${(PAD.t + CH - PAD.b) / 2})`}>
          Mistake
        </text>
      </svg>
    </div>
  )
}

function Arithmetic({
  w,
  now,
  eta,
  setEta,
  steps,
  takeStep,
  startOver,
  atStart,
}: {
  w: number
  now: Measured
  eta: number
  setEta: (eta: number) => void
  steps: number
  takeStep: () => void
  startOver: () => void
  atStart: boolean
}) {
  const step = stepFor(eta, now.steepness)
  const diff = now.nudged - now.mistake
  const y7 = now.fwd.yHat[LABEL]
  const rows: [string, ReactNode][] = [
    [
      'Predict',
      <>
        {LABEL} output <span className="mono">{fmt(y7, 3)}</span> (goal 1)
      </>,
    ],
    [
      'Measure',
      <>
        <span className="mono">{fmt(now.mistake, 5)}</span> average of ten squared gaps
      </>,
    ],
    [
      'Find the slope',
      <>
        nudge to <span className="mono">{fmt(w + NUDGE)}</span>: <span className="mono">{fmt(now.mistake, 5)}</span> →{' '}
        <span className="mono">{fmt(now.nudged, 5)}</span>
        <br />
        <span className="mono">
          {fmt(diff, 5)} ÷ {NUDGE} ≈ {fmt(now.steepness, 4)}
        </span>{' '}
        steepness
      </>,
    ],
    [
      'Step',
      <span className="mono" key="s">
        −({eta}) × ({fmt(now.steepness, 4)}) = {signed(step)}
      </span>,
    ],
  ]
  return (
    <div className="stack">
      <div className="panel">
        <div className="ptitle" style={{ flexWrap: 'wrap' }}>
          <span className="h3">The arithmetic</span>
          <span className="small">Learning rate {eta}</span>
        </div>
        <ol className="arith">
          {rows.map(([k, v]) => (
            <li key={k}>
              <span className="k">{k}</span>
              <span className="v">{v}</span>
            </li>
          ))}
        </ol>
        <div className="sub" id="rate-q" style={{ marginTop: 16, marginBottom: 8 }}>
          Learning rate
        </div>
        <div className="chips" role="group" aria-labelledby="rate-q">
          {RATES.map((r) => (
            <button key={r} className="chip" aria-pressed={eta === r} onClick={() => setEta(r)}>
              {r}
            </button>
          ))}
        </div>
        <div className="btns" style={{ marginTop: 16 }}>
          <button className="btn sm" onClick={takeStep}>
            Take the step
          </button>
          {!atStart && (
            <button className="btn sec sm" onClick={startOver}>
              Start over
            </button>
          )}
          <span className="small" aria-live="polite">
            {steps === 0 ? 'Then repeat: predict, measure, find the slope, step.' : `${steps} step${steps === 1 ? '' : 's'} taken`}
          </span>
        </div>
        <p className="small" style={{ marginTop: 14 }}>
          Why a learning rate of 2? The mistake averages ten outputs, so each steepness is small and the step needs a bigger
          multiplier.
        </p>
      </div>
      <Nudges now={now} />
    </div>
  )
}

/** Smaller nudges get closer to the exact steepness. */
function Nudges({ now }: { now: Measured }) {
  return (
    <div className="panel">
      <div className="h3" style={{ marginBottom: 10 }}>
        Smaller nudges get closer to the exact steepness
      </div>
      {now.nudges.map(([h, s]) => (
        <div className="kv" key={h}>
          <span>
            A nudge of <span className="mono">{h}</span> gives
          </span>
          <span className="m">{fmt(s, 5)}</span>
        </div>
      ))}
      <p className="small" style={{ marginTop: 12 }}>
        Layer 4 computes the exact value, <span className="mono">{fmt(now.exact, 5)}</span>, without nudging.
      </p>
    </div>
  )
}

function YourTurn({ steepness }: { steepness: number }) {
  const [text, setText] = useState('')
  const [checked, setChecked] = useState<boolean | null>(null)
  const answer = stepFor(TURN_RATE, steepness)
  const typed = Number(`0.${text.replace(/\D/g, '')}`)
  return (
    <form
      className="panel turn"
      onSubmit={(e) => {
        e.preventDefault()
        if (text.trim()) setChecked(Math.abs(typed - answer) <= 0.0015)
      }}
    >
      <div className="h3">Your turn</div>
      <p className="body">
        What step would learning rate {TURN_RATE} take? The steepness is still about{' '}
        <span className="mono">{fmt(steepness, 3)}</span>.
      </p>
      <div className="turn-row">
        <label className="mono turn-in">
          +0.
          <input
            inputMode="numeric"
            value={text}
            placeholder="___"
            aria-label={`The step at learning rate ${TURN_RATE}, digits after +0.`}
            onChange={(e) => {
              setText(e.target.value.replace(/[^\d]/g, '').slice(0, 5))
              setChecked(null)
            }}
          />
        </label>
        <button className="btn sec sm" disabled={!text}>
          Check
        </button>
      </div>
      {checked !== null && (
        <p className="feedback" aria-live="polite">
          {checked ? 'Right: ' : `Not quite. `}
          <span className="mono">
            −({TURN_RATE}) × ({fmt(steepness, 3)}) = {signed(answer)}
          </span>
          {checked
            ? `, a quarter of the step at learning rate ${DEFAULT_ETA}. Same direction, smaller step.`
            : '. Multiply the steepness by the learning rate, then flip the sign.'}
        </p>
      )}
    </form>
  )
}
