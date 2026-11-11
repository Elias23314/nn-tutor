import { useMemo, useState } from 'react'
import { LayerHead, LockIcon, QuickCheck, RightIcon, SayItCard, WrongIcon } from '../components/LayerParts'
import { DEFAULT_ETA, DEFAULT_SEED, createNetwork, forward, neuronName, zeroGradients, type Network } from '../engine/network'
import { connectionLabel, fmt, sourceActivity, strength, thousands, type Connection } from '../playground/connection'
import { NetworkDiagram } from '../playground/NetworkDiagram'
import { useTrainer } from '../playground/useTrainer'
import { href } from '../routes'
import { exactSlope, focusConnection, mistakeAt, nudgeSlope, slice, stepFor, withStrength } from './slope'
import '../playground/playground.css'
import '../components/layer.css'
import '../layer1/layer1.css'
import './layer3.css'

const LABEL = 7
/** The nudge you'd try by hand. */
const NUDGE = 0.01
const RATES: [number, string][] = [
  [0.5, 'cautious'],
  [2, 'the default'],
  [10, 'bold'],
  [40, 'very bold'],
]
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
        soFar="One round of practice is guess, measure the mistake, adjust. Adjusting changes the connection strengths that played a part in the mistake, a little, in the direction that makes the mistake smaller."
        lede="But which direction is that, and how big should the change be? This page works it out, with arithmetic, for a single connection."
        loopOn={['Find the slope', 'Take a step']}
        loopCaption="Layer 2’s “Adjust” is really two parts: find the slope, then take a step"
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
    /** The slope measured by nudging: change in mistake ÷ change in strength. */
    slope: (nudged - mistake) / NUDGE,
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
        <div className="sticky-col">
          <NetworkDiagram
            title={`${name} in focus`}
            network={shown}
            fwd={now.fwd}
            top={1}
            selected={c}
            changed={heldStill}
            unchangedLabel="held still"
            changedDescription={`Only ${name} is free to change; every other connection is held still.`}
            marks={[{ layer: 'out', index: LABEL, label: `says ${fmt(now.fwd.yHat[LABEL], 3)}\ngoal 1` }]}
            caption={`The outlined line is ${name}. Everything gray is held still, so you can see what this one connection does. Real practice changes them all at once.`}
          />
        </div>

        <ByHand
          name={name}
          source={source}
          firing={firing}
          w={w}
          w0={w0}
          setW={setW}
          now={now}
          start={start}
          prediction={prediction}
          setPrediction={setPrediction}
          locked={locked}
          lockIn={() => setLocked(true)}
          right={right}
        />
      </div>

      {locked && (
        <>
          <div className="grid2 sec">
            <div className="sticky-col">
              <Downhill net={net} x={example} c={c} name={name} w0={w0} w={w} now={now} eta={eta} />
            </div>
            <div className="stack">
              <FindSlope name={name} w={w} now={now} />
              <TakeStep
                name={name}
                w={w}
                w0={w0}
                now={now}
                start={start}
                eta={eta}
                setEta={setEta}
                steps={steps}
                takeStep={() => {
                  setW(w + stepFor(eta, now.slope))
                  setSteps(steps + 1)
                }}
                startOver={() => {
                  setW(w0)
                  setSteps(0)
                }}
              />
            </div>
          </div>
          <div className="sec">
            <YourTurn slope={start.slope} />
          </div>
        </>
      )}

      <div className="sec">
        <SayItCard
          before="For each connection, it measures which direction makes the mistake smaller (the slope) and takes a small "
          answer="step"
          after=" that way, over and over."
          accepts={(t) => /step|move|change|nudge|jump|adjust/i.test(t)}
          next={{
            label: 'Layer 4: The math',
            text: 'Nudging works for one connection, but this network has 13,002, and testing each one separately would be far too slow. Layer 4 shows how the chain rule finds every slope exactly, in a single pass backward through the network.',
          }}
        />
      </div>

      <div className="exits">
        <a className="btn sec" href={href('layer-2')}>
          Back to layer 2
        </a>
        <a className="btn" href={href('layer-4')}>
          Go deeper: layer 4
        </a>
      </div>
    </>
  )
}

type Measured = ReturnType<typeof measure>

/** Part 1: meet the connection, predict which way it should move, then move it by hand. */
function ByHand({
  name,
  source,
  firing,
  w,
  w0,
  setW,
  now,
  start,
  prediction,
  setPrediction,
  locked,
  lockIn,
  right,
}: {
  name: string
  source: string
  firing: number
  w: number
  w0: number
  setW: (w: number) => void
  now: Measured
  start: Measured
  prediction: Direction | null
  setPrediction: (d: Direction) => void
  locked: boolean
  lockIn: () => void
  right: Direction
}) {
  const y7 = now.fwd.yHat[LABEL]
  const y0 = start.fwd.yHat[LABEL]
  const push = firing * w
  const push0 = firing * w0
  // Keep + and − on a grid of whole nudges from the starting strength, so coming back lands exactly on it.
  const snap = (v: number) => w0 + Math.round((v - w0) / NUDGE) * NUDGE
  const moved = Math.abs(w - w0) > 1e-9
  const stronger = w > w0
  const better = now.mistake < start.mistake
  const box = (label: string, value: string, changed = false, goal?: string) => (
    <div className={`fbox${changed ? ' changed' : ''}`}>
      <span className="fl">{label}</span>
      <span className="fv mono">{value}</span>
      {goal && <span className="fl">{goal}</span>}
    </div>
  )
  return (
    <div className="panel">
      <div className="ptitle">
        <span className="h3">
          <span className="fnum-lg">1</span> One connection, by hand
        </span>
        {moved && (
          <button className="lnk" style={{ fontSize: 14 }} onClick={() => setW(w0)}>
            Put it back
          </button>
        )}
      </div>
      <p className="body">
        We&rsquo;ll hold every other connection still and watch just one: from neuron {source} in hidden layer 2 to the{' '}
        {LABEL} output, written <b>{name}</b>. It&rsquo;s a good one to study: {source} fires hardest for this {LABEL}, and
        the connection leads straight to the {LABEL} output. This is the same {LABEL} as layer 2, before any practice.
      </p>
      <p className="body" style={{ marginTop: 10 }}>
        The connection passes on {source}&rsquo;s firing multiplied by its <b>strength</b>. Follow it through to the
        mistake:
      </p>

      <div className="flow" aria-live="polite">
        {box(`${source} fires`, fmt(firing, 2))}
        <span className="fop">×</span>
        {box('strength', fmt(w), moved)}
        <span className="fop">=</span>
        {box(`push into ${LABEL}`, fmt(push), moved)}
        <span className="fop">→</span>
        {box(`${LABEL} output`, fmt(y7, 3), moved, 'goal 1')}
        <span className="fop">→</span>
        {box('mistake', fmt(now.mistake, 5), moved)}
      </div>
      <p className="small" style={{ marginTop: 8 }}>
        The changes on this page are tiny, so the mistake is shown to five decimal places.
      </p>

      <div className="l3-predict">
        <div className="ptop">
          <span className="pk" id="dir-q">
            {locked ? 'Your prediction' : 'Predict before you try it'}
          </span>
          {locked && (
            <span className="locked">
              <LockIcon /> Locked in
            </span>
          )}
        </div>
        <p className="body">
          The {LABEL} output is at {fmt(y0, 3)} and its goal is 1. To shrink the mistake, should {name} get stronger or
          weaker?
        </p>
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
              {locked && prediction === d && (d === right ? <RightIcon /> : <WrongIcon />)}
            </button>
          ))}
        </div>
        {!locked && (
          <button className="btn sm" style={{ alignSelf: 'flex-start' }} disabled={prediction === null} onClick={lockIn}>
            Lock in and try it
          </button>
        )}
      </div>

      {locked && (
        <>
          <div className="hand" style={{ marginTop: 16 }}>
            <button className="btn sec sm" aria-label={`Weaken ${name} by ${NUDGE}`} onClick={() => setW(snap(w - NUDGE))}>
              −
            </button>
            <div className="hand-val">
              <span className="mono">{fmt(w)}</span>
              <span className="small">Strength of {name}</span>
            </div>
            <button className="btn sec sm" aria-label={`Strengthen ${name} by ${NUDGE}`} onClick={() => setW(snap(w + NUDGE))}>
              +
            </button>
          </div>
          <input
            className="hand-range"
            type="range"
            min={w0 - 2}
            max={w0 + 4}
            step={NUDGE}
            value={w}
            aria-label={`Strength of ${name}`}
            onChange={(e) => setW(Number(e.target.value))}
          />

          <p className="body" style={{ marginTop: 8 }} aria-live="polite">
            {!moved ? (
              <>
                Now test your prediction. Press <b>+</b> to make it a little stronger, or <b>−</b> to make it weaker, and
                watch the boxes above. Each press moves the strength by {NUDGE}.
              </>
            ) : (
              <>
                {stronger ? 'Stronger' : 'Weaker'}: {source} now pushes <span className="mono">{fmt(push)}</span> into the{' '}
                {LABEL} output instead of <span className="mono">{fmt(push0)}</span>, so the {LABEL} output went{' '}
                {y7 > y0 ? 'up' : 'down'} from <span className="mono">{fmt(y0, 3)}</span> to{' '}
                <span className="mono">{fmt(y7, 3)}</span>, {y7 > y0 ? 'closer to' : 'further from'} its goal of 1. The
                mistake went {better ? 'down' : 'up'} from <span className="mono">{fmt(start.mistake, 5)}</span> to{' '}
                <span className="mono">{fmt(now.mistake, 5)}</span>: {better ? 'better' : 'worse'}. So the answer is{' '}
                <b>{right}</b>
                {prediction === right ? ', as you predicted.' : '.'}
              </>
            )}
          </p>
          <p className="bigger">
            <b>The bigger picture:</b> this is 1 of the network&rsquo;s {thousands(13_002)} adjustable numbers. Layer
            2&rsquo;s &ldquo;Adjust&rdquo; is what you just did, done automatically for every one of them: find which
            direction makes the mistake smaller, and move that way. The parts below turn that into arithmetic.
          </p>
        </>
      )}
    </div>
  )
}

/** Part 2: turn "which way is better" into a number. */
function FindSlope({ name, w, now }: { name: string; w: number; now: Measured }) {
  const diff = now.nudged - now.mistake
  const down = now.slope < 0
  return (
    <div className="panel">
      <div className="h3" style={{ marginBottom: 10 }}>
        <span className="fnum-lg">2</span> Find the slope
      </div>
      <p className="body">
        Nudging told you which direction helps. The <b>slope</b> turns that into a number: how much the mistake changes
        for each unit you strengthen {name}. Measure it with one small nudge:
      </p>
      <ol className="arith" style={{ marginTop: 10 }}>
        <li>
          <span className="k">Nudge the strength</span>
          <span className="v">
            <span className="mono">
              {fmt(w)} → {fmt(w + NUDGE)}
            </span>
            , a change of <span className="mono">+{NUDGE}</span>
          </span>
        </li>
        <li>
          <span className="k">See the mistake change</span>
          <span className="v">
            <span className="mono">
              {fmt(now.mistake, 5)} → {fmt(now.nudged, 5)}
            </span>
            , a change of <span className="mono">{signed(diff, 5)}</span>
          </span>
        </li>
        <li>
          <span className="k">Divide</span>
          <span className="v">
            slope = change in mistake ÷ change in strength
            <br />
            <span className="mono">
              {signed(diff, 5)} ÷ {NUDGE} ≈ <b>{fmt(now.slope, 4)}</b>
            </span>
          </span>
        </li>
      </ol>
      <p className="body" style={{ marginTop: 12 }} aria-live="polite">
        The slope is {down ? 'negative' : 'positive'}: {down ? 'strengthening' : 'weakening'} {name} makes the mistake{' '}
        smaller. {down ? 'A positive slope would mean the opposite.' : 'A negative slope would mean the opposite.'} Its size
        says how strongly the mistake responds. On the chart, it&rsquo;s the tilt of the dashed blue line.
      </p>
      <details className="l3-more">
        <summary>How accurate is a nudge?</summary>
        {now.nudges.map(([h, s]) => (
          <div className="kv" key={h}>
            <span>
              A nudge of <span className="mono">{h}</span> gives
            </span>
            <span className="m">{fmt(s, 5)}</span>
          </div>
        ))}
        <p className="small" style={{ marginTop: 10 }}>
          Smaller nudges get closer to the exact slope, <span className="mono">{fmt(now.exact, 5)}</span>. Layer 4 shows how
          the network gets that exact value without nudging at all.
        </p>
      </details>
    </div>
  )
}

/** Part 3: use the slope to move, then do it again. */
function TakeStep({
  name,
  w,
  w0,
  now,
  start,
  eta,
  setEta,
  steps,
  takeStep,
  startOver,
}: {
  name: string
  w: number
  w0: number
  now: Measured
  start: Measured
  eta: number
  setEta: (eta: number) => void
  steps: number
  takeStep: () => void
  startOver: () => void
}) {
  const step = stepFor(eta, now.slope)
  const atStart = Math.abs(w - w0) < 1e-9
  return (
    <div className="panel">
      <div className="h3" style={{ marginBottom: 10 }}>
        <span className="fnum-lg">3</span> Take a step
      </div>
      <p className="body">
        Now use the slope. To make the mistake smaller, move the strength the <b>opposite</b> way to the slope: that is
        what the minus sign does. How far to move is set by the <b>learning rate</b>, a number that says how boldly it
        steps.
      </p>
      <div className="l3-formula mono" aria-live="polite">
        step = −(learning rate) × slope
        <br />
        step = −({eta}) × ({fmt(now.slope, 4)}) = <b>{signed(step)}</b>
      </div>

      <div className="sub" id="rate-q" style={{ marginTop: 16, marginBottom: 8 }}>
        Try a different learning rate
      </div>
      <div className="chips" role="group" aria-labelledby="rate-q">
        {RATES.map(([r, label]) => (
          <button key={r} className="chip" aria-pressed={eta === r} onClick={() => setEta(r)}>
            {r} <span className="rate-note">{label}</span>
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
      </div>
      <p className="body" style={{ marginTop: 12 }} aria-live="polite">
        {steps === 0 ? (
          <>
            Taking the step would move {name} from <span className="mono">{fmt(w)}</span> to{' '}
            <span className="mono">{fmt(w + step)}</span>. Press the button, then press it again: that is the
            &ldquo;Repeat&rdquo; in the loop.
          </>
        ) : (
          <>
            After {steps} step{steps === 1 ? '' : 's'}, the strength is <span className="mono">{fmt(w)}</span> and the
            mistake is <span className="mono">{fmt(now.mistake, 5)}</span>, down from{' '}
            <span className="mono">{fmt(start.mistake, 5)}</span>. The slope is now{' '}
            <span className="mono">{fmt(now.slope, 4)}</span>
            {Math.abs(now.slope) < Math.abs(start.slope) ? ': flatter than before, so the next step is smaller.' : '.'}
          </>
        )}
      </p>
      <p className="callout" style={{ marginTop: 14 }}>
        On this one connection, a bigger learning rate just gets there faster. In real practice all {thousands(13_002)}{' '}
        numbers step at once, and steps that are too big overshoot and make the mistake worse. You can try a learning rate
        of 40 in the playground.
      </p>
      <details className="l3-more">
        <summary>Why does this tutor use 2?</summary>
        <p className="small" style={{ marginTop: 8 }}>
          The mistake is an average over ten outputs, so every slope comes out small. A learning rate of 2 makes up for
          that.
        </p>
      </details>
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
  // The tangent: the slope drawn as a short straight line through the current point.
  const half = 0.5
  const step = stepFor(eta, now.slope)
  const to = Math.min(hi, Math.max(lo, w + step))
  // A key in the empty top-right corner, so labels never sit on the curve.
  const keyX = CW - PAD.r - 230
  const keyY = PAD.t + 6
  return (
    <div className="panel chart">
      <div className="ptitle">
        <span className="h3">The same thing as a picture</span>
      </div>
      <p className="body" style={{ marginBottom: 10 }}>
        Each point on the curve is the mistake for one strength of {name}, with everything else held still. Lower is
        better. The dot is where the connection is now, the dashed line shows the slope there, and the arrow is the step
        it would take: downhill.
      </p>
      <svg
        viewBox={`0 0 ${CW} ${CH}`}
        role="img"
        aria-label={`The mistake falls from ${fmt(ys[0], 3)} to ${fmt(ys.at(-1)!, 3)} as ${name} goes from ${fmt(lo, 1)} to ${fmt(hi, 1)}. Now at ${fmt(w)}, slope about ${fmt(now.slope, 3)}, step ${signed(step)}.`}
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
            {!inView && ' (off this chart)'}
          </text>
          <line x1={keyX} y1={keyY + 28} x2={keyX + 16} y2={keyY + 28} stroke="var(--excite)" strokeWidth={2} strokeDasharray="5 3" />
          <text x={keyX + 22} y={keyY + 32} fill="var(--excite)">
            slope ≈ {fmt(now.slope, 4)}
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

function YourTurn({ slope }: { slope: number }) {
  const [text, setText] = useState('')
  const [checked, setChecked] = useState<boolean | null>(null)
  const [flip, setFlip] = useState<Direction | null>(null)
  const answer = stepFor(TURN_RATE, slope)
  const typed = Number(`0.${text.replace(/\D/g, '')}`)
  return (
    <div className="panel turn">
      <div className="h3">Your turn</div>
      <form
        className="turn"
        onSubmit={(e) => {
          e.preventDefault()
          if (text.trim()) setChecked(Math.abs(typed - answer) <= 0.0015)
        }}
      >
        <p className="body">
          <b>1.</b> Back at the start, the slope was about <span className="mono">{fmt(slope, 3)}</span>. What step would a
          learning rate of {TURN_RATE} take?
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
              −({TURN_RATE}) × ({fmt(slope, 3)}) = {signed(answer)}
            </span>
            {checked
              ? `, a quarter of the step at learning rate ${DEFAULT_ETA}. Same direction, smaller step.`
              : '. Multiply the slope by the learning rate, then flip the sign.'}
          </p>
        )}
      </form>

      <div className="turn" style={{ marginTop: 8 }}>
        <p className="body" id="flip-q">
          <b>2.</b> Suppose the slope had been <span className="mono">+{fmt(Math.abs(slope), 3)}</span> instead. Which way
          should the connection move?
        </p>
        <div className="chips" role="group" aria-labelledby="flip-q">
          {(
            [
              ['stronger', 'Get stronger'],
              ['weaker', 'Get weaker'],
              ['same', 'Stay the same'],
            ] as const
          ).map(([d, label]) => (
            <button key={d} className="chip" aria-pressed={flip === d} disabled={flip === 'weaker'} onClick={() => setFlip(d)}>
              {label}
              {flip === d && (d === 'weaker' ? <RightIcon /> : <WrongIcon />)}
            </button>
          ))}
        </div>
        {flip !== null && (
          <p className="feedback" aria-live="polite">
            {flip === 'weaker'
              ? 'Right. A positive slope means strengthening makes the mistake bigger, so the step goes the other way: −(learning rate) × (a positive slope) is negative, and the connection gets weaker.'
              : 'Not quite. A positive slope means strengthening makes the mistake bigger. The step always goes the opposite way to the slope. Try again.'}
          </p>
        )}
      </div>
    </div>
  )
}
