import { useMemo, useState, type ReactNode } from 'react'
import { LayerHead, QuickCheck, RightIcon, SayItCard, WrongIcon } from '../components/LayerParts'
import { DEFAULT_SEED, IMAGE_SIDE, PIXELS, createNetwork, neuronName, predict } from '../engine/network'
import { fmt, thousands, type Connection } from '../playground/connection'
import { NetworkDiagram, type Mark } from '../playground/NetworkDiagram'
import { PixelImage } from '../playground/PixelImage'
import { useTrainer } from '../playground/useTrainer'
import { WeightCanvas } from '../playground/WeightPictures'
import { href } from '../routes'
import { correct, type Correction } from './correction'
import '../playground/playground.css'
import '../components/layer.css'
import './layer2.css'

const LABEL = 7
/** One round of practice, in plain words. Layer 3 adds "find the slope". */
const ROUND = ['Guess', 'Measure the mistake', 'Adjust', 'Repeat']

export function Layer2() {
  const { status } = useTrainer()
  return (
    <main className="layer">
      <LayerHead
        n={2}
        name="How it works"
        tag="No math"
        title="What changes inside it when it learns?"
        lede="In layer 1, practice turned a wrong guess into a right one. Practice is one small routine, repeated thousands of times: show it a digit along with the right answer, see what it guesses, measure how wrong it was, and adjust. This page slows down a single round."
        loop={ROUND}
        loopCaption="One round of practice. The replay below walks through it."
      />
      <QuickCheck
        before="A neural network turns a picture into "
        after=", passes them through layers of neurons, and its biggest output is its guess."
        options={['rules', 'numbers', 'words']}
        answer="numbers"
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

type Guess = 'ink' | 'blank' | 'both'

/** The pixel → hidden-layer-1 connection this correction changes most: one concrete case to look at. */
function mostChanged(c: Correction): Connection {
  const g = c.grads.layers[0]
  let best = 0
  for (let k = 1; k < g.weights.length; k++) if (Math.abs(g.weights[k]) > Math.abs(g.weights[best])) best = k
  // Its pixel has ink by construction: connections from blank pixels have a gradient of 0.
  return { layer: 0, from: best % PIXELS, to: Math.floor(best / PIXELS) }
}

function Ready({ example }: { example: Float32Array }) {
  // The untrained network everyone starts from, and one correction on this 7.
  const c = useMemo(() => correct(createNetwork(undefined, DEFAULT_SEED), example, LABEL), [example])
  const one = useMemo(() => mostChanged(c), [c])
  const [step, setStep] = useState(1)
  const [guess, setGuess] = useState<Guess | null>(null)
  const [locked, setLocked] = useState(false)
  const [why, setWhy] = useState<number | null>(null)
  const [show, setShow] = useState<'before' | 'after'>('after')
  const adjusted = step === 3

  // What the diagram points at in each step.
  const marks: Mark[] =
    step === 1
      ? [{ layer: 'out', index: c.guessBefore }]
      : step === 2
        ? [
            { layer: 'out', index: c.guessBefore, label: `guess: ${fmt(c.before.yHat[c.guessBefore], 2)}\ngoal 0` },
            { layer: 'out', index: LABEL, label: `says ${fmt(c.before.yHat[LABEL], 2)}\ngoal 1` },
          ]
        : [{ layer: 1, index: one.to }]
  const caption =
    step === 1
      ? `The ${LABEL} flows through the untrained network, and its biggest output, ${c.guessBefore}, is its guess. Only the strongest connection into each neuron is drawn.`
      : step === 2
        ? `The right answer is ${LABEL}, so the ${LABEL} output should be 1 and every other output should be 0.`
        : 'Colored connections changed in this correction. Gray ones didn’t: nothing flowed through them, or the neuron they feed was silent. The outlined one is looked at more closely below.'

  return (
    <>
      <p className="body l2-note">
        This {LABEL} is one of the 60,000 examples it practices on, so it&rsquo;s a different {LABEL} from layer 1&rsquo;s.
        Every one of those examples comes with the right answer attached. The network here hasn&rsquo;t practiced at all
        yet.
      </p>

      <div className="grid2">
        <NetworkDiagram
          title="The network during this round"
          network={c.start}
          fwd={c.before}
          top={adjusted ? 2 : 1}
          changed={adjusted ? c.grads : null}
          unchangedLabel="didn’t change"
          changedDescription="This correction changes connections that had something flowing through them into a firing neuron, and leaves the rest alone."
          selected={adjusted ? one : null}
          marks={marks}
          caption={<span aria-live="polite">{caption}</span>}
        />

        <div className="stack">
          <Replay c={c} step={step} setStep={setStep} locked={locked}>
            <Prediction
              guess={guess}
              setGuess={setGuess}
              locked={locked}
              lockIn={() => {
                setLocked(true)
                setStep(3)
              }}
            />
          </Replay>

          <div className="panel">
            <div className="ptitle" style={{ flexWrap: 'wrap' }}>
              <span className="h3">{step === 1 ? 'Its ten outputs' : step === 2 ? 'Each output against its goal' : 'Its outputs, before and after'}</span>
              {adjusted && (
                <div className="chips" role="group" aria-label="Which outputs to show">
                  <button className="chip" aria-pressed={show === 'before'} onClick={() => setShow('before')}>
                    Before
                  </button>
                  <button className="chip" aria-pressed={show === 'after'} onClick={() => setShow('after')}>
                    After
                  </button>
                </div>
              )}
            </div>
            <GoalBars yHat={adjusted && show === 'after' ? c.after.yHat : c.before.yHat} goals={step >= 2} />
            {step >= 2 && (
              <p className="body" style={{ marginTop: 12 }} aria-live="polite">
                {adjusted && show === 'after' ? (
                  <>
                    After the correction it says {c.guessAfter}, and the mistake is{' '}
                    <span className="mono">{fmt(c.lossAfter)}</span>, down from <span className="mono">{fmt(c.lossBefore)}</span>.
                    Every output moved a little toward its goal.
                  </>
                ) : (
                  <>
                    The blue tick is each output&rsquo;s goal, and the gap is how far the output is from it. The mistake
                    sums up all ten gaps in one number: <span className="mono">{fmt(c.lossBefore)}</span>.
                  </>
                )}
              </p>
            )}
          </div>
        </div>
      </div>

      {adjusted && (
        <div className="panel sec">
          <div className="h3" style={{ marginBottom: 12 }}>
            A closer look at what changed
          </div>

          <p className="body" aria-live="polite">
            {guess === 'ink' ? 'You predicted it: only ' : 'Here is what happened: only '}
            connections from the {c.inkPixels} pixels with ink changed. Not one connection from a blank pixel did.
          </p>

          <WhyBlank why={why} setWhy={setWhy} changed={c.changed} />

          {why !== null && (
            <>
              <p className="callout" style={{ marginTop: 16 }}>
                The rule: a connection changes only if it played a part in the mistake. That means something flowed
                through it, and the neuron it feeds was firing. A blank pixel sends 0, and a silent neuron passes nothing
                on, so either way that connection is left alone.
              </p>

              <OneConnection c={c} one={one} />

              <div className="sub" style={{ margin: '20px 0 8px' }}>
                Where the changes landed: the connections from the pixels into hidden layer 1
              </div>
              <div className="seen-layout">
                <div>
                  <PixelImage image={example} size={140} label={`The ${LABEL} it saw`} />
                  <p className="small" style={{ marginTop: 8 }}>
                    The {LABEL} it saw
                  </p>
                </div>
                <div>
                  <ChangeGrid c={c} />
                  <p className="body" style={{ marginTop: 12 }}>
                    Each square is one neuron in hidden layer 1, and shows how its 784 connections from the pixels
                    changed: blue ones now push it up more, orange ones push it down more. The changes trace the{' '}
                    {LABEL}&rsquo;s ink, and the {16 - c.firing1} silent neurons didn&rsquo;t change at all. That is{' '}
                    {c.inkPixels} ink pixels × {c.firing1} firing neurons = {thousands(c.changedFromPixels)} connections.
                    The other {thousands(c.changed - c.changedFromPixels)} changes are deeper in the network.
                  </p>
                </div>
              </div>

              <p className="body l2-scale">
                <b>One correction is only a nudge.</b> This one happened to be enough to fix this {LABEL}, but a single
                example can&rsquo;t teach it every digit. Practice is {thousands(1875)} rounds of this, each on 32 examples at once.
                That is what turned layer 1&rsquo;s &ldquo;before practice&rdquo; into its &ldquo;after practice&rdquo;.
              </p>
            </>
          )}
        </div>
      )}

      <div className="sec">
        <SayItCard
          before="Learning means changing "
          answer="connection strengths"
          after=": after a mistake, only the connections that played a part in it change."
          accepts={(t) => /connection|weight|strength|synap/i.test(t)}
          next={{
            label: 'Layer 3: With numbers',
            text: 'So a correction changes connection strengths, but by how much, and in which direction? Layer 3 zooms in on one connection. You’ll nudge it by hand, measure how the mistake responds, and work out the step it should take.',
          }}
        />
      </div>

      <div className="exits">
        <a className="btn sec" href={href('layer-1')}>
          Back to layer 1
        </a>
        <a className="btn" href={href('layer-3')}>
          Go deeper: layer 3
        </a>
      </div>
    </>
  )
}

function Replay({
  c,
  step,
  setStep,
  locked,
  children,
}: {
  c: Correction
  step: number
  setStep: (s: number) => void
  locked: boolean
  /** The prediction, asked between measuring and adjusting. */
  children: ReactNode
}) {
  const top = c.before.yHat[c.guessBefore]
  const wrong = c.guessBefore !== LABEL
  const steps = [
    {
      title: 'Guess',
      text: (
        <>
          It&rsquo;s shown the {LABEL}, but not told the answer. The 784 numbers flow through its layers, just as in layer
          1, and its biggest output is {c.guessBefore} (<span className="mono">{fmt(top, 2)}</span>). Its connection
          strengths are still random, so {wrong ? 'the guess is wrong' : 'a right guess is only luck'}.
        </>
      ),
    },
    {
      title: 'Measure the mistake',
      text: (
        <>
          Now the right answer is revealed: {LABEL}. That sets a goal for every output: 1 for the {LABEL}, and 0 for the
          other nine. The <b>mistake</b> is one number that sums up how far all ten outputs are from their goals. 0 would
          be perfect. Here it is <span className="mono">{fmt(c.lossBefore)}</span>.
          <span className="small" style={{ display: 'block', marginTop: 6 }}>
            (It&rsquo;s the average of the ten gaps, each squared. Layer 3 uses that.)
          </span>
        </>
      ),
    },
    {
      title: 'Adjust',
      text: (
        <>
          Each connection strength that played a part in the mistake is changed a little, in the direction that would have
          made the mistake smaller. {thousands(c.changed)} of the network&rsquo;s {thousands(c.total)} adjustable numbers
          changed; almost all of those numbers are connection strengths. Afterwards it says {c.guessAfter} for this{' '}
          {LABEL}.
        </>
      ),
    },
  ]
  return (
    <div className="panel">
      <div className="ptitle">
        <span className="h3">Replay one round</span>
        <span className="small">Part {step} of 3</span>
      </div>
      <ol className="steps" style={{ listStyle: 'none' }}>
        {steps.map((s, i) => {
          const n = i + 1
          const cls = n < step ? 'done' : n === step ? 'cur' : 'todo'
          return (
            <li className={`st ${cls}`} key={s.title} aria-current={n === step ? 'step' : undefined}>
              <span className="n">{n}</span>
              <div>
                <div className="tt">{s.title}</div>
                {n <= step && <div className="tx">{s.text}</div>}
                {n === 2 && step >= 2 && children}
              </div>
            </li>
          )
        })}
      </ol>
      <div className="btns" style={{ marginTop: 16 }}>
        <button className="btn sec sm" disabled={step === 1} onClick={() => setStep(step - 1)}>
          Back
        </button>
        {step === 1 && (
          <button className="btn sm" onClick={() => setStep(2)}>
            Next: measure the mistake
          </button>
        )}
        {step === 2 && locked && (
          <button className="btn sm" onClick={() => setStep(3)}>
            Next: adjust
          </button>
        )}
        {step === 2 && !locked && <span className="small">Make your prediction above to continue.</span>}
        {step === 3 && (
          <button className="btn sm" onClick={() => setStep(1)}>
            Replay
          </button>
        )}
      </div>
    </div>
  )
}

const GUESSES: [Guess, string][] = [
  ['ink', 'Only from pixels with ink'],
  ['blank', 'Only from blank pixels'],
  ['both', 'From both'],
]

function Prediction({
  guess,
  setGuess,
  locked,
  lockIn,
}: {
  guess: Guess | null
  setGuess: (g: Guess) => void
  locked: boolean
  lockIn: () => void
}) {
  return (
    <div className="l2-predict">
      <div className="pk" id="pick-q">
        {locked ? 'Your prediction' : 'Predict before it adjusts'}
      </div>
      <p className="body">
        Every pixel is connected to hidden layer 1. Which of those connections do you think will change?
        {!locked && ' Think back to layer 1: what does a blank pixel send?'}
      </p>
      <div className="chips" role="group" aria-labelledby="pick-q">
        {GUESSES.map(([g, label]) => (
          <button key={g} className="chip" aria-pressed={guess === g} disabled={locked} onClick={() => setGuess(g)}>
            {label}
            {locked && guess === g && (g === 'ink' ? <RightIcon /> : <WrongIcon />)}
          </button>
        ))}
      </div>
      {!locked && (
        <button className="btn sm" style={{ alignSelf: 'flex-start' }} disabled={guess === null} onClick={lockIn}>
          Lock in and adjust
        </button>
      )}
    </div>
  )
}

/** The ten outputs as bars, like layer 1's, with each one's goal and gap once the answer is known. */
function GoalBars({ yHat, goals }: { yHat: ArrayLike<number>; goals: boolean }) {
  const guess = predict(yHat)
  return (
    <div className="gbars" role="list" aria-label="Its ten outputs">
      {Array.from({ length: 10 }, (_, d) => {
        const v = yHat[d]
        const goal = d === LABEL ? 1 : 0
        return (
          <div className={`grow${d === guess ? ' top' : ''}`} key={d} role="listitem">
            <span className="dg">{d}</span>
            <div className="tr">
              <i style={{ width: `${(v * 100).toFixed(1)}%` }} />
              {goals && <span className="goal" style={{ left: `${goal * 100}%` }} />}
            </div>
            <span className="vv">{fmt(v, 2)}</span>
            <span className="nt">
              {goals ? `goal ${goal} · gap ${fmt(Math.abs(v - goal), 2)}` : d === guess ? 'its guess' : ''}
            </span>
          </div>
        )
      })}
    </div>
  )
}

const WHY = [
  'A blank pixel sent 0, so its connections played no part in the mistake',
  'Those connections were already right',
  'It can only change one connection at a time',
]

function WhyBlank({ why, setWhy, changed }: { why: number | null; setWhy: (i: number) => void; changed: number }) {
  const feedback = [
    'Yes. A connection’s change depends on what flowed through it, and nothing flowed from a blank pixel.',
    'Not quite. Their strengths are as random as every other connection’s. They didn’t change because nothing flowed through them: a blank pixel sends 0.',
    `Not quite. This one correction changed ${thousands(changed)} numbers at once. Blank pixels’ connections were left out because nothing flowed through them.`,
  ]
  return (
    <div className="l2-why">
      <div className="q" style={{ marginBottom: 12, fontSize: 18 }} id="why-q">
        Why do you think the connections from blank pixels didn&rsquo;t change?
      </div>
      <div className="opts" role="radiogroup" aria-labelledby="why-q">
        {WHY.map((label, i) => (
          <button
            key={label}
            role="radio"
            aria-checked={why === i}
            className={`opt${why === i ? ' mark' : ''}`}
            style={{ background: 'var(--card)', textAlign: 'left' }}
            disabled={why === 0}
            onClick={() => setWhy(i)}
          >
            {why === i ? i === 0 ? <RightIcon /> : <WrongIcon /> : <span className="r" />}
            {label}
          </button>
        ))}
      </div>
      {why !== null && (
        <p className="feedback" aria-live="polite">
          {feedback[why]}
        </p>
      )}
    </div>
  )
}

/** One connection's strength before and after: the concrete case behind "connection strengths change". */
function OneConnection({ c, one }: { c: Correction; one: Connection }) {
  const at = one.to * PIXELS + one.from
  const before = c.start.layers[0].weights[at]
  const after = c.network.layers[0].weights[at]
  const name = neuronName(1, one.to)
  const row = Math.floor(one.from / IMAGE_SIDE) + 1
  const col = (one.from % IMAGE_SIDE) + 1
  return (
    <div className="l2-one">
      <div className="sub">One connection, before and after</div>
      <div className="l2-one-row">
        <div className="fbox2">
          <span className="fl">before</span>
          <span className="fv mono">{fmt(before)}</span>
        </div>
        <span className="fop" aria-hidden="true">
          →
        </span>
        <div className="fbox2 changed">
          <span className="fl">after</span>
          <span className="fv mono">{fmt(after)}</span>
        </div>
        <p className="body">
          This is the strength of the connection from the pixel at row {row}, column {col} to neuron {name} in hidden layer
          1: the outlined line in the diagram. It went {after > before ? 'up' : 'down'} by{' '}
          <span className="mono">{fmt(Math.abs(after - before))}</span>, so that pixel now pushes neuron {name}{' '}
          {after > before ? (after > 0 ? 'up a little more' : 'down a little less') : after < 0 ? 'down a little more' : 'up a little less'}.
          That small change to a number is what learning
          physically is. This correction made {thousands(c.changed)} of them.
        </p>
      </div>
    </div>
  )
}

/** How each hidden-layer-1 neuron's 784 connections changed in this step (−η · gradient). */
function ChangeGrid({ c }: { c: Correction }) {
  const g = c.grads.layers[0]
  const rows = useMemo(
    () =>
      Array.from({ length: g.outSize }, (_, j) =>
        Float64Array.from(g.weights.subarray(j * PIXELS, (j + 1) * PIXELS), (v) => (v === 0 ? 0 : -v)),
      ),
    [g],
  )
  let max = 1e-12
  for (const r of rows) for (const v of r) max = Math.max(max, Math.abs(v))
  return (
    <div className="wgrid">
      {rows.map((r, j) => {
        const silent = c.before.layers[0].a[j] === 0
        return (
          <div key={j}>
            <WeightCanvas weights={r} max={max} name={`${neuronName(1, j)}${silent ? ' (silent, unchanged)' : ''}`} />
            <div className="cap">
              {neuronName(1, j)}
              {silent ? ' · silent' : ''}
            </div>
          </div>
        )
      })}
    </div>
  )
}
