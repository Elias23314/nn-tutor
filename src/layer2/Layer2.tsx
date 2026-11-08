import { useMemo, useState } from 'react'
import { BrainTable, LayerHead, QuickCheck, RightIcon, SayItCard, WrongIcon } from '../components/LayerParts'
import { DEFAULT_SEED, PIXELS, createNetwork, neuronName } from '../engine/network'
import { fmt, thousands } from '../playground/connection'
import { NetworkDiagram } from '../playground/NetworkDiagram'
import { PixelImage } from '../playground/PixelImage'
import { useTrainer } from '../playground/useTrainer'
import { WeightCanvas } from '../playground/WeightPictures'
import { href } from '../routes'
import { correct, type Correction } from './correction'
import '../playground/playground.css'
import '../components/layer.css'

const LABEL = 7

export function Layer2() {
  const { status } = useTrainer()
  return (
    <main className="layer">
      <LayerHead
        n={2}
        name="How it works"
        tag="No math"
        title="What physically changes when it learns?"
        lede={`One correction, slowed down. Before any practice, it sees a single ${LABEL} from its training examples.`}
        loopOn={['Step']}
        loopCaption="Step 3 of the replay is the “step”"
      />
      <QuickCheck
        before="It got better by being "
        after=" on examples."
        options={['given the rules', 'corrected', 'shown every digit once']}
        answer="corrected"
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

type Pick = 'ink' | 'blank'

function Ready({ example }: { example: Float32Array }) {
  // The untrained network everyone starts from, and one correction on this 7.
  const c = useMemo(() => correct(createNetwork(undefined, DEFAULT_SEED), example, LABEL), [example])
  const [step, setStep] = useState(1)
  const [picks, setPicks] = useState<Pick[]>([])
  const [locked, setLocked] = useState(false)
  const [why, setWhy] = useState<number | null>(null)
  const showChanges = step === 3

  const togglePick = (p: Pick) => setPicks((current) => (current.includes(p) ? current.filter((q) => q !== p) : [...current, p]))

  return (
    <>
      <div className="grid2">
        <NetworkDiagram
          title="The network during this correction"
          network={c.start}
          fwd={c.before}
          top={4}
          changed={showChanges ? c.grads : null}
          caption={
            showChanges
              ? 'Colored connections change in this correction. Gray ones don’t, because they start or end at a silent neuron. Showing the 4 strongest connections into each neuron.'
              : 'The 7 flowing through the untrained network. Showing the 4 strongest connections into each neuron.'
          }
        />

        <div className="stack">
          <Replay c={c} step={step} setStep={setStep} locked={locked} />

          {step >= 2 && (
            <div className="panel">
              <div className="sub" style={{ marginBottom: 8 }} id="pick-q">
                {locked
                  ? 'Before step 3 you picked which connections would change'
                  : 'Before step 3: which connections from the pixels will change? Pick any.'}
              </div>
              <div className="chips" role="group" aria-labelledby="pick-q">
                {(
                  [
                    ['ink', 'From ink pixels'],
                    ['blank', 'From blank pixels'],
                  ] as const
                ).map(([p, label]) => (
                  <button key={p} className="chip" aria-pressed={picks.includes(p)} disabled={locked} onClick={() => togglePick(p)}>
                    {label}
                    {locked && picks.includes(p) && (p === 'ink' ? <RightIcon /> : <WrongIcon />)}
                  </button>
                ))}
              </div>
              {!locked && (
                <button
                  className="btn sm"
                  style={{ marginTop: 12 }}
                  disabled={picks.length === 0}
                  onClick={() => {
                    setLocked(true)
                    setStep(3)
                  }}
                >
                  Lock in and show step 3
                </button>
              )}
              {locked && (
                <p className="feedback" aria-live="polite">
                  {picks.includes('ink') && picks.includes('blank')
                    ? 'One of two. A blank pixel is 0, so its connections played no part in this mistake.'
                    : picks.includes('ink')
                      ? `Right: only connections from the ${c.inkPixels} ink pixels change. A blank pixel is 0, so its connections played no part in this mistake.`
                      : 'The other way round: only connections from ink pixels change. A blank pixel is 0, so its connections played no part in this mistake.'}
                </p>
              )}
            </div>
          )}

          {locked && (
            <WhyBlank
              why={why}
              setWhy={setWhy}
              changed={c.changed}
            />
          )}
        </div>
      </div>

      <div className="panel sec">
        <div className="ptitle" style={{ flexWrap: 'wrap' }}>
          <span className="h3">What changed in hidden layer 1</span>
          {showChanges && (
            <span className="small">
              {c.inkPixels} ink pixels × {c.firing1} firing neurons = {thousands(c.changedFromPixels)} connections
            </span>
          )}
        </div>
        {showChanges ? (
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
                Each square shows how one neuron&rsquo;s 784 connections changed: blue got more exciting, orange more
                inhibiting. Changes appear only where the {LABEL} has ink, and the {16 - c.firing1} silent neurons
                didn&rsquo;t change at all.
              </p>
            </div>
          </div>
        ) : (
          <p className="body">This appears at step 3 of the replay, once you&rsquo;ve picked which connections will change.</p>
        )}
      </div>

      <div className="sec">
        <BrainTable
          topic="The architecture"
          rows={[
            ['Connection strength', 'Synaptic strength', 'A good match in spirit: in both, learning lives in the connections.'],
            [
              'Blue and orange connections',
              'Excitatory and inhibitory synapses',
              'A real neuron’s outgoing synapses are usually all one kind (Dale’s principle). Here one neuron can do both, and a connection can flip.',
            ],
            ['A neuron firing', 'A firing rate', 'Real neurons fire in spikes over time. Here activity is a single number.'],
            [
              'Learning from a mistake',
              'Synaptic plasticity',
              'This network sends an error signal backward from the outputs; most known brain learning rules use local activity. Layer 4 returns to this.',
            ],
          ]}
          so="So: learning lives in the connections in both, but these neurons are far simpler."
        />
      </div>

      <div className="sec">
        <SayItCard
          before="Learning means changing "
          answer="connection strengths"
          after=", and only the connections involved in a mistake change."
          accepts={(t) => /connection|weight|strength|synap/i.test(t)}
        />
      </div>

      <div className="exits">
        <a className="btn sec" href={href('layer-1')}>
          Back to layer 1
        </a>
        <button className="btn" disabled title="Not built yet">
          Go deeper: layer 3 (coming soon)
        </button>
      </div>
    </>
  )
}

function Replay({ c, step, setStep, locked }: { c: Correction; step: number; setStep: (s: number) => void; locked: boolean }) {
  const y7 = c.before.yHat[LABEL]
  const top = c.before.yHat[c.guessBefore]
  const steps = [
    {
      title: 'Situation',
      text: `A ${LABEL} from its training examples. The goal: the ${LABEL} output should fire and the other nine should stay quiet.`,
    },
    {
      title: 'Guess',
      text: (
        <>
          It says {c.guessBefore} (<span className="mono">{fmt(top, 2)}</span>).{' '}
          {c.guessBefore === LABEL ? (
            <>That&rsquo;s right, but not by much.</>
          ) : (
            <>
              The {LABEL} output is {top - y7 < 0.1 ? 'just behind, at' : 'only'} <span className="mono">{fmt(y7, 2)}</span>.
            </>
          )}{' '}
          The mistake, the average of the ten squared gaps, is <span className="mono">{fmt(c.lossBefore)}</span>.
        </>
      ),
    },
    {
      title: 'Adjustment',
      text: (
        <>
          Connections that pushed toward {LABEL} get stronger, and connections that pushed toward wrong digits get
          weaker. Only {thousands(c.changed)} of the {thousands(c.total)} numbers change. After this one step it says{' '}
          {c.guessAfter}, and the mistake drops to <span className="mono">{fmt(c.lossAfter)}</span>.
        </>
      ),
    },
  ]
  return (
    <div className="panel">
      <div className="ptitle">
        <span className="h3">Replay one correction</span>
        <span className="small">Step {step} of 3</span>
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
            Next
          </button>
        )}
        {step === 2 && locked && (
          <button className="btn sm" onClick={() => setStep(3)}>
            Next
          </button>
        )}
        {step === 2 && !locked && <span className="small">Pick which connections will change, below.</span>}
        {step === 3 && (
          <button className="btn sm" onClick={() => setStep(1)}>
            Replay
          </button>
        )}
      </div>
    </div>
  )
}

const WHY = [
  'Those pixels were 0, so they had no part in the mistake',
  'They were already right',
  'It changes one connection at a time',
]

function WhyBlank({ why, setWhy, changed }: { why: number | null; setWhy: (i: number) => void; changed: number }) {
  const feedback = [
    'Yes. A connection’s change is scaled by what flowed through it, and nothing flowed from a blank pixel.',
    'Not quite. Their strengths are as random as every other connection’s. They didn’t change because nothing flowed through them: a blank pixel is 0.',
    `Not quite. This one step changed ${thousands(changed)} numbers at once. Blank pixels’ connections were left out because nothing flowed from them.`,
  ]
  return (
    <div className="panel">
      <div className="q" style={{ marginBottom: 12 }} id="why-q">
        Why didn&rsquo;t the connections from blank pixels change?
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
