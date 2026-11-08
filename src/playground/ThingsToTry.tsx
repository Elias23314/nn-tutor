import { useId, useState, type ReactNode } from 'react'
import { rotate180 } from '../engine/drawing'
import { DEFAULT_ETA, forward, predict, type Network } from '../engine/network'
import { NO_CHANGES } from '../engine/trainer'
import { fmt, pct, thousands } from './connection'
import type { Command, ExperimentResult, Snapshot } from './protocol'

interface CardProps {
  question: string
  options: string[]
  /** Set once the experiment has run: which option turned out to be right. */
  answer: number | null
  verdict: ReactNode
  running: string | null
  disabled?: boolean
  onRun: () => void
  undo?: { label: string; onClick: () => void }
}

function TryCard({ question, options, answer, verdict, running, disabled, onRun, undo }: CardProps) {
  const [choice, setChoice] = useState<number | null>(null)
  const [locked, setLocked] = useState(false)
  const name = useId()
  const done = locked && answer !== null && running === null
  return (
    <div className="try">
      <div className="qq" id={`${name}-q`}>
        {question}
      </div>
      <fieldset className="opts" aria-labelledby={`${name}-q`}>
        {options.map((label, i) => (
          <label key={label} className={`opt${done && i === answer ? ' answer' : ''}`}>
            <input
              type="radio"
              name={name}
              checked={choice === i}
              disabled={locked}
              onChange={() => setChoice(i)}
            />
            {label}
          </label>
        ))}
      </fieldset>
      {!locked && (
        <button
          className="btn sm sec"
          style={{ alignSelf: 'flex-start' }}
          disabled={choice === null || disabled}
          onClick={() => {
            setLocked(true)
            onRun()
          }}
        >
          Lock in, then run it
        </button>
      )}
      {locked && running !== null && <p className="small" aria-live="polite">{running}</p>}
      {done && (
        <div className="verdict" aria-live="polite">
          <b>{choice === answer ? 'You called it.' : `You picked “${options[choice!]}”.`}</b> {verdict}
        </div>
      )}
      {locked && (done || running?.startsWith('Stopped')) && (
        <div className="btns">
          {done && undo && (
            <button className="btn sm sec" onClick={undo.onClick}>
              {undo.label}
            </button>
          )}
          <button
            className="lnk"
            style={{ fontSize: 14 }}
            onClick={() => {
              setLocked(false)
              setChoice(null)
            }}
          >
            Try again
          </button>
        </div>
      )}
    </div>
  )
}

interface Props {
  state: Snapshot
  network: Network
  image: Float32Array
  upsideDown: boolean
  setUpsideDown: (v: boolean) => void
  send: (cmd: Command) => void
}

type Kind = 'eta40' | 'no7'
interface Run {
  id: number
  target: number
}

const top = (yHat: ArrayLike<number>) => {
  const d = predict(yHat)
  return { digit: d, value: yHat[d] }
}

let nextId = 1

export function ThingsToTry({ state, network, image, upsideDown, setUpsideDown, send }: Props) {
  const [runs, setRuns] = useState<Partial<Record<Kind, Run>>>({})
  const [flipResult, setFlipResult] = useState<{ digit: number; value: number } | null>(null)

  // The worker reports each experiment's result when it finishes.
  const resultOf = (kind: Kind): ExperimentResult | null => {
    const run = runs[kind]
    return (run && state.experiments.find((e) => e.id === run.id)) || null
  }
  const progress = (kind: Kind): string | null => {
    const run = runs[kind]
    if (!run || resultOf(kind)) return null
    return state.pending > 0
      ? `Practicing: round ${thousands(state.round)} of ${thousands(run.target)}…`
      : 'Stopped before it finished.'
  }
  const anyRunning = (['eta40', 'no7'] as const).some((k) => runs[k] && !resultOf(k) && state.pending > 0)
  const start = (kind: Kind, cmd: Omit<Extract<Command, { type: 'experiment' }>, 'id' | 'type'>) => {
    const id = nextId++
    setRuns({ ...runs, [kind]: { id, target: cmd.rounds } })
    send({ type: 'experiment', id, ...cmd })
  }

  const eta40Result = resultOf('eta40')
  const no7Result = resultOf('no7')?.probe ?? null
  const no7Rounds = Math.ceil((60_000 - 6_265) / 32)

  return (
    <>
      <div className="h3" style={{ marginTop: 4 }}>
        Things to try
      </div>

      <TryCard
        question="What will happen at learning rate 40?"
        options={['It learns faster', 'Its mistake drops, but it stops reading digits', 'Nothing changes']}
        answer={eta40Result ? (eta40Result.collapsed ? 1 : 0) : null}
        running={progress('eta40')}
        disabled={anyRunning}
        onRun={() => start('eta40', { eta: 40, changes: state.changes, rounds: 100 })}
        verdict={
          eta40Result?.collapsed
            ? `After 100 rounds it reads only ${pct(eta40Result.accuracy)} of test digits. The steps were so big that every output was pushed to 0.`
            : `After 100 rounds it reads ${pct(eta40Result?.accuracy ?? 0)} of test digits. This time the big steps didn't break it.`
        }
        undo={{
          label: 'Set learning rate to 2 and reset',
          onClick: () => {
            send({ type: 'setEta', eta: DEFAULT_ETA })
            send({ type: 'reset' })
          },
        }}
      />

      <TryCard
        question="Remove every 7 from its examples and retrain. What will it say for your 7?"
        options={['Still 7', 'A digit with similar strokes', 'It will refuse to answer']}
        answer={no7Result ? (no7Result.digit === 7 ? 0 : 1) : null}
        running={progress('no7')}
        disabled={anyRunning}
        onRun={() =>
          start('no7', {
            eta: DEFAULT_ETA,
            changes: { removed: [7], relabel: null },
            rounds: no7Rounds,
            probe: image,
          })
        }
        verdict={
          no7Result && (
            <>
              It says {no7Result.digit} ({fmt(no7Result.value, 2)}). With no 7s in its examples, the 7 output&rsquo;s goal
              was always 0, so it never learned to fire. It can&rsquo;t refuse either: it always names its highest output.
            </>
          )
        }
        undo={{ label: 'Back to all 60,000', onClick: () => send({ type: 'setExamples', changes: NO_CHANGES }) }}
      />

      <TryCard
        question="Turn your 7 upside down. What will it say?"
        options={['Still 7', 'Another digit, confidently', 'That it isn’t sure']}
        answer={flipResult ? (flipResult.digit === 7 ? 0 : flipResult.value >= 0.5 ? 1 : 2) : null}
        running={null}
        onRun={() => {
          setUpsideDown(true)
          setFlipResult(top(forward(network, rotate180(image)).yHat))
        }}
        verdict={
          flipResult && (
            <>
              Upside down, it says {flipResult.digit} ({fmt(flipResult.value, 2)}).{' '}
              {flipResult.digit === 7
                ? 'It still finds enough of a 7 in the strokes.'
                : 'It has only ever seen digits the right way up, so it matches the strokes to whatever they now resemble.'}
            </>
          )
        }
        undo={upsideDown ? { label: 'Turn it back', onClick: () => setUpsideDown(false) } : undefined}
      />
    </>
  )
}
