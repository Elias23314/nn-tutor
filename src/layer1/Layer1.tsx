import { useMemo, useState } from 'react'
import { BrainTable, LayerHead, LockIcon, RightIcon, SayItCard, WrongIcon } from '../components/LayerParts'
import { forward, predict, type Network } from '../engine/network'
import { fmt, thousands, weightScale } from '../playground/connection'
import { PixelImage } from '../playground/PixelImage'
import type { Snapshot } from '../playground/protocol'
import { useTrainer, type Send } from '../playground/useTrainer'
import { WeightGrid } from '../playground/WeightPictures'
import { href } from '../routes'
import { PracticeChart } from './PracticeChart'
import { THRESHOLD, confusedWith, finishOrder, mostCommonGuess } from './story'
import { WhatIsIt } from './WhatIsIt'
import '../playground/playground.css'
import '../components/layer.css'

const DIGITS = Array.from({ length: 10 }, (_, d) => d)
/** The digit the opening uses to show pixels becoming numbers. */
const INTRO_DIGIT = 7
const REASONS = ['It has the most curves', 'It looks like other digits', 'Just a guess']
/** Score the test digits often enough to see when each digit crosses 80%. */
const TEST_EVERY = 25

interface Prediction {
  digit: number | null
  reason: string | null
  locked: boolean
}

export function Layer1() {
  const { status, send } = useTrainer([{ type: 'setTestEvery', rounds: TEST_EVERY }])
  return (
    <main className="layer">
      <LayerHead
        n={1}
        name="What it does"
        tag="No math"
        title="Does it actually get better with practice?"
        lede="Its job: name the digit in a 28 × 28 image. Nobody tells it what a 7 looks like. It only gets corrected on examples."
        loopCaption="Each round of practice runs this whole loop once"
      />
      {status.kind === 'loading' && <p className="notice">Loading 70,000 handwritten digits…</p>}
      {status.kind === 'error' && (
        <p className="notice" role="alert">
          Couldn&rsquo;t start the network: {status.message}
        </p>
      )}
      {status.kind === 'ready' && <Ready state={status.state} samples={status.samples} send={send} />}
    </main>
  )
}

function Ready({ state, samples, send }: { state: Snapshot; samples: Float32Array[]; send: Send }) {
  const [prediction, setPrediction] = useState<Prediction>({ digit: null, reason: null, locked: false })
  const order = finishOrder(state.digitHistory, state.round, state.roundsPerPass)
  const busy = state.pending > 0 || state.playing
  const canPractice = prediction.locked && !busy
  // Before the answer is known, the dashed line follows your prediction; after, the real answer.
  const dashed = order.known ? order.last : (prediction.digit ?? order.last)
  // Both weight grids share one color scale, so "before" and "after" can be compared.
  const scale = useMemo(() => weightScale(state.network), [state.network])

  return (
    <>
      <WhatIsIt image={samples[INTRO_DIGIT]} network={state.network} digit={INTRO_DIGIT} />

      <h2 className="layer-h2">Now: does it get better with practice?</h2>
      <PredictCard
        prediction={prediction}
        setPrediction={setPrediction}
        canChange={state.round === 0}
      />

      <TenDigits samples={samples} initial={state.initialNetwork} now={state.network} round={state.round} />

      <div className="grid2 sec">
        <div className="panel">
          <div className="ptitle" style={{ flexWrap: 'wrap' }}>
            <span className="h3">Getting better with practice</span>
            <div className="btns" style={{ gap: 8 }}>
              <button className="btn sm" disabled={!canPractice} onClick={() => send({ type: 'practice', rounds: 1 })}>
                Practice once
              </button>
              <button className="btn sm" disabled={!canPractice} onClick={() => send({ type: 'practice', rounds: 100 })}>
                Practice ×100
              </button>
              <button
                className="btn sec sm"
                disabled={state.round === 0 && !busy}
                onClick={() => {
                  send({ type: 'reset' })
                  setPrediction({ ...prediction, locked: false })
                }}
              >
                Reset
              </button>
            </div>
          </div>
          <PracticeChart
            overall={state.accuracyHistory}
            digit={dashed}
            digitPoints={state.digitHistory[dashed]}
            roundsPerPass={state.roundsPerPass}
          />
          <p className="small" style={{ marginTop: 6 }} aria-live="polite">
            {!prediction.locked
              ? 'Lock in your prediction above to start practice.'
              : `Round ${thousands(state.round)}: ${state.lastTest ? (state.lastTest.accuracy * 100).toFixed(1) : '–'}% of test digits right. Solid line: all 10,000 test digits. Dashed line: test ${dashed}s only${order.known ? '' : ' (your prediction)'}.`}
          </p>
        </div>
        <Reveal state={state} prediction={prediction} />
      </div>

      {state.lastTest?.topMistake && state.round > 0 && state.lastTest.accuracy > 0.5 && (
        <MistakePanel mistake={state.lastTest.topMistake} />
      )}

      <div className="refute sec">
        <div className="h3">You might expect that someone wrote down what a 7 looks like. Nobody did.</div>
        <p className="body">
          All that changes during practice are its 13,002 numbers. Here are 12,544 of them: the connections from the pixels
          to the 16 neurons in hidden layer 1, drawn as pictures. Blue pixels excite the neuron and orange pixels inhibit
          it.
        </p>
        <div className="pair">
          <div>
            <div className="sub" style={{ marginBottom: 8 }}>
              At the start: faint noise
            </div>
            <WeightGrid network={state.initialNetwork} captions={false} scale={scale} />
          </div>
          <div>
            <div className="sub" style={{ marginBottom: 8 }}>
              {state.round === 0 ? 'Now: the same, until it practices' : `After ${thousands(state.round)} rounds`}
            </div>
            <WeightGrid network={state.network} captions={false} scale={scale} />
          </div>
        </div>
      </div>

      <div className="sec">
        <BrainTable
          topic="Practice"
          rows={[
            [
              `Try, get corrected, adjust: ${thousands(state.roundsPerPass)} rounds of 32 examples`,
              'Learning a skill: try, get feedback, adjust',
              'It has only seen these 60,000 digits, so its whole idea of handwriting comes from them.',
            ],
          ]}
          so="So: practice works in both, but this network’s whole world is its examples."
        />
      </div>

      <div className="sec">
        <SayItCard
          before="Nobody told it what a 7 looks like. It got better by being "
          answer="corrected"
          after=" on examples."
          accepts={(t) => /correct|feedback|fix|told.*wrong|mistake/i.test(t)}
        />
      </div>

      <div className="exits">
        <a className="btn sec" href={href('start')}>
          Back to start here
        </a>
        <a className="btn" href={href('layer-2')}>
          Go deeper: layer 2
        </a>
      </div>
    </>
  )
}

function PredictCard({
  prediction,
  setPrediction,
  canChange,
}: {
  prediction: Prediction
  setPrediction: (p: Prediction) => void
  canChange: boolean
}) {
  const { digit, reason, locked } = prediction
  return (
    <section className="predict" aria-labelledby="predict-q">
      <div className="ptop">
        <span className="pk">Predict first</span>
        <span className="locked">
          <LockIcon /> {locked ? 'Locked in before practice' : 'Lock it in before practice'}
        </span>
      </div>
      <div className="q" id="predict-q">
        Which digit will it get right last?
      </div>
      <div className="chips" role="group" aria-labelledby="predict-q">
        {DIGITS.map((d) => (
          <button
            key={d}
            className="chip"
            aria-pressed={digit === d}
            disabled={locked}
            onClick={() => setPrediction({ ...prediction, digit: d })}
          >
            {d}
            {locked && digit === d && <LockIcon />}
          </button>
        ))}
      </div>
      <div className="sub" id="predict-why">
        Why do you think so?
      </div>
      <div className="chips" role="group" aria-labelledby="predict-why">
        {REASONS.map((r) => (
          <button
            key={r}
            className="chip"
            aria-pressed={reason === r}
            disabled={locked}
            onClick={() => setPrediction({ ...prediction, reason: r })}
          >
            {r}
          </button>
        ))}
      </div>
      <div className="btns">
        {!locked && (
          <button className="btn" disabled={digit === null} onClick={() => setPrediction({ ...prediction, locked: true })}>
            Lock in
          </button>
        )}
        {locked && canChange && (
          <button className="lnk" style={{ fontSize: 14.5 }} onClick={() => setPrediction({ ...prediction, locked: false })}>
            Change my prediction
          </button>
        )}
        {!locked && <span className="small">“Right” means it reads that digit correctly at least 80% of the time.</span>}
      </div>
    </section>
  )
}

function TenDigits({ samples, initial, now, round }: { samples: Float32Array[]; initial: Network; now: Network; round: number }) {
  const before = useMemo(() => samples.map((x) => forward(initial, x).yHat), [samples, initial])
  const after = useMemo(() => samples.map((x) => forward(now, x).yHat), [samples, now])
  const beforeGuesses = before.map((y) => predict(y))
  const common = mostCommonGuess(beforeGuesses)
  return (
    <>
      <div className="ptitle sec" style={{ flexWrap: 'wrap' }}>
        <span className="h3">Ten digits it never trained on</span>
        <span className="small">
          Before practice it calls {common.count >= 6 ? 'almost everything' : `${common.count} of these ten`} a {common.digit}
        </span>
      </div>
      <div className="dgrid">
        {samples.map((x, d) => {
          const guessNow = predict(after[d])
          return (
            <div className="dcard" key={d}>
              <PixelImage image={x} size={84} label={`A handwritten ${d} from the test digits`} />
              <div className="r">
                <b>
                  This is {d === 8 ? 'an' : 'a'} {d}
                </b>
                <span className="ln">
                  {beforeGuesses[d] === d ? <RightIcon /> : <WrongIcon />} Round 0: says {beforeGuesses[d]}
                </span>
                {round > 0 && (
                  <>
                    <span className="ln">
                      {guessNow === d ? <RightIcon /> : <WrongIcon />} Round {thousands(round)}: says {guessNow}
                    </span>
                    <span className="small">confidence {fmt(after[d][guessNow], 2)}</span>
                  </>
                )}
              </div>
            </div>
          )
        })}
      </div>
    </>
  )
}

function Reveal({ state, prediction }: { state: Snapshot; prediction: Prediction }) {
  const order = finishOrder(state.digitHistory, state.round, state.roundsPerPass)
  const latest = (d: number) => state.digitHistory[d].at(-1)?.value ?? 0

  if (!prediction.locked || !order.known || prediction.digit === null) {
    const done = 10 - order.behind.length
    return (
      <div className="reveal waiting" aria-live="polite">
        <div className="h3">{state.round === 0 ? 'The answer appears as it practices' : `${done} of 10 digits right so far`}</div>
        <p className="body" style={{ fontSize: 15 }}>
          Each bar is one digit&rsquo;s test examples. A digit counts as right once it reads at least 80% of them
          correctly (the dashed line).
        </p>
        <div className="progress" role="list">
          {DIGITS.map((d) => (
            <div className="d" key={d} role="listitem" aria-label={`${d}: ${(latest(d) * 100).toFixed(0)}% right`}>
              <div className="bar">
                <i className={order.reached[d] !== null ? 'done' : ''} style={{ height: `${latest(d) * 100}%` }} />
                <span className="line" />
              </div>
              <span>{d}</span>
            </div>
          ))}
        </div>
      </div>
    )
  }

  const { last, reached, behind } = order
  const guess = prediction.digit
  const allReached = behind.length === 0
  const byArrival = DIGITS.filter((d) => d !== last && reached[d] !== null).sort((a, b) => reached[b]! - reached[a]!)
  const runnerUp = byArrival[0]
  const mention = [runnerUp, guess].filter((d, i, list) => d !== undefined && d !== last && list.indexOf(d) === i)
  const mixups = state.lastTest ? confusedWith(state.lastTest.confusion, last) : []

  return (
    <div className="reveal" aria-live="polite">
      <div className="h3">
        You predicted {guess}. It was {last}
        {guess === last ? ': you called it.' : '.'}
      </div>
      <div className="slot">
        <span className="k">What happened</span>
        <span className="v">
          {allReached
            ? `${last} was the last digit it read correctly at least ${THRESHOLD * 100}% of the time, at round ${thousands(reached[last]!)}.`
            : `After ${thousands(state.round)} rounds, ${behind.length === 1 ? `${last} is the only digit` : `${behind.join(', ')} are the digits`} still below ${THRESHOLD * 100}%; ${last} is at ${(latest(last) * 100).toFixed(0)}%.`}{' '}
          {mention
            .map((d) =>
              reached[d] !== null
                ? `${d} got there at round ${thousands(reached[d]!)}`
                : `${d} is at ${(latest(d) * 100).toFixed(0)}%`,
            )
            .join(' and ')}
          {mention.length > 0 && '.'}
        </span>
      </div>
      {mixups.length > 0 && (
        <div className="slot">
          <span className="k">Why</span>
          <span className="v">
            In the test set, {last}s are most often read as{' '}
            {mixups.map((m) => `${m.digit === 8 ? 'an' : 'a'} ${m.digit} (${m.count} times)`).join(' or ')}. A digit that
            shares strokes with others takes longest to tell apart.
          </span>
        </div>
      )}
      <div className="slot">
        <span className="k">Watch next</span>
        <span className="v">In layer 2, slow one correction down and see which numbers change.</span>
      </div>
    </div>
  )
}

function MistakePanel({ mistake }: { mistake: NonNullable<NonNullable<Snapshot['lastTest']>['topMistake']> }) {
  const { actual, said, count, image, confidence, outputForActual } = mistake
  return (
    <div className="panel sec mistake">
      <PixelImage image={image} size={110} label={`A handwritten ${actual} it reads as ${said}`} />
      <div>
        <div className="h3">Right on most digits, confidently wrong on some</div>
        <p className="body" style={{ marginTop: 6 }}>
          This is {actual === 8 ? 'an' : 'a'} {actual}. It says {said} with confidence {fmt(confidence, 2)}; its output for{' '}
          {actual} is {fmt(outputForActual, 2)}. Reading {actual}s as {said}s is its most common mistake right now:{' '}
          {thousands(count)} of the 10,000 test digits.
        </p>
      </div>
    </div>
  )
}
