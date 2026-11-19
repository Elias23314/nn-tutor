import { useMemo, useState } from 'react'
import { hasInk, rotate180 } from '../engine/drawing'
import { DEFAULT_ETA, PIXELS, forward, predict } from '../engine/network'
import { AccuracyChart, ConfusionChart, LossChart } from './Charts'
import { CollapseAlert } from './CollapseAlert'
import { ConnectionInspector } from './ConnectionInspector'
import { DrawPad } from './DrawPad'
import { NetworkDiagram } from './NetworkDiagram'
import { OutputBars } from './OutputBars'
import { ThingsToTry } from './ThingsToTry'
import { TrainingExamples } from './TrainingExamples'
import { WeightPictures } from './WeightPictures'
import { G_TO_7, connectionLabel, fmt, pct, thousands, type Connection } from './connection'
import type { Snapshot, Speed } from './protocol'
import { useTrainer, type Send } from './useTrainer'
import './playground.css'

const ETAS = [0.5, 2, 10, 40]

export function Playground() {
  const { status, send } = useTrainer()
  return (
    <main className="pg">
      <div className="pg-head">
        <div>
          <h1>Playground</h1>
          <p className="lede">
            The real network, training in your browser. Draw digits, change its examples, predict what will happen, and
            reset any time.
          </p>
        </div>
        {status.kind === 'ready' && <Stats state={status.state} />}
      </div>
      {status.kind === 'loading' && <p className="notice">Loading 70,000 handwritten digits: 60,000 for it to practice on, and 10,000 kept aside to test it on digits it has never seen…</p>}
      {status.kind === 'error' && (
        <p className="notice" role="alert">
          Couldn&rsquo;t start the network: {status.message}
        </p>
      )}
      {status.kind === 'ready' && <Ready state={status.state} sample7={status.samples[7]} send={send} />}
    </main>
  )
}

function passes(state: Snapshot) {
  if (state.round === 0) return 'before practice'
  const p = state.round / state.roundsPerPass
  if (p === 1) return 'one pass'
  return p < 1 ? `${Math.round(p * 100)}% of one pass` : `${p.toFixed(1)} passes`
}

function Stats({ state }: { state: Snapshot }) {
  return (
    <div className="stats" aria-live="off">
      <div className="stat">
        <div className="k">Practice round</div>
        <div className="v">{thousands(state.round)}</div>
        <div className="s">{passes(state)}</div>
      </div>
      <div className="stat">
        <div className="k">Test digits right</div>
        <div className="v">{state.lastTest ? pct(state.lastTest.accuracy) : '–'}</div>
        <div className="s">of 10,000</div>
      </div>
      <div className="stat">
        <div className="k">Mistake</div>
        <div className="v">{state.recentLoss === null ? '–' : fmt(state.recentLoss)}</div>
        <div className="s">recent average</div>
      </div>
    </div>
  )
}

function Ready({ state, sample7, send }: { state: Snapshot; sample7: Float32Array; send: Send }) {
  const [padImage, setPadImage] = useState<Float32Array | null>(sample7)
  const [image, setImage] = useState<Float32Array>(sample7)
  const [upsideDown, setUpsideDown] = useState(false)
  const [selected, setSelected] = useState<Connection>(G_TO_7)
  const [showMore, setShowMore] = useState(false)
  // A dismissed collapse alert stays hidden until the next reset.
  const [dismissedSession, setDismissedSession] = useState<number | null>(null)

  const seen = useMemo(() => (upsideDown ? rotate180(image) : image), [image, upsideDown])
  const fwd = useMemo(() => forward(state.network, seen), [state.network, seen])
  const inked = hasInk(seen)
  const guess = predict(fwd.yHat)
  const busy = state.playing || state.pending > 0
  const etaIndex = Math.max(0, ETAS.indexOf(state.eta))

  return (
    <>
      <div className="panel" style={{ padding: '16px 20px' }}>
        <div className="ctrl">
          <div className="btns">
            <button className="btn" disabled={busy} onClick={() => send({ type: 'practice', rounds: 1 })}>
              Practice once
            </button>
            <button className="btn sec" onClick={() => send({ type: busy ? 'pause' : 'play' })}>
              {busy ? 'Pause' : 'Play'}
            </button>
            <button className="btn sec" onClick={() => send({ type: 'reset' })}>
              Reset
            </button>
          </div>
          <div className="vsep" />
          <div className="eta">
            <label className="sub" htmlFor="eta">
              Learning rate
            </label>
            <div className="eta-slider">
              <input
                id="eta"
                type="range"
                min={0}
                max={ETAS.length - 1}
                step={1}
                value={etaIndex}
                aria-valuetext={String(ETAS[etaIndex])}
                onChange={(e) => send({ type: 'setEta', eta: ETAS[Number(e.target.value)] })}
              />
              <div className="eta-snaps" aria-hidden="true">
                {ETAS.map((e) => (
                  <span key={e}>{e}</span>
                ))}
              </div>
            </div>
            <span className="eta-value">{state.eta}</span>
          </div>
          <div className="vsep" />
          <button className="lnk" aria-expanded={showMore} onClick={() => setShowMore(!showMore)}>
            {showMore ? 'Fewer controls' : 'More controls'}
          </button>
        </div>
        {showMore && (
          <div className="more">
            <span className="sub">Speed</span>
            <div className="seg" role="group" aria-label="Speed">
              {(['slow', 'normal', 'fast'] as Speed[]).map((s) => (
                <button key={s} aria-pressed={state.speed === s} onClick={() => send({ type: 'setSpeed', speed: s })}>
                  {s[0].toUpperCase() + s.slice(1)}
                </button>
              ))}
            </div>
            <span className="small">
              Each round practices on 32 examples; {thousands(state.roundsPerPass)} rounds make one pass through its{' '}
              {thousands(state.trainCount)} digits.
            </span>
          </div>
        )}
      </div>

      {state.collapse && dismissedSession !== state.session && (
        <CollapseAlert
          collapse={state.collapse}
          onFix={() => {
            send({ type: 'setEta', eta: DEFAULT_ETA })
            send({ type: 'reset' })
          }}
          onDismiss={() => setDismissedSession(state.session)}
        />
      )}

      <div className="pg-main">
        <div className="stack">
          <div className="panel">
            <div className="ptitle">
              <span className="h3">Draw a digit</span>
              <button
                className="lnk"
                style={{ fontSize: 14 }}
                onClick={() => {
                  const blank = new Float32Array(PIXELS)
                  setPadImage(blank)
                  setImage(blank)
                  setUpsideDown(false)
                }}
              >
                Clear
              </button>
            </div>
            <DrawPad
              image={seen}
              seed={padImage}
              upsideDown={upsideDown}
              onChange={(img) => {
                setImage(img)
              }}
            />
            <OutputBars yHat={inked ? fwd.yHat : null} />
            <p className="body" style={{ marginTop: 10, fontSize: 15 }} aria-live="polite">
              {inked
                ? `It reads your ${upsideDown ? 'upside-down ' : ''}digit as ${guess} (${fmt(fwd.yHat[guess], 2)}).`
                : 'Draw a digit to see what it says.'}
            </p>
          </div>
          <TrainingExamples
            counts={state.labelCounts}
            total={state.trainCount}
            changes={state.changes}
            onChange={(changes) => send({ type: 'setExamples', changes })}
          />
        </div>

        <div className="stack">
          <NetworkDiagram
            network={state.network}
            fwd={fwd}
            selected={selected}
            onSelect={setSelected}
            caption={`Showing your digit. ${connectionLabel(selected)} is selected (click any line to pick another). Showing the 3 strongest connections into each neuron.`}
          />
          <WeightPictures network={state.network} round={state.round} />
        </div>

        <div className="stack">
          <ConnectionInspector
            network={state.network}
            initialNetwork={state.initialNetwork}
            fwd={fwd}
            selected={selected}
            onSetWeight={(value) => send({ type: 'setWeight', ...selected, value })}
          />
          <ThingsToTry
            state={state}
            network={state.network}
            image={image}
            upsideDown={upsideDown}
            setUpsideDown={setUpsideDown}
            send={send}
          />
        </div>
      </div>

      <div className="pg-charts">
        <AccuracyChart points={state.accuracyHistory} roundsPerPass={state.roundsPerPass} />
        <LossChart points={state.lossHistory} roundsPerPass={state.roundsPerPass} />
        {state.lastTest && <ConfusionChart confusion={state.lastTest.confusion} />}
      </div>
    </>
  )
}
