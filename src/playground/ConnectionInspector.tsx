import { useState } from 'react'
import type { ForwardResult, Network } from '../engine/network'
import { connectionLabel, fmt, sourceActivity, sourceName, strength, targetName, type Connection } from './connection'

type Tab = 'words' | 'numbers' | 'math'
const STEP = 0.1

interface Props {
  network: Network
  initialNetwork: Network
  fwd: ForwardResult
  selected: Connection
  onSetWeight: (value: number) => void
}

export function ConnectionInspector({ network, initialNetwork, fwd, selected, onSetWeight }: Props) {
  const [tab, setTab] = useState<Tab>('numbers')
  const now = strength(network, selected)
  const start = strength(initialNetwork, selected)
  const flow = sourceActivity(fwd, selected)
  const push = now * flow
  // While typing, the field shows the draft; otherwise it follows the live strength.
  const [draft, setDraft] = useState<string | null>(null)

  const src = sourceName(selected)
  const dst = targetName(selected)
  const isOutput = selected.layer === 2
  const target = isOutput ? `the ${dst} output` : `neuron ${dst}`
  const source = selected.layer === 0 ? `the ${src}` : `neuron ${src}`
  const z = fwd.layers[selected.layer].z[selected.to]
  const a = fwd.layers[selected.layer].a[selected.to]

  const commit = () => {
    const value = Number((draft ?? '').replace('−', '-'))
    if (draft !== null && draft.trim() !== '' && Number.isFinite(value)) onSetWeight(value)
    setDraft(null)
  }

  return (
    <div className="panel">
      <div className="tabs" role="tablist">
        {(['words', 'numbers', 'math'] as const).map((t) => (
          <button key={t} role="tab" aria-selected={tab === t} onClick={() => setTab(t)}>
            {t[0].toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>
      <div className="h3" style={{ marginBottom: 6 }}>
        {connectionLabel(selected)}
      </div>

      {tab === 'words' && (
        <p className="body" style={{ fontSize: 15.5 }}>
          This connection carries {source}&rsquo;s {selected.layer === 0 ? 'ink' : 'firing'} to {target}.{' '}
          {now >= 0 ? `It excites: when ${src} is active, ${target} is pushed up.` : `It inhibits: when ${src} is active, ${target} is pushed down.`}{' '}
          {Math.abs(now - start) < 1e-9
            ? 'Its weight hasn’t changed since the start.'
            : `Its weight is now ${Math.abs(now) > Math.abs(start) ? 'bigger' : 'smaller'} in size than at the start.`}{' '}
          {flow === 0
            ? `For your digit ${src} is ${selected.layer === 0 ? 'blank' : 'silent'}, so this connection plays no part.`
            : `For your digit it ${push > 0 ? 'pushes up' : 'pushes down'} by ${fmt(Math.abs(push), 2)}.`}
        </p>
      )}

      {tab === 'numbers' && (
        <>
          <div className="kv">
            <span>Weight now</span>
            <span className="m">{fmt(now)}</span>
          </div>
          <div className="kv">
            <span>Weight at the start</span>
            <span className="m">{fmt(start)}</span>
          </div>
          <div className="kv">
            <span>
              {src} {selected.layer === 0 ? 'ink' : 'firing'} for your digit
            </span>
            <span className="m">{fmt(flow, 2)}</span>
          </div>
        </>
      )}

      {tab === 'math' && (
        <div className="math">
          <div>
            z<sub>{dst}</sub> = Σ w·a + b = {fmt(z)}
          </div>
          <div>
            this term: w·a = {fmt(now)} × {fmt(flow)} = {fmt(push)}
          </div>
          <div>
            {isOutput ? (
              <>
                ŷ<sub>{dst}</sub> = σ(z) = {fmt(a)}
              </>
            ) : (
              <>
                a<sub>{dst}</sub> = ReLU(z) = {fmt(a)}
              </>
            )}
          </div>
        </div>
      )}

      <div className="btns" style={{ marginTop: 12, gap: 8 }}>
        <button className="iconbtn" aria-label="Decrease the weight by 0.1" onClick={() => onSetWeight(now - STEP)}>
          −
        </button>
        <input
          className="field"
          aria-label={`Weight of ${connectionLabel(selected)}`}
          value={draft ?? fmt(now)}
          onFocus={() => setDraft(fmt(now))}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={commit}
          onKeyDown={(e) => e.key === 'Enter' && e.currentTarget.blur()}
        />
        <button className="iconbtn" aria-label="Increase the weight by 0.1" onClick={() => onSetWeight(now + STEP)}>
          +
        </button>
      </div>
      <p className="small" style={{ marginTop: 8 }}>
        Change it by hand and watch the outputs. Practice will keep adjusting it.
      </p>
    </div>
  )
}
