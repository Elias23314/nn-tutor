import { useState } from 'react'
import { NO_CHANGES, type ExampleChanges } from '../engine/trainer'
import { thousands } from './connection'

interface Props {
  counts: number[]
  total: number
  changes: ExampleChanges
  onChange: (changes: ExampleChanges) => void
}

const ORIGINAL_TOTAL = 60_000

export function TrainingExamples({ counts, total, changes, onChange }: Props) {
  const [mode, setMode] = useState<'none' | 'remove' | 'relabel'>('none')
  const [from, setFrom] = useState(3)
  const [to, setTo] = useState(8)
  const max = Math.max(...counts, 1)
  const changed = changes.removed.length > 0 || changes.relabel !== null

  const toggleRemoved = (d: number) => {
    const removed = changes.removed.includes(d) ? changes.removed.filter((r) => r !== d) : [...changes.removed, d]
    onChange({ ...changes, removed: removed.sort() })
  }

  return (
    <div className="panel">
      <div className="h3">Training examples</div>
      <p className="small" style={{ margin: '4px 0 12px' }}>
        Its whole experience: {thousands(total)} digits
      </p>
      {counts.map((c, d) => (
        <div className={`cnt${changes.removed.includes(d) ? ' off' : ''}`} key={d}>
          <span className="mono">{d}</span>
          <div className="minibar">
            <i style={{ width: `${(c / max) * 100}%` }} />
          </div>
          <span className="mono" style={{ textAlign: 'right' }}>
            {thousands(c)}
          </span>
        </div>
      ))}

      <div className="btns" style={{ marginTop: 14 }}>
        <button
          className="btn sec sm"
          aria-expanded={mode === 'remove'}
          onClick={() => setMode(mode === 'remove' ? 'none' : 'remove')}
        >
          Remove a digit
        </button>
        <button
          className="btn sec sm"
          aria-expanded={mode === 'relabel'}
          onClick={() => setMode(mode === 'relabel' ? 'none' : 'relabel')}
        >
          Relabel some
        </button>
      </div>

      {mode === 'remove' && (
        <div className="picker">
          <span>Take every example of these digits out:</span>
          <div className="digits">
            {counts.map((_, d) => (
              <button key={d} aria-pressed={changes.removed.includes(d)} onClick={() => toggleRemoved(d)}>
                {d}
              </button>
            ))}
          </div>
        </div>
      )}

      {mode === 'relabel' && (
        <div className="picker">
          <label>
            Label half of the{' '}
            <select value={from} onChange={(e) => setFrom(Number(e.target.value))}>
              {counts.map((_, d) => (
                <option key={d}>{d}</option>
              ))}
            </select>
            s as{' '}
            <select value={to} onChange={(e) => setTo(Number(e.target.value))}>
              {counts.map((_, d) => (
                <option key={d}>{d}</option>
              ))}
            </select>
          </label>
          <button
            className="btn sm"
            style={{ alignSelf: 'flex-start' }}
            disabled={from === to}
            onClick={() => onChange({ ...changes, relabel: { from, to, fraction: 0.5 } })}
          >
            Relabel them
          </button>
        </div>
      )}

      {changed && (
        <p className="small" style={{ marginTop: 10 }}>
          {changes.relabel && `Half of its ${changes.relabel.from}s are labeled ${changes.relabel.to}. `}
          Changing its examples starts practice over.
        </p>
      )}
      {changed && (
        <button
          className="lnk"
          style={{ fontSize: 14.5, marginTop: 10 }}
          onClick={() => {
            setMode('none')
            onChange(NO_CHANGES)
          }}
        >
          Back to all {thousands(ORIGINAL_TOTAL)}
        </button>
      )}
    </div>
  )
}
