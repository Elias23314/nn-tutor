import type { Collapse } from '../engine/trainer'
import { fmt, pct } from './connection'

export function CollapseAlert({ collapse, onFix, onDismiss }: { collapse: Collapse; onFix: () => void; onDismiss: () => void }) {
  return (
    <div className="alert" role="alert">
      <svg width="28" height="28" viewBox="0 0 28 28" aria-hidden="true">
        <path d="M14 3l12 21H2z" fill="none" stroke="var(--ink)" strokeWidth="2" strokeLinejoin="round" />
        <path d="M14 11v6" stroke="var(--ink)" strokeWidth="2.2" strokeLinecap="round" />
        <circle cx="14" cy="20.5" r="1.4" fill="var(--ink)" />
      </svg>
      <div className="stack" style={{ gap: 10 }}>
        <div className="h3">Its mistake fell, but it stopped reading digits.</div>
        <p className="body">
          At learning rate {collapse.eta}, the mistake went from <span className="mono">{fmt(collapse.lossFrom)}</span> to{' '}
          <span className="mono">{fmt(collapse.lossTo)}</span> because every output went toward 0. Nine of every ten goals
          are 0, so &ldquo;never fire&rdquo; scores well on the number it was trained to shrink. It now reads{' '}
          {pct(collapse.accuracy)} of test digits correctly
          {collapse.silentHidden2 > 0 &&
            `, and ${collapse.silentHidden2} of the 16 neurons in hidden layer 2 have gone silent for every test digit`}
          .
        </p>
        <div className="btns">
          <button className="btn" onClick={onFix}>
            Set learning rate to 2 and reset
          </button>
          <button className="btn sec" onClick={onDismiss}>
            Keep exploring
          </button>
        </div>
      </div>
    </div>
  )
}
