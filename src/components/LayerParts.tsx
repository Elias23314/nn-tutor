// Pieces every layer page shares, from the mockups.
import { useId, useState, type ReactNode } from 'react'

const LOOP = ['Predict', 'Measure', 'Find the slope', 'Step', 'Repeat']

export function LayerHead({
  n,
  name,
  tag,
  title,
  lede,
  loopOn = [],
  loopCaption,
}: {
  n: number
  name: string
  tag: string
  title: string
  lede: string
  /** Loop stages to highlight. */
  loopOn?: string[]
  /** Leave out to hide the learning loop, on a page that doesn't cover it yet. */
  loopCaption?: string
}) {
  return (
    <div className="head">
      <div>
        <div className="where">
          <div className="num">{n}</div>
          <div className="name">{name}</div>
          <span className="tag">{tag}</span>
        </div>
        <h1>{title}</h1>
        <p className="lede">{lede}</p>
      </div>
      {loopCaption && (
      <div className="loop">
        <div className="row" aria-label={`The learning loop: ${LOOP.join(', then ')}`}>
          {LOOP.map((s, i) => (
            <span key={s} style={{ display: 'contents' }}>
              {i > 0 && (
                <span className="la" aria-hidden="true">
                  →
                </span>
              )}
              <span className={`s${loopOn.includes(s) ? ' on' : ''}`}>{s}</span>
            </span>
          ))}
        </div>
        <div className="cap">{loopCaption}</div>
      </div>
      )}
    </div>
  )
}

export function BrainTable({
  topic,
  rows,
  so,
}: {
  topic: string
  rows: [ReactNode, ReactNode, ReactNode][]
  so: ReactNode
}) {
  return (
    <div className="brain">
      <div className="bh">
        <span className="h3">Is it like a brain? {topic}</span>
        <span className="small">Where the analogy holds, and where it breaks</span>
      </div>
      <div className="brain-scroll">
        <table>
          <thead>
            <tr>
              <th style={{ width: '27%' }}>In this network</th>
              <th style={{ width: '27%' }}>In the brain</th>
              <th>Where it breaks</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i}>
                {r.map((cell, j) => (
                  <td key={j}>{cell}</td>
                ))}
              </tr>
            ))}
            <tr className="so">
              <td colSpan={3}>{so}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  )
}

/** What the next layer is about, shown once the sentence is checked. */
export interface NextUp {
  /** "Layer 2: How it works" */
  label: string
  text: ReactNode
}

/**
 * "Say it in one sentence": fill in the blank, then compare with the model sentence, which sums up
 * the whole layer. Once checked, `next` introduces how the next layer builds on it.
 * `accepts` decides whether an answer counts as matching.
 */
export function SayItCard({
  before,
  after,
  answer,
  accepts,
  next,
}: {
  before: string
  after: string
  answer: string
  accepts: (text: string) => boolean
  next?: NextUp
}) {
  const [text, setText] = useState('')
  const [checked, setChecked] = useState<null | boolean>(null)
  const id = useId()
  const model = `${before}${answer}${after}`
  return (
    <form
      className="repack"
      onSubmit={(e) => {
        e.preventDefault()
        if (text.trim()) setChecked(accepts(text.trim()))
      }}
    >
      <label className="k" htmlFor={id}>
        Say it in one sentence
      </label>
      <div className="qq">
        {before}
        <input
          id={id}
          className="blank"
          value={text}
          placeholder="type a word"
          autoComplete="off"
          onChange={(e) => {
            setText(e.target.value)
            setChecked(null)
          }}
        />
        {after}
      </div>
      {checked === null && (
        <button className="btn sec sm" style={{ alignSelf: 'flex-start' }} disabled={!text.trim()}>
          Check
        </button>
      )}
      {checked !== null && (
        <div className="model" aria-live="polite">
          <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">
            <circle cx="8" cy="8" r="8" fill="#fff" />
            {checked ? (
              <path d="M4.5 8.2l2.3 2.3 4.7-4.9" fill="none" stroke="var(--ink)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
            ) : (
              <path d="M5.5 5.5l5 5M10.5 5.5l-5 5" stroke="var(--ink)" strokeWidth="1.8" strokeLinecap="round" />
            )}
          </svg>
          {checked ? 'Matches the model sentence: ' : 'Compare with the model sentence: '}“{model}”
        </div>
      )}
      {checked !== null && next && (
        <div className="next-up">
          <span className="k">Up next · {next.label}</span>
          <p>{next.text}</p>
        </div>
      )}
    </form>
  )
}

/** "Quick check from the last layer": recall one word from the previous layer's sentence. */
export function QuickCheck({ before, after, options, answer }: { before: string; after: string; options: string[]; answer: string }) {
  const [picked, setPicked] = useState<string | null>(null)
  const right = picked === answer
  return (
    <div className="retr" role="group" aria-label="Quick check from the last layer">
      <span className="k">Quick check from the last layer</span>
      <span>
        {before}___{after}
      </span>
      <div className="chips">
        {options.map((o) => (
          <button
            key={o}
            className={`chip${picked === o && right ? ' right' : ''}`}
            aria-pressed={picked === o}
            disabled={right}
            onClick={() => setPicked(o)}
          >
            {o}
          </button>
        ))}
      </div>
      <span className="small" aria-live="polite">
        {picked === null ? '' : right ? 'Answered' : 'Not quite. Try another.'}
      </span>
    </div>
  )
}

export function LockIcon({ color = 'var(--ink)' }: { color?: string }) {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
      <rect x="2" y="6" width="10" height="7" rx="1.5" fill={color} />
      <path d="M4.5 6V4.5a2.5 2.5 0 0 1 5 0V6" fill="none" stroke={color} strokeWidth="1.5" />
    </svg>
  )
}

export function RightIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">
      <circle cx="8" cy="8" r="8" fill="var(--ink)" />
      <path d="M4.5 8.2l2.3 2.3 4.7-4.9" fill="none" stroke="#fff" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

export function WrongIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">
      <circle cx="8" cy="8" r="7" fill="#fff" stroke="var(--ink)" strokeWidth="1.6" />
      <path d="M5.5 5.5l5 5M10.5 5.5l-5 5" stroke="var(--ink)" strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  )
}
